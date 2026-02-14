# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from base64 import b64encode, b64decode
from contextlib import contextmanager
import numpy as np
from PIL import Image
import copy
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
import torchvision
from PIL import Image
from tqdm import tqdm
from wan.timeutils import ClockContext
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.textimage2video_causal import WanTI2VCausal
import imageio.v3 as iio
from wan.modules.block_attention import get_flex_causal_block_mask_for_prefill, get_flex_block_mask_chunk_prefill


"""
一个有状态的causal diffusion推理代码。
有状态带来的性能提升来自于:
1. 增量decode, 只decode当前预测部分, done
2. 增量encode, 只encode当前需要重新prefill的部分
3. 增量prefill, 只prefill当前新的部分 ccccccccgggg -> cccccccccccc
4. 如果存在可能, 使用vae, diffusion流水线, 将vae和模型分在两张卡上, 重叠vae和diffusion

接口：
prefill: 从头prefill
lastprefill: 给定一部分隐帧, 将其prefill为kv cache的最后一部分
gen: 基于当前kv cache继续生成隐帧, 自动kv cache, 只返回增量的帧
clean_state: 清除所有状态, 包括kv cache, t5 cache, 显存

"""

def extract_first_k_frames(video_path, k):
    # Read the first k frames from the video
    frames = []
    reader = iio.imiter(video_path, plugin="FFMPEG")
    for idx, frame in enumerate(reader):
        if idx >= k:
            break
        frames.append(torch.tensor(frame, dtype=torch.uint8))

    jpeg_message_list = []
    for frame in frames:
        jpeg_tensor = torchvision.io.encode_jpeg(frame.permute(2, 0, 1))
        jpeg_message_list.append(b64encode(jpeg_tensor.numpy().tobytes()).decode("utf-8"))

    return jpeg_message_list


class WanTI2VCausalServer(WanTI2VCausal):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_pil = torchvision.transforms.ToPILImage()
        self.clean_latent_cache = None
        self.clean_image_cache = None

    def generate(self,
                 input_prompt,
                 img=None,
                 size=(1280, 704),
                 max_area=704 * 1280,
                 num_conditional_frames=1,
                 clean_cache=False,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # assert offload_model==False
        return self.t2v(
            input_prompt=input_prompt,
            img=img,
            size=size,
            frame_num=frame_num,
            shift=shift,
            num_conditional_frames=num_conditional_frames,
            clean_cache=clean_cache,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model)
    
    def read_and_process_video(
        self,
        imgs,
        size=(1280, 704),
        num_conditional_frames=1,
        resize=True,
        extract_from_last=False,
    ):
        if type(imgs) == str:
            imgs = extract_first_k_frames(imgs, num_conditional_frames)
        
        # assert len(imgs) == 1 or len(imgs) % 4 == 0, "under incremental decode, the number of frames should be 4n, or frist frame"
        assert num_conditional_frames >= len(imgs), "unser incremental decode, all current bath fames should be considerd"
        
        # return a encoded latent tensor
        # stage1, convert to uint8 space, either from image or video
        with ClockContext(f"{'frame extract':-^30}"):
            if len(imgs) < num_conditional_frames:
                if extract_from_last:
                    imgs = imgs[-num_conditional_frames:]
                else:
                    imgs = imgs[:num_conditional_frames]

            frame_data = []
            for img in imgs:
                img = torch.frombuffer(copy.copy(b64decode(img)), dtype=torch.uint8)
                frame_data.append(torchvision.io.decode_jpeg(img, mode=torchvision.io.ImageReadMode.RGB))

            if resize:
                resized_frames = []
                for frame in frame_data:
                    img = self.to_pil(frame)
                    img = img.resize(size, Image.LANCZOS)
                    resized_frames.append(torch.from_numpy(np.array(img)).permute(2, 0, 1))
                frame_data = resized_frames
                frame_data = torch.stack(frame_data, dim=1) # C, T, H, W
            else:
                frame_data = torch.stack(frame_data, dim=1) # C, T, H, W

        with ClockContext(f"{'vae encode':-^30}"):
            frame_data = frame_data.float().div_(255.0).sub_(0.5).div_(0.5).to(self.device)
            if self.clean_image_cache is None:
                z = self.vae.encode([frame_data])[0].unsqueeze(0)
                self.clean_image_cache = frame_data
            else:
                num_precat = 5
                actual_cated = self.clean_image_cache[:, -num_precat:, ...].shape[1]
                frame_data = torch.cat([self.clean_image_cache[:, -num_precat:, ...], frame_data], dim=1)
                z = self.vae.encode([frame_data])[0].unsqueeze(0)[:, :, (actual_cated - 1) // 4 + 1:]

        # rec = self.vae.decode(list(z))[0]
        # from wan.utils.utils import save_video
        # save_video(rec[None], 'test.mp4')
        return z
    
    def cache_t5_encode(self, input_prompt):
        if self.t5_cache.get(input_prompt) is not None:
            return self.t5_cache[input_prompt]
        else:
            context = self.text_encoder([input_prompt], self.device)
            self.t5_cache[input_prompt] = context
            return context
    
    def clean_state(self):
        self.model.clean_cache()
        self.clean_image_cache = None
        self.clean_latent_cache = None

    def clean_all_state(self):
        self.clean_state()
        self.t5_cache = {}
        torch.cuda.empty_cache()
    
    def prefill(self, cond_latent, block_size, arg_c):
        B, C, T, H, W = cond_latent.shape
        if self.model.kvcache_len() <= T * block_size:
            attention_mask = get_flex_causal_block_mask_for_prefill(T, block_size)
            return self.model(cond_latent, t=torch.zeros(B, (T * H * W) // 4, device=self.device, dtype=torch.float32), **arg_c, prefill=True, block_size=block_size, attention_mask=attention_mask)
        else: # chunk prefill from last, pop kv cache frist
            self.model.pop_kvcache(T * block_size)
            attention_mask = get_flex_block_mask_chunk_prefill(T, self.model.kvcache_len() // block_size, block_size)
            return self.model(cond_latent, t=torch.zeros(B, (T * H * W) // 4, device=self.device, dtype=torch.float32), **arg_c, chunk_prefill=True, block_size=block_size, attention_mask=attention_mask)

    def t2v(self,
            input_prompt,
            img=None,
            size=(1280, 704),
            frame_num=121,
            num_conditional_frames=1,
            clean_cache=False,
            extract_from_last=False,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=50,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            img:
                path of image or video
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        print(f"t2v: num conditional frames {num_conditional_frames}, frame num {frame_num}")

        with ClockContext(f"{'clean cache':-^30}"):
            if num_conditional_frames <= 1 or clean_cache:
                print("state cleaned")
                self.clean_state()

        with ClockContext(f"{'frame extract & vae encode':-^30}"):
            if img is not None:
                cond_latent = self.read_and_process_video(img, size=size, num_conditional_frames=num_conditional_frames, resize=True, extract_from_last=extract_from_last)
                if self.clean_latent_cache is None:
                    self.clean_latent_cache = cond_latent
                else:
                    self.clean_latent_cache = torch.cat((self.clean_latent_cache, cond_latent), dim=2)
                cond_latent_frame = self.clean_latent_cache.shape[2]
            else:
                cond_latent = None
                cond_latent_frame = 0
        # preprocess
        F = frame_num
        num_new_fames = F - num_conditional_frames
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        with ClockContext(f"{'t5 encode':-^30}"):
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context = self.cache_t5_encode(input_prompt)
                context_null = self.cache_t5_encode(n_prompt)
                if offload_model:
                    self.text_encoder.model.cpu()
            else:
                context = self.text_encoder([input_prompt], torch.device('cpu'))
                context_null = self.text_encoder([n_prompt], torch.device('cpu'))
                context = [t.to(self.device) for t in context]
                context_null = [t.to(self.device) for t in context_null]
        noise = torch.randn(1, *target_shape, dtype=torch.float32, device=self.device, generator=seed_g)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise
            B, C, T, H, W = latents.shape
            arg_c = {'context': context}
            arg_null = {'context': context_null}
            block_size = H * W // 4
            if cond_latent is not None:
                with ClockContext(f"{'prefill':-^30}"):
                    self.prefill(cond_latent, block_size, arg_c)

            with ClockContext(f"{'gengen':-^30}"):
                output_frames = []
                for latent_frame_idx in tqdm(range(cond_latent_frame, T)):
                    sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
                    timesteps = sample_scheduler.timesteps
                    latent_model_input = latents[:, :, latent_frame_idx:latent_frame_idx+1, :, :]
                    block_args = dict(block_size=block_size, block_idx=latent_frame_idx)
                    for _, t in enumerate(tqdm(timesteps)):
                        timestep = torch.ones(B, (1 * H * W) // 4, device=self.device, dtype=torch.float32) * t
                        timestep_in = timestep.repeat(2, 1)
                        forward_input = latent_model_input.repeat(2, 1, 1, 1, 1)
                        arg_c_in = {'context': context + context_null}
                        noise_pred = self.model(forward_input, t=timestep_in, **arg_c_in, **block_args)
                        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)

                        noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                        temp_x0 = sample_scheduler.step(
                            noise_pred,
                            t,
                            latent_model_input,
                            return_dict=False,
                            generator=seed_g)[0]
                        latent_model_input = temp_x0
                    output_frames.append(latent_model_input)
                    self.model(latent_model_input, t=torch.zeros_like(timestep), **arg_c, cache=True, **block_args)

            with ClockContext(f"{'vae decode':-^30}"):
                x0 = torch.cat(output_frames, dim=2)  # [B, C, T, H, W]
                num_pre_cat = 2
                x0_cat_last = torch.cat([self.clean_latent_cache[:, :, -num_pre_cat:, ...], x0], dim=2)
                if self.rank == 0:
                    videos_cat_last = self.vae.decode(list(x0_cat_last))[0]
                    videos_inc_last = videos_cat_last[:, -num_new_fames:, ...]
        
        with ClockContext(f"{'epiloge':-^30}"):
            del noise, latents
            del sample_scheduler
            if offload_model:
                gc.collect()
                torch.cuda.synchronize()
            if dist.is_initialized():
                dist.barrier()

        return videos_inc_last if self.rank == 0 else None
