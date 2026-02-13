# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model_causal import WanModelCausal
from .modules.t5 import T5EncoderModel
from .modules.vae2_2 import Wan2_2_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.utils import best_output_size, masks_like


class WanTI2VCausal:

    def __init__(
        self,
        config,
        checkpoint_dir,
        pt_dir=None,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu
        self.t5_cache = {}

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_2_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModelcasual from {checkpoint_dir}")
        self.model = WanModelCausal.from_pretrained(checkpoint_dir)
        if pt_dir is not None:
            logging.info(f"Loading model weights from {pt_dir}")
            self.model.load_state_dict(torch.load(pt_dir, map_location='cpu'), strict=False)
            
        self.model = self._configure_model(
            model=self.model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt
        self.model.to(self.device)
        self.text_encoder.model.to(self.device)

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def generate(self,
                 input_prompt,
                 img=None,
                 size=(1280, 704),
                 max_area=704 * 1280,
                 num_conditional_frames=1,
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
        assert offload_model==False
        return self.t2v(
            input_prompt=input_prompt,
            img=img,
            size=size,
            frame_num=frame_num,
            shift=shift,
            num_conditional_frames=num_conditional_frames,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model)
    
    def generate_dmd(self,
                    input_prompt,
                    img=None,
                    size=(1280, 704),
                    max_area=704 * 1280,
                    num_conditional_frames=1,
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
        assert offload_model==False
        return self.t2v_dmd(
            input_prompt=input_prompt,
            img=img,
            size=size,
            frame_num=frame_num,
            shift=shift,
            num_conditional_frames=num_conditional_frames,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model)
    
    def read_and_process_video(
        self,
        image_or_video_path,
        size=(1280, 704),
        num_conditional_frames=1,
        resize=True,
        extract_from_last=False,
    ):
        # return a encoded latent tensor
        # stage1, convert to uint8 space, either from image or video
        if image_or_video_path.endswith('.mp4'):
            vr = VideoReader(image_or_video_path, ctx=cpu(0), num_threads=2)
            total_len = len(vr)
            start_frame = total_len - num_conditional_frames if extract_from_last else 0
            vr.seek(start_frame)
            frame_ids = list(range(start_frame, start_frame + num_conditional_frames))
            frame_data = vr.get_batch(frame_ids).asnumpy() # T, H, W, C
            if resize:
                resized_frames = []
                for frame in frame_data:
                    img = Image.fromarray(frame)
                    img = img.resize(size, Image.LANCZOS)
                    resized_frames.append(np.array(img))
                frame_data = np.stack(resized_frames)
        elif image_or_video_path.endswith('.jpg') or image_or_video_path.endswith('.png'):
            img = Image.open(image_or_video_path).convert('RGB')
            if resize:
                img = img.resize(size, Image.LANCZOS)
            img = np.array(img).astype(np.uint8)
            frame_data = np.expand_dims(img, axis=0)  # Add batch dimension
        else:
            raise ValueError("Unsupported file format. Please provide a .mp4, .jpg, or .png file.")

        frame_data = torch.from_numpy(frame_data).float().div_(255.0).sub_(0.5).div_(0.5).to(self.device)
        frame_data = frame_data.permute(3, 0, 1, 2) #  C, T, H, W
        z = self.vae.encode([frame_data])[0].unsqueeze(0)
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
        

    def t2v(self,
            input_prompt,
            img=None,
            size=(1280, 704),
            frame_num=121,
            num_conditional_frames=1,
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
        if img is not None:
            cond_latent = self.read_and_process_video(img, size=size, num_conditional_frames=num_conditional_frames, resize=True, extract_from_last=extract_from_last)
            cond_latent_frame = cond_latent.shape[2]
        else:
            cond_latent = None
            cond_latent_frame = 0
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            # self.text_encoder.model.to(self.device)
            context = self.cache_t5_encode(input_prompt)
            context_null = self.cache_t5_encode(n_prompt)
            # if offload_model:
            #     self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = torch.randn(
                1,
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)

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
                from wan.modules.block_attention import get_flex_causal_block_mask_for_prefill
                attention_mask = get_flex_causal_block_mask_for_prefill(cond_latent_frame, block_size)
                self.model(cond_latent, t=torch.zeros(B, (cond_latent_frame * H * W) // 4, device=self.device, dtype=torch.float32), 
                           **arg_c, prefill=True, block_size=block_size, attention_mask=attention_mask)


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
            
            x0 = torch.cat(output_frames, dim=2)  # [B, C, T, H, W]
            if cond_latent is not None:
                x0 = torch.cat([cond_latent, x0], dim=2)  # [B, C, T, H, W]
            self.model.clean_cache()
            if self.rank == 0:
                videos = self.vae.decode(list(x0))

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def t2v_dmd(self,
                input_prompt,
                img=None,
                size=(1280, 704),
                frame_num=121,
                num_conditional_frames=1,
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
        if img is not None:
            cond_latent = self.read_and_process_video(img, size=size, num_conditional_frames=num_conditional_frames, resize=True, extract_from_last=extract_from_last)
            cond_latent_frame = cond_latent.shape[2]
        else:
            cond_latent = None
            cond_latent_frame = 0
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            # self.text_encoder.model.to(self.device)
            context = self.cache_t5_encode(input_prompt)
            context_null = self.cache_t5_encode(n_prompt)
            # if offload_model:
            #     self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = torch.randn(
                1,
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)

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

            # sample videos
            latents = noise
            B, C, T, H, W = latents.shape
            arg_c = {'context': context}
            block_size = H * W // 4
            if cond_latent is not None:
                from wan.modules.block_attention import get_flex_causal_block_mask_for_prefill
                attention_mask = get_flex_causal_block_mask_for_prefill(cond_latent_frame, block_size)
                self.model(cond_latent, t=torch.zeros(B, (cond_latent_frame * H * W) // 4, device=self.device, dtype=torch.float32), 
                           **arg_c, prefill=True, block_size=block_size, attention_mask=attention_mask)


            output_frames = []
            for latent_frame_idx in tqdm(range(cond_latent_frame, T)):
                latent_model_input = latents[:, :, latent_frame_idx:latent_frame_idx+1, :, :]
                block_args = dict(block_size=block_size, block_idx=latent_frame_idx)
                timesteps = [1.0, 15/16, 5/6, 5/8][:sampling_steps]
                timesteps = [*timesteps, 0]
                for _, (t_cur, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
                    timestep = torch.ones(B, (1 * H * W) // 4, device=self.device, dtype=torch.float32) * t_cur * self.num_train_timesteps
                    timestep_in = timestep
                    forward_input = latent_model_input
                    arg_c_in = {'context': context}
                    v_pred = self.model(forward_input, t=timestep_in, **arg_c_in, **block_args)

                    latent_model_input = latent_model_input - t_cur * v_pred
                    if t_next > 1e-5:
                        latent_model_input = (1 - t_next) * latent_model_input + t_next * torch.randn_like(latent_model_input)
                output_frames.append(latent_model_input)
                self.model(latent_model_input, t=torch.zeros_like(timestep), **arg_c, cache=True, **block_args)
            
            x0 = torch.cat(output_frames, dim=2)  # [B, C, T, H, W]
            if cond_latent is not None:
                x0 = torch.cat([cond_latent, x0], dim=2)  # [B, C, T, H, W]
            self.model.clean_cache()
            if self.rank == 0:
                videos = self.vae.decode(list(x0))

        del noise, latents
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
