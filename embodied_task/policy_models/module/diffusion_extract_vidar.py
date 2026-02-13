from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import wan
from diffsynth import ModelManager, WanVideoPipeline, load_state_dict
from transformers import T5EncoderModel
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import best_output_size, masks_like


class DiTFeatureExtractorVidar(nn.Module):
    def __init__(
        self,
        ckpt_dir: str,
        task: str = "ti2v-5B",  #
        pt_dir: Optional[str] = None,  #
        device_id: int = 0,  #
        device=None,
        # offload_model: bool = False,
    ):
        super().__init__()

        # 1.
        cfg = WAN_CONFIGS[task]

        # 2.
        if "t2v" in task:
            model_cls = wan.WanT2V
        elif "ti2v" in task:
            model_cls = wan.WanTI2V
        else:  # i2v
            model_cls = wan.WanI2V

        # 3.
        print(f"ckt_dir: {ckpt_dir} pt_dir: {pt_dir} device_id: {device_id}")
        self.model = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            pt_dir=pt_dir,
            device_id=device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            convert_model_dtype=True,
            # offload_model=False,
        )  #
        self.model.model.to("cpu")
        self.model.vae.model.to("cpu")

        print(f"init checkpoint: {ckpt_dir}, task: {task}, pt_dir: {pt_dir}")
        self.task = task
        self._last_prompt = None  #
        self._last_prompt_embeds = None  #
        self.eval()

    def update_device(self, device: torch.device):
        """Update the internal device_id and move relevant modules to the new device."""
        self.device = device
        self.model.device = device

    @torch.no_grad()
    def _preprocess_for_i2v(
        self,
        input_prompt: str,
        img,
        max_area: int = 640 * 480,
        frame_num: int = 45,
        n_prompt: str = "",
        seed: int = -1,
        offload_model: bool = True,
    ):
        """
        Preprocess inputs for I2V generation.

        Returns:
            dict containing:
                - 'z': encoded image latent [C, T_z, H_z, W_z]
                - 'noise': initial noise [C, T_z, H_z, W_z]
                - 'context': text embedding for prompt
                - 'context_null': text embedding for negative prompt
                - 'seq_len': sequence length for DiT
                - 'ow', 'oh': output width/height
                - 'seed_g': torch.Generator
        """
        import math
        import random
        import sys

        import torchvision.transforms.functional as TF
        from PIL import Image

        # self.device = self.model.device
        print(f"debug input_prompt: {input_prompt}")

        ih, iw = img.height, img.width
        dh, dw = (
            self.model.patch_size[1] * self.model.vae_stride[1],
            self.model.patch_size[2] * self.model.vae_stride[2],
        )
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

        # center-crop
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        assert img.width == ow and img.height == oh

        self.device = self.device
        # to tensor and normalize chw-> 1 c 1 hw
        img_tensor = (
            TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)
        )  # [1, 3, H, W])  # [3, 1, H, W]
        print(f"img device (after t5): {img_tensor.device}")
        print(f"img shape (before vae encode): {img_tensor.shape}")

        # print(f"z")
        z = self.model.vae.encode([img_tensor])  # [C, T=1, H_z, W_z]
        print(f"z length: {len(z)}")
        print(f"z shape: {z[0].shape}")
        # compute sequence length
        F = frame_num
        seq_len = (
            ((F - 1) // self.model.vae_stride[0] + 1)
            * (oh // self.model.vae_stride[1])
            * (ow // self.model.vae_stride[2])
            // (self.model.patch_size[1] * self.model.patch_size[2])
        )
        seq_len = int(math.ceil(seq_len / self.model.sp_size)) * self.model.sp_size

        # initialize noise
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            self.model.vae.model.z_dim,
            (F - 1) // self.model.vae_stride[0] + 1,
            oh // self.model.vae_stride[1],
            ow // self.model.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device,
        )

        # encode prompts
        if input_prompt == self._last_prompt:
            #
            context = [t.to(self.device) for t in self._last_prompt_embeds["context"]]
            context_null = [
                t.to(self.device) for t in self._last_prompt_embeds["context_null"]
            ]
        else:
            #
            if not self.model.t5_cpu:
                self.model.text_encoder.model.to(self.device)
                context = self.model.text_encoder([input_prompt], self.device)
                context_null = self.model.text_encoder([n_prompt], self.device)
                if offload_model:
                    self.model.text_encoder.model.cpu()
            else:
                context = self.model.text_encoder([input_prompt], torch.device("cpu"))
                context_null = self.model.text_encoder([n_prompt], torch.device("cpu"))
                context = [t.to(self.device) for t in context]
                context_null = [t.to(self.device) for t in context_null]

            #
            self._last_prompt = input_prompt
            self._last_prompt_embeds = {
                "context": [t.cpu() for t in context],
                "context_null": [t.cpu() for t in context_null],
            }

        return {
            "z": z,
            "noise": noise,
            "context": context,
            "context_null": context_null,
            "seq_len": seq_len,
            "seed_g": seed_g,
            "frame_num": frame_num,
        }

    def forward(
        self,
        z,
        noise,
        context,
        context_null,
        seq_len: int,
        shift: float = 5.0,
        sample_solver: str = "unipc",
        sampling_steps: int = 20,  # timestep
        guide_scale: float = 5.0,
        offload_model: bool = True,
        seed_g=None,
        step_time: int = 1,
        use_layer_idx: int = 1,
        all_layer: bool = True,
    ):
        """
        Run DiT denoising loop using preprocessed latents and embeddings.

        Returns:
            video_tensor: [C, N, H, W] in [-1, 1]
        """
        import gc

        import torch.distributed as dist
        from tqdm import tqdm
        # self.device = self.model.device

        # Initialize scheduler
        if sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.model.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            # print(f"num_train_timesteps: {self.model.num_train_timesteps}")
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift
            )
            timesteps = sample_scheduler.timesteps

        # Initial latent: blend clean image latent with noise
        # print(f"noise shape: {noise.shape}, z shape: {z[0].shape}")# BCTHW
        B = z[0].shape[0]
        mask1, mask2 = masks_like([noise], zero=True)

        latent = (1.0 - mask2[0]) * z[0] + mask2[0] * noise
        arg_c = {
            "context": [context[0]],
            "seq_len": seq_len,
        }

        arg_null = {
            "context": context_null,
            "seq_len": seq_len,
        }

        if offload_model or getattr(self, "init_on_cpu", False):
            self.model.model.to(self.device)
            torch.cuda.empty_cache()

        # Denoising loop
        with torch.amp.autocast("cuda", dtype=self.model.param_dtype), torch.no_grad():
            for i, t in enumerate(tqdm(timesteps)):
                if step_time is not None and i == step_time - 1:
                    complete = False
                else:
                    complete = True

                latent_model_input = [latent.to(self.device)]
                timestep = torch.stack([t]).to(self.device)

                # Build timestep embedding (match original logic) BCTHW
                temp_ts = (mask2[0][0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat(
                    [temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep]
                )
                timestep_embed = temp_ts.unsqueeze(0)
                timestep_embed = timestep_embed.repeat(B, 1)  # [B, seq_len]

                # Conditional prediction
                noise_pred_cond = self.step_dit(
                    latent_model_input,
                    t=timestep_embed,
                    complete=complete,
                    use_layer_idx=use_layer_idx,
                    all_layer=all_layer,
                    **arg_c,
                )[0]
                if not complete:
                    break

                if offload_model:
                    torch.cuda.empty_cache()
                # Unconditional prediction
                noise_pred_uncond = self.step_dit(
                    latent_model_input,
                    t=timestep_embed,
                    complete=complete,
                    use_layer_idx=use_layer_idx,
                    all_layer=all_layer,
                    **arg_null,
                )[0]

                if offload_model:
                    torch.cuda.empty_cache()
                # Classifier-free guidance
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

                # Scheduler step
                temp_x0 = sample_scheduler.step(
                    noise_pred, t, latent, return_dict=False, generator=seed_g
                )[0]
                latent = temp_x0.squeeze(0)
                latent = (1.0 - mask2[0]) * z[0] + mask2[0] * latent

                del latent_model_input, timestep, timestep_embed

        if offload_model:
            self.model.model.cpu()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        del noise, latent
        del sample_scheduler
        return noise_pred_cond

    def step_dit(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        complete=False,
        use_layer_idx=1,
        all_layer=True,
    ):
        r"""
        Forward pass through the diffusion model
        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model.model.model_type == "i2v":
            assert y is not None
        # params
        device = self.model.model.patch_embedding.weight.device
        if self.model.model.freqs.device != device:
            self.model.model.freqs = self.model.model.freqs.to(device)
        B = t.size(0)
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings BCTHW
        x = [self.model.model.patch_embedding(u) for u in x]
        # print(f"debug: x[0].shape = {x[0].shape}")
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(
            [
                torch.cat([u, u.new_zeros(B, seq_len - u.size(1), u.size(2))], dim=1)
                for u in x
            ]
        )

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.model.model.time_embedding(
                sinusoidal_embedding_1d(self.model.model.freq_dim, t)
                .unflatten(0, (bt, seq_len))
                .float()
            )
            e0 = self.model.model.time_projection(e).unflatten(
                2, (6, self.model.model.dim)
            )
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        # print(f"context[0] shape before padding: {context[0].shape}")
        context = self.model.model.text_embedding(
            torch.stack(
                [
                    torch.cat(
                        [
                            u,
                            u.new_zeros(
                                B, self.model.model.text_len - u.shape[1], u.shape[2]
                            ),
                        ],
                        dim=1,
                    )
                    for u in context
                ]
            )
        )

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.model.model.freqs,
            context=context,
            context_lens=context_lens,
        )
        #
        feature_list = []
        L = len(self.model.model.blocks)
        mid_start = 1 * L // 2  #

        for i, block in enumerate(self.model.model.blocks):
            if not complete and i >= mid_start:
                if len(feature_list) <= use_layer_idx:
                    # print(f"layer {i} shape: {x.shape}")
                    feature_list.append(x)
                if len(feature_list) == use_layer_idx + 1:
                    break

            x = block(x, **kwargs)

        if not complete:
            if all_layer:
                x_feat = torch.cat(feature_list, dim=-1)  # [B, L, D_total]
            else:
                x_feat = feature_list[-1]  # [B, L, D]

            # reshape to [B, F, H, W, D]
            B = x_feat.shape[0]
            f, h, w = grid_sizes[0].tolist()  # assume same shape in batch
            x_feat = x_feat[:, : f * h * w, :].reshape(B, f, h, w, -1)
            return [x_feat]

        # head
        x = self.model.model.head(x, e)

        # unpatchify
        x = self.model.model.unpatchify(x, grid_sizes)
        return [u.float() for u in x]


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x
