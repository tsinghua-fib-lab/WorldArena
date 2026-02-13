from typing import Dict, Optional, Tuple, Union
from diffusers.models import UNetSpatioTemporalConditionModel
from diffusers import TextToVideoSDPipeline, StableVideoDiffusionPipeline
import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import random
from transformers import AutoTokenizer, CLIPTextModelWithProjection
import numpy as np
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)



class Diffusion_feature_extractor(nn.Module):
    def __init__(
        self,
        pipeline=None,
        tokenizer=None,
        text_encoder=None,
        position_encoding=True,
    ):
        super().__init__()
        self.pipeline = pipeline if pipeline is not None else StableVideoDiffusionPipeline()
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained("/cephfs/shared/llm/clip-vit-base-patch32",use_fast=False)
        self.text_encoder = text_encoder if text_encoder is not None else CLIPTextModelWithProjection.from_pretrained("/cephfs/shared/llm/clip-vit-base-patch32")
        self.num_frames = 16
        self.position_encoding = position_encoding

    @torch.no_grad()
    def forward(
            self,
            pixel_values: torch.Tensor,
            texts,
            timestep: Union[torch.Tensor, float, int],
            extract_layer_idx: Union[torch.Tensor, float, int],
            use_latent = False,
            all_layer = False,
            step_time = 1,
            max_length = 20,
    ):

        height = self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor //3
        width = self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor //3
        #height=480
        #width=640
        self.pipeline.vae.eval()
        self.pipeline.image_encoder.eval()
        device = self.pipeline.unet.device
        dtype = self.pipeline.vae.dtype
        #print('dtype:',dtype)
        vae = self.pipeline.vae

        num_videos_per_prompt=1

        batch_size = pixel_values.shape[0]

        pixel_values = rearrange(pixel_values, 'b f c h w-> (b f) c h w').to(dtype)

        with torch.no_grad():
            # texts, tokenizer, text_encoder, img_cond=None, img_cond_mask=None, img_encoder=None, position_encode=True, use_clip=False, max_length=20
            encoder_hidden_states = self.encode_text(texts, self.tokenizer, self.text_encoder, position_encode=self.position_encoding, use_clip=True, max_length=max_length)
        encoder_hidden_states = encoder_hidden_states.to(dtype)
        image_embeddings = encoder_hidden_states

        needs_upcasting = self.pipeline.vae.dtype == torch.float16 and self.pipeline.vae.config.force_upcast
        #if needs_upcasting:
        #    self.pipeline.vae.to(dtype=torch.float32)
        #    pixel_values.to(dtype=torch.float32)
        if pixel_values.shape[-3] == 4:
            image_latents = pixel_values/vae.config.scaling_factor
        else:
            image_latents = self.pipeline._encode_vae_image(pixel_values, device, num_videos_per_prompt, False)
        image_latents = image_latents.to(image_embeddings.dtype)

        #print('dtype:', image_latents.dtype)

        #if needs_upcasting:
        #    self.pipeline.vae.to(dtype=torch.float16)

        #num_frames = self.pipeline.unet.config.num_frames
        num_frames = 16
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        fps=4
        motion_bucket_id=127
        added_time_ids = self.pipeline._get_add_time_ids(
            fps,
            motion_bucket_id,
            0,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            False,
        )
        added_time_ids = added_time_ids.to(device)

        self.pipeline.scheduler.set_timesteps(timestep, device=device)
        timesteps = self.pipeline.scheduler.timesteps

        num_channels_latents = self.pipeline.unet.config.in_channels
        latents = self.pipeline.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            None,
            None,
        )
        print(f"latents shape: {latents.shape} image_latents shape: {image_latents.shape}")
        for i, t in enumerate(timesteps):
            #print('step:',i)
            if i == step_time - 1:
                complete = False
            else:
                complete = True
            #print('complete:',complete)

            latent_model_input = latents
            latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)

            # Concatenate image_latents over channels dimention
            # latent_model_input = torch.cat([mask, latent_model_input, image_latents], dim=2)
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
            #print('latent_model_input_shape:',latent_model_input.shape)
            #print('image_embeddings_shape:',image_embeddings.shape)

            # predict the noise residual
            # print('extract_layer_idx:',extract_layer_idx)
            # print('latent_model_input_shape:',latent_model_input.shape)
            # print('encoder_hidden_states:',image_embeddings.shape)
            feature_pred = self.step_unet(
                latent_model_input,
                t,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                use_layer_idx=extract_layer_idx,
                all_layer = all_layer,
                complete = complete,
            )[0]
            # feature_pred = self.pipeline.unet(
            #     latent_model_input,
            #     t,
            #     encoder_hidden_states=image_embeddings,
            #     added_time_ids=added_time_ids,
            #     return_dict=False,
            # )[0]

            # print('feature_pred_shape:',feature_pred.shape)

            if not complete:
                break

            latents = self.pipeline.scheduler.step(feature_pred, t, latents).prev_sample

        return feature_pred

    def step_unet(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        use_layer_idx: int = 5,
        all_layer: bool = False,
        complete: bool = False,
    ) :
        r"""
        The [`UNetSpatioTemporalConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
            added_time_ids: (`torch.Tensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] instead
                of a plain tuple.
        Returns:
            [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] is
                returned, otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.pipeline.unet.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.pipeline.unet.time_embedding(t_emb)

        time_embeds = self.pipeline.unet.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.pipeline.unet.add_embedding(time_embeds)
        emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0)
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        # 2. pre-process
        sample = self.pipeline.unet.conv_in(sample)

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.pipeline.unet.down_blocks:
            #print('sample_shape:',sample.shape)
            #print('emb_shape:', emb.shape)
            #print('encoder_hidden_states_shape:', encoder_hidden_states.shape)
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.pipeline.unet.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        feature_list = []

        # 5. up
        for i, upsample_block in enumerate(self.pipeline.unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                print(f"sample shape before upsample block {i}: {sample.shape}")
                print(f"res_samples shape before upsample block {i}: {len(res_samples)}")
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )
            if i < use_layer_idx:
                factor = 2**(use_layer_idx - i)
                feature_list.append(torch.nn.functional.interpolate(sample,scale_factor=factor))
            #print('up_sample_idx:',i)
            if i == use_layer_idx and not complete:
                feature_list.append(sample)
                break

        if not complete:
            if all_layer:
                sample = torch.cat(feature_list, dim=1)
                sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])
            else:
                sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])
            # 6. post-process
            return (sample,)

        else:
            sample = self.pipeline.unet.conv_norm_out(sample)
            sample = self.pipeline.unet.conv_act(sample)
            sample = self.pipeline.unet.conv_out(sample)

            # 7. Reshape back to original shape
            sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

            return (sample,)

    @torch.no_grad()
    def encode_text(self, texts, tokenizer, text_encoder, img_cond=None, img_cond_mask=None, img_encoder=None, position_encode=True, use_clip=False, max_length=20):
        def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
            """
            embed_dim: output dimension for each position
            pos: a list of positions to be encoded: size (M,)
            out: (M, D)
            """
            assert embed_dim % 2 == 0
            omega = np.arange(embed_dim // 2, dtype=np.float64)
            omega /= embed_dim / 2.
            omega = 1. / 10000**omega  # (D/2,)

            pos = pos.reshape(-1)  # (M,)
            out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

            emb_sin = np.sin(out) # (M, D/2)
            emb_cos = np.cos(out) # (M, D/2)

            emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
            return emb
        
        # max_length = args.clip_token_length
        with torch.no_grad():
            if use_clip:
                inputs = tokenizer(texts, padding='max_length', return_tensors="pt",truncation=True, max_length=max_length).to(text_encoder.device)
                outputs = text_encoder(**inputs)
                encoder_hidden_states = outputs.last_hidden_state # (batch, 30, 512)
                ###### will be used in the dp ##########
                # self.text_embeds = outputs.text_embeds
                ######################################
                if position_encode:
                    embed_dim, pos_num = encoder_hidden_states.shape[-1], encoder_hidden_states.shape[1]
                    pos = np.arange(pos_num,dtype=np.float64)

                    position_encode = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
                    position_encode = torch.tensor(position_encode, device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype, requires_grad=False)

                    # print("position_encode",position_encode.shape)
                    # print("encoder_hidden_states",encoder_hidden_states.shape)

                    encoder_hidden_states += position_encode
                assert encoder_hidden_states.shape[-1] == 512

                if img_encoder is not None:
                    assert img_cond is not None
                    assert img_cond_mask is not None
                    # print("img_encoder",img_encoder.shape)
                    img_cond = img_cond.to(img_encoder.device)
                    if len(img_cond.shape) == 5:
                        img_cond = img_cond.squeeze(1)
                    
                    img_hidden_states = img_encoder(img_cond).image_embeds
                    img_hidden_states[img_cond_mask] = 0.0
                    img_hidden_states = img_hidden_states.unsqueeze(1).expand(-1,encoder_hidden_states.shape[1],-1)
                    assert img_hidden_states.shape[-1] == 512
                    encoder_hidden_states = torch.cat([encoder_hidden_states, img_hidden_states], dim=-1)
                    assert encoder_hidden_states.shape[-1] == 1024
                else:
                    encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=-1)
            
            else:
                inputs = tokenizer(texts, padding='max_length', return_tensors="pt",truncation=True, max_length=32).to(text_encoder.device)
                outputs = text_encoder(**inputs)
                encoder_hidden_states = outputs.last_hidden_state # (batch, 30, 512)
                assert encoder_hidden_states.shape[1:] == (32,1024)

        return encoder_hidden_states

