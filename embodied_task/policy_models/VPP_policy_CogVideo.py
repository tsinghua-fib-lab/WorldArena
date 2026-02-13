import logging
from typing import Dict, Optional, Tuple
from functools import partial
from torch import einsum, nn
from einops import rearrange, repeat
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import einops
from policy_models.edm_diffusion.score_wrappers import GCDenoiser

from policy_models.module.clip_lang_encoder import LangClip
from policy_models.edm_diffusion.gc_sampling import *
from policy_models.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler
from policy_models.module.Video_Former import Video_Former_2D, Video_Former_3D
from diffusers import StableVideoDiffusionPipeline
from policy_models.module.diffusion_extract import Diffusion_feature_extractor
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from policy_models.module.diffusion_extract_wow import DiTFeatureExtractorWoW
from policy_models.module.diffusion_extract_wow import LightningModelForDataProcess
from policy_models.module.diffusion_extract_vidar import DiTFeatureExtractorVidar
from policy_models.module.diffusion_extract_cogvideo import DiTFeatureExtractorCogVideo

from PIL import Image

# from policy_models.module.diffusion_extract_vidar import DiTFeatureExtractorVidar

logger = logging.getLogger(__name__)


def load_primary_models(pretrained_model_path, eval=False):
    if eval:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            pretrained_model_path, torch_dtype=torch.bfloat16
        )
    else:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path)
    return (
        pipeline,
        None,
        pipeline.feature_extractor,
        pipeline.scheduler,
        pipeline.video_processor,
        pipeline.image_encoder,
        pipeline.vae,
        pipeline.unet,
    )


class VPP_Policy(pl.LightningModule):
    """
    The lightning module used for training.
    """

    def __init__(
        self,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        latent_dim: int = 512,
        multistep: int = 10,
        sampler_type: str = "ddim",
        num_sampling_steps: int = 10,
        sigma_data: float = 0.5,
        sigma_min: float = 0.001,
        sigma_max: float = 80,
        noise_scheduler: str = "exponential",
        sigma_sample_density_type: str = "loglogistic",
        use_lr_scheduler: bool = True,
        act_window_size: int = 10,
        use_text_not_embedding: bool = False,
        seed: int = 42,
        pretrained_model_path: str = "/cephfs/shared/gyj/ckpt/svd_pre/checkpoint-100000",
        text_encoder_path: str = "/home/disk2/gyj/hyc_ckpt/llm/clip-vit-base-patch32",  # 在wow情况下使用在dataprocess模型里
        use_position_encoding: bool = True,
        Former_depth: int = 3,
        Former_heads: int = 8,
        Former_dim_head: int = 64,
        Former_num_time_embeds: int = 13,
        num_latents: int = 3,
        use_Former: str = "3d",
        timestep: int = 20,
        max_length: int = 20,
        extract_layer_idx: int = 1,
        use_all_layer: bool = False,
        obs_seq_len: int = 1,
        action_dim: int = 7,
        action_seq_len: int = 49,
        # 新加入的参数
        video_model: str = "vpp_original",
        base_model_folder: str = "/ML-vePFS/protected/myd/wow-world-model/dit_models/checkpoints/WoW-1-Wan-1.3B-2M",
        custom_dit_path: Optional[str] = None,
        has_image_input: bool = True,
        vae_path: str = None,
        image_encoder_path: str = None,
        tiled=False,
        tile_size_height=32,
        tile_size_width=32,
        tile_stride_height=16,
        tile_stride_width=16,
    ):
        super(VPP_Policy, self).__init__()
        self.latent_dim = latent_dim
        self.use_all_layer = use_all_layer
        self.use_position_encoding = use_position_encoding

        self.act_window_size = act_window_size
        self.action_dim = action_dim

        self.timestep = timestep
        self.extract_layer_idx = extract_layer_idx
        self.use_Former = use_Former
        self.Former_num_time_embeds = Former_num_time_embeds
        self.max_length = max_length
        self.video_model = video_model

        # 替换原来的 condition_dim_list 部分为：
        if self.video_model == "vpp_original":
            # UNet 输出维度（保持原逻辑）
            condition_dim_list = [1280, 1280, 1280, 640]
            sum_dim = sum(
                condition_dim_list[i + 1] for i in range(self.extract_layer_idx + 1)
            )
            condition_dim = (
                condition_dim_list[self.extract_layer_idx + 1]
                if not self.use_all_layer
                else sum_dim
            )
        elif self.video_model == "vpp_wow":
            # DiT 输出维度：假设 dim=1024（❗必须根据实际模型确认❗）
            dit_hidden_dim = 1536  # ←←← 从你的 DiT config 中获取这个值！
            if not self.use_all_layer:
                condition_dim = dit_hidden_dim
            else:
                condition_dim = dit_hidden_dim * (self.extract_layer_idx + 1)
        elif self.video_model == "vpp_vidar":
            # print(f"video_model: {self.video_model} use")
            dit_hidden_dim = 3072  # ←←← 从你的 DiT config 中获取这个值！ 这个打印看看
            if not self.use_all_layer:
                condition_dim = dit_hidden_dim
            else:
                condition_dim = dit_hidden_dim * (self.extract_layer_idx + 1)
        elif self.video_model == "vpp_robomaster":
            # print(f"video_model: {self.video_model} use")
            dit_hidden_dim = 3072  # ←←← 从你的 DiT config 中获取这个值！ 这个打印看看
            if not self.use_all_layer:
                condition_dim = dit_hidden_dim
            else:
                condition_dim = dit_hidden_dim * (self.extract_layer_idx + 1)
        elif self.video_model == "vpp_cogvideo":
            dit_hidden_dim = 3072  # ←←← 从你的 DiT config 中获取这个值！ 这个打印看看
            if not self.use_all_layer:
                condition_dim = dit_hidden_dim
            else:
                condition_dim = dit_hidden_dim * (self.extract_layer_idx + 1)

        if use_Former == "3d":
            self.Video_Former = Video_Former_3D(
                dim=latent_dim,
                depth=Former_depth,
                dim_head=Former_dim_head,
                heads=Former_heads,
                num_time_embeds=Former_num_time_embeds,
                num_latents=num_latents,
                condition_dim=condition_dim,
                use_temporal=True,
                num_frame=Former_num_time_embeds,
            )
        elif use_Former == "2d":
            self.Video_Former = Video_Former_2D(
                dim=latent_dim,
                depth=Former_depth,
                dim_head=Former_dim_head,
                heads=Former_heads,
                num_time_embeds=Former_num_time_embeds,
                num_latents=num_latents,
                condition_dim=condition_dim,
            )
        else:
            self.Video_Former = nn.Linear(condition_dim, latent_dim)

        # print("use_Former:", self.use_Former)
        # print("use_all_layer", self.use_all_layer)

        self.seed = seed
        self.use_lr_scheduler = use_lr_scheduler
        # goal encoders
        if self.video_model == "vpp_original":
            self.language_goal = LangClip(model_name="ViT-B/32").to(self.device)
        elif self.video_model == "vpp_wow":
            self.language_goal = (
                None  # WoW model uses pre-extracted prompt embeddings directly
            )
        elif self.video_model == "vpp_vidar":
            self.language_goal = (
                None  # Vidar model uses pre-extracted context embeddings directly
            )
        elif self.video_model == "vpp_robomaster":
            self.language_goal = (
                None  # Vidar model uses pre-extracted context embeddings directly
            )
        elif self.video_model == "vpp_cogvideo":
            self.language_goal = (
                None  # Vidar model uses pre-extracted context embeddings directly
            )

        if self.video_model == "vpp_original":
            (
                pipeline,
                tokenizer,
                feature_extractor,
                train_scheduler,
                vae_processor,
                text_encoder,
                vae,
                unet,
            ) = load_primary_models(pretrained_model_path, eval=True)

            # text_encoder = CLIPTextModelWithProjection.from_pretrained("/cephfs/shared/llm/clip-vit-base-patch32")
            # tokenizer = AutoTokenizer.from_pretrained("/cephfs/shared/llm/clip-vit-base-patch32", use_fast=False)
            text_encoder = CLIPTextModelWithProjection.from_pretrained(
                text_encoder_path
            )
            tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, use_fast=False)

            text_encoder = text_encoder.to(self.device).eval()

            for param in pipeline.image_encoder.parameters():
                param.requires_grad = False
            for param in text_encoder.parameters():
                param.requires_grad = False

            for param in pipeline.vae.parameters():
                param.requires_grad = False
            for param in pipeline.unet.parameters():
                param.requires_grad = False

            pipeline = pipeline.to(self.device)
            pipeline.unet.eval()
            self.TVP_encoder = Diffusion_feature_extractor(
                pipeline=pipeline,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                position_encoding=self.use_position_encoding,
            )
        elif self.video_model == "vpp_wow":

            self.TVP_encoder = DiTFeatureExtractorWoW(
                base_model_folder=base_model_folder,
                custom_dit_path=custom_dit_path,  # 可选参数，如果无自定义模型路径则可省略或设为None
                position_encoding=self.use_position_encoding,
                has_image_input=has_image_input,
                # num_frames=num_frames
            )
            self.img_encoder_path = image_encoder_path
            for param in self.TVP_encoder.dit.parameters():  # 假设模型属性叫 dit_model
                param.requires_grad = False
            if image_encoder_path is not None:
                self.model_processor = LightningModelForDataProcess(
                    text_encoder_path=text_encoder_path,
                    vae_path=vae_path,
                    image_encoder_path=image_encoder_path,
                    tiled=tiled,
                    tile_size=(tile_size_height, tile_size_width),
                    tile_stride=(tile_stride_height, tile_stride_width),
                    frames=action_seq_len,  # 这个是压缩后的帧数
                )
                for param in self.model_processor.parameters():
                    param.requires_grad = False

        elif self.video_model == "vpp_vidar":
            self.TVP_encoder = DiTFeatureExtractorVidar(
                ckpt_dir=base_model_folder,
                pt_dir=custom_dit_path,  # 可选参数，如果无自定义模型路径则可省略或设为None
                # num_frames=num_frames
            )
            # print(f"device: {self.device} use")
            for param in self.TVP_encoder.pipe.model.parameters():
                param.requires_grad = False
        elif self.video_model == "vpp_robomaster":
            from policy_models.module.diffusion_extract_robomaster import (
                DiTFeatureExtractorRobomaster,
            )

            self.TVP_encoder = DiTFeatureExtractorRobomaster(
                ckpt_dir=base_model_folder,
                pt_dir=custom_dit_path,  # 可选参数，如果无自定义模型路径则可省略或设为None
            )
            for param in self.TVP_encoder.pipe.transformer.parameters():
                param.requires_grad = False
            for param in self.TVP_encoder.pipe.vae.parameters():
                param.requires_grad = False
            for param in self.TVP_encoder.pipe.text_encoder.parameters():
                param.requires_grad = False
        elif self.video_model == "vpp_cogvideo":
            self.TVP_encoder = DiTFeatureExtractorCogVideo(
                base_model_folder=base_model_folder,
                custom_dit_path=custom_dit_path,  # 可选参数，如果无自定义模型路径则可省略或设为None
            )
            for param in self.TVP_encoder.pipe.transformer.parameters():
                param.requires_grad = False
            for param in self.TVP_encoder.pipe.vae.parameters():
                param.requires_grad = False
            for param in self.TVP_encoder.pipe.text_encoder.parameters():
                param.requires_grad = False

        # self.TVP_encoder = self.TVP_encoder.to(self.device)
        # policy network
        if self.video_model == "vpp_original":
            self.model = (
                GCDenoiser(
                    action_dim=action_dim,
                    obs_dim=latent_dim,
                    goal_dim=512,
                    num_tokens=num_latents,
                    goal_window_size=1,
                    obs_seq_len=obs_seq_len,
                    act_seq_len=action_seq_len,
                    device=self.device,
                    sigma_data=0.5,
                )
                .to(self.device)
                .to(torch.bfloat16)
            )
        else:
            self.model = (
                GCDenoiser(
                    action_dim=action_dim,
                    obs_dim=latent_dim,
                    goal_dim=4096,
                    num_tokens=num_latents,
                    goal_window_size=1,
                    obs_seq_len=obs_seq_len,
                    act_seq_len=action_seq_len,
                    device=self.device,
                    sigma_data=0.5,
                )
                .to(self.device)
                .to(torch.bfloat16)
            )

        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_hyperparameters()
        # diffusion stuff
        self.sampler_type = sampler_type
        self.num_sampling_steps = num_sampling_steps
        self.noise_scheduler = noise_scheduler
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_sample_density_type = sigma_sample_density_type
        # for inference
        self.rollout_step_counter = 0
        self.multistep = multistep
        self.latent_goal = None
        self.plan = None
        self.use_text_not_embedding = use_text_not_embedding
        # print_model_parameters(self.perceptual_encoder.perceiver_resampler)
        # for clip loss ground truth plot
        self.ema_callback_idx = None

        for param in self.model.inner_model.proprio_emb.parameters():
            param.requires_grad = False
        for param in self.model.inner_model.goal_emb.parameters():
            param.requires_grad = False
        self.model.inner_model.pos_emb.requires_grad = False

    def process_device(self):
        # vpp_original: 需要移动 pipeline 和 text_encoder
        if self.video_model == "vpp_original":
            self.TVP_encoder.pipeline = self.TVP_encoder.pipeline.to(self.device)
            self.TVP_encoder.text_encoder = self.TVP_encoder.text_encoder.to(
                self.device
            )
        # vpp_wow: DiT 模型本身是 nn.Module，直接移动整个 TVP_encoder
        elif self.video_model == "vpp_wow":
            self.TVP_encoder = self.TVP_encoder.to(self.device)
            self.TVP_encoder.dit = self.TVP_encoder.dit.to(self.device)
            self.TVP_encoder.pipe = self.TVP_encoder.pipe.to(self.device)
            self.model = self.model.to(self.device)
            self.model.inner_model = self.model.inner_model.to(self.device)
            if self.img_encoder_path is not None:
                self.model_processor = self.model_processor.to(self.device)
                self.model_processor.pipe = self.model_processor.pipe.to(self.device)
            self.TVP_encoder.update_device(self.device)

        elif self.video_model == "vpp_vidar":
            self.TVP_encoder = self.TVP_encoder.to(self.device)
            # self.TVP_encoder.pipe = self.TVP_encoder.pipe.to(self.device)
            self.TVP_encoder.pipe.model = self.TVP_encoder.pipe.model.to(self.device)
            self.TVP_encoder.pipe.vae.model = self.TVP_encoder.pipe.vae.model.to(
                self.device
            )
            self.TVP_encoder.pipe.vae.scale = [
                t.to(self.device) if isinstance(t, torch.Tensor) else t
                for t in self.TVP_encoder.pipe.vae.scale
            ]
            self.model = self.model.to(self.device)
            # print(f"move TVP_encoder to device {self.device}")
            self.TVP_encoder.update_device(self.device)
        elif self.video_model == "vpp_robomaster":
            self.TVP_encoder = self.TVP_encoder.to(self.device)
            self.TVP_encoder.pipe.transformer = self.TVP_encoder.pipe.transformer.to(
                self.device
            )
            # self.TVP_encoder.pipe.transformer.time_embedding = self.TVP_encoder.pipe.transformer.time_embedding.to(self.device)
            self.TVP_encoder.pipe.text_encoder = self.TVP_encoder.pipe.text_encoder.to(
                self.device
            )
            self.TVP_encoder.pipe.vae = self.TVP_encoder.pipe.vae.to(self.device)
            self.model = self.model.to(self.device)
            self.TVP_encoder.update_device(self.device)
            # print(f"move TVP_encoder to device {self.device}")
        elif self.video_model == "vpp_cogvideo":
            self.TVP_encoder = self.TVP_encoder.to(self.device)
            self.TVP_encoder.pipe.transformer = self.TVP_encoder.pipe.transformer.to(
                self.device
            )
            self.TVP_encoder.pipe.text_encoder = self.TVP_encoder.pipe.text_encoder.to(
                self.device
            )
            self.TVP_encoder.pipe.vae = self.TVP_encoder.pipe.vae.to(self.device)
            self.model = self.model.to(self.device)
            self.TVP_encoder.update_device(self.device)
            # print(f"move TVP_encoder to device {self.device}")

    def configure_optimizers(self):
        """
        Initialize optimizers and learning rate schedulers based on model configuration.
        """
        # Configuration for models using transformer weight decay
        """optim_groups = self.action_decoder.model.inner_model.get_optim_groups(
            weight_decay=self.optimizer_config.transformer_weight_decay
        )"""
        optim_groups = [
            {
                "params": self.model.inner_model.parameters(),
                "weight_decay": self.optimizer_config.transformer_weight_decay,
            },
            {
                "params": self.Video_Former.parameters(),
                "weight_decay": self.optimizer_config.transformer_weight_decay,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.optimizer_config.learning_rate,
            betas=self.optimizer_config.betas,
        )

        # Optionally initialize the scheduler
        if self.use_lr_scheduler:
            lr_configs = OmegaConf.create(self.lr_scheduler)
            scheduler = TriStageLRScheduler(optimizer, lr_configs)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer

    def on_before_zero_grad(self, optimizer=None):
        total_grad_norm = 0.0
        total_param_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
            total_param_norm += p.norm().item() ** 2
        total_grad_norm = total_grad_norm**0.5
        total_param_norm = total_param_norm**0.5

        self.log(
            "train/grad_norm",
            total_grad_norm,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self.log(
            "train/param_norm",
            total_param_norm,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

    def training_step(
        self,
        dataset_batch: Dict[str, Dict],
    ) -> torch.Tensor:  # type: ignore
        """
        Compute and return the training loss for the MDT Agent.
        The training loss consists of the score matching loss of the diffusion model
        and the contrastive loss of the CLIP model for the multimodal encoder.

        Args:
            batch: Dictionary containing the batch data for each modality.
            batch_idx: Index of the batch. used for compatibility with pytorch lightning.
            dataloader_idx: Index of the dataloader. used for compatibility with pytorch lightning.

        Returns:
            loss tensor
        """
        total_loss, action_loss = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )
        if self.video_model == "vpp_original":
            predictive_feature, latent_goal = self.extract_predictive_feature(
                dataset_batch
            )
        elif self.video_model == "vpp_wow":
            predictive_feature, latent_goal = self.extract_predictive_feature_wow(
                dataset_batch
            )
        elif self.video_model == "vpp_vidar":
            predictive_feature, latent_goal = self.extract_predictive_feature_vidar(
                dataset_batch
            )
        elif self.video_model == "vpp_robomaster":
            predictive_feature, latent_goal = (
                self.extract_predictive_feature_robomaster(dataset_batch)
            )
        elif self.video_model == "vpp_cogvideo":
            predictive_feature, latent_goal = self.extract_predictive_feature_cogvideo(
                dataset_batch
            )
        # predictive_feature, latent_goal= self.extract_predictive_feature(dataset_batch)
        actions = dataset_batch["actions"].squeeze(1)
        # print(f"actions shape:{actions.shape}")
        # print(f"actions shape:{actions.shape}")
        act_loss, sigmas, noise = self.diffusion_loss(
            predictive_feature,
            latent_goal,
            actions,
        )
        print(f"act_loss:{act_loss}")
        # print(f"[DEBUG] act_loss.requires_grad: {act_loss.requires_grad}")
        # print(f"[DEBUG] act_loss.grad_fn: {act_loss.grad_fn}")
        action_loss += act_loss
        total_loss += act_loss

        total_bs = actions.shape[0]
        # print(f"act_loss:{total_loss}")
        self._log_training_metrics(action_loss, total_loss, total_bs)
        return total_loss

    @torch.no_grad()
    def validation_step(
        self, dataset_batch: Dict[str, Dict]
    ) -> Dict[str, torch.Tensor]:  # type: ignore
        """
        Compute and log the validation losses and additional metrics.
        During the validation step, the diffusion model predicts the next action sequence given the current state

        Args:
            batch: Dictionary containing the batch data for each modality.
            batch_idx: Index of the batch. used for compatibility with pytorch lightning.
            dataloader_idx: Index of the dataloader. used for compatibility with pytorch lightning.

        Returns:
            Dictionary containing the sampled plans of plan recognition and plan proposal module, as well as the
            episode indices.
        """
        output = {}
        val_total_act_loss_pp = torch.tensor(0.0).to(self.device)
        # Compute the required embeddings
        # predictive_feature, latent_goal= self.extract_predictive_feature(dataset_batch)
        if self.video_model == "vpp_original":
            predictive_feature, latent_goal = self.extract_predictive_feature(
                dataset_batch
            )
        elif self.video_model == "vpp_wow":
            predictive_feature, latent_goal = self.extract_predictive_feature_wow(
                dataset_batch
            )
        elif self.video_model == "vpp_vidar":
            predictive_feature, latent_goal = self.extract_predictive_feature_vidar(
                dataset_batch
            )
        elif self.video_model == "vpp_robomaster":
            predictive_feature, latent_goal = (
                self.extract_predictive_feature_vidar_robomaster(dataset_batch)
            )

        # predict the next action sequence
        action_pred = self.denoise_actions(
            torch.zeros_like(latent_goal).to(latent_goal.device),
            predictive_feature,
            latent_goal,
            inference=True,
        )
        dataset_batch["actions"] = dataset_batch["actions"].to(action_pred.device)
        actions = dataset_batch["actions"].squeeze(1)
        # compute the mse action loss
        pred_loss = torch.nn.functional.mse_loss(action_pred, actions)
        val_total_act_loss_pp += pred_loss

        output[f"idx:"] = dataset_batch["idx"]
        output["validation_loss"] = val_total_act_loss_pp
        return output

    def extract_predictive_feature(self, dataset_batch, use_single_view=False):
        """
        Compute the required embeddings for the visual ones and the latent goal.
        """
        # 1. extract the revelant visual observations
        use_single_view = True
        if "rgb_obs" in dataset_batch:
            rgb_static = dataset_batch["rgb_obs"]["rgb_static"].to(self.device)
            rgb_gripper = dataset_batch["rgb_obs"]["rgb_gripper"].to(self.device)
        else:
            rgb_static = dataset_batch["first_frame"].to(self.device)
            # rgb_gripper = dataset_batch["rgb_gripper"].to(self.device)
        # 3. we compute the language goal if the language modality is in the scope
        modality = "lang"
        if "lang_text" in dataset_batch:
            language = dataset_batch["lang_text"]
        else:
            dataset_batch["lang_text"] = dataset_batch["text"]
        if self.use_text_not_embedding:
            latent_goal = self.language_goal(dataset_batch["text"]).to(torch.float32)
        else:
            latent_goal = self.language_goal(dataset_batch["lang"]).to(torch.float32)

        language = dataset_batch["text"]

        num_frames = self.Former_num_time_embeds
        rgb_static = rgb_static.to(self.device)
        rgb_static = rearrange(rgb_static, "b h w c -> b c h w")
        # print("rgb_static_shape:", rgb_static.shape)
        # rgb_gripper = rgb_gripper.to(self.device)
        batch = rgb_static.shape[0]

        with torch.no_grad():
            if use_single_view:
                input_rgb = rgb_static.unsqueeze(1)
                # print("input_rgb_shape:", input_rgb.shape)
                perceptual_features = self.TVP_encoder(
                    input_rgb,
                    language,
                    self.timestep,
                    self.extract_layer_idx,
                    all_layer=self.use_all_layer,
                    step_time=1,
                    max_length=self.max_length,
                )
                # print("perceptual_features_before_shape:", perceptual_features.shape)
                perceptual_features = einops.rearrange(
                    perceptual_features, "b f c h w-> b f c (h w)"
                )
                perceptual_features = einops.rearrange(
                    perceptual_features, "b f c l-> b f l c"
                )
                perceptual_features = perceptual_features[:, :num_frames, :, :]
                # print('perceptual_features_shape:', perceptual_features.shape)

                # perceptual_features, gripper_feature = torch.split(perceptual_features, [batch, batch], dim=0)
                # perceptual_features = torch.cat([perceptual_features, gripper_feature], dim=2)
            else:
                input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)
                language = language + language
                perceptual_features = self.TVP_encoder(
                    input_rgb,
                    language,
                    self.timestep,
                    self.extract_layer_idx,
                    all_layer=self.use_all_layer,
                    step_time=1,
                    max_length=self.max_length,
                )

                perceptual_features = einops.rearrange(
                    perceptual_features, "b f c h w-> b f c (h w)"
                )
                perceptual_features = einops.rearrange(
                    perceptual_features, "b f c l-> b f l c"
                )
                perceptual_features = perceptual_features[:, :num_frames, :, :]
                # print('perceptual_features_shape:', perceptual_features.shape)
                perceptual_features, gripper_feature = torch.split(
                    perceptual_features, [batch, batch], dim=0
                )
                perceptual_features = torch.cat(
                    [perceptual_features, gripper_feature], dim=2
                )

        perceptual_features = perceptual_features.to(torch.float32)  # [28, 16, 512, ]
        perceptual_features = self.Video_Former(perceptual_features)  # [28, 224, 384]
        if self.use_Former == "linear":
            perceptual_features = rearrange(perceptual_features, "b T q d -> b (T q) d")
        predictive_feature = {"state_images": perceptual_features}
        predictive_feature["modality"] = modality
        return predictive_feature, latent_goal

    def extract_predictive_feature_wow(self, dataset_batch):
        """
        Compute predictive features using pre-extracted latents and embeddings (from .mp4_wow.tensors.pth).

        Expected keys in dataset_batch:
            - 'latents': [B, C, T, H, W] or similar latent video representation
            - 'prompt_emb': language prompt embedding, used as latent_goal
            - 'image_emb': optional image embedding dict (may be empty)
            - (others like 'states', 'actions' are ignored here)
        """
        # 1. Use precomputed latents as visual input (instead of raw RGB)
        # print(f"self device: {self.device}")
        latents = (
            dataset_batch["latents"].to(self.device).squeeze(1)
        )  # shape: [B, C, T, H, W] or [B, T, C, H, W] — check your saved format!
        # print("latents shape:", latents.shape)
        prompt_emb = dataset_batch["prompt_emb"]

        prompt_emb = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in prompt_emb.items()
        }
        # 2. Language goal is just prompt_emb
        latent_goal = prompt_emb["context"].to(latents.dtype)
        latent_goal = latent_goal.squeeze(1)
        # print(
        #     f"latent_goal dtype after squeeze: {latent_goal.dtype}, shape: {latent_goal.shape}"
        # )

        # 3. Modality
        modality = "lang"

        # 4. Prepare inputs for TVP_encoder
        batch_size = latents.shape[0]
        num_frames = self.Former_num_time_embeds

        latents_input = einops.rearrange(
            latents, "b c t h w -> b t c h w"
        )  # Ensure correct shape for concatenation

        # Check if image_emb exists
        image_emb = dataset_batch.get("image_emb", None)
        if image_emb is not None:
            # image_emb = image_emb.to(self.device)
            # prompt_emb = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in prompt_emb.items()}
            image_emb = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in image_emb.items()
            }
        else:
            print("Warning: No image_emb found. Proceeding without it.")

        # Flatten time into batch to mimic original TVP call style
        latents_input = einops.rearrange(latents_input, "b t c h w -> (b t) c h w")

        # Dummy language list for TVP_encoder interface (will be replaced by actual prompt_emb usage)
        # dummy_lang_text = [""] * latents_input.shape[0]  # TVP may expect list of strings

        # 🔧 CALL TVP_encoder with latents instead of RGB
        with torch.no_grad():
            perceptual_features = self.TVP_encoder(
                texts=prompt_emb,
                latents=latents,
                image_emb=image_emb,
                cfg_scale=5.0,
                num_inference_steps=20,
                step_time=1,
                seed=None,
                use_layer_idx=self.extract_layer_idx,  # 这里设置你想使用的层索引
                all_layer=self.use_all_layer,  # 设置为True或False，根据你是否想拼接所有层
            )

        # Reshape back to [B, T, ...]
        # 🔍 打印 TVP (DiT) 输出维度
        # print(f"[vpp_wow] DiT TVP output shape: {perceptual_features.shape}")  # e.g. [B*T, C, H, W]

        b, f, h, w, c = perceptual_features.shape
        perceptual_features = einops.rearrange(
            perceptual_features, "b t h w c -> b t c (h w)", b=batch_size
        )
        perceptual_features = einops.rearrange(
            perceptual_features, "b t c l -> b t l c"
        )
        # perceptual_features = perceptual_features[:, :num_frames, :, :]

        perceptual_features = perceptual_features.to(torch.float32)
        # 🔍 打印 Video_Former 输入维度
        # print(
        #     f"[Debug] Video_Former input shape: {perceptual_features.shape}"
        # )  # [B, T, L, C]

        perceptual_features = self.Video_Former(perceptual_features)
        # 🔍 打印 Video_Former 输出维度
        # print(f"[vpp_wow] Video_Former output shape: {perceptual_features.shape}")
        if self.use_Former == "linear":
            perceptual_features = einops.rearrange(
                perceptual_features, "b T q d -> b (T q) d"
            )

        # perceptual_features = perceptual_features.to(latents.dtype)
        # latent_goal = latent_goal.to(perceptual_features.dtype).to(perceptual_features.device)
        # print(f"dtype check: latent_goal dtype {latent_goal.dtype}, perceptual_features dtype {perceptual_features.dtype}")
        # perceptual_features = perceptual_features.to(latents.dtype)
        perceptual_features = perceptual_features.to(torch.float32)
        latent_goal = latent_goal.to(torch.float32)
        predictive_feature = {"state_images": perceptual_features, "modality": modality}

        return predictive_feature, latent_goal

    def extract_predictive_feature_vidar(self, dataset_batch):
        # 1. 从预处理数据中直接取出变量（不包装、不转换）
        z = dataset_batch["z"]  # [B, C, 1, H_z, W_z] 现在是list 在内部转换成【0】
        context = dataset_batch["context"]  # List[Tensor], 长度为 B
        context_null = dataset_batch["context_null"]  # List[Tensor], 长度为 B
        noise = dataset_batch["noise"]  # [B, C, 1, H_z, W_z]

        # 处理 z: 如果是 4D，加 batch 维度
        if z[0].dim() == 4:
            z[0] = z[0].unsqueeze(0)  # [C, 1, H, W] -> [1, C, 1, H, W]
        if noise.dim() == 4:
            noise = noise.unsqueeze(0)
        context = [c.unsqueeze(0) if c.dim() == 2 else c for c in context]
        context_null = [c.unsqueeze(0) if c.dim() == 2 else c for c in context_null]

        seq_len = 3600
        # 🔍 Debug: 打印输入 shape

        latent_goal = context[0]
        # print(f"debug type(latent_goal): {type(latent_goal)}")
        # print(f"debug latent_goal len: {len(latent_goal)}")
        # latent_goal = torch.stack(latent_goal, dim=0)  # [L, D] → [1, L, D]
        latent_goal = latent_goal.to(z[0].dtype).to(self.device)
        # print(f"[Debug] latent_goal shape: {latent_goal.shape}")
        latent_goal = latent_goal  # [1, L, D] → 后续可能 broadcast

        seed_g = torch.Generator(device=self.device)
        # noise = dataset_batch["noise"]      # [B, C, 1, H_z, W_z]
        # seq_len 可能是 int 或 tensor，统一转为 int
        if isinstance(seq_len, torch.Tensor):
            seq_len = seq_len.item()

        # 4. 调用封装好的 TVP_encoder (DiTFeatureExtractorVidar)
        with torch.no_grad():
            perceptual_features = self.TVP_encoder(
                z=z,
                noise=noise,
                context=context,
                context_null=context_null,
                seq_len=seq_len,
                step_time=1,  # 只跑前几层，提取中间特征
                use_layer_idx=self.extract_layer_idx,
                all_layer=self.use_all_layer,
                guide_scale=5.0,  # 不影响特征提取（因 complete=False）
                offload_model=False,  # 假设已在 device 上
                seed_g=seed_g,
            )
        # perceptual_features: [B, F, H, W, C] — 来自 DiT 的中间特征

        # 🔍 Debug: 打印 DiT 输出 shape
        print(f"[Debug] DiT perceptual_features shape: {perceptual_features.shape}")

        batch_size = z[0].shape[0]
        # 注意：perceptual_features 已是 [B, F, H, W, C]
        b, f, h, w, c = perceptual_features.shape
        perceptual_features = perceptual_features.to(torch.float32)
        # Reshape for Video_Former: [B, T, L, C]
        perceptual_features = perceptual_features.view(b, f, h * w, c)  # [B, T, L, C]
        # Pass through Video_Former
        perceptual_features = self.Video_Former(perceptual_features)

        if self.use_Former == "linear":
            perceptual_features = einops.rearrange(
                perceptual_features, "b t q d -> b (t q) d"
            )

        perceptual_features = perceptual_features.to(z[0].dtype)

        predictive_feature = {"state_images": perceptual_features, "modality": "lang"}

        # latent_goal 应该是 [B, L, D]？但你之前用的是单个
        # 如果 batch 内 prompt 相同，可 expand；否则需 stack context
        # 这里简化：假设所有样本共享同一 prompt（常见于 chunked 数据）
        # latent_goal = latent_goal.expand(batch_size, -1, -1)  # [B, L, D]

        return predictive_feature, latent_goal

    def extract_predictive_feature_robomaster(self, dataset_batch):
        self.TVP_encoder.pipe.vae.to("cpu")  # 把vae放到cpu上，节省显存
        self.TVP_encoder.pipe.text_encoder.to(
            "cpu"
        )  # 把text_encoder放到cpu上，节省显存
        # 直接解包新格式字段（名字已对齐）
        latents = dataset_batch["latents"].squeeze(1)
        # print(f"prompt_embeds shape: {dataset_batch['prompt_embeds'].shape} dtype: {dataset_batch['prompt_embeds'].dtype}")
        if len(dataset_batch["prompt_embeds"].shape) == 3:
            latent_goal = dataset_batch["prompt_embeds"][1]
            latent_goal = latent_goal.unsqueeze(0)
            prompt_embeds = dataset_batch["prompt_embeds"]
        else:
            latent_goal = dataset_batch["prompt_embeds"][:, 1]
            # print(f"latent_goal {latent_goal.shape} dtype: {latent_goal.dtype}")
            prompt_embeds = dataset_batch["prompt_embeds"].flatten(0, 1)

        # timesteps = dataset_batch["timesteps"]
        # print(f"tiemsteps {timesteps.shape} dtype: {timesteps.dtype}")
        # print(f"inpaint shape: {dataset_batch['inpaint_latents'].shape} dtype: {dataset_batch['inpaint_latents'].dtype}")
        if len(dataset_batch["inpaint_latents"].shape) == 5:
            inpaint_latents = dataset_batch["inpaint_latents"]
        else:
            inpaint_latents = dataset_batch["inpaint_latents"].squeeze(1)
            inpaint_latents = inpaint_latents.flatten(0, 1)
        flow_latents = None
        # image_rotary_emb = dataset_batch["image_rotary_emb"]
        # print(f"image_rotary_emb len {len(image_rotary_emb)} image_rotary_emb[0] shape: {image_rotary_emb[0].shape}")
        extra_step_kwargs = dataset_batch["extra_step_kwargs"]
        print(f"extra_step_kwargs keys: {list(extra_step_kwargs.keys())}")
        # latent_goal = prompt_embeds.chunks(1)[1]
        print(f"latents shape: {latents.shape}")
        print(f"prompt_embeds shape: {prompt_embeds.shape}")
        print(
            f"inpaint_latents shape: {inpaint_latents.shape} device: {inpaint_latents.device}"
        )

        with torch.no_grad():
            perceptual_features = self.TVP_encoder(
                latents=latents,
                prompt_embeds=prompt_embeds,
                # timesteps=timesteps,
                inpaint_latents=inpaint_latents,
                flow_latents=flow_latents,
                # image_rotary_emb=image_rotary_emb,
                extra_step_kwargs=extra_step_kwargs,
                num_inference_steps=20,
                step_time=1,
                use_layer_idx=self.extract_layer_idx,
                all_layer=self.use_all_layer,
            )[0]

        # 🔍 Debug: 打印 DiT 输出 shape
        print(f"[Debug] DiT perceptual_features shape: {perceptual_features.shape}")

        batch_size = latents.shape[0]
        # 注意：perceptual_features 已是 [B, F, H, W, C]
        b, f, h, w, c = perceptual_features.shape
        perceptual_features = perceptual_features.to(torch.float32)
        # Reshape for Video_Former: [B, T, L, C]
        perceptual_features = perceptual_features.view(b, f, h * w, c)  # [B, T, L, C]
        # Pass through Video_Former
        perceptual_features = self.Video_Former(perceptual_features)

        if self.use_Former == "linear":
            perceptual_features = einops.rearrange(
                perceptual_features, "b t q d -> b (t q) d"
            )

        perceptual_features = perceptual_features.to(torch.float32)
        latent_goal = latent_goal.to(torch.float32)

        predictive_feature = {"state_images": perceptual_features, "modality": "lang"}

        # latent_goal 应该是 [B, L, D]？但你之前用的是单个
        # 如果 batch 内 prompt 相同，可 expand；否则需 stack context
        # 这里简化：假设所有样本共享同一 prompt（常见于 chunked 数据）
        # latent_goal = latent_goal.expand(batch_size, -1, -1)  # [B, L, D]

        return predictive_feature, latent_goal

    def extract_predictive_feature_cogvideo(self, dataset_batch):
        """
        Compute predictive features using pre-extracted latents and embeddings (from .mp4_wow.tensors.pth).

        Expected keys in dataset_batch:
            - 'latents': [B, C, T, H, W] or similar latent video representation
            - 'prompt_emb': language prompt embedding, used as latent_goal
            - (others like 'states', 'actions' are ignored here)
        """
        # 1. Use precomputed latents as visual input (instead of raw RGB)
        # print(f"self device: {self.device}")
        latents = (
            dataset_batch["latents"].to(self.device).squeeze(1)
        )  # shape: [B, C, T, H, W] or [B, T, C, H, W] — check your saved format!
        # print("latents shape:", latents.shape)
        prompt_emb = dataset_batch["prompt_emb"].squeeze(1)

        prompt_emb = prompt_emb.to(self.device)
        # 2. Language goal is just prompt_emb
        latent_goal = prompt_emb.to(latents.dtype)
        latent_goal = latent_goal.squeeze(1)  # [B, L, D]
        # print(
        #     f"latent_goal dtype after squeeze: {latent_goal.dtype}, shape: {latent_goal.shape}"
        # )

        image = dataset_batch["image"].squeeze(1)
        image = [einops.rearrange(img, "h w c -> c h w") for img in image]

        # 3. Modality
        modality = "lang"

        # 4. Prepare inputs for TVP_encoder
        batch_size = latents.shape[0]
        num_frames = self.Former_num_time_embeds

        latents = einops.rearrange(
            latents, "b c t h w -> b t c h w"
        )  # Ensure correct shape for concatenation
        latents_noise = torch.randn_like(latents)

        # Flatten time into batch to mimic original TVP call style

        # Dummy language list for TVP_encoder interface (will be replaced by actual prompt_emb usage)
        # dummy_lang_text = [""] * latents_input.shape[0]  # TVP may expect list of strings

        # 🔧 CALL TVP_encoder with latents instead of RGB
        with torch.no_grad():
            perceptual_features = self.TVP_encoder(
                image=image,
                prompt_embeds=prompt_emb,
                latents=latents_noise,
                num_inference_steps=20,
                step_time=1,
                use_layer_idx=self.extract_layer_idx,  # 这里设置你想使用的层索引
                all_layer=self.use_all_layer,  # 设置为True或False，根据你是否想拼接所有层
            )

        # Reshape back to [B, T, ...]
        # 🔍 打印 TVP (DiT) 输出维度
        # print(f"[vpp_wow] DiT TVP output shape: {perceptual_features.shape}")  # e.g. [B*T, C, H, W]

        b, f, h, w, c = perceptual_features.shape
        perceptual_features = einops.rearrange(
            perceptual_features, "b t h w c -> b t c (h w)", b=batch_size
        )
        perceptual_features = einops.rearrange(
            perceptual_features, "b t c l -> b t l c"
        )
        # perceptual_features = perceptual_features[:, :num_frames, :, :]

        perceptual_features = perceptual_features.to(torch.float32)
        # 🔍 打印 Video_Former 输入维度
        # print(
        #     f"[Debug] Video_Former input shape: {perceptual_features.shape}"
        # )  # [B, T, L, C]

        perceptual_features = self.Video_Former(perceptual_features)
        # 🔍 打印 Video_Former 输出维度
        # print(f"[vpp_wow] Video_Former output shape: {perceptual_features.shape}")
        if self.use_Former == "linear":
            perceptual_features = einops.rearrange(
                perceptual_features, "b T q d -> b (T q) d"
            )

        # perceptual_features = perceptual_features.to(latents.dtype)
        # latent_goal = latent_goal.to(perceptual_features.dtype).to(perceptual_features.device)
        # print(f"dtype check: latent_goal dtype {latent_goal.dtype}, perceptual_features dtype {perceptual_features.dtype}")
        # perceptual_features = perceptual_features.to(latents.dtype)
        perceptual_features = perceptual_features.to(torch.float32)
        latent_goal = latent_goal.to(torch.float32)
        predictive_feature = {"state_images": perceptual_features, "modality": modality}

        return predictive_feature, latent_goal

    def extract_predictive_feature_cogvideo_eval(self, dataset_batch):
        """
        Compute predictive features using pre-extracted latents and embeddings (from .mp4_wow.tensors.pth).

        Expected keys in dataset_batch:
            - 'latents': [B, C, T, H, W] or similar latent video representation
            - 'prompt_emb': language prompt embedding, used as latent_goal
            - (others like 'states', 'actions' are ignored here)
        """

        # 3. Modality
        modality = "lang"

        # 🔧 CALL TVP_encoder with latents instead of RGB
        with torch.no_grad():
            perceptual_features = self.TVP_encoder(
                image=dataset_batch["image"],
                prompt=dataset_batch["prompt"],
                num_inference_steps=dataset_batch["num_inference_steps"],
                height=dataset_batch["height"],
                width=dataset_batch["width"],
                step_time=1,
                use_layer_idx=self.extract_layer_idx,  # 这里设置你想使用的层索引
                all_layer=self.use_all_layer,  # 设置为True或False，根据你是否想拼接所有层
            )

        b, f, h, w, c = perceptual_features.shape
        perceptual_features = einops.rearrange(
            perceptual_features, "b t h w c -> b t c (h w)", b=1
        )
        perceptual_features = einops.rearrange(
            perceptual_features, "b t c l -> b t l c"
        )
        # perceptual_features = perceptual_features[:, :num_frames, :, :]

        perceptual_features = perceptual_features.to(torch.float32)
        # 🔍 打印 Video_Former 输入维度
        # print(
        #     f"[Debug] Video_Former input shape: {perceptual_features.shape}"
        # )  # [B, T, L, C]

        perceptual_features = self.Video_Former(perceptual_features)
        # 🔍 打印 Video_Former 输出维度
        # print(f"[vpp_wow] Video_Former output shape: {perceptual_features.shape}")
        if self.use_Former == "linear":
            perceptual_features = einops.rearrange(
                perceptual_features, "b T q d -> b (T q) d"
            )

        perceptual_features = perceptual_features.to(torch.float32)
        latent_goal = self.TVP_encoder.encode_prompt(dataset_batch["prompt"])
        latent_goal = latent_goal.to(torch.float32)
        predictive_feature = {"state_images": perceptual_features, "modality": modality}

        return predictive_feature, latent_goal

    def _log_training_metrics(self, action_loss, total_loss, total_bs):
        """
        Log the training metrics.
        """
        self.log(
            "train/action_loss",
            action_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=total_bs,
        )
        self.log(
            "train/total_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=total_bs,
        )

    def _log_validation_metrics(self, pred_loss, img_gen_loss, val_total_act_loss_pp):
        """
        Log the validation metrics.
        """
        self.log(
            "val_act/action_loss",
            val_total_act_loss_pp
            / len(self.trainer.datamodule.modalities),  # type:ignore
            sync_dist=True,
        )
        self.log(f"val_act/img_gen_loss_pp", img_gen_loss, sync_dist=True)

    def diffusion_loss(
        self,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the score matching loss given the perceptual embedding, latent goal, and desired actions.
        """
        self.model.train()
        self.model.to(latent_goal.dtype)
        self.model.inner_model.to(latent_goal.dtype)
        # print(f"latent_goal.dtype: {latent_goal.dtype}")
        sigmas = self.make_sample_density()(
            shape=(len(actions),), device=self.device
        ).to(self.device)
        noise = torch.randn_like(actions).to(self.device)
        sigmas = sigmas.to(latent_goal.dtype)
        noise = noise.to(latent_goal.dtype)
        actions = actions.to(latent_goal.dtype)
        # for name, var in [
        #    ("perceptual_emb", perceptual_emb),
        #    ("actions", actions),
        #    ("latent_goal", latent_goal),
        #    ("noise", noise),
        #    ("sigmas", sigmas),
        # ]:
        # print(f"[DEBUG] {name}: type={type(var)}, hasattr(shape)={hasattr(var, 'shape')}")
        # if isinstance(var, dict):
        # print(f"    -> keys: {list(var.keys())}")

        loss, _ = self.model.loss(perceptual_emb, actions, latent_goal, noise, sigmas)
        return loss, sigmas, noise

    def denoise_actions(  # type: ignore
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        inference: Optional[bool] = False,
        extra_args={},
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoise the next sequence of actions
        """
        if inference:
            sampling_steps = self.num_sampling_steps
        else:
            sampling_steps = 10
        self.model.eval()
        if len(latent_goal.shape) < len(
            perceptual_emb["state_images"].shape
            if isinstance(perceptual_emb, dict)
            else perceptual_emb.shape
        ):
            latent_goal = latent_goal.unsqueeze(1)  # .expand(-1, seq_len, -1)
        input_state = perceptual_emb
        sigmas = self.get_noise_schedule(sampling_steps, self.noise_scheduler).to(
            self.device
        )
        self.model.to(latent_goal.dtype)
        self.model.inner_model.to(latent_goal.dtype)

        x = (
            torch.randn(
                (len(latent_goal), self.act_window_size, self.action_dim),
                device=self.device,
            )
            * self.sigma_max
        )

        sigmas = sigmas.to(latent_goal.dtype)
        x = x.to(latent_goal.dtype)
        # input_state = input_state.to(latent_goal.dtype)
        latent_plan = latent_plan.to(latent_goal.dtype)

        actions = self.sample_loop(
            sigmas,
            x,
            input_state,
            latent_goal,
            latent_plan,
            self.sampler_type,
            extra_args,
        )

        return actions

    def make_sample_density(self):
        """
        Generate a sample density function based on the desired type for training the model
        We mostly use log-logistic as it has no additional hyperparameters to tune.
        """
        sd_config = []
        if self.sigma_sample_density_type == "lognormal":
            loc = (
                self.sigma_sample_density_mean
            )  # if 'mean' in sd_config else sd_config['loc']
            scale = (
                self.sigma_sample_density_std
            )  # if 'std' in sd_config else sd_config['scale']
            return partial(utils.rand_log_normal, loc=loc, scale=scale)

        if self.sigma_sample_density_type == "loglogistic":
            loc = sd_config["loc"] if "loc" in sd_config else math.log(self.sigma_data)
            scale = sd_config["scale"] if "scale" in sd_config else 0.5
            min_value = (
                sd_config["min_value"] if "min_value" in sd_config else self.sigma_min
            )
            max_value = (
                sd_config["max_value"] if "max_value" in sd_config else self.sigma_max
            )
            return partial(
                utils.rand_log_logistic,
                loc=loc,
                scale=scale,
                min_value=min_value,
                max_value=max_value,
            )

        if self.sigma_sample_density_type == "loguniform":
            min_value = (
                sd_config["min_value"] if "min_value" in sd_config else self.sigma_min
            )
            max_value = (
                sd_config["max_value"] if "max_value" in sd_config else self.sigma_max
            )
            return partial(
                utils.rand_log_uniform, min_value=min_value, max_value=max_value
            )

        if self.sigma_sample_density_type == "uniform":
            return partial(
                utils.rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max
            )

        if self.sigma_sample_density_type == "v-diffusion":
            min_value = self.min_value if "min_value" in sd_config else self.sigma_min
            max_value = (
                sd_config["max_value"] if "max_value" in sd_config else self.sigma_max
            )
            return partial(
                utils.rand_v_diffusion,
                sigma_data=self.sigma_data,
                min_value=min_value,
                max_value=max_value,
            )
        if self.sigma_sample_density_type == "discrete":
            sigmas = self.get_noise_schedule(
                self.num_sampling_steps * 1e5, "exponential"
            )
            return partial(utils.rand_discrete, values=sigmas)
        if self.sigma_sample_density_type == "split-lognormal":
            loc = sd_config["mean"] if "mean" in sd_config else sd_config["loc"]
            scale_1 = (
                sd_config["std_1"] if "std_1" in sd_config else sd_config["scale_1"]
            )
            scale_2 = (
                sd_config["std_2"] if "std_2" in sd_config else sd_config["scale_2"]
            )
            return partial(
                utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2
            )
        else:
            raise ValueError("Unknown sample density type")

    def sample_loop(
        self,
        sigmas,
        x_t: torch.Tensor,
        state: torch.Tensor,
        goal: torch.Tensor,
        latent_plan: torch.Tensor,
        sampler_type: str,
        extra_args={},
    ):
        """
        Main method to generate samples depending on the chosen sampler type. DDIM is the default as it works well in all settings.
        """
        s_churn = extra_args["s_churn"] if "s_churn" in extra_args else 0
        s_min = extra_args["s_min"] if "s_min" in extra_args else 0
        use_scaler = extra_args["use_scaler"] if "use_scaler" in extra_args else False
        keys = ["s_churn", "keep_last_actions"]
        if bool(extra_args):
            reduced_args = {x: extra_args[x] for x in keys}
        else:
            reduced_args = {}
        if use_scaler:
            scaler = self.scaler
        else:
            scaler = None
        # ODE deterministic
        if sampler_type == "lms":
            x_0 = sample_lms(
                self.model,
                state,
                x_t,
                goal,
                sigmas,
                scaler=scaler,
                disable=True,
                extra_args=reduced_args,
            )
        # ODE deterministic can be made stochastic by S_churn != 0
        elif sampler_type == "heun":
            x_0 = sample_heun(
                self.model,
                state,
                x_t,
                goal,
                sigmas,
                scaler=scaler,
                s_churn=s_churn,
                s_tmin=s_min,
                disable=True,
            )
        # ODE deterministic
        elif sampler_type == "euler":
            x_0 = sample_euler(
                self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True
            )
        # SDE stochastic
        elif sampler_type == "ancestral":
            x_0 = sample_dpm_2_ancestral(
                self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True
            )
            # SDE stochastic: combines an ODE euler step with an stochastic noise correcting step
        elif sampler_type == "euler_ancestral":
            x_0 = sample_euler_ancestral(
                self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True
            )
        # ODE deterministic
        elif sampler_type == "dpm":
            x_0 = sample_dpm_2(self.model, state, x_t, goal, sigmas, disable=True)
        # ODE deterministic
        elif sampler_type == "dpm_adaptive":
            x_0 = sample_dpm_adaptive(
                self.model,
                state,
                x_t,
                goal,
                sigmas[-2].item(),
                sigmas[0].item(),
                disable=True,
            )
        # ODE deterministic
        elif sampler_type == "dpm_fast":
            x_0 = sample_dpm_fast(
                self.model,
                state,
                x_t,
                goal,
                sigmas[-2].item(),
                sigmas[0].item(),
                len(sigmas),
                disable=True,
            )
        # 2nd order solver
        elif sampler_type == "dpmpp_2s_ancestral":
            x_0 = sample_dpmpp_2s_ancestral(
                self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True
            )
        # 2nd order solver
        elif sampler_type == "dpmpp_2m":
            x_0 = sample_dpmpp_2m(
                self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True
            )
        elif sampler_type == "dpmpp_2m_sde":
            x_0 = sample_dpmpp_sde(
                self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True
            )
        elif sampler_type == "ddim":
            x_0 = sample_ddim(
                self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True
            )
        elif sampler_type == "dpmpp_2s":
            x_0 = sample_dpmpp_2s(
                self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True
            )
        elif sampler_type == "dpmpp_2_with_lms":
            x_0 = sample_dpmpp_2_with_lms(
                self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True
            )
        else:
            raise ValueError("desired sampler type not found!")
        return x_0

    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
        """
        Get the noise schedule for the sampling steps. Describes the distribution over the noise levels from sigma_min to sigma_max.
        """
        if noise_schedule_type == "karras":
            return get_sigmas_karras(
                n_sampling_steps, self.sigma_min, self.sigma_max, 7, self.device
            )  # rho=7 is the default from EDM karras
        elif noise_schedule_type == "exponential":
            return get_sigmas_exponential(
                n_sampling_steps, self.sigma_min, self.sigma_max, self.device
            )
        elif noise_schedule_type == "vp":
            return get_sigmas_vp(n_sampling_steps, device=self.device)
        elif noise_schedule_type == "linear":
            return get_sigmas_linear(
                n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device
            )
        elif noise_schedule_type == "cosine_beta":
            return cosine_beta_schedule(n_sampling_steps, device=self.device)
        elif noise_schedule_type == "ve":
            return get_sigmas_ve(
                n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device
            )
        elif noise_schedule_type == "iddpm":
            return get_iddpm_sigmas(
                n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device
            )
        raise ValueError("Unknown noise schedule type")

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0

    def forward(self, batch):
        return self.training_step(batch)
        # def training_step(self, batch: Dict[str, Dict], batch_idx: int,
        #                  dataloader_idx: int = 0) -> torch.Tensor

    def eval_forward(self, dataset_batch):
        """
        Method for doing inference with the model.
        """

        if self.video_model == "vpp_original":
            predictive_feature, latent_goal = self.extract_predictive_feature(
                dataset_batch
            )
        elif self.video_model == "vpp_wow":
            predictive_feature, latent_goal = self.extract_predictive_feature_wow(
                dataset_batch
            )
        elif self.video_model == "vpp_vidar":
            predictive_feature, latent_goal = self.extract_predictive_feature_vidar(
                dataset_batch
            )
        elif self.video_model == "vpp_robomaster":
            predictive_feature, latent_goal = (
                self.extract_predictive_feature_robomaster(dataset_batch)
            )
        elif self.video_model == "vpp_cogvideo":
            predictive_feature, latent_goal = (
                self.extract_predictive_feature_cogvideo_eval(dataset_batch)
            )
        else:
            raise ValueError(f"Unknown video_model: {self.video_model}")

        act_seq = self.denoise_actions(
            torch.zeros_like(latent_goal).to(latent_goal.device),
            predictive_feature,
            latent_goal,
            inference=True,
        )
        return act_seq

    def step(self, databatch):
        """
        Do one step of inference with the model. THis method handles the action chunking case.
        Our model is trained to predict a sequence of actions.
        We only compute the sequence once every self.multistep steps.

        Args:
            obs (dict): Observation from environment.
            goal (dict): Goal as visual observation or embedded language instruction.

        Returns:
            Predicted action.
        """
        if self.rollout_step_counter % self.multistep == 0 or True:
            if self.video_model == "vpp_wow":
                databatch = self.model_processor.process(databatch)
                pred_action_seq = self.eval_forward(databatch)
                self.pred_action_seq = pred_action_seq
            if self.video_model == "vpp_vidar":
                text = databatch["text"]  # str
                first_frame = databatch["first_frame"]  # pil
                databatch = self.TVP_encoder._preprocess_for_i2v(
                    input_prompt=text, img=first_frame, frame_num=self.act_window_size
                )
                pred_action_seq = self.eval_forward(databatch)
                self.pred_action_seq = pred_action_seq
            if self.video_model == "vpp_robomaster":
                text = databatch["text"]  # str
                first_frame = databatch["first_frame"]  # pil
                databatch = self.TVP_encoder._preprocess_for_i2v(
                    prompt=text,
                    image_resized=first_frame,
                    height=480,
                    width=640,
                    num_frames=self.act_window_size,
                    num_inference_steps=20,
                )
                pred_action_seq = self.eval_forward(databatch)
                self.pred_action_seq = pred_action_seq
            if self.video_model == "vpp_cogvideo":
                text = databatch["text"]  # str
                first_frame = databatch["first_frame"]  # pil
                databatch = {
                    "prompt": text,
                    "image": Image.fromarray(first_frame),
                    "num_frames": self.act_window_size,
                    "num_inference_steps": 20,
                    "height": 480,
                    "width": 640,
                }
                pred_action_seq = self.eval_forward(databatch)
                self.pred_action_seq = pred_action_seq

            # print("pred_action_seq", pred_action_seq.shape)

        current_action = self.pred_action_seq[0, self.rollout_step_counter]
        if len(current_action.shape) == 2:
            current_action = einops.rearrange(current_action, "b d -> b 1 d")
        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0

        return pred_action_seq

    def on_train_start(self) -> None:

        self.model.to(dtype=self.dtype)

        self.Video_Former.to(dtype=self.dtype)
        if self.language_goal is not None:  # language goal
            self.language_goal.to(dtype=self.dtype)
        # self.language_goal.to(dtype=self.dtype)
        # self.vae.to(dtype=self.dtype)
        self.TVP_encoder.to(dtype=self.dtype)

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)
