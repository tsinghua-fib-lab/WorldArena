import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
import argparse
import json
import os
import socket
from pathlib import Path

import hydra
import imageio
import lightning as pl
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torchvision
import wandb
from diffsynth import ModelManager, WanVideoPipeline, load_state_dict
from einops import rearrange
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
from policy_evaluation.multistep_sequences import get_sequences
from policy_models.utils.utils import get_last_checkpoint
from pytorch_lightning import seed_everything
from termcolor import colored
from torchvision.transforms import v2
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

ROOT = "your dataset path absolute path "  # like /robotwin2.0_clean50_firstinstr_40


class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path,  #
        max_num_frames=121,
        frame_interval=1,
        num_frames=121,
        height=480,
        width=832,
        is_i2v=False,
        obs_seq_len=1,
        act_seq_len=10,
        metadata_file=None,
    ):
        assert metadata_file is not None, "metadata_file must be provided"

        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        self.obs_seq_len = obs_seq_len
        self.act_seq_len = act_seq_len
        self.chunk_frames = obs_seq_len + act_seq_len - 1

        self.frame_process = v2.Compose(
            [
                v2.CenterCrop(size=(height, width)),
                v2.Resize(size=(height, width), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.allow_task = ["adjust_bottle"]  # , click_bell
        # === 1.  metadata  episodes ===
        print(f"Loading dataset from metadata: {metadata_file}")
        with open(metadata_file, "r") as f:
            meta_list = json.load(f)

        self.episodes = []
        for item in meta_list:
            video_path = Path(item["video_path"])

            ep_name = video_path.stem
            ep_idx = int(ep_name.replace("episode", ""))
            if ep_idx >= 25:
                continue

            n_frames = item["n_frames"]

            # Reconstruct other file paths (fixed structure)
            # hypothetical structure: .../{task}/aloha-agilex_clean_50/aloha-agilex_clean_50/{video,instructions,states,actions}
            task_dir = (
                video_path.parent.parent.parent.parent
            )  # Return to the task root directory.
            demo_dir = task_dir / "aloha-agilex_clean_50" / "aloha-agilex_clean_50"
            instr_dir = demo_dir / "instructions"
            states_dir = demo_dir / "states"
            actions_dir = demo_dir / "actions"
            if task_dir.name not in self.allow_task:
                continue
            #
            json_path = instr_dir / f"{ep_name}.json"
            states_path = states_dir / f"{ep_name}.npy"
            actions_path = actions_dir / f"{ep_name}.npy"

            # check existence of all files
            if not (
                json_path.exists() and states_path.exists() and actions_path.exists()
            ):
                continue  # skip this episode if any file is missing

            rel_video_path = video_path.relative_to(ROOT)
            self.episodes.append(
                {
                    "ep_name": ep_name,
                    "video_path": str(video_path),
                    "json_path": str(json_path),
                    "states_path": str(states_path),
                    "actions_path": str(actions_path),
                    "n_frames": n_frames,
                    "rel_video_path": str(rel_video_path),
                }
            )

        # === 2. chunk construction ===
        self.chunks = []

        for ep_idx, ep in enumerate(self.episodes):
            T = ep["n_frames"]
            start = 0
            while start + self.chunk_frames <= T:
                self.chunks.append((ep_idx, start))
                start += 1  # non-overlapping chunks

        print(f"✅ Loaded {len(self.episodes)} episodes → {len(self.chunks)} chunks")

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return image

    def load_frames_using_imageio(
        self,
        file_path,
        max_num_frames,
        start_frame_id,
        interval,
        num_frames,
        frame_process,
    ):
        """
        Load a consecutive sequence of `num_frames` frames starting from `start_frame_id`.
        Used for segmented or sliding-window video loading.

        Args:
            file_path: Path to the video file.
            max_num_frames: (Reserved argument, currently unused.)
            start_frame_id: Index of the starting frame.
            interval: (Reserved argument, currently unused, as frames are loaded consecutively.)
            num_frames: Number of frames to load (segment length).
            frame_process: Function to preprocess each individual frame.

        Returns:
            If self.is_i2v: (frames, first_frame)
            Otherwise: frames

            Here, `frames` is a List[Tensor] with length equal to min(num_frames, remaining_frames).
            However, if the video does not contain enough frames, an error is raised directly
            (the caller is responsible for ensuring valid input).
        """
        # print(f"Loading frames from {file_path} starting from frame {start_frame_id} with interval {interval} for {num_frames} frames")
        reader = imageio.get_reader(file_path)
        total_frames = reader.count_frames()

        # check if requested frames are within bounds
        if start_frame_id + num_frames > total_frames:
            raise IndexError(
                f"Cannot load {num_frames} frames from {file_path} starting at {start_frame_id}. "
                f"Video has only {total_frames} frames."
            )

        frames = []
        first_frame_pil = None

        for i in range(num_frames):
            frame_id = start_frame_id + i
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)

            if first_frame_pil is None:
                first_frame_pil = frame

            frame = frame_process(frame)
            frames.append(frame)

        reader.close()

        # resize and crop the first frame for i2v
        first_frame_cropped = v2.functional.center_crop(
            first_frame_pil, output_size=(self.height, self.width)
        )
        first_frame_np = np.array(first_frame_cropped)

        if self.is_i2v:
            return frames, first_frame_cropped
        else:
            return frames

    def load_video(self, file_path, start_frame_id, num_frames):
        """
        Load a consecutive sequence of `num_frames` frames starting from `start_frame_id`.

        Args:
            file_path (str): Path to the video file.
            start_frame_id (int): Starting frame index (0-based).
            num_frames (int): Number of frames to load.

        Returns:
            - If self.is_i2v: (frames, first_frame)
            - Otherwise: frames

            Where `frames` is a List[Tensor] (after preprocessing), and `first_frame` is a numpy array.
        """
        return self.load_frames_using_imageio(
            file_path=file_path,
            start_frame_id=start_frame_id,
            max_num_frames=None,  # no use
            interval=1,  #
            num_frames=num_frames,
            frame_process=self.frame_process,
        )

    def __getitem__(self, idx):
        ep_idx, start_frame_id = self.chunks[idx]
        episode = self.episodes[ep_idx]

        with open(episode["json_path"], "r") as f:
            instruction = json.load(f)["instruction"]

        prefix = "prompt:In a fixed robotic workspace, generate a rigid, physically consistent embodied robotic arm. The arm maintains high stability with no deformation and enters the frame to"
        instruction = prefix + instruction

        video = self.load_video(
            episode["video_path"], start_frame_id, self.chunk_frames
        )
        states = np.load(episode["states_path"])
        actions = np.load(episode["actions_path"])

        end_frame_id = start_frame_id + self.chunk_frames
        sampled_states = states[start_frame_id:end_frame_id]
        sampled_actions = actions[start_frame_id:end_frame_id]

        rel_video_path = Path(episode["rel_video_path"])
        chunk_name = f"{rel_video_path.stem}{rel_video_path.suffix}"
        full_chunk_path = str(rel_video_path.parent / chunk_name)

        chunk_data = {
            "text": instruction,
            "video": video,
            "states": sampled_states,
            "actions": sampled_actions,
            "path": full_chunk_path,
            "start_frame_id": start_frame_id,
            "rel_video_path": rel_video_path,
            "ep_idx": ep_idx,
        }

        if self.is_i2v:
            frames_list, first_frame = video
            chunk_data["first_frame"] = first_frame
            # print(f"first_frame shape added: {first_frame.shape}")
            chunk_data["video"] = frames_list

        return chunk_data  # back chunks

    def __len__(self):
        return len(self.chunks)  #


def rollout(
    model,
    chunk,
    cfg,
    episode_index=0,
):
    # Get language goal
    # Reset internal state of the model (e.g., rollout_step_counter)
    model.reset()

    # Store actions
    all_actions = []

    with torch.no_grad():
        # Pass the entire chunk dict to model.step()

        action = model.step(chunk)  # Assume model.step expects the same dict format
    print(f"Episode {episode_index}: action shape: {action.shape}")
    # Convert to numpy
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu()
    else:
        raise TypeError(f"Expected torch.Tensor from model.step(), got {type(action)}")

    # Remove batch dimension: [1, T, A] -> [T, A]
    if action.ndim == 3 and action.shape[0] == 1:
        action = action.squeeze(0)  # now [T, A]
    elif action.ndim == 2:
        # Already [T, A], maybe batch was not added — still okay
        pass
    else:
        raise ValueError(
            f"Unexpected action shape: {action.shape}. Expected [1, T, A] or [T, A]."
        )

    all_actions.append(action.float().numpy())
    # Concatenate all actions in this episode
    # Each action_np is [A] or [T_chunk, A]; we assume they are [T_chunk, A] for generality
    # But if each is [A], then we stack along time axis
    if all_actions:
        final_actions = np.concatenate(all_actions, axis=0)  # [T_total, A]
    else:
        final_actions = np.array([]).reshape(0, 0)  # empty but consistent shape

    print(f"Episode {episode_index}: total actions shape = {final_actions.shape}")

    # Save
    if cfg.output_path:
        rel_video_path = chunk["rel_video_path"]  # e.g., "click_bell/ep_001.mp4"
        rel_dir = Path(rel_video_path).parent  # e.g., "click_bell"
        start_frame = chunk["start_frame_id"]

        # Use episode_index from function argument as ep_idx
        ep_idx = chunk["ep_idx"]

        # === 1. Save first frame to dedicated folder ===
        first_frames_base = Path(cfg.output_path)
        video_dir = Path(rel_dir).parent
        first_frame_save_dir = first_frames_base / video_dir / "first_frames"
        first_frame_save_dir.mkdir(parents=True, exist_ok=True)

        first_frame_filename = f"episode{ep_idx}_chunk{start_frame:06d}.png"
        first_frame_path_abs = first_frame_save_dir / first_frame_filename

        # Save image
        first_frame = chunk["first_frame"]  # numpy uint8 [H, W, 3]
        imageio.imwrite(first_frame_path_abs, first_frame)

        # === 2. Prepare metadata for JSON ===
        instruction = chunk["text"]

        # Convert arrays to list for JSON serialization
        def to_list(x):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            if isinstance(x, np.ndarray):
                return x.tolist()
            return x

        pred_actions_list = to_list(final_actions)  # model output
        states = to_list(chunk["actions"])[0]  # ground truth actions

        json_data = {
            "instruction": instruction,
            "actions": pred_actions_list,  # predicted
            "states": states,
            "first_frame": str(first_frame_path_abs),  # abs path string
        }

        # === 3. Save JSON in original task structure ===
        json_save_dir = Path(cfg.output_path) / rel_dir
        json_save_dir.mkdir(parents=True, exist_ok=True)

        json_filename = f"episode{ep_idx}_chunk{start_frame:06d}.json"
        json_path = json_save_dir / json_filename

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        print(f"Saved JSON metadata to {json_path}")
        print(f"Saved first frame to {first_frame_path_abs}")

    return final_actions


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        required=False,
        help="Batchsize.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--custom_dit_path",
        type=str,
        default=None,
        help="Path of custom DiT checkpoint to override default DiT weights.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=15,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--obs_seq_len",
        type=int,
        default=1,
        help="Observation sequence length (frames).",
    )
    parser.add_argument(
        "--act_seq_len",
        type=int,
        default=10,
        help="Action sequence length (frames).",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=None,
        help="Path to the metadata JSON file.",
    )
    parser.add_argument("--action_model_path", type=str, default="")
    parser.add_argument("--is_i2v", type=bool, default=True)
    parser.add_argument("--start_index", type=int, default=True)
    parser.add_argument("--end_index", type=int, default=True)
    # parser.add_argument(
    #     "--checkpoint_every_n_steps",
    #     type=int,
    #     default=None,
    #     help="Save a checkpoint every N training steps (in addition to end of epoch).",
    # )

    args = parser.parse_args()
    return args


from torch.utils.data import DataLoader


# @hydra.main(config_path="../policy_conf", config_name="calvin_evaluate_all")
def main(cfg, start_idx, end_idx):
    torch.cuda.set_device(cfg.device)
    episode_indices = [0, 5, 10]
    seed_everything(0, workers=True)  # type:ignore
    # must like robotwin dataset
    dataset = TextVideoDataset(
        args.dataset_path,
        height=args.height,
        width=args.width,
        is_i2v=args.is_i2v,
        obs_seq_len=args.obs_seq_len,  # ←
        act_seq_len=args.act_seq_len,  # ←
        metadata_file=args.metadata_file,
    )
    # initialize LightningModelForDataProcess

    # evaluate a custom model
    checkpoints = [cfg.ckpt_path]

    print("train_folder", cfg.train_folder)

    for checkpoint in checkpoints:
        print(f"config device: {cfg.device}")

        # ckpt_path = os.path.join(cfg.train_folder,'saved_models')
        print(f"Loading model from {checkpoint}")

        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        device = torch.device(f"cuda:{cfg.device}")
        model = hydra.utils.instantiate(cfg.model)

        model.load_state_dict(state_dict["model"], strict=False)
        model.freeze()
        model = model.to(device)
        model.process_device()

        total_chunks = len(dataset)
        start = max(0, start_idx)
        end = min(total_chunks, end_idx)

        for chunk_idx in tqdm(range(start, end), desc="Generating chunks"):
            print(f"\n=== Processing episode {chunk_idx} ===")

            batch = dataset[chunk_idx]  #

            print(
                cfg.num_sampling_steps,
                cfg.sampler_type,
                cfg.multistep,
                cfg.sigma_min,
                cfg.sigma_max,
                cfg.noise_scheduler,
            )
            model.num_sampling_steps = cfg.num_sampling_steps
            model.sampler_type = cfg.sampler_type
            model.multistep = cfg.multistep
            if cfg.sigma_min is not None:
                model.sigma_min = cfg.sigma_min
            if cfg.sigma_max is not None:
                model.sigma_max = cfg.sigma_max
            if cfg.noise_scheduler is not None:
                model.noise_scheduler = cfg.noise_scheduler

            if cfg.cfg_value != 1:
                raise NotImplementedError("cfg_value != 1 not implemented yet")
            model.process_device()
            model.eval()

            actions_episode = rollout(model, batch, cfg, chunk_idx)

            # run.finish()


from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    import argparse

    from hydra import compose, initialize
    from omegaconf import OmegaConf

    args = parse_args()

    with initialize(config_path="../policy_conf", job_name="wan_evaluate.yaml"):
        cfg = compose(config_name="wan_evaluate.yaml")

    cfg.ckpt_path = args.action_model_path
    cfg.root_data_dir = args.dataset_path
    cfg.output_path = args.output_path
    main(cfg, args.start_index, args.end_index)
