import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
import socket
import json
from pathlib import Path
from wan.configs import MAX_AREA_CONFIGS

ROOT = "The absolute path to your dataset."
class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path,  # 
        height=480,
        width=832,
        is_i2v=False,
        obs_seq_len=1,
        act_seq_len=10,
        metadata_file=None
    ):
        assert metadata_file is not None, "metadata_file must be provided"
        
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        self.obs_seq_len = obs_seq_len
        self.act_seq_len = act_seq_len
        self.chunk_frames = obs_seq_len + act_seq_len-1

        # === 1. 
        print(f"Loading dataset from metadata: {metadata_file}")
        with open(metadata_file, 'r') as f:
            meta_list = json.load(f)

        self.allow_task = ["adjust_bottle","click_bell"] #,"click_bell" "adjust_bottle"

        self.episodes = []
        for item in meta_list:
            video_path = Path(item["video_path"])
            ep_name = video_path.stem
            n_frames = item["n_frames"]


            task_dir = video_path.parent.parent.parent.parent  # 
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
            # 
            try:
                actions = np.load(actions_path)
                # 
                if actions.ndim != 2 or actions.shape[1] != 14 :
                    raise ValueError(f"Invalid shape: {actions.shape}")
                # 
                states = np.load(states_path)
                if states.ndim != 2 or states.shape[0] != actions.shape[0] :
                    raise ValueError(f"States/actions length mismatch: {states.shape} vs {actions.shape}")
                
            except Exception as e:
                print(f"Skipping corrupted file: {actions_path}, error: {e}")
                continue  
                
            # 
            if not (json_path.exists() and states_path.exists() and actions_path.exists()):
                continue  # 
            
            rel_video_path = video_path.relative_to(ROOT)
            self.episodes.append({
                "ep_name": ep_name,
                "video_path": str(video_path),
                "json_path": str(json_path),
                "states_path": str(states_path),
                "actions_path": str(actions_path),
                "n_frames": n_frames,
                "rel_video_path": str(rel_video_path),
            })

        # === 
        self.chunks = []
        for ep_idx, ep in enumerate(self.episodes):
            T = ep["n_frames"]
            start = 0
            while start + self.chunk_frames <= T:
                self.chunks.append((ep_idx, start))
                start += 1  # overlapping chunks
        self.chunks = self.chunks[:]  #
        print(f"✅ Loaded {len(self.episodes)} episodes → {len(self.chunks)} chunks")

    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames):
        reader = imageio.get_reader(file_path)
        total_frames = reader.count_frames()

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

            # 
            frame_resized = frame.resize((self.width, self.height), Image.LANCZOS)
            frame_resized_npy = np.array(frame_resized)
            if first_frame_pil is None:
                #
                first_frame_pil = frame_resized_npy.copy()

            # 
            frames.append(frame_resized_npy)

        reader.close()


        if self.is_i2v:
            return frames, first_frame_pil  # ← 现在是npy
        else:
            return frames

    def load_video(self, file_path, start_frame_id, num_frames):
        return self.load_frames_using_imageio(
            file_path=file_path,
            start_frame_id=start_frame_id,
            max_num_frames = None,  # 
            interval=1,                           # 
            num_frames=num_frames,
        )

    def __getitem__(self, idx):
        ep_idx, start_frame_id = self.chunks[idx]
        episode = self.episodes[ep_idx]
        
        #print(f"debug json_path: {episode['json_path']}")
        with open(episode["json_path"], "r") as f:
            instruction = json.load(f)["instruction"]

        prefix = "prompt:In a fixed robotic workspace, generate a rigid, physically consistent embodied robotic arm. The arm maintains high stability with no deformation and enters the frame to"
        instruction = prefix + instruction 


        video = self.load_video(episode["video_path"], start_frame_id, self.chunk_frames)
        #print(f"[DEBUG GETITEM] actions_path: {episode['actions_path']}")
        states = np.load(episode["states_path"])   # (T, S)
        actions = np.load(episode["actions_path"])  # (T, A)

        end_frame_id = start_frame_id + self.chunk_frames
        sampled_states = states[start_frame_id:end_frame_id]
        sampled_actions = actions[start_frame_id:end_frame_id]

        rel_video_path = Path(episode["rel_video_path"])
        chunk_name = f"{rel_video_path.stem}{rel_video_path.suffix}"
        full_chunk_path = str(rel_video_path.parent / chunk_name)

        data = {
            "text": instruction,
            "video": video,
            "states": sampled_states,
            "actions": sampled_actions,
            "path": full_chunk_path,  # 
            "start_frame_id": start_frame_id,
        }
        
        if self.is_i2v:
            # 
            frames_list, first_frame = video
            data["first_frame"] = first_frame
            
            data["video"] = frames_list

        return data

    def __len__(self):
        return len(self.chunks)

def pad_tensor_to_fixed_size(tensor, target_size=(30, 4096)):

    current_len = tensor.size(0)  #
    target_len = target_size[0]  # 
    target_dim = target_size[1]  # 


    dtype = tensor.dtype
    device = tensor.device


    if current_len < target_len:
        padding = target_len - current_len

        tensor = F.pad(tensor, (0, 0, 0, padding), value=tensor.new_zeros(1).item())  # 

    elif current_len > target_len:
        tensor = tensor[:target_len]  # 

    tensor = tensor.to(dtype=dtype, device=device)

    return tensor

import torch.nn.functional as F
class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path,max_samples=None):
        """
        Args:
            dataset_path (str or Path): 
        """
        self.dataset_path = Path(dataset_path)
        tensor_paths = []
        self.allow_task = ["click_bell"] #,"click_bell" "adjust_bottle"
        for task_dir in self.dataset_path.iterdir():
            if not task_dir.is_dir():
                continue
            
            if task_dir.name not in self.allow_task:
                continue
            # 构造 demo 路径（嵌套两层同名）
            demo_dir = task_dir / "aloha-agilex_clean_50" / "aloha-agilex_clean_50"
            predata_dir = demo_dir / "predata_vidar" 找

            if not predata_dir.exists():
                continue

            for pth_file in predata_dir.glob("episode*_chunk*_vidar.pth"):
                tensor_paths.append(str(pth_file))


        tensor_paths = sorted(tensor_paths)
        if max_samples is not None:
            self.tensor_paths = tensor_paths[:max_samples]
        else:
            self.tensor_paths = tensor_paths
        print(f"Found {len(self.tensor_paths)} valid '_vidar.tensors.pth' files.")
        assert len(self.tensor_paths) > 0, f"No '_vidar.tensors.pth' files found under {dataset_path}!"

    def __getitem__(self, index):
        # 
        for _ in range(5):
            try:
                data_id = index % len(self.tensor_paths)
                path = self.tensor_paths[data_id]

                data = torch.load(path, weights_only=True, map_location="cpu")
                
                required_keys = {"z", "context", "context_null", "actions", "noise"}
                if not required_keys.issubset(data.keys()):
                    missing = required_keys - set(data.keys())
                    raise KeyError(f"Missing keys in {path}: {missing}")
                if data['actions'].shape[1] != 45:
                    raise ValueError(f"actions length is {data['actions'].shape[1]}, expected 45")
                clean_data = {}
                clean_data["z"] = [t.clone().detach() for t in data["z"]]  
                clean_data["context"] = [pad_tensor_to_fixed_size(emb).contiguous().clone().detach() for emb in data["context"]]
                clean_data["context_null"] = [emb.contiguous().clone().detach() for emb in data["context_null"]]

                
                clean_data["actions"] = data["actions"]
                clean_data["noise"] = data["noise"]
                clean_data["raw_text"] = data.get("raw_text", "")
                print(f"raw_text: {data.get('raw_text', '')}")
                
                clean_data["idx"] = data_id
                return clean_data
            except (RuntimeError, OSError, EOFError, KeyError,ValueError) as e:
                #print(f"Corrupted or invalid file: {path}, error: {e}")
                index += 1  # 

    def __len__(self):
        return len(self.tensor_paths)
    
from accelerate import Accelerator
import hydra
class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, cfg,root_path = None,size="832x480"):
        super().__init__()

        self.model = hydra.utils.instantiate(cfg.model)
        self.model.freeze()
        self.model.eval()
        assert hasattr(self.model, 'TVP_encoder'), "Model must have TVP_encoder attribute"
        
        self.root_path = Path(root_path) if root_path else None
        self.size = size
        self._model_on_device = False
    
    
    def test_step(self, batch, batch_idx):

        if not self._model_on_device:
            self.model = self.model.to(self.device)
            self.model.process_device()
            self._model_on_device = True

        device = self.device
        text = batch["text"][0]
        #print(f"text: {text}")
        
        first_frame = batch["first_frame"]  # npy
        #print(f"first)frame shape: {first_frame.shape}")
        first_frame_np = first_frame.cpu().numpy()
        #print(f"first_frame_np shape: {first_frame_np.shape}")
        first_frame_np = np.squeeze(first_frame_np, axis=0)
        first_frame = Image.fromarray(first_frame_np).convert("RGB")
        
        actions = batch["actions"]  # (T, A) tensor
        path = batch["path"][0]  # e.g., "task/.../episode0.mp4"
        start_frame_id = batch["start_frame_id"].item()  # intbatch["first_frame"]

        video_path = Path(path)

        video_name = video_path.stem
        predata_dir = self.root_path / video_path.parent.parent / "predata_vidar"
        predata_dir.mkdir(parents=True, exist_ok=True)
        output_file = predata_dir / f"{video_name}_chunk{start_frame_id:06d}_vidar.pth"

        if output_file.exists():
            try:

                existing_data = torch.load(output_file, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"⚠️ Corrupted file, will reprocess: {output_file}, error: {e}")
                existing_data = {}


            if "raw_text" in existing_data:
                print(f"✅ Skip (has 'raw_text'): {output_file}")
                return  
            else:
                # 
                print(f"🔧 Adding 'raw_text' to existing file: {output_file}")
                existing_data["raw_text"] = text  
                try:
                    torch.save(existing_data, output_file)
                    print(f"💾 Updated with raw_text: {output_file}")
                except Exception as e:
                    print(f"❌ Failed to save updated file: {output_file}, error: {e}")
                return  
            
        print(f"Processing: {text} | First frame type: {type(first_frame)} | Output: {output_file}")


        self.model.TVP_encoder.to(self.device)
        max_area=MAX_AREA_CONFIGS[self.size]
        with torch.no_grad():
            prep_result = self.model.TVP_encoder._preprocess_for_i2v(
                input_prompt=text,
                img=first_frame,          # PIL.Image, 0～255, size=(W, H)
                max_area=max_area,
                frame_num=45,             # 
                n_prompt="",              # 
                seed=-1,
                offload_model=False       # 
            )
        print(f"actions shape: {actions.shape}seq_len: {prep_result['seq_len']}, frame_num: {prep_result['frame_num']}")
        # 
        data = {
            "z": prep_result["z"],               # [C, 1, H_z, W_z] a list
            "context": [emb.cpu() for emb in prep_result["context"]],
            "context_null": [emb.cpu() for emb in prep_result["context_null"]],
            "seq_len": prep_result["seq_len"],
            "frame_num": prep_result["frame_num"],
            "actions": actions.cpu(),                  # (chunk_len, action_dim)
            "noise": prep_result["noise"],       # [C, 1, H_z, W_z] a list
        }


        torch.save(data, output_file)
        print(f"💾 Saved to: {output_file}")





def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
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
    # Multi-node training parameters
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes (machines) to use for distributed training.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs per node to use for training. If None, will use all available GPUs.",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="Rank of the current node in distributed training (0 to num_nodes-1).",
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
    parser.add_argument(
        "--size",
        type=str,
        default="640*480",
        choices=["640*480"], 
        help="Input resolution size key.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=64,
        help="Number of workers for dataloader.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3", "deepspeed_stage_2_offload", "deepspeed_stage_3_offload", "fsdp", "fsdp_native"],
        help="Training strategy",
    )

    
    args = parser.parse_args()
    return args



    


if __name__ == '__main__':
    args = parse_args()
    from hydra import compose, initialize
    
    with initialize(config_path="./policy_conf", job_name="VPP_vidar_train.yaml"):
        cfg = compose(config_name="VPP_vidar_train.yaml")


    if args.task == "data_process":
        dataset = TextVideoDataset(
            args.dataset_path,
            height=args.height,
            width=args.width,
            is_i2v=True,
            obs_seq_len=args.obs_seq_len,       
            act_seq_len=args.act_seq_len,      
            metadata_file=args.metadata_file,
            )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
        )
        model = LightningModelForDataProcess(
            cfg=cfg,
            size=args.size,
            root_path=ROOT,
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices="auto" if args.num_gpus is None else args.num_gpus,
            num_nodes=args.num_nodes,
            precision="bf16",
            strategy=args.training_strategy,
            default_root_dir=args.output_path,
        )
        trainer.test(model, dataloader)
        
    