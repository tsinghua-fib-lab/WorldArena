# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import random
import warnings
import traceback
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms as T
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import imageio
from decord import VideoReader, cpu
from concurrent.futures import ThreadPoolExecutor, as_completed
from einops import rearrange

# from mdt.datasets.utils.dataset_util import euler2rotm, rotm2euler
# from mdt.datasets.utils.video_transforms import Resize_Preprocess, ToTensorVideo
# from mdt.datasets.utils.util import update_paths
from scipy.spatial.transform import Rotation as R  
import decord

class Dataset_xbot(Dataset):
    def __init__(
            self,
            args,
            mode = 'val',
    ):
        """Constructor."""
        super().__init__()
        self.args = args
        self.mode = mode
        data_json_path = args.data_json_path
        data_root_path = args.data_root_path

        # dataset stucture
        # dataset_dir/dataset_name/annotation_name/mode/traj
        # dataset_dir/dataset_name/video/mode/traj
        # dataset_dir/dataset_name/latent_video/mode/traj

        # samles:{'ann_file':xxx, 'frame_idx':xxx, 'dataset_name':xxx}

        # prepare all datasets path
        self.video_path = []
        data_json_path = f'{data_json_path}/{mode}_all.json'
        with open(data_json_path, "r") as f:
            self.samples = json.load(f)
        self.video_path = [os.path.join(data_root_path, sample['dataset_name']) for sample in self.samples]
        
        print(f"ALL dataset, {len(self.samples)} samples in total")

        # with open(f'{self.args.action_json}', "r") as f:
        # self.stat = json.load(f)
        self.a_min = np.array(args.action_01)[None,:]
        self.a_max = np.array(args.action_99)[None,:]
        self.s_min = np.array(args.state_01)[None,:]
        self.s_max = np.array(args.state_99)[None,:]
        print(f"action min: {self.a_min.shape}, action max: {self.a_max.shape}")

    def __len__(self):
        return len(self.samples)

    def _load_latent_video(self, video_path, frame_ids):
        # video_path = video_path.split('/')[:-1]
        # video_path = '/'.join(video_path)+'/0.pt'
        
        # print(video_path)
        with open(video_path,'rb') as file:
            video_tensor = torch.load(file)
            video_tensor.requires_grad = False
        # vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        # print(video_tensor.size(),np.array(frame_ids))
        try:
            assert (np.array(frame_ids) < video_tensor.size()[0]).all()
            assert (np.array(frame_ids) >= 0).all()
        except:
            assert False
        frame_data = video_tensor[frame_ids]
        return frame_data

    def _get_frames(self, label, frame_ids, cam_id, pre_encode, video_dir, use_img_cond=False):
        # directly load videos latent after svd-vae encoder
        assert cam_id is not None
        assert pre_encode == True
        if pre_encode: 
            video_path = label['latent_videos'][cam_id]['latent_video_path']
            try:
                video_path = os.path.join(video_dir,video_path)
                frames = self._load_latent_video(video_path, frame_ids)
            except:
                video_path = video_path.replace("latent_videos", "latent_videos_svd")
                frames = self._load_latent_video(video_path, frame_ids)
        # load original videos
        else: 
            if use_img_cond:
                frame_ids = frame_ids[0]
            video_path = label['videos'][cam_id]['video_path']
            video_path = os.path.join(video_dir,video_path)
            # frames = self._load_video(video_path, frame_ids)
            # frames = mediapy.read_video(video_path)
            vr = decord.VideoReader(video_path)
            frames = vr[frame_ids].asnumpy()
            frames = torch.from_numpy(frames).permute(2,0,1).unsqueeze(0) # (frame, h, w, c) -> (frame, c, h, w)
            # resize the video to self.args.video_size
            frames = self.preprocess(frames)
        return frames

    def _get_obs(self, label, frame_ids, cam_id, pre_encode, video_dir):
        if cam_id is None:
            temp_cam_id = random.choice(self.cam_ids)
        else:
            temp_cam_id = cam_id
        frames = self._get_frames(label, frame_ids, cam_id = temp_cam_id, pre_encode = pre_encode, video_dir=video_dir)
        return frames, temp_cam_id

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def denormalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps=1e-8,
    ) -> np.ndarray:
        clip_range = clip_max - clip_min
        rdata = (data - clip_min) / clip_range * (data_max - data_min) + data_min
        return rdata

    def process_action_xhand(self, label,frame_ids, rel = False):
        num_frames = len(frame_ids)
        frame_ids = frame_ids[:int(self.args.num_frames)] # (f)
        states = np.array(label['states'])[frame_ids] #(f, 38)
        command = np.array(label['actions'])[frame_ids]

        # print(f'states: {states.shape}, actions: {command.shape}')

        state = states[0:1] # current state

        a_dim = command.shape[-1]
        action_base = state[:,:a_dim] #(1,38)
        actions = command - action_base #(self.args.num_frames,38)

        # normalize
        action_scaled = self.normalize_bound(actions, self.a_min, self.a_max)
        state_scaled = self.normalize_bound(state, self.s_min, self.s_max)
        return torch.from_numpy(action_scaled).float(), torch.from_numpy(state_scaled).float()

    def __getitem__(self, index, cam_id = None, return_video = False):

        sample = self.samples[index]
        sampled_video_dir = self.video_path[index]

        ann_file = sample['ann_file']
        # dataset_name = sample['dataset_name']
        ann_file = f'{sampled_video_dir}/{ann_file}'
        frame_ids = sample['frame_ids']
        with open(ann_file, "r") as f:
            label = json.load(f)

        data = dict()
        # action
        data['actions'], data['state_obs'] = self.process_action_xhand(label,frame_ids,rel=self.args.relative)
        # instructions
        data['lang_text'] = label['texts'][0]
        # observation
        static_latent, cam_id = self._get_obs(label, frame_ids[0], cam_id=0, pre_encode=self.args.pre_encode,video_dir=sampled_video_dir)
        gripper_latent, cam_id = self._get_obs(label, frame_ids[0], cam_id=1, pre_encode=self.args.pre_encode,video_dir=sampled_video_dir)
        gripper_latent2, cam_id = self._get_obs(label, frame_ids[0], cam_id=2, pre_encode=self.args.pre_encode,video_dir=sampled_video_dir)
        static_latent = static_latent.unsqueeze(0)
        gripper_latent = gripper_latent.unsqueeze(0) # (1,4,32,32)
        gripper_latent2 = gripper_latent2.unsqueeze(0)

        # one sample
        rgb_obs = {'rgb_static': static_latent, 'rgb_gripper': gripper_latent, 'rgb_gripper2': gripper_latent2}
        data['rgb_obs'] = rgb_obs
        data['ann_file'] = ann_file
        data['frame_ids'] = frame_ids

        return data
        

if __name__ == "__main__":
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    with initialize(config_path="../../conf", job_name="VPP_xbot_train.yaml"):
        cfg = compose(config_name="VPP_xbot_train")
    
    # import sys
    # sys.path.append('/cephfs/cjyyj/code/video_robot_svd-main/mdt')
    # from utils.util import get_args
    # train_args = get_args(cfg.datamodule.args)
    # print(train_args)
    train_dataset = Dataset_xbot(cfg.dataset_args,mode="val")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.dataset_args.batch_size,
        shuffle=cfg.dataset_args.shuffle,
    )
    for data in tqdm(train_loader,total=len(train_loader)):
        print(data['ann_file'])
        print(len(data['rgb_obs']))

    