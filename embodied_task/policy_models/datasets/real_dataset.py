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

class Dataset_policy(Dataset):
    def __init__(
            self,
            args,
            mode = 'val',
            data_json_path = '/localssd/gyj/opensource_robotdata/annotation_all/0407',
            data_root_path = '/localssd/gyj/opensource_robotdata/',
    ):
        """Constructor."""
        super().__init__()
        self.args = args
        self.mode = mode

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

        self.a_min = np.array(args.action_01)[None,:]
        self.a_max = np.array(args.action_99)[None,:]
        self.s_min = np.array(args.state_01)[None,:]
        self.s_max = np.array(args.state_99)[None,:]

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

    def process_action_xhand(self, label,frame_ids, rel = False):
        num_frames = len(frame_ids)
        frame_ids = frame_ids[:int(self.args.num_frames+1)] # (10,)
        states = np.array(label['states'])[frame_ids]
        command = np.array(label['actions'])[frame_ids]

        state_input = states[0:1] #(1,19)
        # always use the set the first item of quat >0
        if state_input[0,3] <0:
            state_input[0,3:7] *= -1 

        states_raw = states if not self.args.learn_command else command

        if not rel:
            state_next = states_raw[:-1] # command
            mu = np.array(self.args.mu)
            std = np.array(self.args.std)

            state_input = (state_input-mu)/std
            action_sclaed = (state_next-mu)/std

        else: # relative ro fiest frame
            xyz, rot, hand = states_raw[:,:3], states_raw[:,3:7], states_raw[:,7:]
            # xyz
            current_xyz = state_input[:,:3]
            delta_xyz = (xyz[:-1]-current_xyz)*self.args.rel_xyz_scale
            # rot
            current_quat = state_input[:,3:7]
            rotm = [R.from_quat(rot[i]).as_matrix() for i in range(len(rot-1))]
            current_rotm = R.from_quat(current_quat[0]).as_matrix()
            rel_rotm = [current_rotm.T @ next_rotm for next_rotm in rotm[:-1]]
            rel_rpy = [R.from_matrix(rot).as_euler('xyz', degrees=False) for rot in rel_rotm]

            rel_rpy = np.array(rel_rpy)*self.args.rel_rot_scale
            # hand
            hand = hand[:-1]*self.args.rel_hand_scale

            action_sclaed = np.concatenate([delta_xyz,rel_rpy,hand],axis=1) # (10,18)

            if self.args.norm_input:
                mu = np.array(self.args.mu)
                std = np.array(self.args.std)
                state_input = (state_input-mu)/std

        return torch.from_numpy(action_sclaed).float(), torch.from_numpy(state_input).float()

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

    def process_action_xhand_v2(self, label,frame_ids, rel = False):
        frame_ids = frame_ids[:int(self.args.num_frames)]
        states = np.array(label['states'])[frame_ids]
        command = np.array(label['actions'])[frame_ids]

        state_input = states[0:1] #(1,19)
        # always use the set the first item of quat >0
        if state_input[0,3] <0:
            state_input[0,3:7] *= -1 

        states_raw = states if not self.args.learn_command else command

        xyz, rot, hand = states_raw[:,:3], states_raw[:,3:7], states_raw[:,7:]
        # xyz
        current_xyz = state_input[:,:3]
        delta_xyz = (xyz-current_xyz)
        # rot
        current_quat = state_input[:,3:7]
        rotm = [R.from_quat(rot[i]).as_matrix() for i in range(len(rot))]
        current_rotm = R.from_quat(current_quat[0]).as_matrix()
        rel_rotm = [current_rotm.T @ next_rotm for next_rotm in rotm]
        rel_rpy = [R.from_matrix(rot).as_euler('xyz', degrees=False) for rot in rel_rotm]
        # hand
        hand = hand

        action = np.concatenate([delta_xyz,rel_rpy,hand],axis=1) # (10,18)
        state = state_input

        action_scaled = self.normalize_bound(action, self.a_min, self.a_max)
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
        if self.args.action_v2:
            data['actions'], data['state_obs'] = self.process_action_xhand_v2(label,frame_ids,rel=self.args.relative)
        else:
            data['actions'], data['state_obs'] = self.process_action_xhand(label,frame_ids,rel=self.args.relative)
        # instructions
        data['lang_text'] = label['texts'][0]
        # observation
        static_latent, cam_id = self._get_obs(label, frame_ids[0], cam_id=0, pre_encode=self.args.pre_encode,video_dir=sampled_video_dir)
        gripper_latent, cam_id = self._get_obs(label, frame_ids[0], cam_id=1, pre_encode=self.args.pre_encode,video_dir=sampled_video_dir)
        static_latent = static_latent.unsqueeze(0)
        gripper_latent = gripper_latent.unsqueeze(0) # (1,4,32,32)

        # one sample
        rgb_obs = {'rgb_static': static_latent, 'rgb_gripper': gripper_latent}
        data['rgb_obs'] = rgb_obs
        data['ann_file'] = ann_file
        data['frame_ids'] = frame_ids

        return data
        

if __name__ == "__main__":
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    with initialize(config_path="../../conf", job_name="VPP_xhand_train.yaml"):
        cfg = compose(config_name="VPP_xhand_train")
    
    # import sys
    # sys.path.append('/cephfs/cjyyj/code/video_robot_svd-main/mdt')
    # from utils.util import get_args
    # train_args = get_args(cfg.datamodule.args)
    # print(train_args)
    train_dataset = Dataset_policy(cfg.dataset_args,mode="val")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.dataset_args.batch_size,
        shuffle=cfg.dataset_args.shuffle,
    )
    for data in tqdm(train_loader,total=len(train_loader)):
        print(data['ann_file'])

    