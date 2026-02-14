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

import sys
sys.path.append('./')
from video_dataset.video_transforms import Resize_Preprocess, ToTensorVideo
import mediapy
import decord
# from dataset.util import update_paths
# from dataset.dataset_util import euler2rotm, rotm2euler


class Dataset_mix(Dataset):
    def __init__(
            self,
            args,
            mode = 'val'
    ):
        """Constructor."""
        super().__init__()
        self.args = args
        self.mode = mode

        # dataset stucture
        # dataset_dir/dataset_name/annotation_name/mode/traj
        # dataset_dir/dataset_name/video/mode/traj
        # dataset_dir/dataset_name/latent_video/mode/traj

        # prepare all datasets path
        dataset_dir = args.dataset_dir
        dataset_names = args.dataset
        dataset_list = dataset_names.split('+')
        self.data_path = [f'{dataset_dir}/{dataset_name}/{args.annotation_name}/{mode}' for dataset_name in dataset_list]
        self.dataset_root = [f'{dataset_dir}/{dataset_name}' for dataset_name in dataset_list]

        # balance the dataset
        self.prob = args.prob
        self.sequence_length = args.sequence_length

        # prepare sample information
        self.samples = []
        self.ann_files = []
        self.samples_num = []
        for dataset_idx, data_path in enumerate(self.data_path):
            if 'calvin' in data_path:
                # train on calvin abc datasets and val on d datasets 
                data_path = data_path.replace('annotation','annotation_abc_13')
                data_path = data_path.replace('val','val_d')

            ann_files = self._init_anns(data_path)

            # for quick debug
            if args.debug:
                ann_files = ann_files[:50]
            
            samples = self._init_sequences(ann_files)

            # for quick validation
            if mode == 'val':
                samples = samples
            
            # gather all sample_info
            self.ann_files.append(ann_files)
            self.samples.append(samples)
            self.samples_num.append(len(samples))
            print(f'dataset_idx: {dataset_idx}, {data_path}')
            print(f'dataset_idx: {dataset_idx}, mode:{mode}, trajectories_num:{len(ann_files)}, sample_num:{len(samples)},')
        
        print(f"ALL dataset, {sum(self.samples_num)} samples in total")
        
        # for clip-vit-base-patch32
        self.preprocess = T.Compose([
            ToTensorVideo(),
            Resize_Preprocess(tuple([args.clip_img_size, args.clip_img_size])), # 224,224
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258,0.27577711], inplace=True)
        ])
        

    def __str__(self):
        return f"{len(self.ann_files)} samples from {self.data_path}"

    def _init_anns(self, data_dir):
        ann_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
        return ann_files

    def _init_sequences(self, ann_files):
        samples = []
        with ThreadPoolExecutor(32) as executor:
            future_to_ann_file = {executor.submit(self._load_and_process_ann_file, ann_file): ann_file for ann_file in ann_files}
            for future in tqdm(as_completed(future_to_ann_file), total=len(ann_files)):
                samples.extend(future.result())
        return samples

    def _load_and_process_ann_file(self, ann_file):

        samples = []
        try:
            with open(ann_file, "r") as f:
                ann = json.load(f)
        except:
            print(f'skip {ann_file}')
            return samples
        try:
            n_frames = len(ann['state'])
        except:
            n_frames = ann['video_length']

        # we directly resize internet video for 1 clip
        if ('bridge' in ann_file) or ('sthv2' in ann_file) or ('rt1' in ann_file):
            # filter out very long video clips
            if n_frames>100:
                return samples

            sample = dict()
            sample['ann_file'] = ann_file
            
            # directly resize internet video to 16 frames
            idx = np.linspace(0,n_frames-1,self.sequence_length).astype(int)
            sample['frame_ids'] = idx.tolist()
            samples.append(sample)
        
        # create multiple samples for robot data  
        else: 
            if 'xhand_1125_pour' in ann_file:
                sequence_interval = 10
                start_interval = 10
            if 'cloth' in ann_file:
                sequence_interval = 5
                start_interval = 5
            else:
                sequence_interval = 2
                start_interval = 2
            # record idx for each clip
            base_idx = np.arange(0,self.sequence_length)*sequence_interval
            max_idx = np.ones_like(base_idx)*(n_frames-1)
            for start_frame in range(0,n_frames,start_interval):
                idx = base_idx + start_frame
                idx = np.minimum(idx,max_idx)
                if len(idx) == self.sequence_length:
                    sample = dict()
                    sample['ann_file'] = ann_file
                    sample['frame_ids'] = idx.tolist()
                    samples.append(sample)

        return samples

    def __len__(self):
        return max(self.samples_num)

    def _load_video(self, video_path, frame_ids):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all()
        assert (np.array(frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).numpy() #(frame, h, w, c)
        # central crop
        h, w = frame_data.shape[1], frame_data.shape[2]
        # if h > w:
        #     margin = (h - w) // 2
        #     frame_data = frame_data[:, margin:margin + w]
        # elif w > h:
        #     margin = (w - h) // 2
        #     frame_data = frame_data[:, :, margin:margin + h]
        return frame_data
    
    def _load_latent_video(self, video_path, frame_ids):
        with open(video_path,'rb') as file:
            video_tensor = torch.load(file)
            video_tensor.requires_grad = False
            assert (np.array(frame_ids) < video_tensor.size()[0]).all()
            assert (np.array(frame_ids) >= 0).all()
        frame_data = video_tensor[frame_ids]
        return frame_data
    
    def _get_frames(self, label, frame_ids, cam_id, pre_encode, video_dir, use_img_cond=False):
        # directly load videos latent after svd-vae encoder
        assert cam_id is not None
        if pre_encode: 
            episode_id = label['episode_id']
            if 'xbot' in video_dir:
                video_path = f"latent_videos/{self.mode}/{episode_id}/{cam_id}.pt"
            else:
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
            vr = decord.VideoReader(video_path)
            frames = vr[frame_ids].asnumpy()
            frames = torch.from_numpy(frames).permute(2,0,1).unsqueeze(0) # (frame, h, w, c) -> (frame, c, h, w)
            # resize the video to self.args.video_size
            frames = self.preprocess(frames)
        return frames

    def _get_obs(self, label, frame_ids, cam_id, pre_encode, video_dir, use_img_cond=False):
        assert cam_id is not None
        temp_cam_id = cam_id
        frames = self._get_frames(label, frame_ids, cam_id = temp_cam_id, pre_encode = pre_encode, video_dir=video_dir, use_img_cond=use_img_cond)
        return frames, temp_cam_id

    def __getitem__(self, index, cam_id = None, return_video = False):
        # sample the ann_file
        dataset_idx = np.random.choice(len(self.samples), p=self.args.prob)
        sampled_dataset = self.samples[dataset_idx]
        sampled_video_dir = self.dataset_root[dataset_idx]

        sampled_idx = index % len(sampled_dataset)
        sample = sampled_dataset[sampled_idx]

        ann_file = sample['ann_file']
        frame_ids = sample['frame_ids']
        with open(ann_file, "r") as f:
            label = json.load(f)
        

        # prepare sample 
        data = dict()
        data['text'] = label['texts'][0] # text for condition

        # prepare rgb sequence
        if 'calvin' in ann_file or 'xhand' in ann_file:
            # 70% selecte 0 and 30% selecte 1
            cam_id = np.random.choice([0,1], p=[0.75,0.25])
            cond_cam_id = int(1-cam_id)
        elif 'xbot_04' in ann_file:
            cam_id = np.random.choice([0,1,2], p=[0.5,0.25,0.25])
        else:
            cam_id = 0
            cond_cam_id = 0

        latent, cam_id = self._get_obs(label, frame_ids, cam_id, pre_encode=True,video_dir=sampled_video_dir)
        data['latent'] = latent.float()
        data['video'] = latent.float()

        if self.args.use_img_cond:
            # load the image condition
            if 'calvin' in ann_file or 'xhand' in ann_file:
                img_cond, cam_id = self._get_obs(label, frame_ids, cond_cam_id, pre_encode=False, video_dir=sampled_video_dir, use_img_cond=True)
                mask = False
            else:
                # h, w = self.args.clip_img_size, self.args.clip_img_size
                # img_cond = torch.zeros([1,3,h,w],dtype=torch.float16)
                # mask = True
                img_cond, cam_id = self._get_obs(label, frame_ids, cond_cam_id, pre_encode=False, video_dir=sampled_video_dir, use_img_cond=True)
                mask = False
            data['img_cond'] = img_cond
            data['img_cond_mask'] = mask

        return data

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/cephfs/cjyyj/code/video_robot_svd/video_conf/train_svd.yaml")
    args = parser.parse_args()
    # data_config = OmegaConf.load("configs/base/data.yaml")
    # diffusion_config = OmegaConf.load("configs/base/diffusion.yaml")
    args = OmegaConf.load(args.config)
    # args = OmegaConf.merge(data_config, args)
    # args = OmegaConf.merge(diffusion_config, args)
    # update_paths(args)
    args = args.train_args
    dataset = Dataset_mix(args,mode='val')

    data_loader = DataLoader(dataset=dataset, 
                                    batch_size=8, 
                                    shuffle=True, 
                                    num_workers=16)
    for data in tqdm(data_loader,total=len(data_loader)):
        print(data['latent'].shape)
        print(data['img_cond'].shape)
        print(data['img_cond_mask'])