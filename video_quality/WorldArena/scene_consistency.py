import io
import os
import cv2
import json
import numpy as np
from PIL import Image
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from .utils import load_video, load_dimension_info, dino_transform, dino_transform_Image
import logging

from .distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor

from omegaconf import OmegaConf
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "submodel/dinov2"))
from dinov2.models import build_model_from_cfg


# logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


def scene_consistency(model, video_list, device):
    sim = 0.0
    cnt = 0
    video_results = []

    image_transform = dino_transform(518)
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        video_sim = 0.0

        images = load_video(video_path)
        B, C, H, W = images.shape
        max_side = max(H, W)
        pad_top = (max_side - H) // 2
        pad_bottom = max_side - H - pad_top
        pad_left = (max_side - W) // 2
        pad_right = max_side - W - pad_left
        padded_images = F.pad(images, (pad_left, pad_right, pad_top, pad_bottom))            
        images = image_transform(padded_images)
        for i in range(len(images)):

            with torch.no_grad():
                image = images[i].unsqueeze(0)
                image = image.to(device)

                image_features = model.forward_features(image)
                image_features = image_features["x_norm_patchtokens"]

                image_features = F.normalize(image_features, dim=-1, p=2)
                
                if i == 0:
                    first_image_features = image_features
                else:
                    # sim_pre = max(0.0, F.cosine_similarity(former_image_features, image_features).item())
                    sim_pre = max(0.0, F.cosine_similarity(former_image_features, image_features, dim=1).mean(dim=-1).item())
                    # sim_fir = max(0.0, F.cosine_similarity(first_image_features, image_features).item())
                    sim_fir = max(0.0, F.cosine_similarity(first_image_features, image_features, dim=1).mean(dim=-1).item())
                    cur_sim = (sim_pre + sim_fir) / 2
                    video_sim += cur_sim
                    cnt += 1
            former_image_features = image_features
        sim_per_images = video_sim / (len(images) - 1)

        sim += video_sim
        video_results.append({'video_path': video_path, 'video_results': sim_per_images})
    # sim_per_video = sim / (len(video_list) - 1)
    sim_per_frame = sim / cnt
    return sim_per_frame, video_results

def compute_scene_consistency(json_dir, submodules_list, **kwargs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    repo_or_dir = './submodel/dinov2'
    config_path = './dino_config.yaml'
    
    checkpoint_path = submodules_list['model']

    if repo_or_dir is None or checkpoint_path is None:
        raise ValueError("Both model code path (`repo_or_dir`) and checkpoint path must be provided in submodules_list")


    cfg = OmegaConf.load(config_path)
    dino_model, _ = build_model_from_cfg(cfg, only_teacher=True)
    dino_model = dino_model.to(device)

    print(f"Loading model weights from: {checkpoint_path}")
    ori_state_dict = dino_model.state_dict()
    state_dict = torch.load(checkpoint_path)
    state_dict_toload = dict()
    for k, v in state_dict.items():
        if k.startswith("teacher"):
            k_toload = k.replace("teacher.", "")
            k_toload = k_toload.replace("backbone.", "")
            if k_toload in ori_state_dict.keys():
                state_dict_toload.update({k_toload: v})
    print(dino_model.load_state_dict(state_dict_toload, strict=False))
    print("Initialize DINO success")


    video_list = load_dimension_info(json_dir, dimension='scene_consistency')
    video_list = distribute_list_to_rank(video_list)

    all_results, video_results = scene_consistency(dino_model, video_list, device)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results
