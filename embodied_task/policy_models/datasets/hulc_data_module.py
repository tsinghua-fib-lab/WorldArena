import logging
import os
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from torch.utils.data import DataLoader
import torchvision
import pytorch_lightning as pl

import policy_models
from policy_models.datasets.utils.episode_utils import load_dataset_statistics
from policy_models.datasets.utils.shared_memory_utils import load_shm_lookup, save_shm_lookup, SharedMemoryLoader

logger = logging.getLogger(__name__)
DEFAULT_TRANSFORM = OmegaConf.create({"train": None, "val": None})
ONE_EP_DATASET_URL = "http://www.informatik.uni-freiburg.de/~meeso/50steps.tar.xz"


class HulcDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        root_data_dir: str = "data",
        num_workers: int = 8,
        transforms: DictConfig = DEFAULT_TRANSFORM,
        shuffle_val: bool = False,
        **kwargs: Dict,
    ):
        super().__init__()
        self.datasets_cfg = datasets
        self.train_datasets = None
        self.val_datasets = None
        self.train_sampler = None
        self.val_sampler = None
        self.num_workers = num_workers
        root_data_path = Path(root_data_dir)
        if not root_data_path.is_absolute():
            root_data_path = Path(policy_models.__file__).parent / root_data_path
        self.training_dir = root_data_path / "training"
        self.val_dir = root_data_path / "validation"
        self.shuffle_val = shuffle_val
        self.modalities: List[str] = []
        self.transforms = transforms
        self.use_shm = False

        #if 'lang_dataset' in self.datasets_cfg:
        #    if "shm_dataset" in self.datasets_cfg.lang_dataset._target_:
        #        self.use_shm = "shm_dataset" in self.datasets_cfg.lang_dataset._target_
        #    else:
        #        self.use_shm = False
        #elif 'shm_dataset' in self.datasets_cfg.vision_dataset._target_:
        #    self.use_shm = True
        #else:
        #    self.use_shm = False

    def prepare_data(self, *args, **kwargs):
        # check if files already exist
        dataset_exist = np.any([len(list(self.training_dir.glob(extension))) for extension in ["*.npz", "*.pkl"]])

        # download and unpack images
        if not dataset_exist:
            if "CI" not in os.environ:
                print(f"No dataset found in {self.training_dir}.")
                print("For information how to download to full CALVIN dataset, please visit")
                print("https://github.com/mees/calvin/tree/main/dataset")
                print("Do you wish to download small debug dataset to continue training?")
                s = input("YES / no")
                if s == "no":
                    exit()
            logger.info(f"downloading dataset to {self.training_dir} and {self.val_dir}")
            torchvision.datasets.utils.download_and_extract_archive(ONE_EP_DATASET_URL, self.training_dir)
            torchvision.datasets.utils.download_and_extract_archive(ONE_EP_DATASET_URL, self.val_dir)

        # if self.use_shm:
        #     # When using shared memory dataset, initialize lookups
        #     train_shmem_loader = SharedMemoryLoader(self.datasets_cfg, self.training_dir)
        #     train_shm_lookup = train_shmem_loader.load_data_in_shared_memory()
        #
        #     val_shmem_loader = SharedMemoryLoader(self.datasets_cfg, self.val_dir)
        #     val_shm_lookup = val_shmem_loader.load_data_in_shared_memory()
        #
        #     save_shm_lookup(train_shm_lookup, val_shm_lookup)

    def setup(self, stage=None):
        transforms = load_dataset_statistics(self.training_dir, self.val_dir, self.transforms)

        # self.train_transforms = {
        #    cam: [hydra.utils.instantiate(transform) for transform in transforms.train[cam]] for cam in transforms.train
        #}
        self.train_transforms = {}
        for cam in transforms.train:
            # print("Processing camera:", cam)
            cam_transforms = []
            for transform in transforms.train[cam]:
                # print("Instantiating transform for camera", cam, ":", transform)
                if transform._target_ == "torchvision.transforms.ColorJitter":
                    instantiated_transform = torchvision.transforms.ColorJitter(
                        brightness=transform.brightness,
                        contrast=tuple(transform.contrast),
                        saturation=tuple(transform.saturation),
                    )
                else:
                    instantiated_transform = hydra.utils.instantiate(transform)
                cam_transforms.append(instantiated_transform)
            self.train_transforms[cam] = cam_transforms

        self.val_transforms = {
            cam: [hydra.utils.instantiate(transform) for transform in transforms.val[cam]] for cam in transforms.val
        }
        self.train_transforms = {key: torchvision.transforms.Compose(val) for key, val in self.train_transforms.items()}
        self.val_transforms = {key: torchvision.transforms.Compose(val) for key, val in self.val_transforms.items()}
        self.train_datasets, self.train_sampler, self.val_datasets, self.val_sampler = {}, {}, {}, {}

        # if self.use_shm:
        #     train_shm_lookup, val_shm_lookup = load_shm_lookup()

        for _, dataset in self.datasets_cfg.items():
            if dataset == 'lang_paraphrase-MiniLM-L3-v2':
                continue
            else:
                train_dataset = hydra.utils.instantiate(
                    dataset, datasets_dir=self.training_dir, transforms=self.train_transforms
                )
                val_dataset = hydra.utils.instantiate(dataset, datasets_dir=self.val_dir, transforms=self.val_transforms)
                # if self.use_shm:
                #     train_dataset.setup_shm_lookup(train_shm_lookup)
                #     val_dataset.setup_shm_lookup(val_shm_lookup)
                key = dataset.key
                self.train_datasets[key] = train_dataset
                self.val_datasets[key] = val_dataset
                self.modalities.append(key)

    def train_dataloader(self):
        return {
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=True,
                shuffle=True,
                prefetch_factor=2,
            )
            for key, dataset in self.train_datasets.items()
        }

    def val_dataloader(self):
        return {
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=True,
            )
            for key, dataset in self.val_datasets.items()
        }

