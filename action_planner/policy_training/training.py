import logging
from pathlib import Path
import sys
from typing import List, Union
import os 
import wandb

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only


import models.models.mdt_agent as models_m
from policy_models.utils.utils import (
    get_git_commit_hash,
    get_last_checkpoint,
    initialize_pretrained_weights,
    print_system_env_info,
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config_abc")
# @hydra.main(config_path="../logs/runs/2023-09-10/17-52-50/.hydra", config_name="config")
def train(cfg: DictConfig) -> None:
    """
    This is called to start a training.

    Args:
        cfg: hydra config
    """
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    os.environ['HYDRA_FULL_ERROR'] = '1'
    seed_everything(cfg.seed, workers=True)  # type: ignore
    log_rank_0(f"Seed for training: {cfg.seed}")
    # new added
    torch.set_float32_matmul_precision('medium')
    
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    chk = get_last_checkpoint(Path.cwd())
    # chk = get_last_checkpoint(Path('/home/temp_store/code/calvin_d/logs/runs/2023-09-10/17-52-50/saved_models/epoch=09_eval_lh/avg_seq_len=2.62.ckpt'))
    # Load Model
    if chk is not None:
        model = getattr(models_m, cfg.model["_target_"].split(".")[-1]).load_from_checkpoint(chk.as_posix())
    else:
        model = hydra.utils.instantiate(cfg.model)
        if "pretrain_chk" in cfg:
            initialize_pretrained_weights(model, cfg)

    log_rank_0(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
    log_rank_0("Repo commit hash: {}".format(get_git_commit_hash(Path(hydra.utils.to_absolute_path(__file__)))))
    log_rank_0(print_system_env_info())

    train_logger = setup_logger(cfg, model)
    callbacks = setup_callbacks(cfg.callbacks)
    lr_logger = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_logger)

    trainer_args = {
        **cfg.trainer,
        "logger": train_logger,
        "callbacks": callbacks,
        "benchmark": False,
        # "check_val_every_n_epoch": 1
    }

    # Configure multi-GPU training
    if is_multi_gpu_training(trainer_args["gpus"]):  # type: ignore
        trainer_args["strategy"] = "ddp"
        if not cfg.slurm:
            modify_argv_hydra()

    trainer = Trainer(**trainer_args)
    # torch.autograd.set_detect_anomaly(True)
    # Start training
    trainer.fit(model, datamodule=datamodule) #, ckpt_path=chk  # type: ignore
    
    wandb.finish()


def setup_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """
    Instantiate all training callbacks.

    Args:
        callbacks_cfg: DictConfig with all callback params

    Returns:
        List of instantiated callbacks.
    """
    callbacks = [hydra.utils.instantiate(cb) for cb in callbacks_cfg.values()]
    return callbacks


def setup_logger(cfg: DictConfig, model: LightningModule):
    """
    Set up the logger (tensorboard or wandb) from hydra config.

    Args:
        cfg: Hydra config
        model: LightningModule

    Returns:
        logger
    """
    pathlib_cwd = Path.cwd()
    if "group" in cfg.logger:
        cfg.logger.group = pathlib_cwd.parent.name
        cfg.logger.name = pathlib_cwd.parent.name + "/" + pathlib_cwd.name
        cfg.logger.id = cfg.logger.name.replace("/", "_")
        train_logger = hydra.utils.instantiate(cfg.logger)
        # train_logger.watch(model)
    else:
        train_logger = hydra.utils.instantiate(cfg.logger)
    return train_logger


def modify_argv_hydra() -> None:
    """
    To make hydra work with pytorch-lightning and ddp, we modify sys.argv for the child processes spawned with ddp.
    This is only used when NOT using slurm.
    """
    cwd = Path.cwd().as_posix()
    cwd = f'"{cwd}"'
    sys.argv = sys.argv[:1]
    sys.argv.extend(
        [
            f"hydra.run.dir={cwd}",
            "hydra/hydra_logging=disabled",
            "hydra/job_logging=disabled",
        ]
    )
    overrides = OmegaConf.load(".hydra/overrides.yaml")
    for o in overrides:
        if "hydra/sweeper" in o:  # type: ignore
            continue

        if "hydra/launcher" in o:  # type: ignore
            continue

        sys.argv.append(o)  # type: ignore


def is_multi_gpu_training(gpus: Union[int, str, ListConfig]) -> bool:
    """
    Parse pytorch-lightning gpu device selection,
    see https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html

    Args:
        gpus: int, str or ListConfig specifying gpu devices

    Returns:
        True if multi-gpu training (ddp), False otherwise.
    """
    return (
        (isinstance(gpus, int) and (gpus > 1 or gpus == -1))
        or (isinstance(gpus, str) and len(gpus) > 1)
        or (isinstance(gpus, ListConfig) and len(gpus) > 1)
    )


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


if __name__ == "__main__":
    # os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    # Set CUDA device IDs

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    os.environ["TOKENIZERS_PARALLELISM"] = 'True'
    train()