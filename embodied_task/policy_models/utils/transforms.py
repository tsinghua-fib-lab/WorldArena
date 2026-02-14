import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleImageTensor(object):
    """Scale tensor of shape (batch, C, H, W) containing images to [0, 1] range

    Args:
        tensor (torch.tensor): Tensor to be scaled.
    Returns:
        Tensor: Scaled tensor.
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return tensor.float().div(255)


class NormalizeVector(object):
    """Normalize a tensor vector with mean and standard deviation."""

    def __init__(self, mean=[0.0], std=[1.0]):
        self.std = torch.Tensor(std)
        self.std[self.std == 0.0] = 1.0
        self.mean = torch.Tensor(mean)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return (tensor - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = torch.tensor(std)
        self.mean = torch.tensor(mean)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


class AddDepthNoise(object):
    """Add multiplicative gamma noise to depth image.
    This is adapted from the DexNet 2.0 code.
    Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/master/gqcnn/training/tf/trainer_tf.py"""

    def __init__(self, shape=1000.0, rate=1000.0):
        self.shape = torch.tensor(shape)
        self.rate = torch.tensor(rate)
        self.dist = torch.distributions.gamma.Gamma(torch.tensor(shape), torch.tensor(rate))

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        multiplicative_noise = self.dist.sample()
        return multiplicative_noise * tensor

    def __repr__(self):
        # return self.__class__.__name__ + f"{self.shape=},{self.rate=},{self.dist=}"
        return self.__class__.__name__ + f"(shape={self.shape}, rate={self.rate}, dist={self.dist})"




# source: https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class RelativeActions(object):
    """Transform absolute actions to relative"""

    def __init__(self, max_pos, max_orn):
        self.max_pos = max_pos
        self.max_orn = max_orn

    @staticmethod
    def batch_angle_between(a, b):
        diff = b - a
        return (diff + np.pi) % (2 * np.pi) - np.pi

    def __call__(self, action_and_obs):
        actions, robot_obs = action_and_obs
        assert isinstance(actions, np.ndarray)
        assert isinstance(robot_obs, np.ndarray)

        rel_pos = actions[:, :3] - robot_obs[:, :3]
        rel_pos = np.clip(rel_pos, -self.max_pos, self.max_pos) / self.max_pos

        rel_orn = self.batch_angle_between(robot_obs[:, 3:6], actions[:, 3:6])
        rel_orn = np.clip(rel_orn, -self.max_orn, self.max_orn) / self.max_orn

        gripper = actions[:, -1:]
        return np.concatenate([rel_pos, rel_orn, gripper], axis=1)

    def __repr__(self):
        return self.__class__.__name__ + f"(max_pos={self.max_pos}, max_orn={self.max_orn})"
