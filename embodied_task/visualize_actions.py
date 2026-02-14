import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def load_actions(predict_path, gt_path):
    predict_actions = torch.load(predict_path).numpy()
    gt_actions = torch.load(gt_path).numpy()
    return predict_actions, gt_actions


def plot_actions_comparison(output_dir, episode_index):
    predict_path = output_dir / f"{episode_index}.pt"
    gt_path = output_dir / f"{episode_index}_gt.pt"

    if not (predict_path.exists() and gt_path.exists()):
        print(f"Skipping Episode {episode_index}: Files do not exist.")
        return

    pred_actions, gt_actions = load_actions(predict_path, gt_path)

    # Ensure the two actions have the same shape
    assert pred_actions.shape == gt_actions.shape, (
        "Prediction and ground truth actions must have the same shape."
    )

    time_steps = pred_actions.shape[0]
    action_dims = pred_actions.shape[1]

    fig, axes = plt.subplots(action_dims, 1, figsize=(10, 2 * action_dims))
    for i in range(action_dims):
        ax = axes[i] if action_dims > 1 else axes
        ax.plot(range(time_steps), pred_actions[:, i], label="Predicted")
        ax.plot(range(time_steps), gt_actions[:, i], label="Ground Truth")
        ax.set_title(f"Action Dim {i}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.legend()

    plt.tight_layout()
    save_fig_path = output_dir / f"{episode_index}_comparison.png"
    plt.savefig(save_fig_path)
    plt.close(fig)
    print(f"Saved comparison plot to {save_fig_path}")


def main(input_folder):
    output_dir = Path(input_folder)
    episodes = set([path.stem.split("_")[0] for path in output_dir.glob("*.pt")])
    for episode in episodes:
        if not episode.endswith("gt"):
            plot_actions_comparison(output_dir, episode)


if __name__ == "__main__":
    input_folder = "your evaluate folder"  #
    main(input_folder)
