import argparse
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# video decode
import decord
from decord import VideoReader, cpu

# resize
import torchvision.transforms.functional as TF

# JEDi
from videojedi import JEDiMetric


def list_mp4_stems(folder: Path):
    """Return {stem: path} for mp4 files in folder (non-recursive)."""
    mp4s = sorted(folder.glob("*.mp4"))
    return {p.stem: p for p in mp4s}


def is_valid_mp4(path: Path) -> bool:
    """Quickly check if a video can be opened by decord."""
    try:
        vr = VideoReader(str(path), ctx=cpu(0))
        _ = len(vr)
        return True
    except Exception:
        return False


def intersect_pairs(real_dir: Path, gen_dir: Path):
    real_map = list_mp4_stems(real_dir)
    gen_map = list_mp4_stems(gen_dir)
    common = sorted(set(real_map.keys()) & set(gen_map.keys()))
    filtered = []
    for k in common:
        rp = real_map[k]
        gp = gen_map[k]
        if is_valid_mp4(rp) and is_valid_mp4(gp):
            filtered.append(k)
        else:
            print(f"[WARN] skip invalid pair: {k}")
    real_paths = [real_map[k] for k in filtered]
    gen_paths = [gen_map[k] for k in filtered]
    return filtered, real_paths, gen_paths


class PairedVideoFolderDataset(Dataset):
    """
    Returns a dict with:
      - "video": FloatTensor [T, 3, H, W] in [0,1]
      - "name":  video stem
    """
    def __init__(self, names, paths, num_frames=16, size=224):
        self.names = names
        self.paths = paths
        self.num_frames = int(num_frames)
        self.size = int(size)

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def _uniform_indices(total_frames: int, num_frames: int):
        if total_frames <= 0:
            return [0] * num_frames
        if total_frames >= num_frames:
            # linspace indices
            idx = torch.linspace(0, total_frames - 1, steps=num_frames)
            return idx.round().long().tolist()
        # pad by repeating last frame
        base = list(range(total_frames))
        base += [total_frames - 1] * (num_frames - total_frames)
        return base

    def _load_video_tensor(self, path: Path):
        try:
            vr = VideoReader(str(path), ctx=cpu(0))
            total = len(vr)
            inds = self._uniform_indices(total, self.num_frames)
            # decord returns NDArray [T, H, W, 3] uint8
            frames = vr.get_batch(inds).asnumpy()

            # to torch float [T, 3, H, W] in [0,1]
            x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

            # resize each frame
            resized = []
            for t in range(x.shape[0]):
                resized.append(TF.resize(x[t], [self.size, self.size], antialias=True))
            x = torch.stack(resized, dim=0)  # [T,3,H,W]
            return x
        except Exception as e:
            print(f"[WARN] failed to read video: {path} -> {e}. Using zero tensor fallback.")
            return torch.zeros(self.num_frames, 3, self.size, self.size, dtype=torch.float32)

    def __getitem__(self, idx):
        path = self.paths[idx]
        name = self.names[idx]
        video = self._load_video_tensor(path)
        return {"video": video, "name": name}


def collate_videos(batch):
    videos = torch.stack([b["video"] for b in batch], dim=0)  # [B,T,3,H,W]
    names = [b["name"] for b in batch]
    return videos, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, required=True, help="Folder of real mp4 videos")
    parser.add_argument("--gen_dir", type=str, required=True, help="Folder of generated mp4 videos")
    parser.add_argument("--num_frames", type=int, default=16, help="Uniformly sampled frames per video")
    parser.add_argument("--size", type=int, default=224, help="Resize shorter side to size x size")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means use all intersected pairs")
    parser.add_argument("--save_intersection", type=str, default="intersection_names.json")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    real_dir = Path(args.real_dir)
    gen_dir = Path(args.gen_dir)

    names, real_paths, gen_paths = intersect_pairs(real_dir, gen_dir)
    if len(names) == 0:
        raise RuntimeError("No intersected mp4 filenames (by stem) found between the two folders.")

    if args.max_samples and args.max_samples > 0:
        names = names[: args.max_samples]
        real_paths = real_paths[: args.max_samples]
        gen_paths = gen_paths[: args.max_samples]

    # save intersection list for reproducibility
    Path(args.save_intersection).write_text(json.dumps(names, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[INFO] real videos: {len(list(real_dir.glob('*.mp4')))}")
    print(f"[INFO] gen  videos: {len(list(gen_dir.glob('*.mp4')))}")
    print(f"[INFO] intersected pairs used for eval: {len(names)}")
    print(f"[INFO] saved intersection names -> {args.save_intersection}")

    real_ds = PairedVideoFolderDataset(names, real_paths, num_frames=args.num_frames, size=args.size)
    gen_ds  = PairedVideoFolderDataset(names, gen_paths,  num_frames=args.num_frames, size=args.size)

    real_loader = DataLoader(
        real_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_videos,
    )
    gen_loader = DataLoader(
        gen_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_videos,
    )

    # JEDiMetric will extract V-JEPA features internally and compute distance
    jedi = JEDiMetric()
    # load_features expects two loaders + num_samples
    num_samples = len(names)
    jedi.load_features(real_loader, gen_loader, num_samples=num_samples)
    score = jedi.compute_metric()
    score = np.exp(-0.4 * score)  # convert distance to similarity in [0,1], higher is better

    print("\n==============================")
    print(f"JEDi score (higher is better): {score}")
    print("==============================\n")
    try:
        gen_dir_name = Path(args.gen_dir).name
        out_json = "Path/to/WorldArena/video_quality/output_JEDi"/Path(f"{gen_dir_name}.json")
        result = {
            "gen_dir": str(args.gen_dir),
            "real_dir": str(args.real_dir),
            "score": float(score) if hasattr(score, "__float__") else score,
        }
        out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[INFO] Saved JEDi score json -> {out_json}")
    except Exception as e:
        print(f"[WARN] Failed to save JEDi score json: {e}")


if __name__ == "__main__":
    # make decord deterministic-ish
    decord.bridge.set_bridge("native")
    main()
