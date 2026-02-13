#!/usr/bin/env python3
import json
from pathlib import Path

import imageio

# ======
DATASET_PATH = "The absolute path to your dataset."
OUTPUT_JSON = "./robotwin_metadata.json"
# =============================

root = Path(DATASET_PATH)
episodes = []

# collect all video paths
for task_dir in root.iterdir():
    if not task_dir.is_dir():
        continue
    demo_dir = task_dir / "aloha-agilex_clean_50" / "aloha-agilex_clean_50"
    video_dir = demo_dir / "video"
    if not video_dir.exists():
        continue
    for video_path in sorted(video_dir.glob("episode*.mp4")):
        episodes.append(str(video_path))

print(f"Found {len(episodes)} videos. Reading frame counts...")

metadata = []
for i, path in enumerate(episodes):
    try:
        reader = imageio.get_reader(path)
        n_frames = reader.count_frames()
        reader.close()
        metadata.append({"video_path": path, "n_frames": n_frames})
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} videos...")
    except Exception as e:
        print(f"❌ Skip {path}: {e}")

# save metadata to JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Done! Metadata saved to: {OUTPUT_JSON}")
print(f"Total valid videos: {len(metadata)}")
