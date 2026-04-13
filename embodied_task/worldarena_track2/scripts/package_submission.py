#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_ROOT = SCRIPT_DIR.parent
SRC = TEMPLATE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from worldarena_track2_template.packaging import build_submission_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package Track 2 submission archive")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", required=True, help="Path to output zip file")
    parser.add_argument("--model-readme", required=True)
    parser.add_argument(
        "--video-dirs",
        nargs="+",
        metavar="VARIANT=DIR",
        help="Policy variant video directories, e.g. 10data=./output/10data 20data=./output/20data",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_dirs = {}
    if args.video_dirs:
        for item in args.video_dirs:
            if "=" in item:
                variant, dir_path = item.split("=", 1)
                video_dirs[variant] = Path(dir_path)
            else:
                video_dirs[Path(item).name] = Path(item)
    build_submission_tree(
        model_name=args.model_name,
        output_zip=Path(args.output),
        model_readme=Path(args.model_readme),
        video_dirs=video_dirs if video_dirs else None,
    )
    print(f"Created: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
