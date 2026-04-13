from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


def _copy_video_tree(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(src_dir.rglob("*.mp4")):
        if not path.is_file():
            continue
        rel = path.relative_to(src_dir)
        target = dst_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


POLICY_VARIANTS = ["10data", "20data", "30data", "50data", "fulldata"]


def build_submission_tree(
    model_name: str,
    output_zip: Path,
    model_readme: Path,
    video_dirs: dict[str, Path] | None = None,
) -> Path:
    """Build submission archive.

    Args:
        model_name: Your world model name.
        output_zip: Path to the output .zip file.
        model_readme: Path to model_README.md.
        video_dirs: Mapping from policy variant name to video directory,
                    e.g. {"10data": Path("output/10data"), ...}.
    """
    root_name = f"{model_name}_eval"
    temp_root = Path(tempfile.mkdtemp(prefix="worldarena_track2_"))
    archive_root = temp_root / root_name
    archive_root.mkdir(parents=True, exist_ok=True)

    if video_dirs:
        for variant, src_dir in video_dirs.items():
            _copy_video_tree(src_dir, archive_root / f"{model_name}_{variant}")

    shutil.copy2(model_readme, archive_root / "model_README.md")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(output_zip.with_suffix("")), "zip", temp_root, root_name)
    return output_zip
