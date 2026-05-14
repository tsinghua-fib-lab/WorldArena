#!/usr/bin/env python3
"""
VLM policy evaluator for robot task videos.

Expected GT layout:
  GT_ROOT/
    task_name/
      videos/episode40.mp4
      instructions/episode40.json

Expected submission layout is configurable with --policy-template. The default
is "{model}/episode{index}.mp4", where index is the 1-based GT ordering index.
For Robohero-style submissions, use:
  --policy-template "{model}/fixed_scene_task_episode{index}.mp4"

The VLM endpoint is assumed to be OpenAI-compatible chat completions with image
URL content blocks. API keys are read from an environment variable only. Use
--api-url and --api-key-env to select your provider.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import requests


DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_API_KEY_ENV = "VLM_API_KEY"
DEFAULT_POLICY_TEMPLATE = "{model}/episode{index}.mp4"
DEFAULT_VLM_MODEL = "Qwen/Qwen3-VL-32B-Instruct"


def episode_key(path: Path) -> tuple[int, str]:
    match = re.search(r"episode(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return sys.maxsize, path.name


def episode_number(episode_name: str) -> str:
    match = re.search(r"episode(\d+)", episode_name)
    return match.group(1) if match else ""


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def parse_models(value: str, submission_root: Path) -> list[str]:
    if value:
        return [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    models = sorted(path.name for path in submission_root.iterdir() if path.is_dir())
    if not models:
        raise ValueError(f"No model directories found under submission root: {submission_root}")
    return models


def collect_gt_pairs(gt_root: Path, order: str, episode_glob: str) -> list[dict[str, Any]]:
    task_dirs = sorted(path for path in gt_root.iterdir() if path.is_dir())
    if order == "folder-major":
        pairs = []
        for task_dir in task_dirs:
            videos_dir = task_dir / "videos"
            if not videos_dir.exists():
                continue
            for video_path in sorted(videos_dir.glob(episode_glob), key=episode_key):
                pairs.append({
                    "index": len(pairs) + 1,
                    "task_name": task_dir.name,
                    "episode_name": video_path.stem,
                    "gt_video": video_path,
                })
        return pairs

    by_episode: dict[int, list[tuple[str, str, Path]]] = {}
    for task_dir in task_dirs:
        videos_dir = task_dir / "videos"
        if not videos_dir.exists():
            continue
        for video_path in sorted(videos_dir.glob(episode_glob), key=episode_key):
            number, _ = episode_key(video_path)
            if number == sys.maxsize:
                continue
            by_episode.setdefault(number, []).append((task_dir.name, video_path.stem, video_path))

    pairs = []
    for number in sorted(by_episode):
        for task_name, episode_name, video_path in sorted(by_episode[number], key=lambda item: item[0]):
            pairs.append({
                "index": len(pairs) + 1,
                "task_name": task_name,
                "episode_name": episode_name,
                "gt_video": video_path,
            })
    return pairs


def load_instruction(gt_root: Path, task_name: str, episode_name: str) -> str:
    instruction_path = gt_root / task_name / "instructions" / f"{episode_name}.json"
    with open(instruction_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for key in ("instruction", "language_instruction", "task_instruction", "text", "prompt"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
    raise KeyError(f"Could not find an instruction string in {instruction_path}")


def resolve_policy_video(submission_root: Path, policy_template: str, model: str, item: dict[str, Any]) -> Path:
    rendered = policy_template.format(
        model=model,
        index=item["index"],
        task=item["task_name"],
        episode=item["episode_name"],
        episode_number=episode_number(item["episode_name"]),
    )
    path = Path(rendered)
    return path if path.is_absolute() else submission_root / path


def extract_frames_base64(video_path: Path, num_frames: int) -> list[str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    if total_frames < num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = [0]
        if num_frames > 2:
            step = (total_frames - 1) / (num_frames - 1)
            frame_indices.extend(int(i * step) for i in range(1, num_frames - 1))
        if num_frames > 1:
            frame_indices.append(total_frames - 1)

    frames_b64 = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        ok, buffer = cv2.imencode(".jpg", frame)
        if ok:
            frames_b64.append(base64.b64encode(buffer.tobytes()).decode("ascii"))

    cap.release()
    return frames_b64


def build_prompt(instruction: str) -> str:
    return f"""You are a robot task execution judge. Determine whether the policy model correctly executed the instruction.

**Task Instruction**: {instruction}

**Input Description**:
- The first images are uniformly sampled GT video frames showing the correct execution.
- The later images are uniformly sampled policy video frames to be judged.

**Evaluation Criteria**:
1. If the instruction explicitly requires the left or right arm, the correct arm must be used; otherwise judge failure.
2. Compare the final GT frame and final policy frame to determine whether the final task state is basically consistent.
3. Judge whether the policy motion and intent are consistent with the instruction semantics.

**Tolerable Differences**:
- Minor visual hallucinations, object deformation, or color shifts caused by the world model renderer.
- Small differences in trajectory or video length.

**Judge success (1)**:
- Correct arm is used.
- Final state is similar to GT and the task is basically completed.
- Action intent is correct.

**Judge failure (0)**:
- Wrong arm is used.
- Final state differs significantly from GT, or the task is not completed / completed incorrectly.
- Motion direction is completely wrong, or the wrong object is grasped/operated.

Focus on comparing the final frames of GT and policy. Reply in this format:
thinking: [brief analysis]
answer: [0 or 1]"""


def parse_vlm_response(response_text: str) -> dict[str, Any]:
    thinking_match = re.search(r"thinking:\s*(.+?)(?=answer:|$)", response_text, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r"answer:\s*\[?\s*([01])\s*\]?", response_text, re.IGNORECASE)
    thinking = thinking_match.group(1).strip() if thinking_match else response_text.strip()
    answer = int(answer_match.group(1)) if answer_match else -1
    return {"thinking": thinking, "answer": answer, "raw_response": response_text}


def call_vlm(
    gt_frames: list[str],
    policy_frames: list[str],
    instruction: str,
    api_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_backoff: float,
    trust_env_proxy: bool,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": build_prompt(instruction)}]
    gt_label = "--- GT Video Frames (Reference) ---"
    policy_label = "--- Policy Video Frames (To Judge) ---"

    content.append({"type": "text", "text": f"\n\n{gt_label}"})
    for frame_b64 in gt_frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}})

    content.append({"type": "text", "text": f"\n\n{policy_label}"})
    for frame_b64 in policy_frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    retryable_status_codes = {429, 500, 502, 503, 504}

    session = requests.Session()
    session.trust_env = trust_env_proxy
    for attempt in range(retries):
        try:
            response = session.post(api_url, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                response_text = data["choices"][0]["message"]["content"]
                return parse_vlm_response(response_text)
            if response.status_code in retryable_status_codes and attempt < retries - 1:
                wait_seconds = retry_backoff * (attempt + 1)
                print(f"    API returned {response.status_code}; retrying in {wait_seconds:.1f}s ({attempt + 2}/{retries})")
                time.sleep(wait_seconds)
                continue
            raise RuntimeError(f"API request failed: {response.status_code}\n{response.text}")
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ProxyError,
            requests.exceptions.ConnectionError,
            requests.exceptions.SSLError,
        ) as exc:
            if attempt < retries - 1:
                wait_seconds = retry_backoff * (attempt + 1)
                print(f"    API request failed with {type(exc).__name__}; retrying in {wait_seconds:.1f}s ({attempt + 2}/{retries})")
                time.sleep(wait_seconds)
                continue
            raise RuntimeError(f"API request failed after {retries} retries: {exc}") from exc

    raise RuntimeError("API request failed without a response")


def load_checkpoint(path: Path) -> tuple[list[dict[str, Any]], set[tuple[int, str]]]:
    if not path.exists():
        return [], set()
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    completed = set()
    for row in results:
        index = row.get("eval_index")
        model = row.get("policy_model")
        if index and model:
            completed.add((int(index), str(model)))
    return results, completed


def save_checkpoint(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "eval_index",
        "task_name",
        "episode_name",
        "policy_model",
        "instruction",
        "vlm_answer",
        "thinking",
        "policy_video",
        "gt_video",
        "error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_result_stub(item: dict[str, Any], model: str, policy_video: Path, instruction: str = "") -> dict[str, Any]:
    return {
        "eval_index": item["index"],
        "task_name": item["task_name"],
        "episode_name": item["episode_name"],
        "policy_model": model,
        "instruction": instruction,
        "policy_video": str(policy_video),
        "gt_video": str(item["gt_video"]),
    }


def judge_item(args: argparse.Namespace, item: dict[str, Any], model: str, api_key: str) -> dict[str, Any]:
    policy_video = resolve_policy_video(args.submission_root, args.policy_template, model, item)
    result = build_result_stub(item, model, policy_video)

    if not item["gt_video"].exists():
        result.update({"vlm_answer": -1, "error": f"GT video not found: {item['gt_video']}"})
        return result
    if not policy_video.exists():
        result.update({"vlm_answer": -1, "error": f"Policy video not found: {policy_video}"})
        return result

    instruction = load_instruction(args.gt_root, item["task_name"], item["episode_name"])
    result["instruction"] = instruction

    print(f"    GT: {item['gt_video']}")
    gt_frames = extract_frames_base64(item["gt_video"], args.num_frames)
    print(f"    Policy: {policy_video}")
    policy_frames = extract_frames_base64(policy_video, args.num_frames)
    if not gt_frames:
        result.update({"vlm_answer": -1, "error": f"No GT frames extracted: {item['gt_video']}"})
        return result
    if not policy_frames:
        result.update({"vlm_answer": -1, "error": f"No policy frames extracted: {policy_video}"})
        return result

    vlm_result = call_vlm(
        gt_frames=gt_frames,
        policy_frames=policy_frames,
        instruction=instruction,
        api_url=args.api_url,
        api_key=api_key,
        model=args.vlm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        retry_backoff=args.retry_backoff,
        trust_env_proxy=args.trust_env_proxy,
    )
    result.update(vlm_result)
    result["vlm_answer"] = vlm_result.get("answer", -1)
    return result


def print_summary(results: list[dict[str, Any]], models: list[str]) -> None:
    print("\nSummary")
    for model in models:
        rows = [row for row in results if row.get("policy_model") == model and not row.get("error")]
        success = sum(1 for row in rows if row.get("vlm_answer") == 1)
        if rows:
            print(f"  {model}: {success}/{len(rows)} = {success / len(rows) * 100:.1f}% success")
        else:
            print(f"  {model}: 0/0")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate robot policy videos with an OpenAI-compatible VLM judge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gt-root", type=Path, required=True, help="GT root containing task/videos and task/instructions subdirectories.")
    parser.add_argument("--submission-root", type=Path, required=True, help="Root directory containing policy model output videos.")
    parser.add_argument("--models", default="", help="Space- or comma-separated model directory names. If omitted, subdirectories under --submission-root are used.")
    parser.add_argument("--policy-template", default=DEFAULT_POLICY_TEMPLATE, help="Policy video path template relative to --submission-root. Variables: {model}, {index}, {task}, {episode}, {episode_number}.")
    parser.add_argument("--gt-order", choices=["folder-major", "episode-major"], default="folder-major", help="Ordering used to map 1-based submission indices to GT videos.")
    parser.add_argument("--episode-glob", default="episode*.mp4", help="Glob used to discover GT episode videos inside each task/videos directory.")
    parser.add_argument("--start-episode", type=positive_int, default=1, help="1-based GT mapping index to start from.")
    parser.add_argument("--max-episodes", type=int, default=0, help="Maximum number of GT mapping entries to evaluate; 0 means all.")
    parser.add_argument("--num-frames", type=positive_int, default=5, help="Number of uniformly sampled frames per video.")
    parser.add_argument("--api-url", default=os.getenv("VLM_API_URL", DEFAULT_API_URL), help="OpenAI-compatible chat completions endpoint.")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV, help="Environment variable name containing the API key.")
    parser.add_argument("--vlm-model", default=os.getenv("VLM_MODEL", DEFAULT_VLM_MODEL), help="VLM model name. Can also be set by VLM_MODEL.")
    parser.add_argument("--temperature", type=float, default=0.0, help="VLM sampling temperature. Use 0 for more deterministic judging when supported.")
    parser.add_argument("--max-tokens", type=positive_int, default=1024, help="Maximum VLM response tokens.")
    parser.add_argument("--timeout", type=positive_int, default=180, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=positive_int, default=5, help="Maximum API attempts per item.")
    parser.add_argument("--retry-backoff", type=float, default=10.0, help="Linear retry backoff in seconds.")
    parser.add_argument("--trust-env-proxy", action="store_true", help="Allow requests to use proxy variables from the environment.")
    parser.add_argument("--output-csv", type=Path, default=None, help="CSV output path.")
    parser.add_argument("--checkpoint-json", type=Path, default=None, help="JSON checkpoint path for resume.")
    parser.add_argument("--run-name", default="", help="Name used for default output file names.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved mappings without calling the VLM API.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first unexpected exception.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.gt_root = args.gt_root.expanduser().resolve()
    args.submission_root = args.submission_root.expanduser().resolve()

    if not args.gt_root.exists():
        raise FileNotFoundError(f"GT root does not exist: {args.gt_root}")
    if not args.submission_root.exists():
        raise FileNotFoundError(f"Submission root does not exist: {args.submission_root}")
    if not args.vlm_model and not args.dry_run:
        raise ValueError("VLM model is required. Set --vlm-model or VLM_MODEL.")

    api_key = os.getenv(args.api_key_env, "")
    if not api_key and not args.dry_run:
        raise ValueError(f"API key is required in environment variable {args.api_key_env}")

    models = parse_models(args.models, args.submission_root)
    gt_pairs = collect_gt_pairs(args.gt_root, args.gt_order, args.episode_glob)
    if args.start_episode > 1:
        gt_pairs = [item for item in gt_pairs if item["index"] >= args.start_episode]
    if args.max_episodes > 0:
        gt_pairs = gt_pairs[:args.max_episodes]
    if not gt_pairs:
        raise ValueError("No GT videos found for evaluation")

    run_name = sanitize_name(args.run_name or f"vlm_policy_eval_{args.vlm_model or 'dry_run'}")
    output_csv = args.output_csv or Path.cwd() / f"{run_name}.csv"
    checkpoint_json = args.checkpoint_json or Path.cwd() / f"{run_name}.json"

    results, completed = load_checkpoint(checkpoint_json)
    selected = {(item["index"], model) for item in gt_pairs for model in models}
    completed_selected = completed & selected
    total = len(selected)
    current = len(completed_selected)

    print(f"GT root: {args.gt_root}")
    print(f"Submission root: {args.submission_root}")
    print(f"GT order: {args.gt_order}")
    print(f"Policy template: {args.policy_template}")
    print(f"GT pairs: {len(gt_pairs)}")
    print(f"Models: {models}")
    print(f"VLM model: {args.vlm_model or '<dry-run>'}")
    print(f"Checkpoint: {checkpoint_json}")
    print(f"Output CSV: {output_csv}")
    print(f"Completed selected: {len(completed_selected)}/{total}")

    if args.dry_run:
        for item in gt_pairs[: min(len(gt_pairs), 20)]:
            for model in models:
                policy_video = resolve_policy_video(args.submission_root, args.policy_template, model, item)
                print(f"  {item['index']}: {item['task_name']}/{item['episode_name']} | {model} -> {policy_video}")
        print("Dry run finished; no API calls were made.")
        return 0

    for item in gt_pairs:
        print(f"\nEvaluation index {item['index']} -> {item['task_name']}/{item['episode_name']}")
        for model in models:
            key = (item["index"], model)
            if key in completed:
                print(f"  [{model}] skip")
                continue
            current += 1
            print(f"  [{current}/{total}] [{model}] judging")
            try:
                result = judge_item(args, item, model, api_key)
                if result.get("error"):
                    print(f"    error: {result['error']}")
                else:
                    print(f"    VLM answer: {result.get('vlm_answer', -1)}")
                results.append(result)
                completed.add(key)
                save_checkpoint(checkpoint_json, results)
            except Exception as exc:
                print(f"    fatal error: {exc}")
                traceback.print_exc()
                save_checkpoint(checkpoint_json, results)
                if args.fail_fast:
                    raise

    write_csv(output_csv, results)
    print(f"\nSaved JSON: {checkpoint_json}")
    print(f"Saved CSV: {output_csv}")
    print_summary(results, models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
