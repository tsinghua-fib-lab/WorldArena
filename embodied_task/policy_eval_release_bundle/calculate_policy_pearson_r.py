#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_SIMULATOR_SCORES = {
    "10data": 28.6,
    "20data": 34.58,
    "30data": 37.78,
    "50data": 43.52,
    "fulldata": 46.8,
}


DEFAULT_EVAL_PATH = Path(
    "/ML-vePFS/protected/wzy/WorldModel/Ctrl-World/output/results/"
    "chill_evaluation_checkpoint_Qwen3_VL_32B_Instruct.json"
)


MODEL_KEYS = ("10data", "20data", "30data", "50data", "fulldata")


def normalize_model_name(name: str) -> str:
    compact = name.strip().lower()
    compact = compact.replace("-", "_")
    compact = compact.replace("radiodata", "data")
    compact = compact.replace("robotwin_all_clean_wnorm_wowrist_", "")
    compact = compact.replace("robohero_0_", "")
    compact = compact.replace("my_model_", "")
    compact = compact.replace("kinetra_", "")
    compact = compact.replace("/", "_")
    for key in MODEL_KEYS:
        if key in compact:
            return key
    return compact


def parse_answer(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_eval_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return data
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported evaluation file type: {path.suffix}")


def load_simulator_scores(path: Path | None) -> dict[str, float]:
    if path is None:
        return DEFAULT_SIMULATOR_SCORES.copy()
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {normalize_model_name(k): float(v) for k, v in data.items()}
        if isinstance(data, list):
            scores = {}
            for row in data:
                model = row.get("model") or row.get("policy_model") or row.get("WorldModel_Policy_Model")
                score = row.get("score") or row.get("simulator_score") or row.get("Simulator_pi05结果")
                if model is not None and score is not None:
                    scores[normalize_model_name(str(model))] = float(score)
            return scores
        raise ValueError(f"Unsupported simulator JSON format: {path}")
    if path.suffix.lower() == ".csv":
        scores = {}
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row.get("model") or row.get("policy_model") or row.get("WorldModel_Policy_Model")
                score = row.get("score") or row.get("simulator_score") or row.get("Simulator_pi05结果")
                if model is not None and score is not None:
                    scores[normalize_model_name(str(model))] = float(score)
        return scores
    raise ValueError(f"Unsupported simulator file type: {path.suffix}")


def aggregate_vlm_scores(rows: list[dict[str, Any]], mode: str) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for row in rows:
        model = row.get("policy_model") or row.get("model") or row.get("WorldModel_Policy_Model")
        if not model:
            continue
        key = normalize_model_name(str(model))
        item = stats.setdefault(key, {"rows": 0.0, "valid": 0.0, "success": 0.0, "errors": 0.0})
        item["rows"] += 1
        answer = parse_answer(row.get("vlm_answer", row.get("answer")))
        has_error = bool(row.get("error"))
        if has_error:
            item["errors"] += 1
        is_valid = answer in (0, 1) and not has_error
        if is_valid:
            item["valid"] += 1
        if answer == 1 and not has_error:
            item["success"] += 1
    for item in stats.values():
        denominator = item["rows"] if mode == "errors-as-fail" else item["valid"]
        item["rate"] = item["success"] / denominator * 100.0 if denominator else math.nan
    return stats


def pearson_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 2:
        raise ValueError("Need at least two paired values")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return math.nan
    return numerator / (denom_x * denom_y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Pearson R between VLM policy scores and simulator scores.")
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL_PATH, help="VLM evaluation JSON checkpoint or CSV.")
    parser.add_argument("--sim-scores", type=Path, default=None, help="Optional simulator scores CSV/JSON.")
    parser.add_argument(
        "--mode",
        choices=("valid-only", "errors-as-fail"),
        default="valid-only",
        help="Use only valid VLM answers or count error rows in the denominator.",
    )
    args = parser.parse_args()

    rows = load_eval_rows(args.eval)
    simulator_scores = load_simulator_scores(args.sim_scores)
    vlm_stats = aggregate_vlm_scores(rows, args.mode)

    pairs = []
    for key in MODEL_KEYS:
        if key in simulator_scores and key in vlm_stats and not math.isnan(vlm_stats[key]["rate"]):
            pairs.append((key, vlm_stats[key], simulator_scores[key]))

    if len(pairs) < 2:
        raise RuntimeError("Need at least two matched models to compute Pearson R")

    vlm_values = [item["rate"] for _, item, _ in pairs]
    sim_values = [score for _, _, score in pairs]
    r_value = pearson_r(vlm_values, sim_values)

    print(f"Evaluation file: {args.eval}")
    print(f"Mode: {args.mode}")
    print("")
    print(f"{'model':<10} {'vlm_success_%':>14} {'simulator_%':>12} {'success':>10} {'valid':>8} {'rows':>8} {'errors':>8}")
    print("-" * 82)
    for key, item, sim_score in pairs:
        print(
            f"{key:<10} {item['rate']:>14.4f} {sim_score:>12.4f} "
            f"{int(item['success']):>10} {int(item['valid']):>8} {int(item['rows']):>8} {int(item['errors']):>8}"
        )
    print("")
    print(f"Pearson R = {r_value:.6f}")


if __name__ == "__main__":
    main()
