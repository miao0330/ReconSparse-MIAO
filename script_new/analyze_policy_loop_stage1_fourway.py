from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import fmean


METRICS = ("pdms", "rc", "hdscore", "nc", "dac", "ttc", "c")
MODELS = ("ori_latest", "baseline_latest", "static_latest", "policy_loop_latest")


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    position = max(0.0, min(1.0, probability)) * (len(values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    if not values:
        return float("nan"), float("nan")
    estimates = [
        fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(samples)
    ]
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def sign_p(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(trials, k) for k in range(tail + 1)) / (2**trials)
    return min(1.0, 2.0 * probability)


def load_rows(path: Path) -> dict[str, dict[tuple[int, str], dict[str, float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {model: {} for model in MODELS}
    for task in (payload.get("tasks", {}) or {}).values():
        if task.get("status") != "completed":
            continue
        model = str(task.get("display_name", ""))
        if model not in rows:
            continue
        metrics = task.get("metrics", {})
        key = (int(task["repeat"]), Path(str(task["scenario_path"])).stem)
        rows[model][key] = {metric: float(metrics[metric]) for metric in METRICS}
    return rows


def analyze(rows: dict[str, dict[tuple[int, str], dict[str, float]]], samples: int) -> dict:
    shared = sorted(set.intersection(*(set(rows[model]) for model in MODELS)))
    if not shared:
        raise RuntimeError("No fully paired cells")
    means = {
        model: {metric: fmean(rows[model][key][metric] for key in shared) for metric in METRICS}
        for model in MODELS
    }
    comparisons = {}
    for comparator in ("ori_latest", "baseline_latest", "static_latest"):
        result = {}
        for index, metric in enumerate(METRICS):
            diffs = [
                rows["policy_loop_latest"][key][metric] - rows[comparator][key][metric]
                for key in shared
            ]
            wins = sum(value > 1e-12 for value in diffs)
            losses = sum(value < -1e-12 for value in diffs)
            ci = bootstrap_ci(diffs, samples, 20260719 + index)
            result[metric] = {
                "policy_mean": means["policy_loop_latest"][metric],
                "comparator_mean": means[comparator][metric],
                "difference": fmean(diffs),
                "relative_percent": 100.0 * fmean(diffs) / means[comparator][metric]
                if abs(means[comparator][metric]) > 1e-12 else None,
                "ci95": ci,
                "wins": wins,
                "ties": len(diffs) - wins - losses,
                "losses": losses,
                "sign_p": sign_p(wins, losses),
            }
        comparisons[f"policy_loop_vs_{comparator.removesuffix('_latest')}"] = result
    return {"paired_cells": len(shared), "model_means": means, "comparisons": comparisons}


def markdown(result: dict) -> str:
    lines = ["# Four-way Policy-loop analysis", "", f"Paired cells: {result['paired_cells']}", ""]
    for name, metrics in result["comparisons"].items():
        lines += [f"## {name}", "", "| Metric | Policy | Comparator | Difference | Relative | 95% CI | W/T/L | Sign p |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for metric in METRICS:
            row = metrics[metric]
            ci = row["ci95"]
            lines.append(
                f"| {metric} | {row['policy_mean']:.6f} | {row['comparator_mean']:.6f} | "
                f"{row['difference']:+.6f} | {row['relative_percent']:+.2f}% | "
                f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | {row['wins']}/{row['ties']}/{row['losses']} | {row['sign_p']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    args = parser.parse_args()
    result = analyze(load_rows(args.status_json.resolve()), int(args.bootstrap_samples))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = markdown(result)
    (args.output_dir / "paired_analysis.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
