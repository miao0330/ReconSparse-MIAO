from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import fmean
from typing import Any


METRICS = ("pdms", "rc", "hdscore", "nc", "dac", "ttc", "c")
MODEL_NAMES = ("baseline_latest", "static_latest", "policy_loop_latest")


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return float("nan")
    position = max(0.0, min(1.0, float(probability))) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _bootstrap_ci(differences: list[float], *, samples: int, seed: int) -> tuple[float, float]:
    if not differences:
        return float("nan"), float("nan")
    rng = random.Random(int(seed))
    count = len(differences)
    estimates = [
        fmean(differences[rng.randrange(count)] for _ in range(count))
        for _ in range(int(samples))
    ]
    estimates.sort()
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _sign_test_p(wins: int, losses: int) -> float:
    trials = int(wins) + int(losses)
    if trials <= 0:
        return 1.0
    tail = min(int(wins), int(losses))
    probability = sum(math.comb(trials, value) for value in range(tail + 1)) / (2**trials)
    return min(1.0, 2.0 * float(probability))


def _load_rows(path: Path) -> dict[str, dict[tuple[int, str], dict[str, float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", {})
    if not isinstance(tasks, dict):
        raise TypeError("status.json tasks must be a mapping")
    rows: dict[str, dict[tuple[int, str], dict[str, float]]] = {name: {} for name in MODEL_NAMES}
    for task in tasks.values():
        if not isinstance(task, dict) or task.get("status") != "completed":
            continue
        name = str(task.get("display_name", ""))
        if name not in rows:
            continue
        metrics = task.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        key = (int(task["repeat"]), Path(str(task["scenario_path"])).stem)
        rows[name][key] = {metric: float(metrics[metric]) for metric in METRICS}
    return rows


def _analyze(rows: dict[str, dict[tuple[int, str], dict[str, float]]], *, samples: int) -> dict[str, Any]:
    shared_keys = sorted(set.intersection(*(set(rows[name]) for name in MODEL_NAMES)))
    if not shared_keys:
        raise RuntimeError("No completed paired evaluation cells were found")
    model_means = {
        name: {
            metric: fmean(rows[name][key][metric] for key in shared_keys)
            for metric in METRICS
        }
        for name in MODEL_NAMES
    }
    comparisons: dict[str, Any] = {}
    for comparator in ("baseline_latest", "static_latest"):
        metric_rows: dict[str, Any] = {}
        for metric_index, metric in enumerate(METRICS):
            differences = [
                rows["policy_loop_latest"][key][metric] - rows[comparator][key][metric]
                for key in shared_keys
            ]
            wins = sum(value > 1.0e-12 for value in differences)
            losses = sum(value < -1.0e-12 for value in differences)
            ties = len(differences) - wins - losses
            lower, upper = _bootstrap_ci(
                differences,
                samples=int(samples),
                seed=20260717 + int(metric_index),
            )
            comparator_mean = float(model_means[comparator][metric])
            mean_difference = float(fmean(differences))
            metric_rows[metric] = {
                "policy_mean": float(model_means["policy_loop_latest"][metric]),
                "comparator_mean": comparator_mean,
                "mean_difference": mean_difference,
                "relative_difference_percent": (
                    100.0 * mean_difference / comparator_mean if abs(comparator_mean) > 1.0e-12 else None
                ),
                "bootstrap_95_ci": [float(lower), float(upper)],
                "wins": int(wins),
                "ties": int(ties),
                "losses": int(losses),
                "sign_test_two_sided_p": float(_sign_test_p(wins, losses)),
            }
        comparisons[f"policy_loop_vs_{comparator.removesuffix('_latest')}"] = metric_rows
    return {
        "paired_cells": len(shared_keys),
        "bootstrap_samples": int(samples),
        "model_means": model_means,
        "comparisons": comparisons,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Policy-loop stage-1 paired analysis",
        "",
        f"Paired scene-repeat cells: {int(result['paired_cells'])}",
        "",
    ]
    for comparison_name, comparison in result["comparisons"].items():
        lines.extend(
            [
                f"## {comparison_name}",
                "",
                "| Metric | Policy | Comparator | Difference | Relative | 95% bootstrap CI | W/T/L | Sign p |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for metric in METRICS:
            row = comparison[metric]
            relative = row["relative_difference_percent"]
            relative_text = "n/a" if relative is None else f"{float(relative):.2f}%"
            ci = row["bootstrap_95_ci"]
            lines.append(
                f"| {metric} | {row['policy_mean']:.6f} | {row['comparator_mean']:.6f} | "
                f"{row['mean_difference']:+.6f} | {relative_text} | "
                f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | "
                f"{row['wins']}/{row['ties']}/{row['losses']} | {row['sign_test_two_sided_p']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired analysis for the stage-1 three-arm evaluation")
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    args = parser.parse_args()
    rows = _load_rows(args.status_json.resolve())
    result = _analyze(rows, samples=int(args.bootstrap_samples))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paired_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    markdown = _markdown(result)
    (output_dir / "paired_analysis.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
