from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_DIR = Path("/root/clone/HUGSIM-ORI/configs/scenarios/nuscenes")
DEFAULT_INITIAL_CKPT = Path(
    "/OpenDataset/ReconDreamer-RL/outputs/actor_learner/"
    "20260701_035459_HUGSIM_Template_ReinforcePP_NoGRPO_NoAux_Substeps1/"
    "weights/latest.ckpt"
)
DEFAULT_CONFIG_DIR = REPO_ROOT / "script/configs/sparsedrive_v2/experiments"
DEFAULT_RUN_ROOT = Path("/OpenDataset/ReconDreamer-RL/outputs/experiments/full_nuscenes_88_20260721")
DEFAULT_TRAIN_PYTHON = Path("/root/miniconda3/envs/recondreamerNew-rl/bin/python")
DEFAULT_HUGSIM_TEMPLATE = Path(
    "/root/clone/HUGSIM-ORI/configs/sim/nuscenes_eval_sparsedrive_v2_ppo_grpo_ver14.yaml"
)
DEFAULT_SLOTS = ("0:0", "1:1", "2:2", "3:3", "4:4", "5:5", "6:6", "7:7")


def _scenario_names(scenario_dir: Path) -> list[str]:
    names = sorted(path.name for path in scenario_dir.glob("*.yaml"))
    if len(names) != 88:
        raise RuntimeError(f"Expected exactly 88 HUGSIM scenario YAMLs under {scenario_dir}, found {len(names)}")
    return names


def _common_config(*, scenarios: list[str], run_root: Path, arm: str, initial_ckpt: Path) -> dict[str, Any]:
    return {
        "extends": [
            "../template/base.yaml",
            "../template/env/base.yaml",
            "../template/env/rewards/step_path.yaml",
            "../template/train/actor_learner.yaml",
            "../template/train/reinforcepp.yaml",
            "../template/agent/sparsedrive_v2.yaml",
        ],
        "env": {
            "use_all_scenes": False,
            "scene_sampling": "random",
            "max_steps": 36,
            "start_frame": {"mode": "zero", "allow_short_tail": False, "min": 0},
            "hugsim": {
                "scenes": list(scenarios),
                "output_root": str(run_root / "hugsim" / arm),
            },
            "reward": {
                "step_path": {
                    "collision": {
                        "mode": "dense_penalty",
                        "w_static": 500.0,
                        "w_dynamic": 500.0,
                    }
                }
            },
        },
        "train": {
            "wandb": {
                "enabled": False,
                "group": f"FullNuScenes88_{arm}",
            },
            "actor_learner": {
                "buffer_dir": str(run_root / "train" / arm / "actor_learner"),
                "timestamp_buffer_dir": False,
                "learner_gpu_id": 0,
                "learner_gpu_ids": [0],
                "actor_gpu_pool": [1, 2, 3, 4, 5, 6, 7],
                "actors_per_gpu": 1,
                "auto_max_inflight_per_actor": False,
                "max_inflight_per_actor": 64,
                "actor_horizon": 32,
                "num_envs_per_actor": 1,
                "vec_env_mode": "serial",
                "samples_per_update": 786,
                "shards_per_update": 48,
                "shard_collect_timeout_s": 1800,
                "actor_shard_stall_timeout_s": 3600,
                "allow_partial_updates_after_timeout": True,
                "restart_failed_actors": False,
                "max_actor_restarts": 0,
                "max_updates": 32,
            },
        },
        "agent": {"ckpt": str(initial_ckpt)},
    }


def build_experiment_configs(
    *,
    scenarios: list[str],
    config_dir: Path,
    run_root: Path,
    initial_ckpt: Path,
    overwrite: bool,
) -> dict[str, Path]:
    configs: dict[str, dict[str, Any]] = {}

    baseline = _common_config(
        scenarios=scenarios,
        run_root=run_root,
        arm="baseline_collision500",
        initial_ckpt=initial_ckpt,
    )
    configs["baseline_collision500"] = baseline

    static = copy.deepcopy(baseline)
    # Keep the Reinforce++ optimizer defaults and add static GRPO on top.
    static["extends"] = [*static["extends"][:-1], "../template/train/grpo.yaml", "../template/agent/sparsedrive_v2.yaml"]
    static["train"]["grpo"] = {
        "enable": True,
        "coef": 0.1,
        "num_candidates": 4,
        "candidate_select": "topk",
        "objective": "grpo",
        "norm_eps": 1.0e-6,
        "use_rank_adv": False,
    }
    static["train"]["wandb"]["group"] = "FullNuScenes88_static_GRPO"
    static["env"]["hugsim"]["output_root"] = str(run_root / "hugsim" / "static_grpo")
    static["train"]["actor_learner"]["buffer_dir"] = str(run_root / "train" / "static_grpo" / "actor_learner")
    configs["static_grpo"] = static

    policy_loop = copy.deepcopy(baseline)
    policy_loop["extends"] = [
        *policy_loop["extends"][:-1],
        "../template/env/rewards/policy_loop_dense.yaml",
        "../template/train/policy_loop_grpo.yaml",
        "../template/agent/sparsedrive_v2.yaml",
    ]
    policy_loop["train"]["wandb"]["group"] = "FullNuScenes88_policy_loop_GRPO"
    policy_loop["train"]["actor_learner"]["buffer_dir"] = str(run_root / "train" / "policy_loop_grpo" / "actor_learner")
    policy_loop["env"]["hugsim"]["output_root"] = str(run_root / "hugsim" / "policy_loop_grpo")
    policy_loop["train"]["policy_loop_grpo"] = {
        "enable": True,
        "coef": 0.1,
        "num_candidates": 4,
        "branch_horizon_steps": 16,
        "branch_parallelism": 4,
        "branch_bootstrap": "auto",
        "keyframe_times_s": [4.0, 8.0],
        "rl_step_s": 0.5,
        "intervention_steps": 2,
        "severe_offroad_lateral_m": 3.5,
        "continuation_mode": "argmax",
        "deterministic_seed": True,
        "norm_eps": 1.0e-6,
        "train_metric_heads_only": True,
        "metric_heads_prefix": "_trajectory_head.decoder.layers.1.metric_heads",
    }
    configs["policy_loop_grpo"] = policy_loop

    config_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, payload in configs.items():
        path = config_dir / f"full_nuscenes_88_{name}.yaml"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing experiment config: {path}")
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        paths[name] = path
    return paths


def _run_one(
    *,
    name: str,
    config_path: Path,
    args: argparse.Namespace,
    sequence_log: Path,
) -> None:
    cmd = [
        str(args.train_python),
        "-u",
        str(REPO_ROOT / "script/train_eval_pipeline.py"),
        "--reinforcepp-config",
        str(config_path),
        "--reinforcepp-algo-tag",
        f"full_nuscenes_88_{name}",
        "--run-root",
        str(args.run_root / name),
        "--hugsim-template",
        str(args.hugsim_template),
        "--scenario-dir",
        str(args.scenario_dir),
        "--max-scenes",
        "88",
        "--repeat-evals",
        str(args.repeat_evals),
        "--retry-count",
        str(args.retry_count),
        "--slots",
        *args.slots,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with sequence_log.open("a", encoding="utf-8") as handle:
        handle.write(f"START {name} {started} config={config_path}\n")
        handle.flush()
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
    finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with sequence_log.open("a", encoding="utf-8") as handle:
        handle.write(f"END {name} {finished} returncode={result.returncode}\n")
    if result.returncode != 0:
        raise RuntimeError(f"Full 88-scene experiment {name} failed with code {result.returncode}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and sequentially run the full 88-scene HUGSIM matrix.")
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--initial-ckpt", type=Path, default=DEFAULT_INITIAL_CKPT)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--train-python", type=Path, default=DEFAULT_TRAIN_PYTHON)
    parser.add_argument("--hugsim-template", type=Path, default=DEFAULT_HUGSIM_TEMPLATE)
    parser.add_argument("--repeat-evals", type=int, default=2)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--slots", nargs="+", default=list(DEFAULT_SLOTS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("baseline_collision500", "static_grpo", "policy_loop_grpo"),
        default=["baseline_collision500", "static_grpo", "policy_loop_grpo"],
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.scenario_dir = args.scenario_dir.resolve()
    args.initial_ckpt = args.initial_ckpt.resolve()
    args.config_dir = args.config_dir.resolve()
    args.run_root = args.run_root.resolve()
    args.train_python = args.train_python.resolve()
    args.hugsim_template = args.hugsim_template.resolve()
    if not args.scenario_dir.is_dir():
        raise FileNotFoundError(args.scenario_dir)
    if not args.initial_ckpt.is_file():
        raise FileNotFoundError(args.initial_ckpt)
    if not args.train_python.is_file():
        raise FileNotFoundError(args.train_python)
    if not args.hugsim_template.is_file():
        raise FileNotFoundError(args.hugsim_template)
    scenarios = _scenario_names(args.scenario_dir)
    paths = build_experiment_configs(
        scenarios=scenarios,
        config_dir=args.config_dir,
        run_root=args.run_root,
        initial_ckpt=args.initial_ckpt,
        overwrite=bool(args.overwrite),
    )
    print(f"[full-88] scenarios={len(scenarios)} configs={args.config_dir}", flush=True)
    for name, path in paths.items():
        print(f"[full-88] config arm={name} path={path}", flush=True)
    if bool(args.generate_only):
        return 0

    args.run_root.mkdir(parents=True, exist_ok=True)
    sequence_log = args.run_root / "sequence.log"
    for name in args.arms:
        _run_one(name=name, config_path=paths[name], args=args, sequence_log=sequence_log)
        print(f"[full-88] completed arm={name}", flush=True)
    print(f"[full-88] all arms completed; sequence_log={sequence_log}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
