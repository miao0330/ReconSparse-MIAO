from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_DIR = Path("/root/clone/HUGSIM-ORI/configs/scenarios/nuscenes")
DEFAULT_INITIAL_CKPT = Path(
    "/OpenDataset/ReconDreamer-RL/outputs/actor_learner/"
    "20260701_035459_HUGSIM_Template_ReinforcePP_NoGRPO_NoAux_Substeps1/"
    "weights/latest.ckpt"
)
DEFAULT_CONFIG_DIR = REPO_ROOT / "script/configs/sparsedrive_v2/experiments"
DEFAULT_OUTPUT_ROOT = Path("/OpenDataset/ReconDreamer-RL/outputs/experiments/full_nuscenes_88_collision_only")


def scenario_names(scenario_dir: Path) -> list[str]:
    names = sorted(path.name for path in scenario_dir.glob("*.yaml"))
    if len(names) != 88:
        raise RuntimeError(f"Expected 88 HUGSIM NuScenes YAMLs, found {len(names)} in {scenario_dir}")
    return names


def collision_only_reward(*, collision_weight: float) -> dict[str, Any]:
    return {
        "step_path": {
            "CRAFT": {"enable": False},
            "path": {
                "interp_ds_m": 0.25,
                "w_progress": 0.0,
                "progress_forward_cap_m": 2.0,
                "progress_backward_cap_m": 0.5,
                "w_lateral": 0.0,
                "lateral_free_m": 0.3,
                "lateral_huber_delta_m": 0.5,
                "w_yaw": 0.0,
                "yaw_free_deg": 5.0,
                "yaw_huber_delta_deg": 10.0,
                "severe_gate_scale": 0.0,
            },
            "comfort": {
                "w_longitudinal_jerk": 0.0,
                "w_yaw_jerk": 0.0,
                "longitudinal_jerk_free": 1.5,
                "longitudinal_jerk_delta": 2.0,
                "yaw_jerk_free": 0.2,
                "yaw_jerk_delta": 0.3,
                "jerk_clip": 50.0,
            },
            "collision": {
                "mode": "dense_penalty",
                "w_static": float(collision_weight),
                "w_dynamic": float(collision_weight),
                "gate_scale": 0.0,
            },
            "safety": {
                "enable": False,
                "lookahead_m": 15.0,
                "corridor_half_width_m": 2.5,
                "safe_gap_m": 8.0,
                "safe_ttc_s": 3.0,
                "w_clearance": 0.0,
                "w_ttc": 0.0,
                "progress_gate_strength": 0.0,
                "min_progress_gate": 1.0,
            },
        }
    }


def common_config(*, scenarios: list[str], output_root: Path, arm: str, initial_ckpt: Path, collision_weight: float) -> dict[str, Any]:
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
                "output_root": str(output_root / "hugsim" / arm),
            },
            "reward": collision_only_reward(collision_weight=collision_weight),
        },
        "train": {
            "wandb": {"enabled": False, "group": f"FullNuScenes88_CollisionOnly_{arm}"},
            "actor_learner": {
                "buffer_dir": str(output_root / "train" / arm / "actor_learner"),
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


def policy_loop_branch_collision_only(*, collision_weight: float) -> dict[str, Any]:
    reward = collision_only_reward(collision_weight=collision_weight)["step_path"]
    return {
        "reward": {
            "mode": "step_path",
            "common": {
                "dt": 0.5,
                "ea": {"enable": False, "weight": 0.0},
                "terminal": {
                    "enable": True,
                    "penalty": 0.0,
                    "success_bonus": 0.0,
                    "apply_on_failure": True,
                    "apply_on_timeout": False,
                    "apply_on_env_done": False,
                },
            },
            "step_path": copy.deepcopy(reward),
        }
    }


def write_yaml(path: Path, payload: dict[str, Any], *, generated_at: str) -> None:
    path.write_text(
        f"# Generated at: {generated_at}\n# Reward: collision-only dense penalty; all progress/tracking/comfort/safety weights are zero.\n"
        + yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def generate(*, scenario_dir: Path, config_dir: Path, output_root: Path, initial_ckpt: Path, collision_weight: float, overwrite: bool) -> list[Path]:
    scenarios = scenario_names(scenario_dir)
    baseline = common_config(
        scenarios=scenarios,
        output_root=output_root,
        arm="baseline",
        initial_ckpt=initial_ckpt,
        collision_weight=collision_weight,
    )
    static = copy.deepcopy(baseline)
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
    static["train"]["wandb"]["group"] = "FullNuScenes88_CollisionOnly_static_GRPO"
    static["train"]["actor_learner"]["buffer_dir"] = str(output_root / "train" / "static_grpo" / "actor_learner")
    static["env"]["hugsim"]["output_root"] = str(output_root / "hugsim" / "static_grpo")

    policy = copy.deepcopy(baseline)
    policy["extends"] = [
        *policy["extends"][:-1],
        "../template/train/policy_loop_grpo.yaml",
        "../template/agent/sparsedrive_v2.yaml",
    ]
    policy["train"]["wandb"]["group"] = "FullNuScenes88_CollisionOnly_policy_loop_GRPO"
    policy["train"]["actor_learner"]["buffer_dir"] = str(output_root / "train" / "policy_loop_grpo" / "actor_learner")
    policy["env"]["hugsim"]["output_root"] = str(output_root / "hugsim" / "policy_loop_grpo")
    policy["train"]["policy_loop_grpo"] = {
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
        **policy_loop_branch_collision_only(collision_weight=collision_weight),
    }

    payloads = {
        "full_nuscenes_88_collision_only_baseline.yaml": baseline,
        "full_nuscenes_88_collision_only_static_grpo.yaml": static,
        "full_nuscenes_88_collision_only_policy_loop_grpo.yaml": policy,
    }
    config_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    paths: list[Path] = []
    for filename, payload in payloads.items():
        path = config_dir / filename
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
        write_yaml(path, payload, generated_at=generated_at)
        paths.append(path)
    print(f"generated_at={generated_at} scenarios={len(scenarios)} collision_weight={collision_weight}")
    for path in paths:
        print(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate 88-scene collision-only SparseDriveV2 experiment YAMLs.")
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--initial-ckpt", type=Path, default=DEFAULT_INITIAL_CKPT)
    parser.add_argument("--collision-weight", type=float, default=500.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate(
        scenario_dir=args.scenario_dir.resolve(),
        config_dir=args.config_dir.resolve(),
        output_root=args.output_root,
        initial_ckpt=args.initial_ckpt.resolve(),
        collision_weight=float(args.collision_weight),
        overwrite=bool(args.overwrite),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
