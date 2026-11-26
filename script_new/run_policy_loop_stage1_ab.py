from __future__ import annotations

import argparse
import copy
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = (
    REPO_ROOT
    / "script/configs/sparsedrive_v2/20260717_HUGSM_policy_loop_grpo_stage1_smoke.yaml"
)
DEFAULT_INITIAL_CKPT = Path(
    "/OpenDataset/ReconDreamer-RL/outputs/actor_learner/"
    "20260701_035459_HUGSIM_Template_ReinforcePP_NoGRPO_NoAux_Substeps1/"
    "weights/latest.ckpt"
)
DEFAULT_RUN_ROOT = REPO_ROOT / "outputs/experiments/policy_loop_stage1_ab_20260717"

ARMS = {
    "baseline": {"learner_gpu": 0, "actor_gpu": 1, "static_grpo": False, "policy_loop": False},
    "static": {"learner_gpu": 2, "actor_gpu": 3, "static_grpo": True, "policy_loop": False},
    "policy_loop": {"learner_gpu": 4, "actor_gpu": 5, "static_grpo": False, "policy_loop": True},
}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}")
    return payload


def _arm_config(
    base: dict[str, Any],
    *,
    arm: str,
    run_root: Path,
    initial_ckpt: Path,
    max_updates: int,
    actor_horizon: int,
    samples_per_update: int,
    shards_per_update: int,
    max_inflight_per_actor: int,
    env_max_steps: int | None,
    scene_sampling: str | None,
    single_scene: str | None,
    collision_penalty: float | None,
    actor_gpu_pool: list[int] | None,
) -> dict[str, Any]:
    spec = ARMS[arm]
    cfg = copy.deepcopy(base)
    env_cfg = cfg.setdefault("env", {})
    if env_max_steps is not None:
        env_cfg["max_steps"] = int(env_max_steps)
    if scene_sampling is not None:
        env_cfg["scene_sampling"] = str(scene_sampling)
    if single_scene:
        scene_name = str(single_scene)
        if not scene_name.endswith(".yaml"):
            scene_name = f"{scene_name}.yaml"
        env_cfg["use_all_scenes"] = False
        env_cfg.setdefault("hugsim", {})["scenes"] = [scene_name]
    if collision_penalty is not None:
        reward_cfg = env_cfg.setdefault("reward", {})
        step_path_cfg = reward_cfg.setdefault("step_path", {})
        collision_cfg = step_path_cfg.setdefault("collision", {})
        collision_cfg["mode"] = "dense_penalty"
        collision_cfg["w_static"] = float(collision_penalty)
        collision_cfg["w_dynamic"] = float(collision_penalty)
    env_cfg.setdefault("hugsim", {})["output_root"] = str(run_root / "hugsim" / arm)

    train_cfg = cfg.setdefault("train", {})
    train_cfg["algo"] = (
        "reinforcepp-grpo-noaux" if bool(spec["static_grpo"]) else "reinforcepp-nogrpo-noaux"
    )
    train_cfg.setdefault("wandb", {})["enabled"] = False
    train_cfg["wandb"]["group"] = f"PolicyLoopStage1AB_{arm}"

    actor_learner = train_cfg.setdefault("actor_learner", {})
    actor_learner.update(
        {
            "buffer_dir": str(run_root / "train" / arm),
            "timestamp_buffer_dir": False,
            "learner_gpu_id": int(spec["learner_gpu"]),
            "learner_gpu_ids": [int(spec["learner_gpu"])],
            "actor_gpu_pool": (
                [int(gpu) for gpu in actor_gpu_pool]
                if actor_gpu_pool
                else [int(spec["actor_gpu"])]
            ),
            "actors_per_gpu": 1,
            "auto_max_inflight_per_actor": False,
            "max_inflight_per_actor": int(max_inflight_per_actor),
            "actor_horizon": int(actor_horizon),
            "num_envs_per_actor": 1,
            "vec_env_mode": "serial",
            # samples_per_update is the hard lower bound used by the async
            # learner; a 786 target with 32-step shards selects 25 shards
            # (800 actual steps), preserving the legacy step floor.
            "samples_per_update": int(samples_per_update),
            "shards_per_update": int(shards_per_update),
            "shard_collect_timeout_s": 1800,
            "actor_shard_stall_timeout_s": 3600,
            "allow_partial_updates_after_timeout": True,
            "restart_failed_actors": False,
            "max_actor_restarts": 0,
            "max_updates": int(max_updates),
        }
    )

    grpo_cfg = train_cfg.setdefault("grpo", {})
    grpo_cfg.update(
        {
            "enable": bool(spec["static_grpo"]),
            "coef": 0.1 if bool(spec["static_grpo"]) else 0.0,
            "num_candidates": 4 if bool(spec["static_grpo"]) else 0,
            "candidate_select": "topk",
            "objective": "grpo",
            "norm_eps": 1.0e-6,
            "use_rank_adv": False,
        }
    )

    policy_loop_cfg = train_cfg.setdefault("policy_loop_grpo", {})
    policy_loop_cfg.update(
        {
            "enable": bool(spec["policy_loop"]),
            "coef": 0.1,
            "num_candidates": 4,
            "branch_horizon_steps": 16,
            "branch_parallelism": 4,
            "keyframe_times_s": [4.0, 8.0],
            "rl_step_s": 0.5,
            "intervention_steps": 2,
            "severe_offroad_lateral_m": 3.5,
            "continuation_mode": "argmax",
            "deterministic_seed": True,
            "train_metric_heads_only": True,
            "metric_heads_prefix": "_trajectory_head.decoder.layers.1.metric_heads",
        }
    )

    agent_cfg = cfg.setdefault("agent", {})
    agent_cfg["type"] = "sparsedrive_v2"
    agent_cfg["ckpt"] = str(initial_ckpt)
    agent_cfg["trainable_prefixes"] = ["_trajectory_head.decoder.layers.1.metric_heads"]
    return cfg


def _tail(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return "waiting for log"
    return lines[-1] if lines else "empty log"


def _publish_named_checkpoints(run_root: Path) -> None:
    checkpoint_root = run_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        source = run_root / "train" / arm / "weights/latest.ckpt"
        target = checkpoint_root / f"{arm}_latest.ckpt"
        if target.exists():
            target.unlink()
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the stage-1 Policy-loop three-arm training matrix")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--initial-ckpt", type=Path, default=DEFAULT_INITIAL_CKPT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--max-updates", type=int, default=4)
    parser.add_argument(
        "--actor-horizon",
        type=int,
        default=32,
        help="RL steps per actor shard; legacy SparseDriveV2 default is 32.",
    )
    parser.add_argument(
        "--samples-per-update",
        type=int,
        default=786,
        help="Minimum replay steps consumed per learner update (786 -> 800 with 32-step shards).",
    )
    parser.add_argument(
        "--shards-per-update",
        type=int,
        default=48,
        help="Shard target used when sample sizing is disabled or as async capacity.",
    )
    parser.add_argument(
        "--max-inflight-per-actor",
        type=int,
        default=64,
        help="Maximum unconsumed shards per actor; must cover the 786-step sample target.",
    )
    parser.add_argument(
        "--env-max-steps",
        type=int,
        default=36,
        help="Maximum RL steps in the source environment; use 20 for the smoke config.",
    )
    parser.add_argument(
        "--scene-sampling",
        default="random",
        help="Source scene sampling mode; legacy training uses random.",
    )
    parser.add_argument(
        "--single-scene",
        default=None,
        help="Restrict training to one HUGSIM scenario, e.g. scene-0411-medium-00.yaml.",
    )
    parser.add_argument(
        "--collision-penalty",
        type=float,
        default=None,
        help="Override step_path dense collision penalty for both static and dynamic collisions.",
    )
    parser.add_argument(
        "--actor-gpus",
        type=int,
        nargs="+",
        default=None,
        help="Override actor GPU pool; use only when running one arm to avoid GPU contention.",
    )
    parser.add_argument("--poll-s", type=float, default=10.0)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    base_config = args.base_config.resolve()
    initial_ckpt = args.initial_ckpt.resolve()
    run_root = args.run_root.resolve()
    if not initial_ckpt.is_file():
        raise FileNotFoundError(initial_ckpt)
    if run_root.exists() and not bool(args.resume):
        raise FileExistsError(f"Refusing to reuse experiment directory: {run_root}")
    if not run_root.exists() and bool(args.resume):
        raise FileNotFoundError(f"Cannot resume missing experiment directory: {run_root}")
    if int(args.max_updates) <= 0:
        raise ValueError("--max-updates must be positive")

    base = _load_yaml(base_config)
    config_root = run_root / "configs"
    log_root = run_root / "logs"
    config_root.mkdir(parents=True, exist_ok=bool(args.resume))
    log_root.mkdir(parents=True, exist_ok=bool(args.resume))

    selected_arms = list(dict.fromkeys(str(arm) for arm in args.arms))
    for arm in selected_arms:
        config_path = config_root / f"{arm}.yaml"
        config = _arm_config(
            base,
            arm=arm,
            run_root=run_root,
            initial_ckpt=initial_ckpt,
            max_updates=int(args.max_updates),
            actor_horizon=int(args.actor_horizon),
            samples_per_update=int(args.samples_per_update),
            shards_per_update=int(args.shards_per_update),
            max_inflight_per_actor=int(args.max_inflight_per_actor),
            env_max_steps=None if args.env_max_steps is None else int(args.env_max_steps),
            scene_sampling=None if args.scene_sampling is None else str(args.scene_sampling),
            single_scene=None if args.single_scene is None else str(args.single_scene),
            collision_penalty=None if args.collision_penalty is None else float(args.collision_penalty),
            actor_gpu_pool=None if args.actor_gpus is None else [int(gpu) for gpu in args.actor_gpus],
        )
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if bool(args.generate_only):
        print(f"[matrix] generated configs={config_root}", flush=True)
        return 0

    processes: dict[str, subprocess.Popen[str]] = {}
    handles: dict[str, Any] = {}
    try:
        for arm in selected_arms:
            config_path = config_root / f"{arm}.yaml"
            log_path = log_root / f"{arm}.log"
            handle = log_path.open("a" if bool(args.resume) else "w", encoding="utf-8")
            if bool(args.resume):
                handle.write(f"\n[matrix] resume max_updates={int(args.max_updates)}\n")
                handle.flush()
            handles[arm] = handle
            command = [
                sys.executable,
                "-u",
                str(REPO_ROOT / "script/train_actor_learner_v2.py"),
                "--role",
                "orchestrator",
                "--config",
                str(config_path),
            ]
            processes[arm] = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(f"[matrix] launched arm={arm} pid={processes[arm].pid} config={config_path}", flush=True)

        while processes:
            time.sleep(max(1.0, float(args.poll_s)))
            for arm, process in list(processes.items()):
                code = process.poll()
                print(f"[matrix] arm={arm} code={code} tail={_tail(log_root / f'{arm}.log')}", flush=True)
                if code is None:
                    continue
                handles[arm].close()
                del handles[arm]
                del processes[arm]
                if int(code) != 0:
                    raise RuntimeError(f"Training arm {arm} failed with exit code {code}")
    except BaseException:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    finally:
        for handle in handles.values():
            handle.close()

    _publish_named_checkpoints(run_root)
    print(f"[matrix] complete checkpoints={run_root / 'checkpoints'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
