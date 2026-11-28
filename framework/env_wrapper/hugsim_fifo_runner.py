from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from framework.env_wrapper.fifo_io import read_fifo_payload, write_fifo_payload


BRANCH_STATE_VERSION = 1
_BRANCH_STATE_OP = "capture_branch_state"
_RESTORE_STATE_OP = "restore_branch_state"
_ENV_STATE_FIELDS = (
    "vr",
    "vab",
    "velo",
    "steer",
    "last_accel",
    "last_steer_rate",
    "timestamp",
)
_CONTROLLER_STATE_FIELDS = ("exec_traj", "exec_pointer", "t")
_WRAPPER_STATE_FIELDS = ("_elapsed_steps", "_has_reset")


def _clone_state_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone_state_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_value(item) for item in value)
    return copy.deepcopy(value)


def _restore_state_value(value: Any, current: Any = None) -> Any:
    if torch.is_tensor(value):
        if torch.is_tensor(current):
            return value.detach().to(device=current.device, dtype=current.dtype).clone()
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        dtype = current.dtype if isinstance(current, np.ndarray) else value.dtype
        return np.asarray(value, dtype=dtype).copy()
    if isinstance(value, dict):
        current_dict = current if isinstance(current, dict) else {}
        return {
            key: _restore_state_value(item, current_dict.get(key, None))
            for key, item in value.items()
        }
    if isinstance(value, list):
        current_list = current if isinstance(current, list) else []
        return [
            _restore_state_value(item, current_list[idx] if idx < len(current_list) else None)
            for idx, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        current_tuple = current if isinstance(current, tuple) else ()
        return tuple(
            _restore_state_value(item, current_tuple[idx] if idx < len(current_tuple) else None)
            for idx, item in enumerate(value)
        )
    return copy.deepcopy(value)


def _env_wrapper_chain(env: Any) -> list[Any]:
    chain: list[Any] = []
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        child = getattr(current, "env", None)
        if child is None or child is current:
            break
        current = child
    return chain


def capture_hugsim_env_state(env: Any) -> dict[str, Any]:
    """Capture the mutable HUGSIM state without serializing scene/model assets."""
    raw_env = getattr(env, "unwrapped", env)
    missing = [field for field in _ENV_STATE_FIELDS if not hasattr(raw_env, field)]
    if missing:
        raise RuntimeError(f"HUGSIM environment does not expose snapshot fields: {missing}")
    planner = getattr(raw_env, "planner", None)
    if planner is None or not isinstance(getattr(planner, "stats", None), dict):
        raise RuntimeError("HUGSIM environment does not expose planner.stats for snapshotting")

    controller_state: dict[str, dict[str, Any]] = {}
    for controller_id, controller in dict(getattr(planner, "controller", {})).items():
        fields = {
            field: _clone_state_value(getattr(controller, field))
            for field in _CONTROLLER_STATE_FIELDS
            if hasattr(controller, field)
        }
        if fields:
            controller_state[str(controller_id)] = fields

    render_kwargs = getattr(raw_env, "render_kwargs", None)
    if not isinstance(render_kwargs, dict) or "planning" not in render_kwargs:
        raise RuntimeError("HUGSIM environment does not expose render_kwargs['planning']")

    wrapper_state: list[dict[str, Any]] = []
    for layer in _env_wrapper_chain(env):
        wrapper_state.append(
            {
                "class": f"{type(layer).__module__}.{type(layer).__qualname__}",
                "fields": {
                    field: _clone_state_value(getattr(layer, field))
                    for field in _WRAPPER_STATE_FIELDS
                    if hasattr(layer, field)
                },
            }
        )

    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "version": BRANCH_STATE_VERSION,
        "env": {field: _clone_state_value(getattr(raw_env, field)) for field in _ENV_STATE_FIELDS},
        "planner_stats": _clone_state_value(planner.stats),
        "controller_state": controller_state,
        "planning": _clone_state_value(render_kwargs["planning"]),
        "wrapper_state": wrapper_state,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [state.cpu().clone() for state in cuda_rng_state],
        },
    }


def restore_hugsim_env_state(env: Any, state: dict[str, Any]) -> None:
    """Restore a state captured by :func:`capture_hugsim_env_state`."""
    if not isinstance(state, dict) or int(state.get("version", -1)) != BRANCH_STATE_VERSION:
        raise ValueError(
            f"Unsupported HUGSIM branch state version: {state.get('version', None) if isinstance(state, dict) else None}"
        )
    raw_env = getattr(env, "unwrapped", env)
    env_state = state.get("env", {})
    for field in _ENV_STATE_FIELDS:
        if field not in env_state:
            raise ValueError(f"HUGSIM branch state is missing env field {field!r}")
        setattr(raw_env, field, _restore_state_value(env_state[field], getattr(raw_env, field, None)))

    planner = getattr(raw_env, "planner", None)
    if planner is None:
        raise RuntimeError("HUGSIM environment does not expose planner for state restore")
    planner.stats = _restore_state_value(state.get("planner_stats", {}), getattr(planner, "stats", {}))
    controllers = dict(getattr(planner, "controller", {}))
    for controller_id, fields in dict(state.get("controller_state", {})).items():
        if controller_id not in controllers:
            raise ValueError(f"HUGSIM branch state references unknown controller {controller_id!r}")
        controller = controllers[controller_id]
        for field, value in dict(fields).items():
            setattr(controller, field, _restore_state_value(value, getattr(controller, field, None)))

    render_kwargs = getattr(raw_env, "render_kwargs", None)
    if not isinstance(render_kwargs, dict):
        raise RuntimeError("HUGSIM environment does not expose render_kwargs for state restore")
    render_kwargs["planning"] = _restore_state_value(
        state.get("planning", None),
        render_kwargs.get("planning", None),
    )

    layers = _env_wrapper_chain(env)
    saved_layers = list(state.get("wrapper_state", []))
    if len(layers) != len(saved_layers):
        raise ValueError(
            "HUGSIM wrapper chain changed between snapshot and restore: "
            f"snapshot={len(saved_layers)} current={len(layers)}"
        )
    for layer, saved in zip(layers, saved_layers):
        current_class = f"{type(layer).__module__}.{type(layer).__qualname__}"
        if str(saved.get("class", "")) != current_class:
            raise ValueError(
                "HUGSIM wrapper chain class mismatch: "
                f"snapshot={saved.get('class', None)!r} current={current_class!r}"
            )
        for field, value in dict(saved.get("fields", {})).items():
            setattr(layer, field, _restore_state_value(value, getattr(layer, field, None)))

    rng = dict(state.get("rng", {}))
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(torch.as_tensor(rng["torch_cpu"], dtype=torch.uint8).cpu())
    cuda_states = list(rng.get("torch_cuda", []))
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.as_tensor(item, dtype=torch.uint8).cpu() for item in cuda_states])


def ensure_hugsim_import_paths(hugsim_repo: str | Path | None = None) -> None:
    repo = Path(hugsim_repo) if hugsim_repo is not None else Path.cwd()
    for path in (repo, repo / "sim"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HUGSIM env-only FIFO backend.")
    parser.add_argument("--scenario_path", required=True)
    parser.add_argument("--base_path", required=True)
    parser.add_argument("--camera_path", required=True)
    parser.add_argument("--kinematic_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ad", default="sparsedrive_v2")
    parser.add_argument("--substeps_per_rl_step", type=int, default=2)
    parser.add_argument("--fifo_timeout_s", type=float, default=300.0)
    parser.add_argument("--fifo_poll_interval_s", type=float, default=0.2)
    return parser


def reset_hugsim_for_official_episode_start(env: Any) -> tuple[Any, dict[str, Any]]:
    """Match official HUGSIM closed_loop.py double-reset initialization."""
    env.reset()
    obs, info = env.reset()
    return obs, dict(info)


def execute_control_substeps(
    *,
    env: Any,
    plan: np.ndarray,
    initial_info: dict[str, Any],
    traj2control: Any,
    substeps_per_rl_step: int,
    status_fn: Any | None = None,
) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    substeps = max(1, int(substeps_per_rl_step))
    start_ts = _safe_float(initial_info.get("timestamp", None))
    controller_info = dict(initial_info)
    step_dt = getattr(getattr(env, "unwrapped", env), "dt", None)
    if step_dt is not None:
        controller_info["dt"] = float(step_dt)
    acc, steer_rate = traj2control(np.asarray(plan, dtype=np.float32), controller_info)
    # Match official HUGSIM: pass the shared controller's native action through.
    action = {"acc": float(acc), "steer_rate": float(steer_rate)}

    obs = None
    info: dict[str, Any] = dict(initial_info)
    total_reward = 0.0
    terminated = False
    truncated = False
    executed = 0
    substep_rewards: list[float] = []
    for _idx in range(substeps):
        if callable(status_fn):
            status_fn(state="executing_substep", substep_idx=int(_idx), substeps_per_rl_step=int(substeps))
        obs, reward, terminated, truncated, info = env.step(dict(action))
        info = dict(info)
        reward_f = float(reward)
        total_reward += reward_f
        substep_rewards.append(reward_f)
        executed += 1
        if callable(status_fn):
            status_fn(
                state="executed_substep",
                substep_idx=int(_idx),
                substeps_per_rl_step=int(substeps),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        if bool(terminated or truncated):
            break

    end_ts = _safe_float(info.get("timestamp", None))
    info["reward"] = float(total_reward)
    info["terminated"] = bool(terminated)
    info["truncated"] = bool(truncated)
    info["hugsim_substeps_per_rl_step"] = int(substeps)
    info["hugsim_executed_substeps"] = int(executed)
    info["hugsim_substep_rewards"] = substep_rewards
    if start_ts is not None:
        info["hugsim_rl_step_start_timestamp"] = float(start_ts)
    if end_ts is not None:
        info["hugsim_rl_step_end_timestamp"] = float(end_ts)
    if start_ts is not None and end_ts is not None:
        info["hugsim_rl_step_delta_s"] = float(end_ts - start_ts)
    return obs, float(total_reward), bool(terminated), bool(truncated), info


def run_fifo_env(args: argparse.Namespace) -> None:
    ensure_hugsim_import_paths()

    import gymnasium  # type: ignore
    import hugsim_env  # noqa: F401
    from sim.utils.ad_config import normalize_ad_name  # type: ignore
    from sim.utils.config_loader import load_closed_loop_cfg  # type: ignore
    from sim.utils.sim_utils import traj2control  # type: ignore
    from sim.utils.sparsedrive_vehicle import load_sparsedrive_wheelbase  # type: ignore

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    obs_pipe = output_dir / "obs_pipe"
    plan_pipe = output_dir / "plan_pipe"
    for pipe in (obs_pipe, plan_pipe):
        if pipe.exists():
            pipe.unlink()
        os.mkfifo(pipe)

    _write_status(output_dir, state="starting", pid=os.getpid(), started_at=time.time())
    env = None
    try:
        cfg, _output = load_closed_loop_cfg(
            scenario_path=str(args.scenario_path),
            base_path=str(args.base_path),
            camera_path=str(args.camera_path),
            kinematic_path=str(args.kinematic_path),
            ad=str(args.ad),
        )
        if normalize_ad_name(str(args.ad)) == "sparsedrive_v2" and os.environ.get("HUGSIM_USE_AGENT_WHEELBASE", "").strip() == "1":
            wheelbase = load_sparsedrive_wheelbase(
                getattr(cfg.base, "sparsedrive_v2_repo", ""),
                getattr(cfg.base, "sparsedrive_v2_nuplan_repo", None),
            )
            if wheelbase is not None:
                cfg.kinematic.Lf = float(wheelbase) / 2.0
                cfg.kinematic.Lr = float(wheelbase) / 2.0
                print(f"[hugsim_fifo_runner] Using SparseDriveV2 wheelbase override: {wheelbase}", flush=True)
        _write_status(output_dir, state="making_env", pid=os.getpid(), started_at=time.time())
        env = gymnasium.make("hugsim_env/HUGSim-v0", cfg=cfg, output=str(output_dir))
        _write_status(output_dir, state="resetting", pid=os.getpid(), started_at=time.time())
        obs, info = reset_hugsim_for_official_episode_start(env)
        substeps = max(1, int(args.substeps_per_rl_step))
        print(f"[hugsim_fifo_runner] substeps_per_rl_step={substeps}", flush=True)
        _write_status(
            output_dir,
            state="running",
            pid=os.getpid(),
            started_at=time.time(),
            substeps_per_rl_step=int(substeps),
        )

        _write_status(output_dir, state="writing_obs", pid=os.getpid(), updated_at=time.time())
        write_fifo_payload(
            obs_pipe,
            (obs, info),
            timeout_s=float(args.fifo_timeout_s),
            poll_interval_s=float(args.fifo_poll_interval_s),
        )
        while True:
            _write_status(output_dir, state="waiting_plan", pid=os.getpid(), updated_at=time.time())
            request = read_fifo_payload(
                plan_pipe,
                timeout_s=float(args.fifo_timeout_s),
                poll_interval_s=float(args.fifo_poll_interval_s),
            )
            if request is None or (isinstance(request, str) and request == "STOP"):
                _write_status(output_dir, state="stopped", pid=os.getpid(), completed_at=time.time())
                return

            if isinstance(request, dict) and request.get("op", None) == _BRANCH_STATE_OP:
                snapshot = {
                    "version": BRANCH_STATE_VERSION,
                    "sim_state": capture_hugsim_env_state(env),
                    "obs": _clone_state_value(obs),
                    "info": _clone_state_value(info),
                }
                write_fifo_payload(
                    obs_pipe,
                    {"op": _BRANCH_STATE_OP, "snapshot": snapshot},
                    timeout_s=float(args.fifo_timeout_s),
                    poll_interval_s=float(args.fifo_poll_interval_s),
                )
                continue

            if isinstance(request, dict) and request.get("op", None) == _RESTORE_STATE_OP:
                snapshot = request.get("snapshot", None)
                if not isinstance(snapshot, dict) or int(snapshot.get("version", -1)) != BRANCH_STATE_VERSION:
                    raise ValueError("restore_branch_state requires a compatible snapshot mapping")
                restore_hugsim_env_state(env, dict(snapshot["sim_state"]))
                obs = _restore_state_value(snapshot["obs"])
                info = _restore_state_value(snapshot["info"])
                write_fifo_payload(
                    obs_pipe,
                    (obs, info),
                    timeout_s=float(args.fifo_timeout_s),
                    poll_interval_s=float(args.fifo_poll_interval_s),
                )
                continue

            obs, reward, terminated, truncated, info = execute_control_substeps(
                env=env,
                plan=np.asarray(request, dtype=np.float32),
                initial_info=dict(info),
                traj2control=traj2control,
                substeps_per_rl_step=int(substeps),
                status_fn=lambda **payload: _write_status(
                    output_dir,
                    pid=os.getpid(),
                    updated_at=time.time(),
                    **payload,
                ),
            )
            info = dict(info)
            write_fifo_payload(
                obs_pipe,
                (obs, info),
                timeout_s=float(args.fifo_timeout_s),
                poll_interval_s=float(args.fifo_poll_interval_s),
            )
            if bool(terminated or truncated):
                _write_status(
                    output_dir,
                    state="completed",
                    pid=os.getpid(),
                    completed_at=time.time(),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
                return
    except BaseException as exc:
        _write_status(
            output_dir,
            state="error",
            pid=os.getpid(),
            error=repr(exc),
            traceback=traceback.format_exc(),
            completed_at=time.time(),
        )
        raise
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_fifo_env(args)


def _write_status(output_dir: Path, **payload: Any) -> None:
    status_path = output_dir / "status.json"
    status_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


if __name__ == "__main__":
    main()
