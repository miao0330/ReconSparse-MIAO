from __future__ import annotations

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np
import torch

from framework.replay_schema import (
    get_env_plan_xyyaw,
    get_grpo_candidates,
    set_policy_loop_grpo_result,
)


@dataclass
class _BranchState:
    candidate_index: int
    env: Any
    obs: Any
    candidate_plan: np.ndarray
    total_return: float = 0.0
    actual_steps: int = 0
    done: bool = False
    valid: bool = True
    termination_reason: str = "horizon"
    restored_state_signature: str = ""


def keyframe_steps_from_config(config: Mapping[str, Any]) -> tuple[tuple[int, float], ...]:
    rl_step_s = float(config.get("rl_step_s", 0.5))
    if rl_step_s <= 0.0:
        raise ValueError("policy_loop_grpo.rl_step_s must be > 0")
    out: list[tuple[int, float]] = []
    for raw_time in config.get("keyframe_times_s", (4.0, 8.0)):
        time_s = float(raw_time)
        step = int(round(time_s / rl_step_s))
        if step <= 0 or not math.isclose(step * rl_step_s, time_s, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError(
                f"policy_loop_grpo keyframe time {time_s} is not aligned to rl_step_s={rl_step_s}"
            )
        out.append((step, time_s))
    return tuple(out)


def _as_plan_array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    plan = np.asarray(value, dtype=np.float32)
    if plan.ndim != 2 or int(plan.shape[0]) < 1 or int(plan.shape[1]) < 2:
        raise ValueError(f"Expected a non-empty trajectory shaped (horizon, dims>=2), got {plan.shape}")
    if int(plan.shape[1]) == 2:
        xy = plan[:, :2]
        prev = np.concatenate([np.zeros((1, 2), dtype=np.float32), xy[:-1]], axis=0)
        yaw = np.arctan2((xy - prev)[:, 1], (xy - prev)[:, 0])[:, None]
        plan = np.concatenate([xy, yaw], axis=1)
    return np.ascontiguousarray(plan[:, :3], dtype=np.float32)


def _remaining_candidate_plan(plan: np.ndarray, intervention_step: int) -> np.ndarray:
    idx = min(max(0, int(intervention_step)), int(plan.shape[0]) - 1)
    if idx == 0:
        return plan
    anchor = plan[idx - 1]
    remaining = plan[idx:].copy()
    delta = remaining[:, :2] - anchor[None, :2]
    cos_yaw = math.cos(float(anchor[2]))
    sin_yaw = math.sin(float(anchor[2]))
    remaining[:, 0] = cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1]
    remaining[:, 1] = -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1]
    remaining[:, 2] = (remaining[:, 2] - float(anchor[2]) + math.pi) % (2.0 * math.pi) - math.pi
    return remaining


def _action_from_plan(plan: np.ndarray) -> tuple[float, float, float, int]:
    first = plan[0]
    return float(first[0]), float(first[1]), float(first[2]), 2

#NOTE 环境封装
def _step_with_plan(env: Any, plan: Any) -> tuple[Any, float, bool, bool, Dict[str, Any]]:
    plan_array = _as_plan_array(plan)
    inject = getattr(env, "set_external_plan_local_xyyaw", None)
    if callable(inject):
        inject(plan_array)
    obs, reward, terminated, truncated, info = env.step(_action_from_plan(plan_array))
    return obs, float(reward), bool(terminated), bool(truncated), dict(info or {})


def _prefix_state_hash(
    *,
    scenario_index: int,
    start_frame: int,
    prefix_replays: Sequence[Mapping[str, Any]],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"scene={int(scenario_index)};start={int(start_frame)};".encode("utf-8"))
    for replay in prefix_replays:
        digest.update(_as_plan_array(get_env_plan_xyyaw(replay)).tobytes(order="C"))
    return digest.hexdigest()


def _restored_state_signature(obs: Any, info: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    stable_keys = (
        "sample_token",
        "scene",
        "scene_id",
        "scenario_index",
        "start_frame",
        "frame_idx",
        "now_frame",
        "step",
    )
    for source_name, source in (("obs", obs), ("info", info)):
        if not isinstance(source, Mapping):
            continue
        for key in stable_keys:
            if key not in source or source[key] is None:
                continue
            value = source[key]
            digest.update(f"{source_name}:{key}:".encode("utf-8"))
            try:
                if torch.is_tensor(value):
                    value = value.detach().cpu().numpy()
                array = np.asarray(value)
                if array.dtype != object and int(array.size) <= 256:
                    digest.update(str(array.dtype).encode("ascii"))
                    digest.update(str(tuple(array.shape)).encode("ascii"))
                    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
                    continue
            except Exception:
                pass
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()

#NOTE 从episode起点重新回放
def _reset_and_replay_prefix(
    env: Any,
    *,
    scenario_index: int,
    start_frame: int,
    prefix_replays: Sequence[Mapping[str, Any]],
) -> tuple[Any, str]:
    reset_to = getattr(env, "reset_to", None)
    if not callable(reset_to):
        raise RuntimeError("Policy-loop branch environment must expose reset_to(scene_id, start_frame=...)")
    obs, info = reset_to(int(scenario_index), start_frame=int(start_frame))
    info = dict(info or {})
    for replay in prefix_replays:
        obs, _reward, terminated, truncated, info = _step_with_plan(env, get_env_plan_xyyaw(replay))
        if bool(terminated or truncated):
            reason = str(info.get("done_reason", info.get("terminal_kind", "done")))
            raise RuntimeError(f"Counterfactual prefix replay terminated before keyframe: {reason}")
    return obs, _restored_state_signature(obs, info)


def _capture_policy_parameter_versions(agent: Any) -> list[tuple[torch.nn.Parameter, int]]:
    parameters_fn = getattr(agent, "parameters", None)
    if not callable(parameters_fn):
        return []
    return [
        (parameter, int(getattr(parameter, "_version", -1)))
        for parameter in parameters_fn()
        if torch.is_tensor(parameter)
    ]


def _assert_policy_parameters_unchanged(snapshot: Sequence[tuple[torch.nn.Parameter, int]]) -> None:
    changed = [
        index
        for index, (parameter, version) in enumerate(snapshot)
        if int(getattr(parameter, "_version", -1)) != int(version)
    ]
    if changed:
        raise RuntimeError(
            "Frozen old policy parameters changed during policy-loop rollout; "
            f"changed_parameter_indices={changed[:8]}"
        )


def _termination_reason(info: Mapping[str, Any], *, truncated: bool) -> str:
    if "done_reason" in info:
        return str(info["done_reason"])
    if "termination_reason" in info:
        return str(info["termination_reason"])
    if "terminal_kind" in info:
        return str(info["terminal_kind"])
    return "truncated" if bool(truncated) else "terminated"


def _explicit_collision(info: Mapping[str, Any]) -> bool:
    return any(
        bool(info.get(key, False))
        for key in ("collision", "static_collision", "dynamic_collision")
    )


def _explicit_severe_offroad(info: Mapping[str, Any], *, lateral_threshold_m: float) -> bool:
    if any(
        bool(info.get(key, False))
        for key in ("severe_offroad", "severe_off_route", "offroad_terminated", "off_route_terminated")
    ):
        return True
    lateral_error = info.get("centerline_lateral_error_m", None)
    if lateral_error is None:
        return False
    try:
        return abs(float(lateral_error)) >= float(lateral_threshold_m)
    except (TypeError, ValueError):
        return False

#强制干预结束后，使用agent继续生成Plan
def _continuation_actions(
    agent: Any,
    states: Sequence[_BranchState],
    *,
    continuation_mode: str,
) -> list[tuple[Any, Mapping[str, Any]]]:
    observations = [state.obs for state in states]
    act_batch = getattr(agent, "act_batch", None)
    if callable(act_batch):
        with torch.inference_mode():
            actions, _logps, replays = act_batch(
                observations,
                eta=1.0,
                mode_idx=-1,
                mode_select=str(continuation_mode),
            )
        if len(actions) != len(observations) or len(replays) != len(observations):
            raise RuntimeError(
                "Policy-loop continuation act_batch output length mismatch: "
                f"observations={len(observations)} actions={len(actions)} replays={len(replays)}"
            )
        return list(zip(actions, replays))
    out: list[tuple[Any, Mapping[str, Any]]] = []
    for observation in observations:
        with torch.inference_mode():
            action, _logp, replay = agent.act(
                observation,
                eta=1.0,
                mode_idx=-1,
                mode_select=str(continuation_mode),
            )
        out.append((action, replay))
    return out


def _evaluate_candidate_group(
    candidate_indices: Sequence[int],
    *,
    candidate_plans: np.ndarray,
    agent: Any,
    env_factory: Callable[[int], Any],
    scenario_index: int,
    start_frame: int,
    prefix_replays: Sequence[Mapping[str, Any]],
    branch_snapshot: Mapping[str, Any] | None,
    branch_horizon_steps: int,
    intervention_steps: int,
    continuation_mode: str,
    severe_offroad_lateral_m: float,
) -> list[_BranchState]:
    def build_state(candidate_index: int) -> _BranchState:
        env = None
        try:
            env = env_factory(int(candidate_index))
            if branch_snapshot is not None:
                restore = getattr(env, "restore_branch_state", None)
                if not callable(restore):
                    raise RuntimeError("Policy-loop branch environment must expose restore_branch_state(snapshot)")
                obs, info = restore(dict(branch_snapshot))
                state_signature = _restored_state_signature(obs, dict(info or {}))
            else:
                obs, state_signature = _reset_and_replay_prefix(
                    env,
                    scenario_index=int(scenario_index),
                    start_frame=int(start_frame),
                    prefix_replays=prefix_replays,
                )
            return _BranchState(
                candidate_index=int(candidate_index),
                env=env,
                obs=obs,
                candidate_plan=_as_plan_array(candidate_plans[int(candidate_index)]),
                restored_state_signature=state_signature,
            )
        except Exception as exc:
            if env is not None:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
            return _BranchState(
                candidate_index=int(candidate_index),
                env=None,
                obs=None,
                candidate_plan=_as_plan_array(candidate_plans[int(candidate_index)]),
                valid=False,
                done=True,
                termination_reason=f"invalid:{type(exc).__name__}:{exc}",
            )

    def advance_state(item: tuple[_BranchState, Any]) -> None:
        state, plan = item
        try:
            state.obs, reward, terminated, truncated, info = _step_with_plan(state.env, plan)
            state.total_return += float(reward)
            state.actual_steps += 1
            collision = _explicit_collision(info)
            severe_offroad = _explicit_severe_offroad(
                info,
                lateral_threshold_m=float(severe_offroad_lateral_m),
            )
            if bool(terminated or truncated or collision or severe_offroad):
                state.done = True
                if severe_offroad:
                    state.termination_reason = "severe_offroad"
                elif collision and not bool(terminated or truncated):
                    state.termination_reason = str(info.get("done_reason", "collision"))
                else:
                    state.termination_reason = _termination_reason(info, truncated=bool(truncated))
        except Exception as exc:
            state.valid = False
            state.done = True
            state.termination_reason = f"invalid:{type(exc).__name__}:{exc}"

    worker_count = max(1, len(candidate_indices))
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="policy-loop-env") as executor:
        states = list(executor.map(build_state, candidate_indices))
        try:
            for branch_step in range(int(branch_horizon_steps)):
                active = [state for state in states if state.valid and not state.done]
                if not active:
                    break
                continuation: dict[int, tuple[Any, Mapping[str, Any]]] = {}
                if branch_step >= int(intervention_steps):
                    try:
                        outputs = _continuation_actions(
                            agent,
                            active,
                            continuation_mode=str(continuation_mode),
                        )
                        continuation = {id(state): output for state, output in zip(active, outputs)}
                    except Exception as exc:
                        for state in active:
                            state.valid = False
                            state.done = True
                            state.termination_reason = f"invalid:{type(exc).__name__}:{exc}"
                        break

                plans: list[Any] = []
                for state in active:
                    if branch_step < int(intervention_steps):
                        plans.append(_remaining_candidate_plan(state.candidate_plan, branch_step))
                    else:
                        _action, replay = continuation[id(state)]
                        plans.append(get_env_plan_xyyaw(replay))
                list(executor.map(advance_state, zip(active, plans)))
        finally:
            def close_state(state: _BranchState) -> None:
                close = getattr(state.env, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

            list(executor.map(close_state, states))
    return states


def collect_policy_loop_grpo_for_shard(
    shard: Dict[str, Any],
    *,
    agent: Any,
    env_factory: Callable[[int, int], Any],
    config: Mapping[str, Any],
    branch_snapshots: Mapping[int, Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Attach policy-in-the-loop returns to eligible keyframe replay entries."""
    if not bool(config.get("enabled", False)):
        return {"keyframes": 0, "valid_candidates": 0, "skipped": "disabled"}

    replay_entries = shard.get("replay", None)
    meta = shard.get("meta", None)
    if not isinstance(replay_entries, list) or not isinstance(meta, dict):
        raise TypeError("Policy-loop GRPO requires a shard with list replay and mapping meta")
    if not bool(meta.get("policy_loop_episode_start", False)):
        return {"keyframes": 0, "valid_candidates": 0, "skipped": "shard_not_episode_start"}

    start_info = meta.get("policy_loop_episode_start_info", {}) or {}
    scenario_index = int(start_info.get("scenario_index", start_info.get("scene", 0)))
    start_frame = int(start_info.get("start_frame", 0))
    if start_frame != 0:
        return {"keyframes": 0, "valid_candidates": 0, "skipped": "nonzero_start_frame"}

    episode_length = len(replay_entries)
    done = shard.get("done", None)
    if torch.is_tensor(done):
        done_indices = (done.detach().cpu().view(-1) > 0.5).nonzero(as_tuple=False).view(-1)
        if int(done_indices.numel()) > 0:
            episode_length = min(episode_length, int(done_indices[0].item()) + 1)

    num_candidates = int(config.get("num_candidates", 4))
    parallelism = max(1, int(config.get("branch_parallelism", 4)))
    branch_bootstrap = str(config.get("branch_bootstrap", "auto")).strip().lower()
    if branch_bootstrap not in {"auto", "snapshot", "prefix_replay"}:
        raise ValueError(f"Unsupported policy-loop branch_bootstrap={branch_bootstrap!r}")
    branch_horizon_steps = int(config.get("branch_horizon_steps", 16))
    intervention_steps = int(config.get("intervention_steps", 2))
    severe_offroad_lateral_m = float(config.get("severe_offroad_lateral_m", 3.5))
    continuation_mode = str(config.get("continuation_mode", "argmax"))
    rl_step_s = float(config.get("rl_step_s", 0.5))
    episode_id = str(meta.get("policy_loop_episode_id", f"actor{meta.get('actor_id', 0)}-shard{meta.get('shard_idx', 0)}"))
    weights_version = int(meta.get("weights_version", 0))

    keyframe_count = 0
    valid_candidate_count = 0
    snapshot_keyframe_count = 0
    prefix_replay_keyframe_count = 0
    for keyframe_step, keyframe_time_s in keyframe_steps_from_config(config):
        if keyframe_step >= episode_length:
            continue
        keyframe_replay = replay_entries[keyframe_step]
        candidates = get_grpo_candidates(keyframe_replay)
        candidate_plans = np.asarray(candidates["traj_xyyaw"], dtype=np.float32)
        mode_indices = torch.as_tensor(candidates["mode_indices"], dtype=torch.long).view(-1)
        old_log_probs = torch.as_tensor(candidates["old_log_probs"], dtype=torch.float32).view(-1)
        available = min(int(candidate_plans.shape[0]), int(mode_indices.numel()), int(old_log_probs.numel()))
        if available < num_candidates:
            raise RuntimeError(
                f"Policy-loop keyframe requires {num_candidates} candidates, only {available} are available"
            )
        candidate_plans = candidate_plans[:num_candidates]
        mode_indices = mode_indices[:num_candidates]
        prefix_replays = replay_entries[:keyframe_step]
        snapshot = None
        if branch_bootstrap != "prefix_replay" and branch_snapshots is not None:
            snapshot = branch_snapshots.get(int(keyframe_step), None)
        if branch_bootstrap == "snapshot" and snapshot is None:
            raise RuntimeError(
                f"Policy-loop keyframe step {keyframe_step} requires an exact branch snapshot"
            )
        if snapshot is None:
            prefix_replay_keyframe_count += 1
        else:
            snapshot_keyframe_count += 1
        prefix_hash = _prefix_state_hash(
            scenario_index=scenario_index,
            start_frame=start_frame,
            prefix_replays=prefix_replays,
        )

        policy_parameter_versions = _capture_policy_parameter_versions(agent)
        states: list[_BranchState] = []
        try:
            for group_start in range(0, num_candidates, parallelism):
                group_indices = list(range(group_start, min(num_candidates, group_start + parallelism)))
                states.extend(
                    _evaluate_candidate_group(
                        group_indices,
                        candidate_plans=candidate_plans,
                        agent=agent,
                        env_factory=lambda candidate_index, keyframe_step=keyframe_step: env_factory(
                            int(keyframe_step), int(candidate_index)
                        ),
                        scenario_index=scenario_index,
                        start_frame=start_frame,
                        prefix_replays=prefix_replays,
                        branch_snapshot=snapshot,
                        branch_horizon_steps=branch_horizon_steps,
                        intervention_steps=intervention_steps,
                        continuation_mode=continuation_mode,
                        severe_offroad_lateral_m=severe_offroad_lateral_m,
                    )
                )
        finally:
            _assert_policy_parameters_unchanged(policy_parameter_versions)
        states.sort(key=lambda state: state.candidate_index)

        restored_state_signatures = [state.restored_state_signature for state in states]
        valid_signatures = {
            signature
            for state, signature in zip(states, restored_state_signatures)
            if state.valid and signature
        }
        if len(valid_signatures) > 1:
            for state in states:
                if state.valid:
                    state.valid = False
                    state.done = True
                    state.termination_reason = "invalid:prefix_state_mismatch"

        candidate_returns = torch.tensor([state.total_return for state in states], dtype=torch.float32)
        actual_steps = torch.tensor([state.actual_steps for state in states], dtype=torch.long)
        terminated = torch.tensor([state.done and state.valid for state in states], dtype=torch.bool)
        termination_time_s = actual_steps.to(dtype=torch.float32) * float(rl_step_s)
        termination_reason = [state.termination_reason for state in states]
        valid_mask = torch.tensor([state.valid for state in states], dtype=torch.bool)
        set_policy_loop_grpo_result(
            keyframe_replay,
            episode_id=episode_id,
            keyframe_time_s=float(keyframe_time_s),
            keyframe_step=int(keyframe_step),
            weights_version=weights_version,
            mode_indices=mode_indices.clone(),
            candidate_returns=candidate_returns,
            actual_steps=actual_steps,
            terminated=terminated,
            termination_time_s=termination_time_s,
            termination_reason=termination_reason,
            valid_mask=valid_mask,
            prefix_state_hash=prefix_hash,
            restored_state_signatures=restored_state_signatures,
        )
        keyframe_count += 1
        valid_candidate_count += int(valid_mask.sum().item())

    summary = {
        "keyframes": int(keyframe_count),
        "valid_candidates": int(valid_candidate_count),
        "episode_length": int(episode_length),
        "snapshot_keyframes": int(snapshot_keyframe_count),
        "prefix_replay_keyframes": int(prefix_replay_keyframe_count),
    }
    meta["policy_loop_grpo"] = dict(summary)
    return summary


__all__ = ["collect_policy_loop_grpo_for_shard", "keyframe_steps_from_config"]