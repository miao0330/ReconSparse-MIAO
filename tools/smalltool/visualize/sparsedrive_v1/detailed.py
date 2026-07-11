#!/usr/bin/env python3
"""Detailed SparseDrive-v1 HUGSIM rollout visualizer.

Saves the rollout mp4 plus candidate trajectory screenshots, candidate JSON,
candidate CSV, selected trajectory CSV, reward CSV, and a manifest.

Command:
cd /root/clone/ReconDreamer-RL
python tools/smalltool/visualize/sparsedrive_v1/detailed.py \
  --config /root/clone/ReconDreamer-RL/script/configs/sparsedrive/20260630_template_HUGSIM_sparsedrive_v1.yaml \
  --ckpt /root/clone/ReconDreamer-RL/egoADs/SparseDrive/ckpt/sparsedrive_stage2.pth \
  --sparsedrive-config /root/clone/ReconDreamer-RL/egoADs/SparseDrive/projects/configs/sparsedrive_small_stage2.py \
  --hugsim-scene scene-0038-easy-00 \
  --horizon 32 \
  --fps 2 \
  --cuda 0 \
  --mode-select greedy \
  --top-k 8

This script is intentionally v1-specific. SparseDrive-v1 does not expose the
SparseDriveV2 GRPO candidate adapter, so candidates are read from the raw
planning head logits/trajectories and converted into the repo canonical ego
frame: x forward, y left, yaw radians.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio
import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework.utils.repo_paths import resolve_ego_ads_subdir, resolve_repo_path
from tools.smalltool.visualize.sparsedrive_v2.detailed import (
    _grid_frame,
    _obs_for_policy,
    _overlay_debug_text,
    build_candidate_score_payload,
    overlay_top_right_inset,
    render_bev_debug_image,
)

_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "outputs" / "RewardCheckandVideo" / "HUGSIM" / "sparsedrive-v1"
_DEFAULT_SPARSEDRIVE_ROOT = Path(resolve_ego_ads_subdir("SparseDrive"))
_DEFAULT_SPARSEDRIVE_CONFIG = _DEFAULT_SPARSEDRIVE_ROOT / "projects" / "configs" / "sparsedrive_small_stage2.py"


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    artifacts: Path
    frames: Path
    bev: Path
    candidate_grid: Path
    candidate_json: Path
    video: Path
    selected_csv: Path
    candidate_csv: Path
    reward_csv: Path
    summary_json: Path
    manifest: Path
    debug_shard: Path


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return _json_safe(arr.item())
        return [_json_safe(v) for v in arr.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _write_json(path: str | Path, payload: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _write_csv_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(k) for row in rows for k in row.keys()})
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows([dict(row) for row in rows])
    return out


def _softmax_np(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float32)
    finite = np.isfinite(arr)
    if not bool(finite.any()):
        return np.full(arr.shape, 1.0 / float(arr.size), dtype=np.float32)
    safe = np.where(finite, arr, -np.inf)
    shifted = safe - float(np.max(safe))
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if not math.isfinite(total) or total <= 0.0:
        return np.full(arr.shape, 1.0 / float(arr.size), dtype=np.float32)
    return (exp / total).astype(np.float32)


def _sort_candidate_indices(logits: Sequence[float] | np.ndarray) -> list[int]:
    arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    # Python sort is stable, so equal logits keep ascending mode index.
    return sorted(range(int(arr.shape[0])), key=lambda idx: (-float(arr[idx]), int(idx)))


def _v1_trajs_to_xyyaw(policy: Any, trajs: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(trajs):
        raw = trajs.detach().cpu().numpy()
    else:
        raw = np.asarray(trajs)
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[-1] < 2:
        raise RuntimeError(f"Expected v1 trajectories shape (modes,horizon,2+), got {tuple(raw.shape)}")
    converted = []
    convert = getattr(policy, "_sparsedrive_lidar_xy_to_hugsim_action_xyyaw", None)
    if not callable(convert):
        raise RuntimeError("SparseDrive-v1 policy missing _sparsedrive_lidar_xy_to_hugsim_action_xyyaw")
    for mode_idx in range(int(raw.shape[0])):
        converted.append(np.asarray(convert(raw[mode_idx, :, :2]), dtype=np.float32)[:, :3])
    return np.stack(converted, axis=0).astype(np.float32)


def _candidate_rows(
    *,
    step: int,
    frame_idx: int,
    cmd_idx: int,
    selected_mode_idx: int,
    traj_xyyaw: np.ndarray,
    logits: Sequence[float] | np.ndarray,
    probs: Sequence[float] | np.ndarray,
) -> list[dict[str, Any]]:
    traj = np.asarray(traj_xyyaw, dtype=np.float32)
    logits_arr = np.asarray(logits, dtype=np.float32).reshape(-1)
    probs_arr = np.asarray(probs, dtype=np.float32).reshape(-1)
    if traj.ndim != 3 or traj.shape[-1] < 3:
        raise RuntimeError(f"Expected traj_xyyaw shape (modes,horizon,3+), got {tuple(traj.shape)}")
    if int(logits_arr.shape[0]) != int(traj.shape[0]) or int(probs_arr.shape[0]) != int(traj.shape[0]):
        raise RuntimeError("Candidate logits/probs length must match trajectory modes")
    rank_by_mode = {int(idx): rank + 1 for rank, idx in enumerate(_sort_candidate_indices(logits_arr))}

    rows: list[dict[str, Any]] = []
    for mode_idx in range(int(traj.shape[0])):
        for point_idx in range(int(traj.shape[1])):
            rows.append(
                {
                    "step": int(step),
                    "frame_idx": int(frame_idx),
                    "cmd_idx": int(cmd_idx),
                    "mode_idx": int(mode_idx),
                    "point_idx": int(point_idx),
                    "is_selected": bool(int(mode_idx) == int(selected_mode_idx)),
                    "rank_by_logit": int(rank_by_mode[int(mode_idx)]),
                    "logit": float(logits_arr[int(mode_idx)]),
                    "prob": float(probs_arr[int(mode_idx)]),
                    "local_x": float(traj[int(mode_idx), int(point_idx), 0]),
                    "local_y": float(traj[int(mode_idx), int(point_idx), 1]),
                    "local_yaw": float(traj[int(mode_idx), int(point_idx), 2]),
                }
            )
    return rows


def _selected_rows(
    *,
    step: int,
    frame_idx: int,
    cmd_idx: int,
    mode_idx: int,
    logp: Any,
    traj_xyyaw: np.ndarray,
) -> list[dict[str, Any]]:
    traj = np.asarray(traj_xyyaw, dtype=np.float32)
    logp_v = float(logp.detach().cpu().item()) if torch.is_tensor(logp) else float(logp)
    rows = []
    for point_idx in range(int(traj.shape[0])):
        rows.append(
            {
                "step": int(step),
                "frame_idx": int(frame_idx),
                "cmd_idx": int(cmd_idx),
                "mode_idx": int(mode_idx),
                "point_idx": int(point_idx),
                "logp": float(logp_v),
                "local_x": float(traj[point_idx, 0]),
                "local_y": float(traj[point_idx, 1]),
                "local_yaw": float(traj[point_idx, 2]),
            }
        )
    return rows


def _recompute_plan_yaw(plan_xyyaw: np.ndarray) -> np.ndarray:
    out = np.asarray(plan_xyyaw, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[0] <= 0 or out.shape[1] < 3:
        return out
    prev_xy = np.concatenate([np.zeros((1, 2), dtype=np.float32), out[:-1, :2]], axis=0)
    dxy = out[:, :2] - prev_xy
    out[:, 2] = np.arctan2(dxy[:, 1], dxy[:, 0]).astype(np.float32)
    return out


def _smooth_plan_with_previous(
    *,
    previous_plan: np.ndarray | None,
    current_plan: np.ndarray,
    alpha: float,
) -> np.ndarray:
    cur = np.asarray(current_plan, dtype=np.float32)
    a = float(alpha)
    if previous_plan is None or a <= 0.0:
        return cur.astype(np.float32, copy=True)
    prev = np.asarray(previous_plan, dtype=np.float32)
    if prev.shape != cur.shape:
        return cur.astype(np.float32, copy=True)
    smoothed = cur.astype(np.float32, copy=True)
    smoothed[:, :2] = (1.0 - a) * cur[:, :2] + a * prev[:, :2]
    return _recompute_plan_yaw(smoothed)


def _sign_flip_count(values: Sequence[float], *, eps: float = 1.0e-3) -> int:
    signs: list[int] = []
    for value in values:
        v = float(value)
        if abs(v) <= float(eps):
            continue
        signs.append(1 if v > 0.0 else -1)
    return sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])


def _compute_jitter_summary(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    reward_rows: Sequence[Mapping[str, Any]],
    mode_select: str,
    requested_mode_idx: int,
    smooth_alpha: float,
) -> dict[str, Any]:
    by_step: dict[int, list[Mapping[str, Any]]] = {}
    for row in selected_rows:
        by_step.setdefault(int(row["step"]), []).append(row)
    steps = sorted(by_step)
    first_points: list[Mapping[str, Any]] = []
    full_plans: list[np.ndarray] = []
    for step in steps:
        rows = sorted(by_step[step], key=lambda r: int(r["point_idx"]))
        if not rows:
            continue
        first_points.append(rows[0])
        full_plans.append(
            np.asarray(
                [[float(r["local_x"]), float(r["local_y"]), float(r["local_yaw"])] for r in rows],
                dtype=np.float32,
            )
        )

    modes = [int(row["mode_idx"]) for row in first_points]
    first_x = [float(row["local_x"]) for row in first_points]
    first_y = [float(row["local_y"]) for row in first_points]
    first_yaw = [float(row["local_yaw"]) for row in first_points]
    dx = np.diff(np.asarray(first_x, dtype=np.float32)) if len(first_x) > 1 else np.zeros((0,), dtype=np.float32)
    dy = np.diff(np.asarray(first_y, dtype=np.float32)) if len(first_y) > 1 else np.zeros((0,), dtype=np.float32)
    dyaw = np.diff(np.unwrap(np.asarray(first_yaw, dtype=np.float32))) if len(first_yaw) > 1 else np.zeros((0,), dtype=np.float32)

    plan_l2: list[float] = []
    plan_y_delta: list[float] = []
    for idx in range(1, len(full_plans)):
        prev = full_plans[idx - 1]
        cur = full_plans[idx]
        if prev.shape == cur.shape:
            plan_l2.append(float(np.linalg.norm(cur[:, :2] - prev[:, :2], axis=1).mean()))
            plan_y_delta.append(float(np.abs(cur[:, 1] - prev[:, 1]).mean()))

    best_by_step: dict[int, int] = {}
    margin_by_step: dict[int, float] = {}
    if candidate_rows:
        per_step_logits: dict[int, dict[int, float]] = {}
        for row in candidate_rows:
            if int(row.get("point_idx", -1)) != 0:
                continue
            per_step_logits.setdefault(int(row["step"]), {})[int(row["mode_idx"])] = float(row["logit"])
        for step, logits_by_mode in per_step_logits.items():
            ordered = sorted(logits_by_mode.items(), key=lambda kv: (-kv[1], kv[0]))
            if ordered:
                best_by_step[int(step)] = int(ordered[0][0])
                margin_by_step[int(step)] = float(ordered[0][1] - ordered[1][1]) if len(ordered) > 1 else float("nan")
    best_modes = [best_by_step[s] for s in sorted(best_by_step)]
    margins = [v for _, v in sorted(margin_by_step.items()) if math.isfinite(float(v))]

    reward_sum = float(sum(float(row.get("reward", 0.0)) for row in reward_rows))
    done = bool(reward_rows[-1].get("done", False)) if reward_rows else False
    return {
        "steps": int(len(first_points)),
        "mode_select": str(mode_select),
        "requested_mode_idx": int(requested_mode_idx),
        "smooth_alpha": float(smooth_alpha),
        "reward_sum": reward_sum,
        "done": done,
        "selected_mode_sequence": modes,
        "selected_mode_switches": int(sum(1 for i in range(1, len(modes)) if modes[i] != modes[i - 1])),
        "selected_mode_unique": sorted({int(v) for v in modes}),
        "best_logit_mode_sequence": best_modes,
        "best_logit_mode_switches": int(sum(1 for i in range(1, len(best_modes)) if best_modes[i] != best_modes[i - 1])),
        "best_logit_margin_mean": float(np.mean(margins)) if margins else None,
        "best_logit_margin_min": float(np.min(margins)) if margins else None,
        "first_y_sign_flips": int(_sign_flip_count(first_y)),
        "first_y_abs_delta_mean": float(np.mean(np.abs(dy))) if dy.size else 0.0,
        "first_y_abs_delta_max": float(np.max(np.abs(dy))) if dy.size else 0.0,
        "first_x_abs_delta_mean": float(np.mean(np.abs(dx))) if dx.size else 0.0,
        "first_yaw_abs_delta_mean": float(np.mean(np.abs(dyaw))) if dyaw.size else 0.0,
        "first_yaw_abs_delta_max": float(np.max(np.abs(dyaw))) if dyaw.size else 0.0,
        "full_plan_xy_l2_delta_mean": float(np.mean(plan_l2)) if plan_l2 else 0.0,
        "full_plan_y_abs_delta_mean": float(np.mean(plan_y_delta)) if plan_y_delta else 0.0,
        "first_y_values": first_y,
        "first_yaw_values": first_yaw,
    }


def _resolve_output_paths(
    *,
    output_root: str | Path,
    out_dir: str | Path | None,
    label: str,
    timestamp: str,
) -> OutputPaths:
    safe_label = str(label or "sparsedrive_v1").strip().replace(os.sep, "_").replace(" ", "_")
    root = Path(out_dir) if out_dir is not None else Path(output_root) / f"{safe_label}-{timestamp}"
    artifacts = root / "artifacts"
    stem = f"{safe_label}_{timestamp}_sparsedrivev1"
    return OutputPaths(
        root=root,
        artifacts=artifacts,
        frames=artifacts / "frames",
        bev=artifacts / "bev",
        candidate_grid=artifacts / "candidate_grid",
        candidate_json=artifacts / "candidate_json",
        video=artifacts / f"{stem}_hugsim_online.mp4",
        selected_csv=artifacts / f"{stem}_selected_plan.csv",
        candidate_csv=artifacts / f"{stem}_candidate_trajs.csv",
        reward_csv=artifacts / f"{stem}_step_reward.csv",
        summary_json=artifacts / f"{stem}_jitter_summary.json",
        manifest=root / "run_info.md",
        debug_shard=artifacts / f"{stem}_debug_shard.pt",
    )


def _first_hugsim_label(cfg: Mapping[str, Any], fallback: str = "hugsim_ori") -> str:
    env_cfg = cfg.get("env", {}) if isinstance(cfg, Mapping) else {}
    hugsim_cfg = env_cfg.get("hugsim", {}) if isinstance(env_cfg, Mapping) else {}
    scenes = hugsim_cfg.get("scenes", None) if isinstance(hugsim_cfg, Mapping) else None
    if isinstance(scenes, (list, tuple)) and scenes:
        return str(scenes[0])
    return str(fallback)


def _format_hugsim_scene_name(scene: Any) -> str:
    text = str(scene).strip()
    if not text:
        raise ValueError("Empty HUGSIM scene name")
    if text.startswith("scene-"):
        return text
    try:
        return f"scene-{int(text):04d}"
    except ValueError:
        return text


def _apply_hugsim_scene_override(cfg: dict[str, Any], args: argparse.Namespace) -> str | None:
    scene_arg = getattr(args, "hugsim_scene", None)
    if scene_arg is None:
        scene_arg = getattr(args, "scene_name", None)
    if scene_arg is None:
        return None
    scene_name = _format_hugsim_scene_name(scene_arg)
    env_cfg = cfg.setdefault("env", {})
    if not isinstance(env_cfg, dict):
        raise RuntimeError("Config field `env` must be a mapping")
    hugsim_cfg = env_cfg.setdefault("hugsim", {})
    if not isinstance(hugsim_cfg, dict):
        raise RuntimeError("Config field `env.hugsim` must be a mapping")
    hugsim_cfg["scenes"] = [scene_name]
    return scene_name


def _apply_scenario_path_override(cfg: dict[str, Any], scenario_path: str | None) -> str | None:
    if scenario_path is None:
        return None
    path = Path(resolve_repo_path(str(scenario_path)))
    if not path.exists():
        raise FileNotFoundError(f"HUGSIM scenario path not found: {path}")
    scene_name = path.stem
    env_cfg = cfg.setdefault("env", {})
    if not isinstance(env_cfg, dict):
        raise RuntimeError("Config field `env` must be a mapping")
    hugsim_cfg = env_cfg.setdefault("hugsim", {})
    if not isinstance(hugsim_cfg, dict):
        raise RuntimeError("Config field `env.hugsim` must be a mapping")
    hugsim_cfg["scenario_dir"] = str(path.parent)
    hugsim_cfg["scenes"] = [scene_name]
    return scene_name


def _apply_agent_overrides(
    cfg: dict[str, Any],
    *,
    ckpt_path: str,
    sparsedrive_config_path: str,
) -> None:
    agent_cfg = cfg.setdefault("agent", {})
    if not isinstance(agent_cfg, dict):
        raise RuntimeError("Config field `agent` must be a mapping")
    agent_cfg["type"] = "sparsedrive"
    agent_cfg["ckpt"] = str(ckpt_path)
    agent_cfg["config"] = str(sparsedrive_config_path)


def _apply_scene_index_override(cfg: dict[str, Any], scene_index: int | None) -> str | None:
    if scene_index is None:
        return None
    env_cfg = cfg.setdefault("env", {})
    if not isinstance(env_cfg, dict):
        raise RuntimeError("Config field `env` must be a mapping")
    hugsim_cfg = env_cfg.setdefault("hugsim", {})
    if not isinstance(hugsim_cfg, dict):
        raise RuntimeError("Config field `env.hugsim` must be a mapping")
    from framework.runner.env_factory import discover_hugsim_scenarios
    from framework.utils.repo_paths import resolve_hugsim_path

    scenario_dir = str(
        resolve_hugsim_path(
            hugsim_cfg.get("scenario_dir", None),
            "configs",
            "scenarios",
            "nuscenes",
        )
    )
    scenarios = discover_hugsim_scenarios(scenario_dir)
    idx = int(scene_index)
    if idx < 0 or idx >= len(scenarios):
        raise RuntimeError(f"HUGSIM scene index out of range: {idx}; discovered={len(scenarios)} under {scenario_dir}")
    scene_name = str(scenarios[idx].official_scene_name)
    hugsim_cfg["scenes"] = [scene_name]
    return scene_name


def _frame_idx_from_replay_or_info(replay: Mapping[str, Any], info: Mapping[str, Any] | None = None) -> int:
    info_map = dict(info or {})
    for source in (replay, info_map):
        for key in ("frame_idx", "now_frame", "frame_before"):
            if key in source:
                try:
                    return int(source[key])
                except Exception:
                    pass
    return -1


def _extract_v1_candidates(policy: Any, observation: Mapping[str, Any]) -> dict[str, Any]:
    img_t, data = policy._build_batch(dict(observation))
    model_device = getattr(policy, "device", torch.device("cpu"))
    img_t = img_t.to(model_device)
    data_dev = policy._to_model_metas(data, model_device)
    model = getattr(policy, "_model", None)
    if model is None:
        raise RuntimeError("SparseDrive-v1 policy missing _model")
    model.eval()
    with torch.inference_mode():
        logits_all, traj_all = policy._forward_raw_planning(img=img_t, data_dev=data_dev)
    logits_all = logits_all.detach().cpu().to(dtype=torch.float32)
    traj_all = traj_all.detach().cpu().to(dtype=torch.float32)

    cmd = data["gt_ego_fut_cmd"].detach().cpu().reshape(-1)
    cmd_idx = int(torch.argmax(cmd[:3]).item()) if int(cmd.numel()) >= 3 else 0
    logits = logits_all[int(cmd_idx)].reshape(-1).numpy().astype(np.float32)
    traj_xyyaw = _v1_trajs_to_xyyaw(policy, traj_all[int(cmd_idx)].numpy())
    probs = _softmax_np(logits)
    return {
        "cmd_idx": int(cmd_idx),
        "logits": logits,
        "probs": probs,
        "traj_xyyaw": traj_xyyaw,
        "mode_indices": np.arange(int(traj_xyyaw.shape[0]), dtype=np.int64),
    }


def _select_v1_action_from_candidates(
    candidates: Mapping[str, Any],
    *,
    mode_idx: int = -1,
    mode_select: str = "greedy",
) -> tuple[tuple[float, float, float, int], torch.Tensor, dict[str, Any]]:
    logits = np.asarray(candidates["logits"], dtype=np.float32).reshape(-1)
    traj_xyyaw = np.asarray(candidates["traj_xyyaw"], dtype=np.float32)
    if traj_xyyaw.ndim != 3 or traj_xyyaw.shape[-1] < 3:
        raise RuntimeError(f"Expected candidate traj_xyyaw shape (modes,horizon,3+), got {tuple(traj_xyyaw.shape)}")
    if int(logits.shape[0]) != int(traj_xyyaw.shape[0]):
        raise RuntimeError("Candidate logits length must match trajectory modes")

    if int(mode_idx) < 0:
        sel = str(mode_select).strip().lower()
        if sel in {"greedy", "max", "argmax"}:
            selected_mode_idx = int(np.argmax(logits))
        else:
            probs = _softmax_np(logits)
            if probs.shape[0] > 0 and np.isfinite(probs).all() and float(probs.sum()) > 0.0:
                selected_mode_idx = int(np.random.choice(np.arange(probs.shape[0]), p=probs.astype(np.float64)))
            else:
                selected_mode_idx = int(np.argmax(logits))
    else:
        selected_mode_idx = max(0, min(int(mode_idx), int(logits.shape[0]) - 1))

    log_probs = torch.log_softmax(torch.as_tensor(logits, dtype=torch.float32), dim=0)
    logp = log_probs[int(selected_mode_idx)].detach().cpu().clone()
    selected_traj = traj_xyyaw[int(selected_mode_idx), :, :3].astype(np.float32, copy=True)
    first = selected_traj[0]
    action = (float(first[0]), float(first[1]), float(first[2]), 2)
    traj_tensor = torch.from_numpy(selected_traj).detach().cpu().clone()
    replay: dict[str, Any] = {
        "cmd_idx": int(candidates.get("cmd_idx", -1)),
        "mode_idx": int(selected_mode_idx),
        "traj_xyyaw": traj_tensor.clone(),
        "execute_mode": "continuous",
    }
    try:
        from framework.replay_schema import set_env_plan

        set_env_plan(replay, plan_xyyaw=traj_tensor.clone(), plan_frame="ego", action_flag=2)
    except Exception:
        pass
    return action, logp, replay


def _render_candidate_grid_image(
    *,
    sample_detail: Mapping[str, Any],
    traj_xyyaw: np.ndarray,
    scores: Sequence[float],
    top_k: int,
    selected_index: int | None,
    panel_size: int,
    columns: int,
) -> np.ndarray:
    traj = np.asarray(traj_xyyaw, dtype=np.float32)
    score_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    order = _sort_candidate_indices(score_arr)
    count = int(traj.shape[0])
    display = order[: max(1, min(count, int(top_k)))]
    if selected_index is not None and int(selected_index) not in display and 0 <= int(selected_index) < count:
        display.append(int(selected_index))
    if not display:
        display = list(range(min(count, max(1, int(top_k)))))

    panel = max(160, int(panel_size))
    cols = max(1, int(columns))
    rows = int(math.ceil(len(display) / float(cols)))
    canvas = np.full((rows * panel, cols * panel, 3), 246, dtype=np.uint8)
    for rank, cand_idx in enumerate(display):
        r = rank // cols
        c = rank % cols
        one_detail = dict(sample_detail)
        one_detail["sample_token"] = f"mode {cand_idx}"
        img = render_bev_debug_image(
            sample_detail=one_detail,
            traj_xyyaw=traj[int(cand_idx) : int(cand_idx) + 1],
            scores=np.asarray([score_arr[int(cand_idx)]], dtype=np.float32),
            top_k=1,
            width=panel,
            height=panel,
            selected_index=0 if selected_index is not None and int(cand_idx) == int(selected_index) else None,
            max_display_candidates=1,
        )
        lines = [
            f"rank={rank + 1} mode={cand_idx}",
            f"logit={float(score_arr[int(cand_idx)]):+.4f}",
            "SELECTED" if selected_index is not None and int(cand_idx) == int(selected_index) else "",
        ]
        img = _overlay_debug_text(img, [line for line in lines if line])
        canvas[r * panel : (r + 1) * panel, c * panel : (c + 1) * panel] = img
    return canvas


def _build_sample_detail(replay: Mapping[str, Any], current_pose: np.ndarray | None = None) -> dict[str, Any]:
    del current_pose
    return {
        "sample_token": str(replay.get("sample_token", "")),
        "map_patch_radius_m": 25.0,
        "gt_xy": np.zeros((0, 2), dtype=np.float32),
        "gt_history_xy": np.zeros((0, 2), dtype=np.float32),
        "ego_history_xy": np.zeros((0, 2), dtype=np.float32),
        "scene_objects": [],
        "render_layers": {},
    }


def _build_reward_row(
    *,
    step: int,
    frame_before: int,
    frame_after: int,
    replay: Mapping[str, Any],
    logp: Any,
    reward: float,
    cum_reward: float,
    done: bool,
    info: Mapping[str, Any],
) -> dict[str, Any]:
    logp_v = float(logp.detach().cpu().item()) if torch.is_tensor(logp) else float(logp)
    return {
        "step": int(step),
        "scene_id": int(info.get("scene_id", replay.get("scene_id", -1)) or -1),
        "frame_before": int(frame_before),
        "frame_after": int(frame_after),
        "sample_token": str(replay.get("sample_token", "")),
        "sample_token_after": str(info.get("sample_token", "")),
        "cmd_idx": int(replay.get("cmd_idx", -1)),
        "mode_idx": int(replay.get("mode_idx", -1)),
        "old_logp": float(logp_v),
        "reward": float(reward),
        "cum_reward": float(cum_reward),
        "done": bool(done),
        "done_reason": str(info.get("done_reason", "")),
        "progress_reward": float(info.get("progress_reward", 0.0) or 0.0),
        "cost_reward": float(info.get("cost_reward", 0.0) or 0.0),
        "collision": bool(info.get("collision", False)),
        "dynamic_collision": bool(info.get("dynamic_collision", False)),
        "static_collision": bool(info.get("static_collision", False)),
    }


def _write_manifest(
    *,
    paths: OutputPaths,
    config_path: str,
    ckpt_path: str,
    sparsedrive_config_path: str,
    label: str,
    steps: int,
    detailed_outputs: bool = True,
) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# SparseDrive-v1 HUGSIM {'detailed ' if bool(detailed_outputs) else ''}visualization\n",
        f"- run_dir: `{paths.root}`\n",
        f"- label: {label}\n",
        f"- config: `{config_path}`\n",
        f"- ckpt: `{ckpt_path}`\n",
        f"- sparsedrive_config: `{sparsedrive_config_path}`\n",
        f"- steps: {int(steps)}\n",
        f"- video: `{paths.video}`\n",
        f"- selected_csv: `{paths.selected_csv}`\n",
        f"- reward_csv: `{paths.reward_csv}`\n",
    ]
    if bool(detailed_outputs):
        lines.extend(
            [
                f"- candidate_csv: `{paths.candidate_csv}`\n",
                f"- bev_dir: `{paths.bev}`\n",
                f"- candidate_grid_dir: `{paths.candidate_grid}`\n",
            ]
        )
    paths.manifest.write_text("".join(lines), encoding="utf-8")


def run_online_hugsim_scene(
    *,
    config_path: str | Path,
    ckpt_path: str | Path,
    sparsedrive_config_path: str | Path,
    output_root: str | Path = _DEFAULT_OUTPUT_ROOT,
    out_dir: str | Path | None = None,
    hugsim_scene: str | None = None,
    scenario_path: str | None = None,
    scene_index: int | None = None,
    timestamp: str | None = None,
    cuda: int = 0,
    horizon: int = 32,
    fps: float = 2.0,
    mode_idx: int = -1,
    mode_select: str = "greedy",
    top_k: int = 8,
    bev_size: int = 520,
    grid_panel_size: int = 360,
    grid_columns: int = 4,
    worker_id: int | None = None,
    save_shard: bool = True,
    detailed_outputs: bool = True,
    smooth_alpha: float = 0.0,
) -> dict[str, Any]:
    from framework.runner.agent_factory import build_agent
    from framework.runner.env_factory import build_actor_env

    config_resolved = str(resolve_repo_path(str(config_path)))
    ckpt_resolved = str(resolve_repo_path(str(ckpt_path)))
    sd_config_resolved = str(resolve_repo_path(str(sparsedrive_config_path)))
    cfg = _load_yaml(config_resolved)
    _apply_agent_overrides(cfg, ckpt_path=ckpt_resolved, sparsedrive_config_path=sd_config_resolved)
    selected_label = _apply_scenario_path_override(cfg, scenario_path) or None
    selected_label = _apply_hugsim_scene_override(cfg, argparse.Namespace(hugsim_scene=hugsim_scene, scene_name=None)) or selected_label
    if selected_label is None:
        selected_label = _apply_scene_index_override(cfg, scene_index)
    label = selected_label or _first_hugsim_label(cfg, fallback=f"hugsim_scene_{scene_index}" if scene_index is not None else "hugsim_ori")
    ts = timestamp or time.strftime("%Y%m%d-%H%M%S")
    paths = _resolve_output_paths(output_root=output_root, out_dir=out_dir, label=label, timestamp=ts)
    output_dirs = [paths.artifacts]
    if bool(detailed_outputs):
        output_dirs.extend([paths.frames, paths.bev, paths.candidate_grid, paths.candidate_json])
    for directory in output_dirs:
        _ensure_dir(directory)

    device = torch.device(f"cuda:{int(cuda)}" if torch.cuda.is_available() and int(cuda) >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(int(cuda))

    env = build_actor_env(
        cfg,
        cuda=max(0, int(cuda)),
        actor_id=(int(scene_index) if scene_index is not None else 0),
        worker_id=(None if worker_id is None else int(worker_id)),
        total_actors=1,
    )
    agent = build_agent(cfg, device=device)
    obs, reset_info = env.reset()
    sim = getattr(env, "env", env)
    max_steps = None if int(horizon) <= 0 else int(horizon)

    writer = imageio.get_writer(
        str(paths.video),
        mode="I",
        fps=float(fps),
        macro_block_size=1,
        codec="libx264",
        ffmpeg_log_level="error",
        input_params=["-framerate", str(float(fps))],
        output_params=["-pix_fmt", "yuv420p"],
    )

    selected_rows: list[dict[str, Any]] = []
    candidate_rows_all: list[dict[str, Any]] = []
    reward_rows: list[dict[str, Any]] = []
    debug_shard: dict[str, Any] = {"reward": [], "done": [], "old_logp": [], "replay": [], "meta": []}
    reward_sum = 0.0
    done = False
    step = 0
    previous_selected_traj: np.ndarray | None = None
    smooth_alpha_v = max(0.0, min(float(smooth_alpha), 0.99))
    try:
        while (max_steps is None or step < max_steps) and not done:
            obs_for_policy = _obs_for_policy(obs, sim)
            frame_before = int(getattr(sim, "now_frame", _frame_idx_from_replay_or_info({}, {})))
            candidates = _extract_v1_candidates(agent, obs_for_policy) if bool(detailed_outputs) else None
            if bool(detailed_outputs):
                assert candidates is not None
                action, logp, replay = _select_v1_action_from_candidates(
                    candidates,
                    mode_idx=int(mode_idx),
                    mode_select=str(mode_select),
                )
            else:
                action, logp, replay = agent.act(
                    obs_for_policy,
                    mode_idx=int(mode_idx),
                    mode_select=str(mode_select),
                )
            selected_plan = replay.get("traj_xyyaw")
            if torch.is_tensor(selected_plan):
                selected_traj = selected_plan.detach().cpu().numpy()
            else:
                selected_traj = np.asarray(selected_plan, dtype=np.float32)
            selected_traj = np.asarray(selected_traj, dtype=np.float32)
            if selected_traj.ndim != 2 or selected_traj.shape[-1] < 3:
                raise RuntimeError(f"Invalid selected traj_xyyaw shape: {tuple(selected_traj.shape)}")
            selected_traj = selected_traj[:, :3]
            selected_traj = _smooth_plan_with_previous(
                previous_plan=previous_selected_traj,
                current_plan=selected_traj,
                alpha=smooth_alpha_v,
            )
            previous_selected_traj = selected_traj.astype(np.float32, copy=True)
            if smooth_alpha_v > 0.0:
                replay["traj_xyyaw"] = torch.from_numpy(selected_traj).detach().cpu().clone()
                try:
                    from framework.replay_schema import set_env_plan

                    set_env_plan(replay, plan_xyyaw=replay["traj_xyyaw"].clone(), plan_frame="ego", action_flag=2)
                except Exception:
                    pass

            setter = getattr(env, "set_external_plan_local_xyyaw", None)
            if callable(setter):
                setter(selected_traj.astype(np.float32))
            else:
                setattr(sim, "_external_plan_local_xyyaw", selected_traj.astype(np.float64).copy())

            selected_mode_idx = int(replay.get("mode_idx", -1))
            cmd_idx = int(replay.get("cmd_idx", -1))
            if bool(detailed_outputs):
                assert candidates is not None
                cmd_idx = int(replay.get("cmd_idx", candidates["cmd_idx"]))
                candidate_traj = np.asarray(candidates["traj_xyyaw"], dtype=np.float32)
                logits = np.asarray(candidates["logits"], dtype=np.float32)
                probs = np.asarray(candidates["probs"], dtype=np.float32)
                mode_indices = np.asarray(candidates["mode_indices"], dtype=np.int64)
                selected_candidate_idx = selected_mode_idx if 0 <= selected_mode_idx < int(candidate_traj.shape[0]) else None
                sample_detail = _build_sample_detail(replay)

                bev_img = render_bev_debug_image(
                    sample_detail=sample_detail,
                    traj_xyyaw=candidate_traj,
                    scores=logits,
                    top_k=int(top_k),
                    width=int(bev_size),
                    height=int(bev_size),
                    selected_index=selected_candidate_idx,
                    max_display_candidates=max(8, min(int(candidate_traj.shape[0]), int(top_k) + 6)),
                )
                bev_path = paths.bev / f"step_{step:06d}_bev.png"
                imageio.imwrite(bev_path, bev_img)

                grid_img = _render_candidate_grid_image(
                    sample_detail=sample_detail,
                    traj_xyyaw=candidate_traj,
                    scores=logits,
                    top_k=int(top_k),
                    selected_index=selected_candidate_idx,
                    panel_size=int(grid_panel_size),
                    columns=int(grid_columns),
                )
                grid_path = paths.candidate_grid / f"step_{step:06d}_candidates.png"
                imageio.imwrite(grid_path, grid_img)

                candidate_payload = build_candidate_score_payload(
                    scene=int(scene_index if scene_index is not None else -1),
                    step=int(step),
                    frame_idx=int(frame_before),
                    sample_token=str(replay.get("sample_token", "")),
                    traj_xyyaw=candidate_traj,
                    scores=probs,
                    score_logits=logits,
                    mode_indices=mode_indices,
                    top_k=int(top_k),
                )
                candidate_payload["selected_mode_idx"] = int(selected_mode_idx)
                candidate_payload["selected_candidate_idx"] = None if selected_candidate_idx is None else int(selected_candidate_idx)
                _write_json(paths.candidate_json / f"step_{step:06d}_candidates.json", candidate_payload)

                candidate_rows_all.extend(
                    _candidate_rows(
                        step=int(step),
                        frame_idx=int(frame_before),
                        cmd_idx=int(cmd_idx),
                        selected_mode_idx=int(selected_mode_idx),
                        traj_xyyaw=candidate_traj,
                        logits=logits,
                        probs=probs,
                    )
                )
            selected_rows.extend(
                _selected_rows(
                    step=int(step),
                    frame_idx=int(frame_before),
                    cmd_idx=int(cmd_idx),
                    mode_idx=int(selected_mode_idx),
                    logp=logp,
                    traj_xyyaw=selected_traj,
                )
            )

            obs_next, reward_v, terminated, truncated, info_raw = env.step(action)
            info = dict(info_raw or {})
            done = bool(terminated or truncated)
            reward_sum += float(reward_v)
            frame_after = int(info.get("now_frame", info.get("frame_idx", getattr(sim, "now_frame", frame_before))))
            reward_row = _build_reward_row(
                step=int(step),
                frame_before=int(frame_before),
                frame_after=int(frame_after),
                replay=replay,
                logp=logp,
                reward=float(reward_v),
                cum_reward=float(reward_sum),
                done=bool(done),
                info=info,
            )
            reward_rows.append(reward_row)
            if bool(detailed_outputs):
                debug_shard["reward"].append(float(reward_v))
                debug_shard["done"].append(bool(done))
                debug_shard["old_logp"].append(float(reward_row["old_logp"]))
                debug_shard["replay"].append(replay)
                debug_shard["meta"].append(
                    {
                        "step": int(step),
                        "frame_before": int(frame_before),
                        "frame_after": int(frame_after),
                        "reset_info": _json_safe(reset_info),
                        "info": _json_safe(info),
                        "bev": str(bev_path),
                        "candidate_grid": str(grid_path),
                    }
                )

            frame = _grid_frame(obs_next)
            if bool(detailed_outputs):
                best_mode = int(_sort_candidate_indices(logits)[0]) if int(logits.shape[0]) > 0 else -1
                logp_v = float(logp.detach().cpu().item()) if torch.is_tensor(logp) else float(logp)
                frame = _overlay_debug_text(
                    frame,
                    [
                        f"sparsedrive-v1 scene={label} step={step} frame={frame_before}->{frame_after}",
                        f"cmd={cmd_idx} selected_mode={selected_mode_idx} best_logit_mode={best_mode} logp={logp_v:+.5f}",
                        f"reward={float(reward_v):+.5f} cum={reward_sum:+.5f} done={done}",
                        f"first_plan_xy=({selected_traj[0, 0]:+.2f},{selected_traj[0, 1]:+.2f}) yaw={selected_traj[0, 2]:+.2f}",
                    ],
                )
                frame = overlay_top_right_inset(frame, bev_img, inset_width=420, margin=12, border_px=2)
            writer.append_data(frame)
            if bool(detailed_outputs):
                imageio.imwrite(paths.frames / f"step_{step:06d}.png", frame)
            obs = obs_next
            step += 1
    finally:
        writer.close()
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    _write_csv_rows(paths.selected_csv, selected_rows)
    _write_csv_rows(paths.reward_csv, reward_rows)
    if bool(detailed_outputs):
        _write_csv_rows(paths.candidate_csv, candidate_rows_all)
    summary = _compute_jitter_summary(
        selected_rows=selected_rows,
        candidate_rows=candidate_rows_all,
        reward_rows=reward_rows,
        mode_select=str(mode_select),
        requested_mode_idx=int(mode_idx),
        smooth_alpha=float(smooth_alpha_v),
    )
    _write_json(paths.summary_json, summary)
    if bool(save_shard) and bool(detailed_outputs):
        torch.save(debug_shard, paths.debug_shard)
    _write_manifest(
        paths=paths,
        config_path=config_resolved,
        ckpt_path=ckpt_resolved,
        sparsedrive_config_path=sd_config_resolved,
        label=label,
        steps=int(step),
        detailed_outputs=bool(detailed_outputs),
    )
    result = {
        "run_dir": str(paths.root),
        "video": str(paths.video),
        "selected_csv": str(paths.selected_csv),
        "reward_csv": str(paths.reward_csv),
        "summary_json": str(paths.summary_json),
        "steps": int(step),
        "reward_sum": float(reward_sum),
    }
    if bool(detailed_outputs):
        result.update(
            {
                "bev_dir": str(paths.bev),
                "candidate_grid_dir": str(paths.candidate_grid),
                "candidate_json_dir": str(paths.candidate_json),
                "candidate_csv": str(paths.candidate_csv),
                "debug_shard": str(paths.debug_shard) if bool(save_shard) else None,
            }
        )
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SparseDrive-v1 HUGSIM detailed rollout visualizer")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--sparsedrive-config", type=str, default=str(_DEFAULT_SPARSEDRIVE_CONFIG))
    parser.add_argument("--output-root", type=str, default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--out-dir", "--out", dest="out_dir", type=str, default=None)
    parser.add_argument("--hugsim-scene", type=str, default=None, help="Official HUGSIM scene name, e.g. scene-0661-medium-01.")
    parser.add_argument("--scene-name", type=str, default=None, help="Alias used by some visualizers.")
    parser.add_argument("--scenario-path", type=str, default=None, help="Exact HUGSIM scenario yaml path.")
    parser.add_argument("--scene-index", type=int, default=None)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--horizon", "--max-steps", dest="horizon", type=int, default=32)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--mode-idx", type=int, default=-1)
    parser.add_argument("--mode-select", type=str, default="greedy", choices=["greedy", "sample"])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--bev-size", type=int, default=520)
    parser.add_argument("--grid-panel-size", type=int, default=360)
    parser.add_argument("--grid-columns", type=int, default=4)
    parser.add_argument("--worker-id", type=int, default=None)
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.0,
        help="EMA weight for previous selected local plan xy. 0 disables smoothing.",
    )
    parser.add_argument("--save-shard", dest="save_shard", action="store_true", default=True)
    parser.add_argument("--no-save-shard", dest="save_shard", action="store_false")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    result = run_online_hugsim_scene(
        config_path=args.config,
        ckpt_path=args.ckpt,
        sparsedrive_config_path=args.sparsedrive_config,
        output_root=args.output_root,
        out_dir=args.out_dir,
        hugsim_scene=args.hugsim_scene or args.scene_name,
        scenario_path=args.scenario_path,
        scene_index=args.scene_index,
        timestamp=args.timestamp,
        cuda=int(args.cuda),
        horizon=int(args.horizon),
        fps=float(args.fps),
        mode_idx=int(args.mode_idx),
        mode_select=str(args.mode_select),
        top_k=int(args.top_k),
        bev_size=int(args.bev_size),
        grid_panel_size=int(args.grid_panel_size),
        grid_columns=int(args.grid_columns),
        worker_id=args.worker_id,
        save_shard=bool(args.save_shard),
        smooth_alpha=float(args.smooth_alpha),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
