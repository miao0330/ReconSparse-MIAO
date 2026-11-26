#!/usr/bin/env python3
"""Detailed SparseDriveV2 HUGSIM-ORI candidate visualizer.

Saves the rollout mp4 plus per-step camera frames, BEV images, candidate
trajectory grid screenshots, candidate score JSON, step JSON, step CSV, and a
manifest.

Command:
cd /root/clone/ReconDreamer-RL
CUDA_VISIBLE_DEVICES=1 python tools/smalltool/visualize/sparsedrive_v2/detailed.py \
  --config /root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/20260616_template_HUGSM_algo-dsl.yaml \
  --ckpt /root/clone/ReconDreamer-RL/egoADs/SparseDriveV2/ckpt/sparsedrive_navsimv2.ckpt \
  --scenario-path /root/clone/HUGSIM-ORI/configs/scenarios/nuscenes/scene-0062-medium-00.yaml \
  --max-steps 42 \
  --num-candidates 8 \
  --top-k 5 \
  --mode-select greedy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework.runner.env_factory import discover_hugsim_scenarios
from framework.replay_schema import get_env_plan_xyyaw
from framework.utils.repo_paths import resolve_hugsim_path, resolve_hugsim_root
from framework.utils.repo_paths import resolve_repo_path



# Candidate trajectory visualization helpers imported from the previous CRAFT visualizer.
def _stable_desc_sort_indices(scores: Sequence[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.argsort(-arr, kind="stable").astype(np.int64, copy=False)


def candidate_visual_styles(
    *,
    scores: Sequence[float],
    top_k: int,
    min_alpha: float = 0.12,
    max_alpha: float = 0.70,
) -> list[dict[str, Any]]:
    ranked_indices = _stable_desc_sort_indices(scores)
    score_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    total = int(score_arr.shape[0])
    if total <= 0:
        return []

    top_k_count = max(1, min(int(top_k), total))
    out: list[dict[str, Any]] = []
    denom = max(1, total - 1)
    for rank, candidate_index in enumerate(ranked_indices.tolist(), start=1):
        score_rank = rank - 1
        fade = float(score_rank) / float(denom)
        alpha = float(max(min_alpha, max_alpha - 0.50 * fade))
        linewidth = float(max(1.0, 2.8 - 1.25 * fade))
        color = _score_to_rgb(float(score_arr[int(candidate_index)]), min_score=float(score_arr.min()), max_score=float(score_arr.max()))
        out.append(
            {
                "candidate_index": int(candidate_index),
                "rank": int(rank),
                "score": float(score_arr[int(candidate_index)]),
                "alpha": float(alpha),
                "linewidth": float(linewidth),
                "color": tuple(int(v) for v in color),
                "is_top_k": bool(rank <= top_k_count),
            }
        )
    return out


def format_pdm_score_percent(score: float, *, digits: int = 2) -> str:
    return f"{float(score) * 100.0:.{max(0, int(digits))}f}"


def select_bev_candidate_indices(
    *,
    scores: Sequence[float],
    top_k: int,
    selected_index: int | None = None,
    max_candidates: int = 8,
) -> list[int]:
    score_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    total = int(score_arr.shape[0])
    if total <= 0:
        return []
    limit = max(1, min(int(max_candidates), total))
    ranked = _stable_desc_sort_indices(score_arr).tolist()
    keep: list[int] = []

    def add(idx: int | None) -> None:
        if idx is None:
            return
        idx_int = int(idx)
        if 0 <= idx_int < total and idx_int not in keep and len(keep) < limit:
            keep.append(idx_int)

    for idx in ranked[: max(1, min(int(top_k), total))]:
        add(int(idx))
    add(selected_index)
    if len(keep) < limit:
        add(int(ranked[-1]))
    if len(keep) < limit and total > 2:
        quantile_positions = np.linspace(0, total - 1, num=limit, dtype=np.int64).tolist()
        for pos in quantile_positions:
            add(int(ranked[int(pos)]))
            if len(keep) >= limit:
                break
    return keep


def _score_to_rgb(score: float, *, min_score: float, max_score: float) -> tuple[int, int, int]:
    if not math.isfinite(score):
        return (180, 180, 180)
    span = max(1.0e-6, float(max_score) - float(min_score))
    norm = float(np.clip((float(score) - float(min_score)) / span, 0.0, 1.0))
    low = np.asarray([210, 220, 230], dtype=np.float32)
    high = np.asarray([235, 96, 52], dtype=np.float32)
    rgb = (1.0 - norm) * low + norm * high
    return tuple(int(np.clip(v, 0, 255)) for v in np.rint(rgb))


def build_candidate_score_payload(
    *,
    scene: int,
    step: int,
    frame_idx: int,
    sample_token: str,
    traj_xyyaw: np.ndarray,
    scores: Sequence[float],
    score_logits: Sequence[float] | None,
    mode_indices: Sequence[int] | None,
    top_k: int,
    candidate_details: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    traj = np.asarray(traj_xyyaw, dtype=np.float32)
    if traj.ndim != 3 or traj.shape[-1] < 3:
        raise RuntimeError(f"Expected traj_xyyaw shape (candidates,horizon,3), got {tuple(traj.shape)}")

    score_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if int(score_arr.shape[0]) != int(traj.shape[0]):
        raise RuntimeError(
            "Candidate count mismatch between scores and trajectories: "
            f"scores={int(score_arr.shape[0])} traj={int(traj.shape[0])}"
        )
    logits_arr = None if score_logits is None else np.asarray(score_logits, dtype=np.float32).reshape(-1)
    if logits_arr is not None and int(logits_arr.shape[0]) != int(traj.shape[0]):
        raise RuntimeError("score_logits length mismatch")
    mode_idx_arr = None if mode_indices is None else np.asarray(mode_indices, dtype=np.int64).reshape(-1)
    if mode_idx_arr is not None and int(mode_idx_arr.shape[0]) != int(traj.shape[0]):
        raise RuntimeError("mode_indices length mismatch")

    styles = candidate_visual_styles(scores=score_arr, top_k=top_k)
    style_by_candidate = {int(item["candidate_index"]): dict(item) for item in styles}
    ranked_indices = [int(item["candidate_index"]) for item in styles]
    details_by_candidate = {int(key): dict(value) for key, value in dict(candidate_details or {}).items()}
    payload_candidates: list[dict[str, Any]] = []
    for candidate_index in ranked_indices:
        visual = dict(style_by_candidate[int(candidate_index)])
        item = {
            "candidate_index": int(candidate_index),
            "rank": int(visual["rank"]),
            "score": float(visual["score"]),
            "visual": {
                "alpha": float(visual["alpha"]),
                "linewidth": float(visual["linewidth"]),
                "color": list(visual["color"]),
                "is_top_k": bool(visual["is_top_k"]),
            },
            "traj_xyyaw": traj[int(candidate_index)].astype(np.float32).tolist(),
        }
        if logits_arr is not None:
            item["score_logit"] = float(logits_arr[int(candidate_index)])
        if mode_idx_arr is not None:
            item["mode_index"] = int(mode_idx_arr[int(candidate_index)])
        if int(candidate_index) in details_by_candidate:
            item["score_breakdown"] = details_by_candidate[int(candidate_index)]
        payload_candidates.append(item)

    return {
        "scene": int(scene),
        "step": int(step),
        "frame_idx": int(frame_idx),
        "sample_token": str(sample_token),
        "top_k": int(max(1, int(top_k))),
        "top_k_candidate_indices": [int(item["candidate_index"]) for item in styles[: max(1, int(top_k))]],
        "candidates": payload_candidates,
    }


def overlay_top_right_inset(
    frame: np.ndarray,
    inset: np.ndarray,
    *,
    inset_width: int = 320,
    margin: int = 12,
    border_px: int = 2,
) -> np.ndarray:
    out = np.asarray(frame).copy()
    inset_arr = np.asarray(inset)
    if out.ndim != 3 or inset_arr.ndim != 3:
        raise RuntimeError("overlay_top_right_inset expects HWC RGB arrays")
    if int(out.shape[2]) < 3 or int(inset_arr.shape[2]) < 3:
        raise RuntimeError("overlay_top_right_inset expects 3-channel images")

    try:
        import cv2
    except Exception:
        return out

    h, w = out.shape[:2]
    inset_w = max(16, min(int(inset_width), max(16, w - 2 * int(margin))))
    scale = float(inset_w) / float(max(1, int(inset_arr.shape[1])))
    inset_h = max(16, int(round(float(inset_arr.shape[0]) * scale)))
    max_h = max(16, h - 2 * int(margin))
    if inset_h > max_h:
        inset_h = max_h
        scale = float(inset_h) / float(max(1, int(inset_arr.shape[0])))
        inset_w = max(16, int(round(float(inset_arr.shape[1]) * scale)))

    resized = cv2.resize(inset_arr, (int(inset_w), int(inset_h)), interpolation=cv2.INTER_LINEAR)
    x1 = int(w - margin - inset_w)
    y1 = int(margin)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x1 + int(inset_w))
    y2 = min(h, y1 + int(inset_h))
    patch = resized[: max(0, y2 - y1), : max(0, x2 - x1)]
    if patch.size == 0:
        return out

    roi = out[y1:y2, x1:x2]
    out[y1:y2, x1:x2] = patch
    if int(border_px) > 0:
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), (245, 245, 245), int(border_px))
    return out


def write_score_payload(out_dir: str | Path, payload: dict[str, Any]) -> Path:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    step = int(payload.get("step", 0))
    path = root / f"step_{step:06d}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _grid_frame(observation: dict[str, np.ndarray]) -> np.ndarray:
    keys = ["front_left", "front", "front_right", "back_left", "back", "back_right"]
    imgs = [observation[k] for k in keys]
    row1 = np.concatenate(imgs[:3], axis=1)
    row2 = np.concatenate(imgs[3:], axis=1)
    return np.concatenate([row1, row2], axis=0)


def _overlay_debug_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    if len(lines) == 0:
        return frame
    try:
        import cv2
    except Exception:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    box_h = min(h - 10, 26 + 22 * len(lines))
    box_w = min(w - 10, 920)
    x0, y0 = 8, 8
    roi = out[y0 : y0 + box_h, x0 : x0 + box_w].copy()
    shade = roi.copy()
    cv2.rectangle(shade, (0, 0), (box_w - 1, box_h - 1), (16, 16, 16), thickness=-1)
    out[y0 : y0 + box_h, x0 : x0 + box_w] = cv2.addWeighted(shade, 0.50, roi, 0.50, 0)
    y = y0 + 22
    for line in lines:
        cv2.putText(out, str(line), (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (236, 236, 236), 1, cv2.LINE_AA)
        y += 22
    return out


def _obs_for_policy(obs: dict[str, Any], sim: Any) -> dict[str, Any]:
    out = dict(obs)
    out.setdefault("timestamp", np.float32(float(getattr(sim, "now_frame", 0)) * 0.1))
    if "ego_pose" not in out:
        out["ego_pose"] = np.asarray(getattr(sim, "start_ego", np.eye(4)), dtype=np.float32)
    if "cam2ego" not in out:
        cam2ego = getattr(sim, "cam2ego", None)
        if isinstance(cam2ego, list) and len(cam2ego) >= 3:
            out["cam2ego"] = np.asarray(np.stack(cam2ego, axis=0), dtype=np.float32)
    if "cam_intrinsics" not in out:
        all_cams = getattr(sim, "all_cams", None)
        if isinstance(all_cams, list) and len(all_cams) >= 3:
            intr = []
            hw = []
            for cam in all_cams:
                intr.append(np.asarray(cam.get("intrinsics"), dtype=np.float32))
                hw.append([float(cam.get("height", sim.h)), float(cam.get("width", sim.w))])
            out["cam_intrinsics"] = np.asarray(np.stack(intr, axis=0), dtype=np.float32)
            out.setdefault("cam_hw", np.asarray(hw, dtype=np.float32))
    if "driving_command" not in out:
        out["driving_command"] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if "ego_status" not in out:
        vel = np.asarray(out.get("ego_velocity", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        acc = np.asarray(out.get("ego_acceleration", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        cmd = np.asarray(out.get("driving_command", np.zeros((4,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        cmd4 = np.zeros((4,), dtype=np.float32)
        vel2 = np.zeros((2,), dtype=np.float32)
        acc2 = np.zeros((2,), dtype=np.float32)
        cmd4[: min(4, cmd.shape[0])] = cmd[: min(4, cmd.shape[0])]
        vel2[: min(2, vel.shape[0])] = vel[: min(2, vel.shape[0])]
        acc2[: min(2, acc.shape[0])] = acc[: min(2, acc.shape[0])]
        out["ego_status"] = np.concatenate([cmd4, vel2, acc2], axis=0).astype(np.float32)
    return out


def _extract_plan_and_scores(policy: Any, replay: dict[str, Any], *, num_candidates: int, candidate_select: str) -> dict[str, Any]:
    candidate_fn = getattr(policy, "sample_counterfactual_trajectories_from_replay_batch", None)
    if not callable(candidate_fn):
        raise RuntimeError("SparseDriveV2Policy missing sample_counterfactual_trajectories_from_replay_batch")
    candidates = candidate_fn([replay], num_candidates=int(num_candidates), candidate_select=str(candidate_select))
    traj_xyyaw = np.asarray(candidates["traj_xyyaw"].detach().cpu().numpy(), dtype=np.float32)[0]
    raw_score_logits = candidates.get("score_logits", np.zeros((1, traj_xyyaw.shape[0]), dtype=np.float32))
    if torch.is_tensor(raw_score_logits):
        score_logits = np.asarray(raw_score_logits.detach().cpu().numpy(), dtype=np.float32)[0]
    else:
        score_logits = np.asarray(raw_score_logits, dtype=np.float32)[0]
    log_probs = np.asarray(candidates["log_probs"].detach().cpu().numpy(), dtype=np.float32)[0]
    return {
        "traj_xyyaw": traj_xyyaw,
        "score_logits": score_logits,
        "log_probs": log_probs,
        "mode_indices": np.asarray(candidates["mode_indices"].detach().cpu().numpy(), dtype=np.int64)[0],
        "candidate_bundle": candidates,
    }


def _jsonable_scalar_or_list(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating,)):
            return float(item)
        if isinstance(item, (np.bool_,)):
            return bool(item)
        return item
    return arr.astype(np.float32, copy=False).tolist() if np.issubdtype(arr.dtype, np.number) else arr.tolist()


def candidate_details_from_sample_detail(sample_detail: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for item in (dict(sample_detail or {}).get("candidates", []) or []):
        if not isinstance(item, dict) or "candidate_index" not in item:
            continue
        idx = int(item["candidate_index"])
        detail: dict[str, Any] = {}
        for key in (
            "score_terms",
            "weighted_metrics",
            "multiplicative_metrics",
            "weighted_score",
            "multiplicative_product",
            "progress_ratio",
            "mean_error_m",
            "final_error_m",
            "first_error_m",
            "mean_yaw_error_rad",
            "smoothness_penalty_raw",
            "ttc_earliest_risk_time_s",
            "driving_direction_oncoming_progress_m",
            "ea_gate_max_ea",
            "ea_gate_evaluated_pairs",
        ):
            if key in item:
                value = item[key]
                if isinstance(value, dict):
                    detail[key] = {str(k): _jsonable_scalar_or_list(v) for k, v in value.items()}
                else:
                    detail[key] = _jsonable_scalar_or_list(value)
        if detail:
            out[idx] = detail
    return out


def candidate_details_from_term_matrices(
    terms: dict[str, Any] | None,
    *,
    candidate_scores: Sequence[float] | None = None,
) -> dict[int, dict[str, Any]]:
    term_dict = dict(terms or {})
    candidate_count = 0
    if candidate_scores is not None:
        candidate_count = int(np.asarray(candidate_scores, dtype=np.float32).reshape(-1).shape[0])
    for value in term_dict.values():
        arr = np.asarray(value)
        if arr.ndim >= 1:
            candidate_count = max(candidate_count, int(arr.shape[0]))
    out: dict[int, dict[str, Any]] = {}
    if candidate_count <= 0:
        return out
    scores = None if candidate_scores is None else np.asarray(candidate_scores, dtype=np.float32).reshape(-1)
    for idx in range(candidate_count):
        score_terms: dict[str, Any] = {}
        step_terms: dict[str, Any] = {}
        for key, value in term_dict.items():
            arr = np.asarray(value)
            if arr.ndim == 0:
                score_terms[str(key)] = _jsonable_scalar_or_list(arr)
            elif arr.shape[0] > idx:
                if arr.ndim == 1:
                    score_terms[str(key)] = _jsonable_scalar_or_list(arr[idx])
                else:
                    row = arr[idx]
                    step_terms[str(key)] = _jsonable_scalar_or_list(row)
                    try:
                        score_terms[f"{key}_sum"] = float(np.asarray(row, dtype=np.float32).sum())
                        score_terms[f"{key}_mean"] = float(np.asarray(row, dtype=np.float32).mean())
                    except Exception:
                        pass
        detail: dict[str, Any] = {"score_terms": score_terms}
        if step_terms:
            detail["step_terms"] = step_terms
        if scores is not None and idx < int(scores.shape[0]):
            detail["final_score"] = float(scores[idx])
        out[idx] = detail
    return out


def extract_candidate_score_details(
    policy: Any,
    replay: dict[str, Any],
    traj_xyyaw: np.ndarray,
    candidate_scores: Sequence[float],
    sample_detail: dict[str, Any] | None = None,
) -> dict[int, dict[str, Any]]:
    details = candidate_details_from_sample_detail(sample_detail)
    if details:
        return details
    scorer_fn = getattr(policy, "_ensure_counterfactual_scorer_backend", None)
    scorer = scorer_fn() if callable(scorer_fn) else None
    terms = getattr(scorer, "_last_terms", None)
    return candidate_details_from_term_matrices(terms, candidate_scores=candidate_scores)


def _build_bev_sample_detail(policy: Any, replay: dict[str, Any], traj_xyyaw: np.ndarray) -> dict[str, Any]:
    scorer = getattr(policy, "_ensure_counterfactual_scorer_backend", None)
    if callable(scorer):
        backend = scorer()
        detail_backends = [
            backend,
            getattr(backend, "_delegate", None),
            getattr(getattr(backend, "_pdm", None), "_delegate", None),
        ]
        for detail_backend in detail_backends:
            score_with_details = getattr(detail_backend, "score_with_details", None)
            if callable(score_with_details):
                _scores, details = score_with_details([replay], torch.as_tensor(traj_xyyaw[None, ...], dtype=torch.float32))
                if len(details) > 0:
                    detail = dict(details[0])
                    detail["render_layers"] = detail.get("render_layers", {}) or {}
                    return detail
    return {
        "sample_token": str(replay.get("sample_token", "")),
        "gt_xy": np.asarray(replay.get("gt_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32),
        "gt_history_xy": np.asarray(replay.get("gt_history_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32),
        "ego_history_xy": np.asarray(replay.get("ego_history_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32),
        "scene_objects": [],
        "render_layers": {},
        "map_patch_radius_m": 20.0,
    }


def _bev_points_to_px(
    points_xy: np.ndarray,
    *,
    width: int,
    height: int,
    patch_radius: float,
) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    cx = float(width) * 0.5
    cy = float(height) * 0.55
    scale = float(min(width, height) * 0.40 / max(1.0e-6, patch_radius))
    px = np.zeros((pts.shape[0], 2), dtype=np.int32)
    px[:, 0] = np.rint(cx - pts[:, 1] * scale).astype(np.int32)
    px[:, 1] = np.rint(cy - pts[:, 0] * scale).astype(np.int32)
    return px


def bev_render_scale(*, width: int, height: int) -> dict[str, float | int]:
    base = max(1.0, min(float(width), float(height)) / 420.0)
    scale = float(np.clip(base, 1.0, 4.5))
    text_scale = float(np.clip(0.75 + 0.25 * math.sqrt(scale), 1.0, 1.7))
    return {
        "scale": scale,
        "font_scale": min(0.75, 0.42 * text_scale),
        "small_font_scale": min(0.66, 0.34 * text_scale),
        "header_font_scale": min(0.82, 0.50 * text_scale),
        "font_thickness": max(1, int(round(0.75 * text_scale))),
        "trajectory_line_px": max(2, int(round(2.2 * scale))),
        "selected_extra_line_px": max(1, int(round(1.0 * scale))),
        "gt_line_px": max(3, int(round(3.0 * scale))),
        "history_line_px": max(2, int(round(2.0 * scale))),
        "ego_radius_px": max(6, int(round(6.0 * scale))),
        "point_radius_px": max(4, int(round(4.0 * scale))),
        "legend_width_px": min(360, max(230, int(round(170.0 * text_scale)))),
        "legend_row_h_px": min(28, max(18, int(round(16.0 * text_scale)))),
        "legend_pad_px": min(12, max(6, int(round(6.0 * text_scale)))),
    }


def _draw_map_layers_on_bev(frame: np.ndarray, render_layers: dict[str, Any], *, patch_radius: float) -> None:
    try:
        import cv2
    except Exception:
        return

    h, w = frame.shape[:2]

    def to_px(points_xy: np.ndarray) -> np.ndarray:
        return _bev_points_to_px(points_xy, width=int(w), height=int(h), patch_radius=patch_radius)

    for polygon in render_layers.get("drivable_polygons", []) or []:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 3:
            cv2.fillPoly(frame, [to_px(pts).reshape(-1, 1, 2)], (228, 226, 220))
    for polygon in render_layers.get("road_surface_polygons", []) or []:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 3:
            cv2.fillPoly(frame, [to_px(pts).reshape(-1, 1, 2)], (220, 217, 211))
    for polygon in render_layers.get("lane_fill_polygons", []) or []:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 3:
            cv2.fillPoly(frame, [to_px(pts).reshape(-1, 1, 2)], (212, 210, 206))
    for polygon in render_layers.get("walkway_polygons", []) or []:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 3:
            cv2.fillPoly(frame, [to_px(pts).reshape(-1, 1, 2)], (199, 221, 194))
    for polygon in render_layers.get("crossing_polygons", []) or []:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 3:
            cv2.polylines(frame, [to_px(pts).reshape(-1, 1, 2)], True, (170, 170, 165), 1, cv2.LINE_AA)
    for line in render_layers.get("road_edge_lines", []) or []:
        pts = np.asarray(line, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 2:
            cv2.polylines(frame, [to_px(pts).reshape(-1, 1, 2)], False, (140, 136, 130), 1, cv2.LINE_AA)
    for line in render_layers.get("lane_boundary_lines", []) or []:
        pts = np.asarray(line, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 2:
            cv2.polylines(frame, [to_px(pts).reshape(-1, 1, 2)], False, (168, 164, 156), 1, cv2.LINE_AA)
    for line in render_layers.get("lane_marking_lines", []) or []:
        pts = np.asarray(line, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 2:
            cv2.polylines(frame, [to_px(pts).reshape(-1, 1, 2)], False, (110, 110, 110), 1, cv2.LINE_AA)
    for line in render_layers.get("lane_centerlines", []) or []:
        pts = np.asarray(line, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 2:
            cv2.polylines(frame, [to_px(pts).reshape(-1, 1, 2)], False, (72, 149, 217), 1, cv2.LINE_AA)


def _draw_scene_objects_on_bev(frame: np.ndarray, scene_objects: Sequence[dict[str, Any]], *, patch_radius: float) -> None:
    try:
        import cv2
    except Exception:
        return

    h, w = frame.shape[:2]

    def to_px(points_xy: np.ndarray) -> np.ndarray:
        return _bev_points_to_px(points_xy, width=int(w), height=int(h), patch_radius=patch_radius)

    for obj in scene_objects or []:
        corners = np.asarray(obj.get("corners_xy", []), dtype=np.float32)
        if corners.ndim != 2 or corners.shape[0] < 3:
            continue
        color_name = str(obj.get("category", "car"))
        if "pedestrian" in color_name.lower():
            color = (142, 68, 173)
        elif "bicycle" in color_name.lower() or "motorcycle" in color_name.lower():
            color = (39, 174, 96)
        elif "bus" in color_name.lower() or "truck" in color_name.lower() or "trailer" in color_name.lower():
            color = (211, 84, 0)
        else:
            color = (231, 76, 60)
        cv2.fillPoly(frame, [to_px(corners).reshape(-1, 1, 2)], color)
        cv2.polylines(frame, [to_px(corners).reshape(-1, 1, 2)], True, (40, 40, 40), 1, cv2.LINE_AA)
        center = np.asarray(obj.get("center_xy", np.mean(corners[:, :2], axis=0)), dtype=np.float32).reshape(2)
        cpx = to_px(center[None, :])[0]
        cv2.circle(frame, tuple(cpx), 2, (25, 25, 25), -1, cv2.LINE_AA)
        yaw = float(obj.get("yaw_rad", 0.0))
        heading = np.asarray([[center[0], center[1]], [center[0] + 1.5 * np.cos(yaw), center[1] + 1.5 * np.sin(yaw)]], dtype=np.float32)
        hpx = to_px(heading)
        cv2.arrowedLine(frame, tuple(hpx[0]), tuple(hpx[1]), (30, 30, 30), 1, cv2.LINE_AA, tipLength=0.25)


def render_bev_debug_image(
    *,
    sample_detail: dict[str, Any],
    traj_xyyaw: np.ndarray,
    scores: Sequence[float],
    top_k: int,
    width: int = 420,
    height: int = 420,
    selected_index: int | None = None,
    max_display_candidates: int = 8,
) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("render_bev_debug_image requires opencv-python") from exc

    detail = dict(sample_detail)
    traj = np.asarray(traj_xyyaw, dtype=np.float32)
    if traj.ndim != 3 or traj.shape[-1] < 3:
        raise RuntimeError(f"Expected traj_xyyaw shape (candidates,horizon,3), got {tuple(traj.shape)}")

    gt_xy = np.asarray(detail.get("gt_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    gt_history_xy = np.asarray(detail.get("gt_history_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    ego_history_xy = np.asarray(detail.get("ego_history_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    scene_objects = list(detail.get("scene_objects", []) or [])
    render_layers = dict(detail.get("render_layers", {}) or {})
    if not render_layers and isinstance(detail.get("map_layers", None), dict):
        try:
            from framework.algorithms.nuscenes_scorer_utils import NuScenesScorerUtils

            render_layers = NuScenesScorerUtils._build_render_layers(dict(detail.get("map_layers", {}) or {}))
        except Exception:
            render_layers = {}
    patch_radius = float(detail.get("map_patch_radius_m", 20.0))
    scale_cfg = bev_render_scale(width=int(width), height=int(height))
    font_scale = float(scale_cfg["font_scale"])
    small_font_scale = float(scale_cfg["small_font_scale"])
    header_font_scale = float(scale_cfg["header_font_scale"])
    font_thickness = int(scale_cfg["font_thickness"])
    traj_line_base = int(scale_cfg["trajectory_line_px"])
    selected_extra_line = int(scale_cfg["selected_extra_line_px"])
    ego_radius = int(scale_cfg["ego_radius_px"])
    point_radius = int(scale_cfg["point_radius_px"])
    pad = int(scale_cfg["legend_pad_px"])

    out = np.full((int(height), int(width), 3), 248, dtype=np.uint8)
    out[:] = np.asarray([245, 243, 238], dtype=np.uint8)
    _draw_map_layers_on_bev(out, render_layers, patch_radius=patch_radius)
    _draw_scene_objects_on_bev(out, scene_objects, patch_radius=patch_radius)

    cx = float(width) * 0.5
    cy = float(height) * 0.55

    def to_px(points_xy: np.ndarray) -> np.ndarray:
        return _bev_points_to_px(points_xy, width=int(width), height=int(height), patch_radius=patch_radius)

    cv2.circle(out, (int(cx), int(cy)), ego_radius, (35, 35, 35), -1)
    cv2.circle(out, (int(cx), int(cy)), max(ego_radius + 4, int(round(10 * float(scale_cfg["scale"])))), (255, 255, 255), max(1, font_thickness))
    cv2.putText(out, "EGO", (int(cx) + 2 * ego_radius, int(cy) - ego_radius), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (35, 35, 35), font_thickness, cv2.LINE_AA)

    if gt_history_xy.ndim == 2 and gt_history_xy.shape[0] > 0:
        pts = to_px(gt_history_xy)
        if pts.shape[0] >= 2:
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, (135, 135, 135), int(scale_cfg["history_line_px"]), cv2.LINE_AA)
    if gt_xy.ndim == 2 and gt_xy.shape[0] > 0:
        pts = to_px(gt_xy)
        if pts.shape[0] >= 2:
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, (18, 18, 18), int(scale_cfg["gt_line_px"]), cv2.LINE_AA)
    if ego_history_xy.ndim == 2 and ego_history_xy.shape[0] > 0:
        pts = to_px(ego_history_xy)
        if pts.shape[0] >= 2:
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, (33, 102, 172), int(scale_cfg["gt_line_px"]), cv2.LINE_AA)
        cv2.circle(out, tuple(pts[-1]), point_radius + 1, (33, 102, 172), -1, cv2.LINE_AA)

    display_indices = set(
        select_bev_candidate_indices(
            scores=scores,
            top_k=top_k,
            selected_index=selected_index,
            max_candidates=max_display_candidates,
        )
    )
    style_rows = [
        row
        for row in candidate_visual_styles(scores=scores, top_k=top_k, min_alpha=0.35, max_alpha=0.92)
        if int(row["candidate_index"]) in display_indices
    ]
    for row in style_rows:
        idx = int(row["candidate_index"])
        cand = np.asarray(traj[idx], dtype=np.float32)
        cand_pts = to_px(cand[:, :2])
        color = tuple(int(v) for v in row["color"])
        is_selected = selected_index is not None and int(idx) == int(selected_index)
        alpha = 1.0 if is_selected else float(row["alpha"])
        linewidth = max(traj_line_base, int(round(float(row["linewidth"]) * float(scale_cfg["scale"])))) + (selected_extra_line if is_selected else 0)
        if cand_pts.shape[0] >= 2:
            overlay = out.copy()
            cv2.polylines(overlay, [cand_pts.reshape(-1, 1, 2)], False, color, linewidth, cv2.LINE_AA)
            out[:] = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0)
        if cand_pts.shape[0] > 0:
            cv2.circle(out, tuple(cand_pts[0]), point_radius, color, -1, cv2.LINE_AA)
            cv2.circle(out, tuple(cand_pts[-1]), point_radius + (1 if is_selected else 0), color, -1, cv2.LINE_AA)

    if style_rows:
        legend_w = int(scale_cfg["legend_width_px"])
        legend_x = max(pad, int(width) - legend_w)
        legend_y = pad + int(round(20 * float(scale_cfg["scale"])))
        row_h = int(scale_cfg["legend_row_h_px"])
        title_h = int(round(18 * float(scale_cfg["scale"])))
        box_h = min(int(height) // 3, title_h + row_h * len(style_rows) + pad)
        x0, y0 = legend_x - pad, pad
        x1, y1 = int(width) - pad, pad + box_h
        roi = out[y0:y1, x0:x1].copy()
        panel = roi.copy()
        cv2.rectangle(panel, (0, 0), (max(1, x1 - x0) - 1, max(1, y1 - y0) - 1), (252, 251, 247), -1)
        out[y0:y1, x0:x1] = cv2.addWeighted(panel, 0.72, roi, 0.28, 0.0)
        cv2.rectangle(out, (x0, y0), (x1, y1), (150, 150, 150), max(1, font_thickness), cv2.LINE_AA)
        cv2.putText(out, "shown", (legend_x, pad + title_h - 2), cv2.FONT_HERSHEY_SIMPLEX, small_font_scale, (50, 50, 50), font_thickness, cv2.LINE_AA)
        for row_idx, row in enumerate(style_rows[: max(0, (box_h - 30) // row_h)]):
            y = legend_y + row_idx * row_h
            color = tuple(int(v) for v in row["color"])
            idx = int(row["candidate_index"])
            cv2.line(out, (legend_x, y), (legend_x + int(round(14 * float(scale_cfg["scale"]))), y), color, max(3, traj_line_base), cv2.LINE_AA)
            mark = "*" if selected_index is not None and idx == int(selected_index) else " "
            text = f"{mark}#{idx} r{int(row['rank'])} {format_pdm_score_percent(float(row['score']))}"
            cv2.putText(out, text, (legend_x + int(round(18 * float(scale_cfg["scale"]))), y + max(4, int(round(4 * float(scale_cfg["scale"]))))), cv2.FONT_HERSHEY_SIMPLEX, small_font_scale, (45, 45, 45), font_thickness, cv2.LINE_AA)

    cv2.rectangle(out, (0, 0), (width - 1, height - 1), (120, 120, 120), max(1, font_thickness))
    header = f"{str(detail.get('sample_token', ''))[:14]} shown={len(style_rows)}/{int(traj.shape[0])}"
    cv2.putText(out, header, (pad + 2, int(round(20 * float(scale_cfg["scale"])))), cv2.FONT_HERSHEY_SIMPLEX, header_font_scale, (70, 70, 70), font_thickness, cv2.LINE_AA)
    return out


_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "outputs" / "RewardCheckandVideo" / "HUGSIM" / "sparsedrive-v2"


@dataclass(frozen=True)
class HUGSIMSelection:
    scene_index: int
    official_scene_name: str
    scenario_path: str


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    video: Path
    frames_dir: Path
    bev_dir: Path
    candidate_grid_dir: Path
    scores_dir: Path
    steps_dir: Path
    manifest: Path
    step_csv: Path


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
        if value.ndim == 0:
            return _json_safe(value.detach().cpu().item())
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


def _ensure_output_paths(out_dir: str | Path, *, scene_index: int) -> OutputPaths:
    root = Path(out_dir)
    video = root / f"hugsim_scene_{int(scene_index):03d}_grpo.mp4"
    paths = OutputPaths(
        root=root,
        video=video,
        frames_dir=root / "frames",
        bev_dir=root / "bev",
        candidate_grid_dir=root / "candidate_grid",
        scores_dir=root / "scores",
        steps_dir=root / "steps",
        manifest=root / "run_manifest.json",
        step_csv=root / "step_summary.csv",
    )
    for path in [paths.root, paths.frames_dir, paths.bev_dir, paths.candidate_grid_dir, paths.scores_dir, paths.steps_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _sanitize_path_part(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "hugsim_scene"


def build_timestamped_output_paths(
    *,
    output_root: str | Path,
    official_scene_name: str,
    timestamp: str | None = None,
) -> OutputPaths:
    scene_slug = _sanitize_path_part(str(official_scene_name))
    ts = str(timestamp or datetime.now().strftime("%Y%m%d-%H%M%S"))
    root = Path(output_root) / f"{scene_slug}_{ts}"
    paths = OutputPaths(
        root=root,
        video=root / f"{scene_slug}_grpo.mp4",
        frames_dir=root / "frames",
        bev_dir=root / "bev",
        candidate_grid_dir=root / "candidate_grid",
        scores_dir=root / "scores",
        steps_dir=root / "steps",
        manifest=root / "run_manifest.json",
        step_csv=root / "step_summary.csv",
    )
    for path in [paths.root, paths.frames_dir, paths.bev_dir, paths.candidate_grid_dir, paths.scores_dir, paths.steps_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def select_hugsim_scenario(
    *,
    config: Mapping[str, Any],
    scene: int | None,
    scene_name: str | None,
    scenario_path: str | None = None,
) -> HUGSIMSelection:
    env_cfg = dict(config.get("env", {}) or {})
    hugsim_cfg = dict(env_cfg.get("hugsim", {}) or {})
    scenario_dir = str(
        resolve_hugsim_path(
            hugsim_cfg.get("scenario_dir", None),
            "configs",
            "scenarios",
            "nuscenes",
        )
    )
    scenarios = discover_hugsim_scenarios(scenario_dir)
    if not scenarios:
        raise RuntimeError(f"No HUGSIM scenarios discovered under {scenario_dir}")

    if scenario_path:
        wanted_path = Path(resolve_repo_path(str(scenario_path))).resolve()
        for idx, spec in enumerate(scenarios):
            if Path(spec.scenario_path).resolve() == wanted_path:
                return HUGSIMSelection(
                    scene_index=int(idx),
                    official_scene_name=str(spec.official_scene_name),
                    scenario_path=str(spec.scenario_path),
                )
        try:
            import yaml

            with wanted_path.open("r", encoding="utf-8") as f:
                payload = yaml.safe_load(f) or {}
            official = str(payload.get("scene_name", wanted_path.stem))
        except Exception:
            official = wanted_path.stem
        return HUGSIMSelection(
            scene_index=-1,
            official_scene_name=official,
            scenario_path=str(wanted_path),
        )

    if scene_name:
        wanted = str(scene_name)
        for idx, spec in enumerate(scenarios):
            path = Path(spec.scenario_path)
            if wanted in {spec.official_scene_name, path.stem, path.name, spec.scenario_path}:
                return HUGSIMSelection(
                    scene_index=int(idx),
                    official_scene_name=str(spec.official_scene_name),
                    scenario_path=str(spec.scenario_path),
                )
        raise RuntimeError(f"HUGSIM scene name not found: {wanted}")

    idx = int(scene or 0)
    if idx < 0 or idx >= len(scenarios):
        raise RuntimeError(f"HUGSIM scene index out of range: {idx}; discovered={len(scenarios)}")
    spec = scenarios[idx]
    return HUGSIMSelection(
        scene_index=int(idx),
        official_scene_name=str(spec.official_scene_name),
        scenario_path=str(spec.scenario_path),
    )


def build_run_manifest_payload(
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    ckpt_path: str | Path,
    out_dir: str | Path,
    scene_index: int,
    official_scene_name: str,
    scenario_path: str,
    cuda: int,
    num_candidates: int,
    candidate_select: str,
    mode_select: str,
) -> dict[str, Any]:
    env_cfg = dict(config.get("env", {}) or {})
    agent_cfg = dict(config.get("agent", {}) or {})
    train_cfg = dict(config.get("train", {}) or {})
    grpo_cfg = dict(train_cfg.get("grpo", {}) or {})
    scorer_cfg = dict(agent_cfg.get("nuscenes_scorer", {}) or {})
    return {
        "config_path": str(config_path),
        "ckpt_path": str(ckpt_path),
        "out_dir": str(out_dir),
        "cuda": int(cuda),
        "env_backend": str(env_cfg.get("backend", "recon")),
        "scene_index": int(scene_index),
        "official_scene_name": str(official_scene_name),
        "scenario_path": str(scenario_path),
        "scorer_backend": str(scorer_cfg.get("backend", "nuscenes_pdm")),
        "scorer_config": _json_safe(scorer_cfg),
        "grpo": {
            "num_candidates": int(num_candidates),
            "candidate_select": str(candidate_select),
            "config_num_candidates": int(grpo_cfg.get("num_candidates", num_candidates)),
            "config_candidate_select": str(grpo_cfg.get("candidate_select", candidate_select)),
        },
        "policy": {
            "mode_select": str(mode_select),
            "execute_mode": str(train_cfg.get("policy_execute_mode", "first_step")),
        },
    }


def extract_reward_terms(reward_info: Mapping[str, Any]) -> dict[str, float | bool | str]:
    interesting = (
        "reward",
        "positive",
        "cost",
        "progress",
        "lateral",
        "yaw",
        "collision",
        "off_road",
        "route",
        "front_obstacle",
        "terminal",
        "done_reason",
        "reward_mode",
        "hugsim_base_reward",
        "xz_err",
        "yaw_err",
    )
    out: dict[str, float | bool | str] = {}
    for key, value in reward_info.items():
        key_text = str(key)
        if not any(part in key_text for part in interesting):
            continue
        if isinstance(value, (str, bool, int, float, np.integer, np.floating, np.bool_)):
            out[key_text] = _json_safe(value)
    return out


def _score_summary(scores: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if arr.size <= 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _polygon_heading_length_width(poly_xy: np.ndarray) -> tuple[float, float, float]:
    poly = np.asarray(poly_xy, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[0] < 2 or poly.shape[1] < 2:
        return 0.0, 1.0, 1.0
    if poly.shape[0] >= 4:
        front_mid = 0.5 * (poly[0, :2] + poly[1, :2])
        rear_mid = 0.5 * (poly[2, :2] + poly[3, :2])
        heading = front_mid - rear_mid
        yaw = float(math.atan2(float(heading[1]), float(heading[0]))) if float(np.linalg.norm(heading)) > 1.0e-9 else 0.0
        length = float(max(1.0e-6, np.linalg.norm(front_mid - rear_mid)))
        width = float(max(1.0e-6, np.linalg.norm(poly[0, :2] - poly[1, :2])))
        return yaw, length, width
    delta = poly[1, :2] - poly[0, :2]
    yaw = float(math.atan2(float(delta[1]), float(delta[0]))) if float(np.linalg.norm(delta)) > 1.0e-9 else 0.0
    return yaw, float(max(1.0e-6, np.linalg.norm(delta))), 1.0


def _localize_recon_global_poly(
    poly_xy: Any,
    *,
    ego_xy: np.ndarray,
    world_to_ego_rot: np.ndarray,
) -> np.ndarray | None:
    try:
        poly = np.asarray(poly_xy, dtype=np.float64)
    except Exception:
        return None
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] < 2:
        return None
    rel = poly[:, :2] - np.asarray(ego_xy, dtype=np.float64).reshape(1, 2)
    return (rel @ np.asarray(world_to_ego_rot, dtype=np.float64).T).astype(np.float32)


def _plan_for_hugsim_object(token: str, index: int, scenario_plan_list: Sequence[Any] | None) -> Any | None:
    plans = list(scenario_plan_list or [])
    if index < len(plans):
        return plans[index]
    match = re.search(r"(\d+)$", str(token))
    if match is None:
        return None
    plan_idx = int(match.group(1))
    return plans[plan_idx] if plan_idx < len(plans) else None


def _constant_planner_future_from_object(
    *,
    raw: Mapping[str, Any],
    local_poly: np.ndarray,
    plan: Any,
    ego_xy: np.ndarray,
    world_to_ego_rot: np.ndarray,
    ego_yaw: float,
    future_horizon: int,
    future_dt_s: float,
) -> tuple[list[list[float]], list[float], list[float]] | None:
    plan_items = list(plan) if isinstance(plan, (list, tuple)) else []
    if len(plan_items) < 7 or str(plan_items[6]) != "ConstantPlanner":
        return None
    try:
        speed = float(plan_items[4])
    except Exception:
        speed = float(raw.get("speed_mps", 0.0))
    yaw_world = float(raw.get("yaw_rad", plan_items[3] if len(plan_items) > 3 else 0.0))
    center_local = np.mean(local_poly[:, :2], axis=0).astype(np.float64)
    center_world = np.asarray(ego_xy, dtype=np.float64).reshape(2) + center_local @ np.asarray(world_to_ego_rot, dtype=np.float64)
    a = float(center_world[0])
    b = float(center_world[1])
    dt = max(1.0e-6, float(future_dt_s))
    future_xy_world: list[list[float]] = []
    future_yaw_local: list[float] = []
    future_mask: list[float] = []
    for _ in range(1, max(1, int(future_horizon))):
        a = a - speed * math.sin(yaw_world) * dt
        b = b + speed * math.cos(yaw_world) * dt
        future_xy_world.append([float(a), float(b)])
        future_yaw_local.append(float(math.atan2(math.sin(yaw_world - float(ego_yaw)), math.cos(yaw_world - float(ego_yaw)))))
        future_mask.append(1.0)
    if not future_xy_world:
        return None
    future_local = (np.asarray(future_xy_world, dtype=np.float64) - ego_xy.reshape(1, 2)) @ world_to_ego_rot.T
    return future_local.astype(np.float32).tolist(), future_yaw_local, future_mask


def build_hugsim_grpo_object_context(
    info: Mapping[str, Any],
    *,
    scenario_plan_list: Sequence[Any] | None = None,
    future_horizon: int = 8,
    future_dt_s: float = 0.5,
) -> dict[str, Any]:
    """Build scorer-local object overrides from HUGSIM aligned Recon-global context."""
    ego_poly = info.get("hugsim_ego_box_recon_global_poly", None)
    try:
        ego_poly_arr = np.asarray(ego_poly, dtype=np.float64)
    except Exception:
        ego_poly_arr = np.zeros((0, 2), dtype=np.float64)
    if ego_poly_arr.ndim != 2 or ego_poly_arr.shape[0] < 4 or ego_poly_arr.shape[1] < 2:
        return {
            "available": False,
            "scene_objects": [],
            "ea_agent_states": [],
            "hugsim_object_count": 0,
            "recon_cache_object_count": 0,
        }

    ego_xy = np.mean(ego_poly_arr[:, :2], axis=0)
    front_mid = 0.5 * (ego_poly_arr[0, :2] + ego_poly_arr[1, :2])
    rear_mid = 0.5 * (ego_poly_arr[2, :2] + ego_poly_arr[3, :2])
    heading = front_mid - rear_mid
    ego_yaw = float(math.atan2(float(heading[1]), float(heading[0]))) if float(np.linalg.norm(heading)) > 1.0e-9 else 0.0
    c, s = float(math.cos(-ego_yaw)), float(math.sin(-ego_yaw))
    world_to_ego_rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)

    def convert_items(items: Any, *, default_source: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for idx, raw in enumerate([] if items is None else items):
            if not isinstance(raw, Mapping):
                continue
            local_poly = _localize_recon_global_poly(
                raw.get("poly", raw.get("corners_xy", None)),
                ego_xy=ego_xy,
                world_to_ego_rot=world_to_ego_rot,
            )
            if local_poly is None:
                continue
            yaw, length, width = _polygon_heading_length_width(local_poly)
            velocity_xy = np.asarray(raw.get("velocity_xy", [0.0, 0.0]), dtype=np.float32).reshape(-1)
            if velocity_xy.size < 2:
                velocity_xy = np.zeros((2,), dtype=np.float32)
            velocity_local = (
                velocity_xy[:2].astype(np.float64).reshape(1, 2) @ world_to_ego_rot.T
            ).reshape(2).astype(np.float32)
            source = str(raw.get("source", default_source))
            token = str(raw.get("token", raw.get("id", f"{source}_{idx}")))
            obj = {
                "source": source,
                "token": token,
                "category": str(raw.get("category", raw.get("name", "vehicle.car"))),
                "center_xy": np.mean(local_poly[:, :2], axis=0).astype(np.float32).tolist(),
                "corners_xy": local_poly[:, :2].astype(np.float32).tolist(),
                "length_m": float(raw.get("length_m", length)),
                "width_m": float(raw.get("width_m", width)),
                "yaw_rad": float(raw.get("yaw_rad", yaw)),
                "velocity_xy": velocity_local.tolist(),
                "speed_mps": float(raw.get("speed_mps", np.linalg.norm(velocity_local))),
            }
            has_explicit_future = False
            for future_key in ("future_xy", "future_centers_xy", "future_trajectory_xy", "future_traj_xy"):
                if future_key not in raw:
                    continue
                try:
                    future_world = np.asarray(raw.get(future_key), dtype=np.float64)
                except Exception:
                    future_world = np.zeros((0, 2), dtype=np.float64)
                if future_world.ndim == 2 and future_world.shape[0] > 0 and future_world.shape[1] >= 2:
                    future_local = (future_world[:, :2] - ego_xy.reshape(1, 2)) @ world_to_ego_rot.T
                    obj["future_xy"] = future_local.astype(np.float32).tolist()
                    has_explicit_future = True
                break
            for yaw_key in ("future_yaw", "future_yaw_rad", "future_headings_rad", "future_heading_rad"):
                if yaw_key in raw:
                    future_yaw_world = np.asarray(raw.get(yaw_key), dtype=np.float32).reshape(-1)
                    future_yaw_local = np.arctan2(
                        np.sin(future_yaw_world - float(ego_yaw)),
                        np.cos(future_yaw_world - float(ego_yaw)),
                    ).astype(np.float32)
                    obj["future_yaw"] = future_yaw_local.tolist()
                    break
            for mask_key in ("future_mask", "future_masks", "future_valid", "future_valid_mask"):
                if mask_key in raw:
                    obj["future_mask"] = np.asarray(raw.get(mask_key), dtype=np.float32).reshape(-1).tolist()
                    break
            if "future_dt_s" in raw:
                obj["future_dt_s"] = float(raw.get("future_dt_s", 0.5))
            elif "future_dt" in raw:
                obj["future_dt_s"] = float(raw.get("future_dt", 0.5))
            elif "dt_s" in raw:
                obj["future_dt_s"] = float(raw.get("dt_s", 0.5))
            if (
                not has_explicit_future
                and source == "hugsim_inserted"
                and scenario_plan_list is not None
            ):
                rollout = _constant_planner_future_from_object(
                    raw=raw,
                    local_poly=local_poly,
                    plan=_plan_for_hugsim_object(token, idx, scenario_plan_list),
                    ego_xy=ego_xy,
                    world_to_ego_rot=world_to_ego_rot,
                    ego_yaw=ego_yaw,
                    future_horizon=future_horizon,
                    future_dt_s=future_dt_s,
                )
                if rollout is not None:
                    obj["future_xy"], obj["future_yaw"], obj["future_mask"] = rollout
                    obj["future_dt_s"] = float(future_dt_s)
            out.append(obj)
        return out

    hugsim_objects = convert_items(info.get("hugsim_obj_boxes_recon_global", []), default_source="hugsim_inserted")
    recon_objects = convert_items(info.get("recon_cache_dynamic_objects", []), default_source="recon_cache")
    scene_objects = [*hugsim_objects, *recon_objects]
    return {
        "available": True,
        "scene_objects": scene_objects,
        "ea_agent_states": [dict(item) for item in scene_objects],
        "hugsim_object_count": len(hugsim_objects),
        "recon_cache_object_count": len(recon_objects),
    }


def build_hugsim_step_payload(
    *,
    step: int,
    scene_index: int,
    official_scene_name: str,
    frame_idx: int,
    sample_token: str,
    selected_mode_index: int | None,
    selected_logp: float,
    closed_loop_reward: float,
    closed_loop_reward_sum: float,
    reward_info: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    grpo_object_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = list(candidate_payload.get("candidates", []) or [])
    scores = [float(item.get("score", 0.0)) for item in candidates]
    object_context = dict(grpo_object_context or {})
    return {
        "step": int(step),
        "scene_index": int(scene_index),
        "official_scene_name": str(official_scene_name),
        "frame_idx": int(frame_idx),
        "sample_token": str(sample_token),
        "selected": {
            "mode_index": None if selected_mode_index is None else int(selected_mode_index),
            "logp": float(selected_logp),
        },
        "grpo_score_summary": _score_summary(scores),
        "closed_loop_reward": {
            "step_reward": float(closed_loop_reward),
            "cumulative_reward": float(closed_loop_reward_sum),
        },
        "reward_terms": extract_reward_terms(reward_info),
        "reward_info": _json_safe(reward_info),
        "grpo_object_context": _json_safe(object_context),
        "candidate_scores": _json_safe(candidate_payload),
    }


def render_candidate_grid_image(
    *,
    sample_detail: Mapping[str, Any],
    traj_xyyaw: np.ndarray,
    scores: Sequence[float],
    top_k: int,
    panel_size: int = 320,
    columns: int = 4,
) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("render_candidate_grid_image requires opencv-python") from exc

    traj = np.asarray(traj_xyyaw, dtype=np.float32)
    score_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if traj.ndim != 3 or traj.shape[-1] < 3:
        raise RuntimeError(f"Expected traj_xyyaw shape (candidates,horizon,3), got {tuple(traj.shape)}")
    count = min(int(traj.shape[0]), int(score_arr.shape[0]))
    if count <= 0:
        return np.full((int(panel_size), int(panel_size), 3), 248, dtype=np.uint8)

    cols = max(1, int(columns))
    rows = int(math.ceil(float(count) / float(cols)))
    panel = max(96, int(panel_size))
    canvas = np.full((rows * panel, cols * panel, 3), 245, dtype=np.uint8)
    ranked = np.argsort(-score_arr[:count], kind="stable").astype(np.int64)
    rank_by_idx = {int(idx): int(rank + 1) for rank, idx in enumerate(ranked.tolist())}
    summary = _score_summary(score_arr[:count])

    for idx in range(count):
        row = idx // cols
        col = idx % cols
        detail = dict(sample_detail)
        detail["sample_token"] = f"cand={idx} rank={rank_by_idx.get(idx, -1)} score={format_pdm_score_percent(float(score_arr[idx]))}"
        one_traj = traj[idx : idx + 1]
        one_score = np.asarray([score_arr[idx]], dtype=np.float32)
        tile = render_bev_debug_image(
            sample_detail=detail,
            traj_xyyaw=one_traj,
            scores=one_score,
            top_k=1 if rank_by_idx.get(idx, count + 1) <= int(top_k) else 0,
            width=panel,
            height=panel,
        )
        x0 = col * panel
        y0 = row * panel
        canvas[y0 : y0 + panel, x0 : x0 + panel] = tile
        title = f"#{idx} r{rank_by_idx.get(idx, -1)} s={format_pdm_score_percent(float(score_arr[idx]))}"
        cv2.rectangle(canvas, (x0, y0), (x0 + panel - 1, y0 + 25), (255, 255, 255), -1)
        cv2.putText(canvas, title, (x0 + 7, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)

    footer = (
        f"GRPO mean={format_pdm_score_percent(summary['mean'])} "
        f"std={format_pdm_score_percent(summary['std'])} "
        f"min={format_pdm_score_percent(summary['min'])} "
        f"max={format_pdm_score_percent(summary['max'])}"
    )
    cv2.rectangle(canvas, (0, max(0, canvas.shape[0] - 30)), (canvas.shape[1] - 1, canvas.shape[0] - 1), (255, 255, 255), -1)
    cv2.putText(canvas, footer, (10, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_step_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(str(key))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key, "")) for key in keys})


def _selected_mode_index(replay: Mapping[str, Any], fallback: Any) -> int | None:
    for key in ["mode_idx", "mode_index", "selected_mode_index"]:
        if key in replay:
            try:
                return int(replay[key])
            except Exception:
                pass
    try:
        arr = np.asarray(fallback).reshape(-1)
        if arr.size > 0:
            return int(arr[0])
    except Exception:
        pass
    return None


def _make_hugsim_env(*, config: Mapping[str, Any], selection: HUGSIMSelection, cuda: int):
    from framework.env_wrapper.hugsim_adapter import HUGSIMReconEnv
    from framework.env_wrapper.hugsim_scene_index import HUGSIMSceneIndex

    env_cfg = dict(config.get("env", {}) or {})
    hugsim_cfg = dict(env_cfg.get("hugsim", {}) or {})
    alignment_cfg = dict(hugsim_cfg.get("alignment", {}) or {})
    scene_index = HUGSIMSceneIndex(
        nuscenes_root=hugsim_cfg.get("nuscenes_root", "assets/nuscenes/v1.0-trainval"),
        frame2token_dir=hugsim_cfg.get("frame2token_dir", "assets/nus/information/frame2token"),
    )
    kwargs = {
        "scenario_name": selection.official_scene_name,
        "scenario_path": selection.scenario_path,
        "scene_index": scene_index,
        "reward_cfg": env_cfg.get("reward", {}) or {},
        "output_root": hugsim_cfg.get("output_root", "outputs/hugsim_rl_visualize"),
        "hugsim_repo": resolve_hugsim_path(hugsim_cfg.get("repo", None)) or resolve_hugsim_root(),
        "base_path": resolve_hugsim_path(
            hugsim_cfg.get("base_path", None),
            "configs",
            "sim",
            "nuscenes_eval_sparsedrive_v2_ppo_grpo_ver14.yaml",
        ),
        "camera_path": resolve_hugsim_path(hugsim_cfg.get("camera_path", None), "configs", "sim", "nuscenes_camera.yaml"),
        "kinematic_path": resolve_hugsim_path(hugsim_cfg.get("kinematic_path", None), "configs", "sim", "kinematic.yaml"),
        "substeps_per_rl_step": int(hugsim_cfg.get("substeps_per_rl_step", 2)),
        "recon_data_root": hugsim_cfg.get("recon_data_root", "assets/nus/data"),
        "hugsim_model_base": hugsim_cfg.get("model_base", None),
        "launch_mode": hugsim_cfg.get("launch_mode", "direct"),
        "pixi_cmd": hugsim_cfg.get("pixi_cmd", "pixi"),
        "fifo_timeout_s": float(hugsim_cfg.get("fifo_timeout_s", 300.0)),
        "fifo_poll_interval_s": float(hugsim_cfg.get("fifo_poll_interval_s", 0.2)),
        "cuda": int(cuda),
        "min_gt_route_points": int(hugsim_cfg.get("min_gt_route_points", 2)),
        "alignment_enabled": bool(alignment_cfg.get("enabled", True)),
        "alignment_max_rmse_m": float(alignment_cfg.get("max_rmse_m", 2.0)),
        "use_recon_cache_objects": bool(alignment_cfg.get("use_recon_cache_objects", True)),
        "use_hugsim_inserted_objects": bool(alignment_cfg.get("use_hugsim_inserted_objects", True)),
    }
    if hugsim_cfg.get("fifo_runner_path", None) is not None:
        kwargs["fifo_runner_path"] = hugsim_cfg.get("fifo_runner_path")
    return HUGSIMReconEnv(**kwargs)


def _make_policy(*, config: Mapping[str, Any], ckpt_path: str | Path, cuda: int):
    from framework.agent.sparsedrive_v2 import SparseDriveV2Policy

    agent_cfg = dict(config.get("agent", {}) or {})
    train_cfg = dict(config.get("train", {}) or {})
    scorer_cfg = dict(agent_cfg.get("nuscenes_scorer", {}) or {})
    for key in ("scene_cache_root", "agent_state_cache_root", "ea_project_src", "nuscenes_dataroot", "token2vad_path"):
        if key in scorer_cfg and scorer_cfg[key] is not None:
            scorer_cfg[key] = str(resolve_repo_path(str(scorer_cfg[key])))
    device = f"cuda:{int(cuda)}" if torch.cuda.is_available() else "cpu"
    return SparseDriveV2Policy(
        ckpt_path=str(resolve_repo_path(str(ckpt_path))),
        device=device,
        execute_mode=str(train_cfg.get("policy_execute_mode", "first_step")),
        rl_lr=float(train_cfg.get("policy_lr", 1.0e-5)),
        trainable_prefixes=agent_cfg.get("trainable_prefixes", None),
        frozen_prefixes=agent_cfg.get("frozen_prefixes", None),
        nuscenes_scorer_config=scorer_cfg,
    )


def run_online_hugsim_scene(
    *,
    config_path: str | Path,
    ckpt_path: str | Path,
    out_dir: str | Path | None = None,
    output_root: str | Path = _DEFAULT_OUTPUT_ROOT,
    scene: int | None = None,
    scene_name: str | None = None,
    scenario_path: str | None = None,
    timestamp: str | None = None,
    cuda: int = 0,
    fps: float = 2.0,
    duration_s: float | None = None,
    max_steps: int | None = None,
    num_candidates: int | None = None,
    top_k: int = 5,
    bev_size: int = 1600,
    grid_panel_size: int = 360,
    grid_columns: int = 4,
    candidate_select: str | None = None,
    mode_select: str | None = None,
) -> dict[str, Any]:
    resolved_config = resolve_repo_path(str(config_path))
    config = _load_yaml(resolved_config)
    selection = select_hugsim_scenario(
        config=config,
        scene=scene,
        scene_name=scene_name,
        scenario_path=scenario_path,
    )
    if out_dir is None:
        paths = build_timestamped_output_paths(
            output_root=output_root,
            official_scene_name=selection.official_scene_name,
            timestamp=timestamp,
        )
    else:
        paths = _ensure_output_paths(out_dir, scene_index=selection.scene_index)

    train_cfg = dict(config.get("train", {}) or {})
    grpo_cfg = dict(train_cfg.get("grpo", {}) or {})
    n_candidates = int(num_candidates if num_candidates is not None else grpo_cfg.get("num_candidates", 8))
    cand_select = str(candidate_select if candidate_select is not None else grpo_cfg.get("candidate_select", "topk"))
    policy_mode_select = str(mode_select if mode_select is not None else train_cfg.get("policy_mode_select", "sample"))
    try:
        scenario_payload = _load_yaml(selection.scenario_path)
    except Exception:
        scenario_payload = {}
    scenario_plan_list = list(scenario_payload.get("plan_list", []) or [])

    manifest = build_run_manifest_payload(
        config=config,
        config_path=resolved_config,
        ckpt_path=resolve_repo_path(str(ckpt_path)),
        out_dir=paths.root,
        scene_index=selection.scene_index,
        official_scene_name=selection.official_scene_name,
        scenario_path=selection.scenario_path,
        cuda=int(cuda),
        num_candidates=n_candidates,
        candidate_select=cand_select,
        mode_select=policy_mode_select,
    )
    _write_json(paths.manifest, manifest)

    env = _make_hugsim_env(config=config, selection=selection, cuda=int(cuda))
    policy = _make_policy(config=config, ckpt_path=ckpt_path, cuda=int(cuda))

    obs, reset_info = env.reset()
    current_info = dict(reset_info or {})
    step_limit = max_steps
    if step_limit is None and duration_s is not None:
        step_limit = max(1, int(math.ceil(float(duration_s) * float(fps))))
    if step_limit is None:
        step_limit = int((config.get("env", {}) or {}).get("max_steps", 36))

    writer = imageio.get_writer(
        str(paths.video),
        mode="I",
        fps=float(fps),
        macro_block_size=1,
        codec="libx264",
        ffmpeg_log_level="error",
        output_params=["-pix_fmt", "yuv420p"],
    )

    score_paths: list[str] = []
    step_paths: list[str] = []
    candidate_grid_paths: list[str] = []
    csv_rows: list[dict[str, Any]] = []
    reward_sum = 0.0
    done = False
    try:
        for step in range(int(step_limit)):
            if done:
                break
            obs_in = _obs_for_policy(obs, env)
            action, logp, replay = policy.sample_sparsedrivev2_with_replay(
                obs_in,
                mode_idx=-1,
                mode_select=policy_mode_select,
            )
            grpo_object_context = build_hugsim_grpo_object_context(
                current_info,
                scenario_plan_list=scenario_plan_list,
                future_horizon=8,
                future_dt_s=0.5,
            )
            if bool(grpo_object_context.get("available", False)):
                replay["scene_objects_override"] = list(grpo_object_context.get("scene_objects", []))
                replay["ea_agent_states_override"] = list(grpo_object_context.get("ea_agent_states", []))
                replay["ttc_agent_states_override"] = list(grpo_object_context.get("scene_objects", []))
                replay["grpo_object_context"] = {
                    "source": "hugsim_aligned_recon_global",
                    "hugsim_object_count": int(grpo_object_context.get("hugsim_object_count", 0)),
                    "recon_cache_object_count": int(grpo_object_context.get("recon_cache_object_count", 0)),
                    "scene_object_count": len(grpo_object_context.get("scene_objects", []) or []),
                }
            gt_sample_token = current_info.get("grpo_gt_sample_token", current_info.get("recon_cache_sample_token", None))
            if gt_sample_token is not None and str(gt_sample_token):
                replay["gt_sample_token_override"] = str(gt_sample_token)
            gt_frame_idx = current_info.get("grpo_gt_frame_idx", current_info.get("recon_cache_frame_idx", None))
            if gt_frame_idx is not None:
                try:
                    replay["gt_frame_idx_override"] = int(gt_frame_idx)
                except Exception:
                    pass
            replay_out = _extract_plan_and_scores(
                policy,
                replay,
                num_candidates=n_candidates,
                candidate_select=cand_select,
            )
            traj_xyyaw = np.asarray(replay_out["traj_xyyaw"], dtype=np.float32)
            mode_indices = np.asarray(replay_out["mode_indices"], dtype=np.int64)
            score_logits = np.asarray(replay_out["score_logits"], dtype=np.float32)
            score_tensor = policy.pdm_score_counterfactuals_from_replay_batch(
                [replay],
                torch.as_tensor(traj_xyyaw[None, ...], dtype=torch.float32),
            )
            if torch.is_tensor(score_tensor):
                candidate_scores = np.asarray(score_tensor.detach().cpu().numpy(), dtype=np.float32)[0]
            else:
                candidate_scores = np.asarray(score_tensor, dtype=np.float32)[0]

            sample_detail = _build_bev_sample_detail(policy, replay, traj_xyyaw)
            selected_mode = _selected_mode_index(replay, np.asarray([mode_indices[0] if mode_indices.size else -1]))
            selected_candidate_idx = None
            if selected_mode is not None and mode_indices.size:
                matches = np.where(mode_indices == int(selected_mode))[0]
                if int(matches.shape[0]) > 0:
                    selected_candidate_idx = int(matches[0])
            bev_img = render_bev_debug_image(
                sample_detail=sample_detail,
                traj_xyyaw=traj_xyyaw,
                scores=candidate_scores,
                top_k=int(top_k),
                width=int(bev_size),
                height=int(bev_size),
                selected_index=selected_candidate_idx,
                max_display_candidates=max(4, min(10, int(top_k) + 3)),
            )
            imageio.imwrite(paths.bev_dir / f"step_{step:06d}_bev.png", bev_img)
            candidate_grid_img = render_candidate_grid_image(
                sample_detail=sample_detail,
                traj_xyyaw=traj_xyyaw,
                scores=candidate_scores,
                top_k=int(top_k),
                panel_size=int(grid_panel_size),
                columns=int(grid_columns),
            )
            candidate_grid_path = paths.candidate_grid_dir / f"step_{step:06d}_candidates.png"
            imageio.imwrite(candidate_grid_path, candidate_grid_img)
            candidate_grid_paths.append(str(candidate_grid_path))

            frame_idx = int(current_info.get("now_frame", current_info.get("frame_idx", -1)))
            sample_token = str(replay.get("sample_token", current_info.get("sample_token", "")))
            candidate_payload = build_candidate_score_payload(
                scene=int(selection.scene_index),
                step=int(step),
                frame_idx=int(frame_idx),
                sample_token=sample_token,
                traj_xyyaw=traj_xyyaw,
                scores=candidate_scores,
                score_logits=score_logits,
                mode_indices=mode_indices,
                top_k=int(top_k),
                candidate_details=extract_candidate_score_details(
                    policy,
                    replay,
                    traj_xyyaw,
                    candidate_scores,
                    sample_detail=sample_detail,
                ),
            )
            score_path = write_score_payload(paths.scores_dir, candidate_payload)
            score_paths.append(str(score_path))

            try:
                selected_plan_raw = get_env_plan_xyyaw(replay)
            except RuntimeError:
                selected_plan_raw = np.zeros((0, 3), dtype=np.float32)
            selected_plan = np.asarray(
                selected_plan_raw.detach().cpu().numpy() if torch.is_tensor(selected_plan_raw) else selected_plan_raw,
                dtype=np.float32,
            )
            if selected_plan.ndim == 2 and selected_plan.shape[0] > 0:
                env.set_external_plan_local_xyyaw(selected_plan[:, :3])
            obs_next, reward_v, terminated, truncated, reward_info = env.step(action)
            reward_info = dict(reward_info or {})
            reward_sum += float(reward_v)
            done = bool(terminated or truncated)
            next_frame_idx = int(reward_info.get("now_frame", reward_info.get("frame_idx", frame_idx)))

            logp_float = float(logp.detach().cpu().item()) if torch.is_tensor(logp) else float(logp)
            step_payload = build_hugsim_step_payload(
                step=int(step),
                scene_index=int(selection.scene_index),
                official_scene_name=selection.official_scene_name,
                frame_idx=int(frame_idx),
                sample_token=str(reward_info.get("sample_token", sample_token)),
                selected_mode_index=selected_mode,
                selected_logp=logp_float,
                closed_loop_reward=float(reward_v),
                closed_loop_reward_sum=float(reward_sum),
                reward_info=reward_info,
                candidate_payload=candidate_payload,
                grpo_object_context=grpo_object_context,
            )
            step_path = paths.steps_dir / f"step_{step:06d}.json"
            _write_json(step_path, step_payload)
            step_paths.append(str(step_path))

            summary = step_payload["grpo_score_summary"]
            frame = _grid_frame(obs)
            frame = _overlay_debug_text(
                frame,
                [
                    f"hugsim_scene={selection.scene_index:03d} {selection.official_scene_name}",
                    f"step={step} frame={frame_idx}->{next_frame_idx} reward={float(reward_v):.3f} sum={reward_sum:.3f}",
                    f"grpo mean={summary['mean']:.3f} min={summary['min']:.3f} max={summary['max']:.3f}",
                    f"mode={selected_mode} logp={logp_float:.4f} done={done}",
                ],
            )
            frame = overlay_top_right_inset(frame, bev_img, inset_width=390, margin=12, border_px=2)
            writer.append_data(frame)
            imageio.imwrite(paths.frames_dir / f"step_{step:06d}.png", frame)

            row = {
                "step": int(step),
                "scene_index": int(selection.scene_index),
                "official_scene_name": selection.official_scene_name,
                "frame_idx": int(frame_idx),
                "next_frame_idx": int(next_frame_idx),
                "sample_token": str(reward_info.get("sample_token", sample_token)),
                "reward": float(reward_v),
                "reward_sum": float(reward_sum),
                "grpo_score_mean": float(summary["mean"]),
                "grpo_score_std": float(summary["std"]),
                "grpo_score_min": float(summary["min"]),
                "grpo_score_max": float(summary["max"]),
                "selected_mode_index": "" if selected_mode is None else int(selected_mode),
                "selected_logp": logp_float,
                "done": bool(done),
                "done_reason": reward_info.get("done_reason", ""),
                "reward_mode": reward_info.get("reward_mode", ""),
                "grpo_context_available": bool(grpo_object_context.get("available", False)),
                "grpo_hugsim_object_count": int(grpo_object_context.get("hugsim_object_count", 0)),
                "grpo_recon_cache_object_count": int(grpo_object_context.get("recon_cache_object_count", 0)),
                "grpo_scene_object_count": len(grpo_object_context.get("scene_objects", []) or []),
            }
            for key, value in extract_reward_terms(reward_info).items():
                if key not in row:
                    row[key] = value
            csv_rows.append(row)
            obs = obs_next
            current_info = dict(reward_info)
    finally:
        writer.close()
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    _write_step_csv(paths.step_csv, csv_rows)
    return {
        "video": str(paths.video),
        "manifest": str(paths.manifest),
        "step_csv": str(paths.step_csv),
        "frames_dir": str(paths.frames_dir),
        "bev_dir": str(paths.bev_dir),
        "candidate_grid_dir": str(paths.candidate_grid_dir),
        "scores_dir": str(paths.scores_dir),
        "steps_dir": str(paths.steps_dir),
        "score_json": score_paths,
        "step_json": step_paths,
        "candidate_grid": candidate_grid_paths,
        "steps": len(csv_rows),
        "reward_sum": float(reward_sum),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="SparseDriveV2 HUGSIM GRPO reward visualizer")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output-root", type=str, default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--out", type=str, default=None, help="Exact output run directory. Overrides timestamped output-root layout.")
    parser.add_argument("--scene-index", type=int, default=None, help="Index in HUGSIM scenario_dir discovery order.")
    parser.add_argument("--scene", type=int, default=None, help="Alias for --scene-index.")
    parser.add_argument("--scene-name", type=str, default=None, help="Official scene_name or scenario yaml stem.")
    parser.add_argument("--scenario-path", type=str, default=None, help="Exact HUGSIM scenario yaml path.")
    parser.add_argument("--timestamp", type=str, default=None, help="Override run timestamp for output folder naming.")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bev-size", type=int, default=1600)
    parser.add_argument("--grid-panel-size", type=int, default=360)
    parser.add_argument("--grid-columns", type=int, default=4)
    parser.add_argument("--candidate-select", type=str, default=None, choices=["topk", "all"])
    parser.add_argument("--mode-select", type=str, default=None, choices=["greedy", "sample"])
    args = parser.parse_args()

    result = run_online_hugsim_scene(
        config_path=args.config,
        ckpt_path=args.ckpt,
        out_dir=args.out,
        output_root=args.output_root,
        scene=args.scene_index if args.scene_index is not None else args.scene,
        scene_name=args.scene_name,
        scenario_path=args.scenario_path,
        timestamp=args.timestamp,
        cuda=int(args.cuda),
        fps=float(args.fps),
        duration_s=args.duration_s,
        max_steps=args.max_steps,
        num_candidates=args.num_candidates,
        top_k=int(args.top_k),
        bev_size=int(args.bev_size),
        grid_panel_size=int(args.grid_panel_size),
        grid_columns=int(args.grid_columns),
        candidate_select=args.candidate_select,
        mode_select=args.mode_select,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
