#!/usr/bin/env python3
"""SparseDrive-v1 HUGSIM detection/BEV/combine visualizer.

This online visualizer runs a HUGSIM scene with SparseDrive-v1 and renders the
model's own prediction outputs:

- camera prediction with projected 3D detection boxes
- BEV prediction with detection boxes and track temporal anchor boxes
- combined camera + BEV frames
- mp4 video from the combined frames

It intentionally does not read or draw GT boxes.

Example:
cd /root/clone/ReconDreamer-RL
python tools/smalltool/visualize/sparsedrive_v1/hugsim_det_bev_combine.py \
  --config script/configs/sparsedrive/20260630_template_HUGSIM_sparsedrive_v1.yaml \
  --ckpt egoADs/SparseDrive/ckpt/sparsedrive_stage2.pth \
  --hugsim-scene scene-0166-easy-00 \
  --horizon 32 \
  --fps 2 \
  --cuda 0
"""

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import types

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework.utils.repo_paths import resolve_ego_ads_subdir, resolve_repo_path
from tools.smalltool.visualize.sparsedrive_v1 import detailed as v1_detailed
from tools.smalltool.visualize.sparsedrive_v2 import detailed as v2_detailed

_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "outputs" / "RewardCheckandVideo" / "HUGSIM" / "sparsedrive-v1-det-bev"
_SPARSEDRIVE_ROOT = Path(resolve_ego_ads_subdir("SparseDrive"))
_DEFAULT_SPARSEDRIVE_CONFIG = _SPARSEDRIVE_ROOT / "projects" / "configs" / "sparsedrive_small_stage2.py"

_HUGSIM_CAMERA_TO_INDEX = {
    "front": 0,
    "front_left": 1,
    "front_right": 2,
    "back_left": 3,
    "back": 4,
    "back_right": 5,
}
_SPARSEDRIVE_ORDER = ("front", "front_right", "front_left", "back", "back_left", "back_right")
_SPARSEDRIVE_CAM_NAMES = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
_HUGSIM_MARKER_KEYS = ("hugsim_step_idx", "hugsim_timestamp", "official_scene_name")


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    images: Path
    cam: Path
    bev: Path
    combine: Path
    video: Path
    manifest: Path


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
        if value.ndim == 0:
            return _json_safe(value.item())
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _resolve_output_paths(
    *,
    output_root: str | Path,
    out_dir: str | Path | None,
    label: str,
    timestamp: str,
) -> OutputPaths:
    safe_label = str(label or "hugsim").strip().replace(os.sep, "_").replace(" ", "_")
    root = Path(out_dir) if out_dir is not None else Path(output_root) / f"{safe_label}-{timestamp}"
    artifacts = root / "artifacts"
    stem = f"{safe_label}_{timestamp}_det_bev_combine"
    return OutputPaths(
        root=root,
        images=artifacts / "hugsim_images",
        cam=artifacts / "cam_pred",
        bev=artifacts / "bev_pred",
        combine=artifacts / "combine",
        video=artifacts / f"{stem}.mp4",
        manifest=root / "run_info.md",
    )


def _ensure_dirs(paths: OutputPaths) -> None:
    for path in [paths.root, paths.images, paths.cam, paths.bev, paths.combine]:
        path.mkdir(parents=True, exist_ok=True)


def _camera_indices() -> list[int]:
    return [_HUGSIM_CAMERA_TO_INDEX[name] for name in _SPARSEDRIVE_ORDER]


def _is_hugsim_observation(observation: Mapping[str, Any]) -> bool:
    return any(key in observation for key in _HUGSIM_MARKER_KEYS)


def _as_rgb_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image with 3 channels, got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _scaled_intrinsic(
    intrinsic: np.ndarray,
    *,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> np.ndarray:
    K = np.asarray(intrinsic, dtype=np.float32)
    if K.shape == (4, 4):
        K = K[:3, :3]
    if K.shape != (3, 3):
        raise ValueError(f"Expected camera intrinsic shape (3,3) or (4,4), got {K.shape}")
    out = K.copy()
    src_h, src_w = float(source_hw[0]), float(source_hw[1])
    dst_h, dst_w = float(target_hw[0]), float(target_hw[1])
    out[0, 0] *= dst_w / max(1.0, src_w)
    out[0, 2] *= dst_w / max(1.0, src_w)
    out[1, 1] *= dst_h / max(1.0, src_h)
    out[1, 2] *= dst_h / max(1.0, src_h)
    return out.astype(np.float32)


def _ordered_cam2lidar(observation: Mapping[str, Any]) -> np.ndarray:
    cam2ego = np.asarray(observation["cam2ego"], dtype=np.float32)
    cam2ego = cam2ego[_camera_indices()]
    if _is_hugsim_observation(observation):
        from framework.agent.sparsedrive_v1.adapter import _hugsim_cam2ego_to_sparsedrive_v1_cam2lidar

        return _hugsim_cam2ego_to_sparsedrive_v1_cam2lidar(cam2ego)
    return cam2ego.astype(np.float32)


def _orthonormalized_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _s, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    out = u @ vt
    if float(np.linalg.det(out)) < 0.0:
        u[:, -1] *= -1.0
        out = u @ vt
    return out


def _cam2lidar_to_sparse_lidar2cam_row(cam2lidar: np.ndarray) -> np.ndarray:
    """Convert standard cam->lidar matrices to SparseDrive renderer layout.

    SparseDrive's original visualizer expects a row-vector lidar2cam matrix:
    points_cam = points_lidar @ M[:3, :3] + M[3, :3].
    """
    c2l = np.asarray(cam2lidar, dtype=np.float64)
    if c2l.ndim != 3 or c2l.shape[-2:] != (4, 4):
        raise ValueError(f"Expected cam2lidar shape (N,4,4), got {c2l.shape}")
    out = np.repeat(np.eye(4, dtype=np.float64)[None], int(c2l.shape[0]), axis=0)
    for idx in range(int(c2l.shape[0])):
        mat = c2l[idx]
        if np.allclose(mat[3, :], np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64), atol=1e-4):
            cam_to_lidar_rot = _orthonormalized_rotation(mat[:3, :3])
            cam_in_lidar = np.asarray(mat[:3, 3], dtype=np.float64)
            lidar_to_cam_row_rot = cam_to_lidar_rot
            lidar_to_cam_row_trans = -(cam_in_lidar @ lidar_to_cam_row_rot)
            out[idx, :3, :3] = lidar_to_cam_row_rot
            out[idx, 3, :3] = lidar_to_cam_row_trans
            continue

        # Some simulator matrices are already row-vector affine transforms.
        # In that case the normal matrix inverse preserves the expected row
        # layout: rotation in [:3, :3], translation in [3, :3].
        row_inv = np.linalg.inv(mat.astype(np.float64))
        row_inv[:3, :3] = _orthonormalized_rotation(row_inv[:3, :3])
        out[idx] = row_inv
    return out


def _prepare_hugsim_render_data(
    observation: Mapping[str, Any],
    *,
    image_dir: str | Path,
    step: int,
    render_size: tuple[int, int] = (1600, 900),
) -> dict[str, Any]:
    """Build SparseDrive's original renderer data from a HUGSIM observation.

    Returned keys deliberately contain prediction-rendering inputs only; GT
    fields are not added here.
    """
    image_root = Path(image_dir)
    image_root.mkdir(parents=True, exist_ok=True)
    render_w, render_h = int(render_size[0]), int(render_size[1])
    target_hw = (render_h, render_w)

    cam_intrinsics = np.asarray(observation["cam_intrinsics"], dtype=np.float32)[_camera_indices()]
    cam_hw = observation.get("cam_hw", None)
    if cam_hw is not None:
        cam_hw_arr = np.asarray(cam_hw, dtype=np.float32)[_camera_indices()]
    else:
        cam_hw_arr = None
    lidar2cam = _cam2lidar_to_sparse_lidar2cam_row(_ordered_cam2lidar(observation))

    filenames: list[str] = []
    scaled_K: list[np.ndarray] = []
    for cam_name, obs_key, K0, idx in zip(_SPARSEDRIVE_CAM_NAMES, _SPARSEDRIVE_ORDER, cam_intrinsics, range(6)):
        img = _as_rgb_uint8(observation[obs_key])
        if cam_hw_arr is not None:
            source_hw = (int(cam_hw_arr[idx, 0]), int(cam_hw_arr[idx, 1]))
        else:
            source_hw = (int(img.shape[0]), int(img.shape[1]))
        scaled_K.append(_scaled_intrinsic(K0, source_hw=source_hw, target_hw=target_hw))

        if img.shape[1] != render_w or img.shape[0] != render_h:
            img = cv2.resize(img, (render_w, render_h), interpolation=cv2.INTER_LINEAR)
        path = image_root / f"step_{int(step):06d}_{cam_name}.jpg"
        imageio.imwrite(path, img)
        filenames.append(str(path))

    return {
        "cam_intrinsic": np.stack(scaled_K, axis=0).astype(np.float32),
        "img_filename": filenames,
        "lidar2cam": lidar2cam,
    }


def _unwrap_model(policy: Any) -> Any:
    model = getattr(policy, "_model", None)
    if model is None:
        raise RuntimeError("SparseDrive-v1 policy missing _model")
    return getattr(model, "module", model)


def _forward_sparse_prediction(policy: Any, observation: Mapping[str, Any]) -> dict[str, Any]:
    img_t, data = policy._build_batch(dict(observation))
    model_device = getattr(policy, "device", torch.device("cpu"))
    img_t = img_t.to(model_device)
    data_dev = policy._to_model_metas(data, model_device)
    model = _unwrap_model(policy)
    model.eval()

    with torch.inference_mode():
        feature_maps = model.extract_feat(img_t)
        model_outs = model.head(feature_maps, data_dev)
        post_results = model.head.post_process(model_outs, data_dev)

    if not post_results:
        raise RuntimeError("SparseDrive-v1 post_process returned no prediction results")
    result = dict(post_results[0])
    _det_out, _map_out, _motion_out, planning_out = model_outs
    if planning_out is None:
        raise RuntimeError("SparseDrive-v1 planning output is None")

    cls_raw = planning_out["classification"][-1]
    reg_raw = planning_out["prediction"][-1]
    if cls_raw.ndim == 2:
        cls_raw = cls_raw.unsqueeze(0)
    if reg_raw.ndim == 4:
        reg_raw = reg_raw.unsqueeze(0)
    bs = int(cls_raw.shape[0])
    logits_all = cls_raw.reshape(bs, 3, int(policy._ego_fut_mode))[0].detach().cpu().to(dtype=torch.float32)
    traj_all = (
        reg_raw.reshape(bs, 3, int(policy._ego_fut_mode), int(policy._ego_fut_ts), 2)[0]
        .cumsum(dim=-2)
        .detach()
        .cpu()
        .to(dtype=torch.float32)
    )

    cmd = data["gt_ego_fut_cmd"].detach().cpu().reshape(-1)
    cmd_idx = int(torch.argmax(cmd[:3]).item()) if int(cmd.numel()) >= 3 else 0
    logits = logits_all[int(cmd_idx)].reshape(-1).numpy().astype(np.float32)
    traj_lidar_xy = traj_all[int(cmd_idx)].numpy().astype(np.float32)
    traj_xyyaw = v1_detailed._v1_trajs_to_xyyaw(policy, traj_lidar_xy)
    candidates = {
        "cmd_idx": int(cmd_idx),
        "logits": logits,
        "probs": v1_detailed._softmax_np(logits),
        "traj_lidar_xy": traj_lidar_xy,
        "traj_xyyaw": traj_xyyaw,
        "mode_indices": np.arange(int(traj_xyyaw.shape[0]), dtype=np.int64),
    }
    return {
        "result": result,
        "model_data": data,
        "candidates": candidates,
    }


def _merge_prediction_results(
    *,
    det_result: Mapping[str, Any],
    motion_result: Mapping[str, Any],
    planning_result: Mapping[str, Any],
    final_planning: np.ndarray | torch.Tensor,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    result.update(dict(det_result))
    result.update(dict(motion_result))
    result.update(dict(planning_result))
    if torch.is_tensor(final_planning):
        result["final_planning"] = final_planning.detach().cpu().numpy().astype(np.float32)
    else:
        result["final_planning"] = np.asarray(final_planning, dtype=np.float32)
    return result


def _sparsedrive_lidar_xy_to_debug_bev_xy(points_xy: Any) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    out = np.zeros_like(pts[..., :2], dtype=np.float32)
    out[..., 0] = pts[..., 1]
    out[..., 1] = -pts[..., 0]
    return out


def _box_bev_corners_lidar(box: np.ndarray) -> np.ndarray:
    arr = np.asarray(box, dtype=np.float32).reshape(-1)
    if int(arr.shape[0]) < 7:
        return np.zeros((0, 2), dtype=np.float32)
    x, y = float(arr[0]), float(arr[1])
    w, l = float(abs(arr[3])), float(abs(arr[4]))
    yaw = float(arr[6])
    local = np.asarray(
        [
            [l * 0.5, w * 0.5],
            [l * 0.5, -w * 0.5],
            [-l * 0.5, -w * 0.5],
            [-l * 0.5, w * 0.5],
        ],
        dtype=np.float32,
    )
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot.T + np.asarray([x, y], dtype=np.float32)


def _build_scorer_replay_from_observation(
    replay: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    sample_token = observation.get("sample_token", observation.get("token", replay.get("sample_token", None)))
    out: dict[str, Any] = {}
    if sample_token is not None:
        out["sample_token"] = str(sample_token)
    for source_key, target_key in (
        ("grpo_gt_sample_token", "gt_sample_token_override"),
        ("recon_cache_sample_token", "gt_sample_token_override"),
        ("grpo_gt_frame_idx", "gt_frame_idx_override"),
        ("recon_cache_frame_idx", "gt_frame_idx_override"),
    ):
        if target_key in out or source_key not in observation:
            continue
        value = observation.get(source_key)
        if value is None:
            continue
        try:
            out[target_key] = int(value) if "frame" in target_key else str(value)
        except Exception:
            out[target_key] = value
    for key in ("scene_id", "frame_idx", "now_frame"):
        if key in observation:
            try:
                out[key] = int(observation[key])
            except Exception:
                pass

    try:
        object_context = v2_detailed.build_hugsim_grpo_object_context(observation)
    except Exception:
        object_context = {}
    if bool(object_context.get("available", False)):
        out["scene_objects_override"] = list(object_context.get("scene_objects", []) or [])
        out["ea_agent_states_override"] = list(object_context.get("ea_agent_states", []) or [])
        out["ttc_agent_states_override"] = list(object_context.get("scene_objects", []) or [])
    return out


def _build_context_bev_sample_detail(
    *,
    policy: Any,
    replay: Mapping[str, Any],
    observation: Mapping[str, Any],
    traj_xyyaw: np.ndarray,
) -> dict[str, Any]:
    scorer_replay = _build_scorer_replay_from_observation(replay, observation)
    if "sample_token" in scorer_replay:
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
                if not callable(score_with_details):
                    continue
                try:
                    _scores, details = score_with_details(
                        [scorer_replay],
                        torch.as_tensor(np.asarray(traj_xyyaw, dtype=np.float32)[None, ...], dtype=torch.float32),
                    )
                except Exception:
                    continue
                if len(details) > 0:
                    detail = dict(details[0])
                    detail["render_layers"] = detail.get("render_layers", {}) or {}
                    return detail
    return {
        "sample_token": str(scorer_replay.get("sample_token", "")),
        "gt_xy": np.zeros((0, 2), dtype=np.float32),
        "gt_history_xy": np.zeros((0, 2), dtype=np.float32),
        "ego_history_xy": np.zeros((0, 2), dtype=np.float32),
        "scene_objects": [],
        "render_layers": {},
        "map_patch_radius_m": 25.0,
    }


def _draw_sparsedrive_pred_boxes_on_context_bev(
    image: np.ndarray,
    *,
    result: Mapping[str, Any],
    det_score_thresh: float,
    patch_radius: float,
) -> None:
    boxes = result.get("boxes_3d", None)
    scores = result.get("scores_3d", None)
    anchors = result.get("anchor_queue", None)
    periods = result.get("period", None)
    if boxes is None or scores is None:
        return
    boxes_np = _to_numpy_float32(boxes)
    scores_np = _to_numpy_float32(scores).reshape(-1)
    anchors_np = _to_numpy_float32(anchors) if anchors is not None else None
    periods_np = np.asarray(periods.detach().cpu().numpy() if torch.is_tensor(periods) else periods).reshape(-1) if periods is not None else None
    if boxes_np.ndim != 2 or int(scores_np.shape[0]) == 0:
        return

    def to_px(points_xy: np.ndarray) -> np.ndarray:
        return v2_detailed._bev_points_to_px(
            np.asarray(points_xy, dtype=np.float32),
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            patch_radius=float(patch_radius),
        )

    for idx in range(min(int(boxes_np.shape[0]), int(scores_np.shape[0]))):
        if float(scores_np[idx]) < float(det_score_thresh):
            continue
        corners = _sparsedrive_lidar_xy_to_debug_bev_xy(_box_bev_corners_lidar(boxes_np[idx]))
        if corners.shape[0] >= 3:
            pts = to_px(corners).reshape(-1, 1, 2)
            cv2.polylines(image, [pts], True, (40, 180, 40), 2, cv2.LINE_AA)
        if anchors_np is None or periods_np is None or anchors_np.ndim < 3:
            continue
        period = max(0, min(int(periods_np[idx]), int(anchors_np.shape[1])))
        for j in range(period):
            anchor_corners = _sparsedrive_lidar_xy_to_debug_bev_xy(_box_bev_corners_lidar(anchors_np[idx, -1 - j]))
            if anchor_corners.shape[0] >= 3:
                pts = to_px(anchor_corners).reshape(-1, 1, 2)
                cv2.polylines(image, [pts], True, (35, 120, 230), 1, cv2.LINE_AA)


def _to_numpy_float32(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def _load_module_from_file(module_name: str, path: str | Path) -> Any:
    module_path = Path(path)
    existing = sys.modules.get(module_name)
    if existing is not None and Path(getattr(existing, "__file__", "") or "") == module_path:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_sparse_visualization_renderers(*, sparse_root: str | Path = _SPARSEDRIVE_ROOT) -> tuple[type, type]:
    sparse_root_path = Path(sparse_root)
    vis_dir = sparse_root_path / "tools" / "visualization"
    if str(sparse_root_path) not in sys.path:
        sys.path.insert(0, str(sparse_root_path))

    vis_pkg = sys.modules.get("tools.visualization")
    if vis_pkg is None:
        vis_pkg = types.ModuleType("tools.visualization")
        vis_pkg.__path__ = [str(vis_dir)]
        sys.modules["tools.visualization"] = vis_pkg
    else:
        vis_pkg.__path__ = [str(vis_dir)]

    tools_pkg = sys.modules.get("tools")
    if tools_pkg is not None:
        setattr(tools_pkg, "visualization", vis_pkg)

    bev_mod = _load_module_from_file("tools.visualization.bev_render", vis_dir / "bev_render.py")
    cam_mod = _load_module_from_file("tools.visualization.cam_render", vis_dir / "cam_render.py")
    return bev_mod.BEVRender, cam_mod.CamRender


def _apply_sparse_visualization_thresholds(*, det_score_thresh: float, map_score_thresh: float) -> None:
    for module_name in ("tools.visualization.bev_render", "tools.visualization.cam_render"):
        mod = sys.modules.get(module_name)
        if mod is None:
            continue
        if hasattr(mod, "SCORE_THRESH"):
            setattr(mod, "SCORE_THRESH", float(det_score_thresh))
        if hasattr(mod, "MAP_SCORE_THRESH"):
            setattr(mod, "MAP_SCORE_THRESH", float(map_score_thresh))


@contextlib.contextmanager
def _temporary_cwd(path: str | Path):
    old = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(old)


def _render_prediction_images(
    *,
    policy: Any,
    replay: Mapping[str, Any],
    observation: Mapping[str, Any],
    selected_traj_xyyaw: np.ndarray,
    render_data: dict[str, Any],
    result: dict[str, Any],
    out_dir: str | Path,
    step: int,
    det_score_thresh: float,
    map_score_thresh: float,
    draw_map: bool,
) -> tuple[Path, Path, Path]:
    BEVRender, CamRender = _load_sparse_visualization_renderers()
    _apply_sparse_visualization_thresholds(
        det_score_thresh=float(det_score_thresh),
        map_score_thresh=float(map_score_thresh),
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_choices = {
        "draw_pred": True,
        "det": True,
        "track": True,
        "motion": True,
        "map": bool(draw_map),
        "planning": True,
    }
    with _temporary_cwd(_SPARSEDRIVE_ROOT):
        cam_render = CamRender(plot_choices, str(out))
        cam_path = Path(cam_render.render(render_data, result, int(step)))

        bev_path = out / "bev_pred" / f"{int(step):04d}.jpg"
        try:
            traj_one = np.asarray(selected_traj_xyyaw, dtype=np.float32)[None, :, :3]
            sample_detail = _build_context_bev_sample_detail(
                policy=policy,
                replay=replay,
                observation=observation,
                traj_xyyaw=traj_one,
            )
            bev_img = v2_detailed.render_bev_debug_image(
                sample_detail=sample_detail,
                traj_xyyaw=traj_one,
                scores=np.asarray([1.0], dtype=np.float32),
                top_k=1,
                width=900,
                height=900,
                selected_index=0,
                max_display_candidates=1,
            )
            _draw_sparsedrive_pred_boxes_on_context_bev(
                bev_img,
                result=result,
                det_score_thresh=float(det_score_thresh),
                patch_radius=float(sample_detail.get("map_patch_radius_m", 25.0)),
            )
            imageio.imwrite(bev_path, bev_img)
        except Exception:
            bev_render = BEVRender(plot_choices, str(out))
            bev_render.reset_canvas()
            bev_render.draw_detection_pred(result)
            bev_render.draw_track_pred(result)
            bev_render.draw_motion_pred(result)
            bev_render.draw_planning_pred(render_data, result)
            bev_render._render_sdc_car()
            bev_render._render_command(render_data)
            bev_render._render_legend()
            bev_render.save_fig(str(bev_path))

    combine_path = out / "combine" / f"{int(step):04d}.jpg"
    combine_path.parent.mkdir(parents=True, exist_ok=True)
    _write_combined_image(cam_path=cam_path, bev_path=bev_path, save_path=combine_path)
    return cam_path, bev_path, combine_path


def _write_combined_image(*, cam_path: str | Path, bev_path: str | Path, save_path: str | Path) -> np.ndarray:
    cam_image = cv2.imread(str(cam_path), cv2.IMREAD_COLOR)
    bev_image = cv2.imread(str(bev_path), cv2.IMREAD_COLOR)
    if cam_image is None:
        raise RuntimeError(f"Failed to read rendered camera image: {cam_path}")
    if bev_image is None:
        raise RuntimeError(f"Failed to read rendered BEV image: {bev_path}")
    cam_h = int(cam_image.shape[0])
    bev_w = max(1, int(round(float(bev_image.shape[1]) * cam_h / max(1.0, float(bev_image.shape[0])))))
    bev_resized = cv2.resize(bev_image, (bev_w, cam_h), interpolation=cv2.INTER_AREA)
    merged = cv2.hconcat([cam_image, bev_resized])
    pad_h = int(merged.shape[0] % 2)
    pad_w = int(merged.shape[1] % 2)
    if pad_h or pad_w:
        merged = cv2.copyMakeBorder(
            merged,
            0,
            pad_h,
            0,
            pad_w,
            borderType=cv2.BORDER_REPLICATE,
        )
    cv2.imwrite(str(save_path), merged)
    return merged


def _write_manifest(
    *,
    paths: OutputPaths,
    config_path: str,
    ckpt_path: str,
    sparsedrive_config_path: str,
    label: str,
    steps: int,
    reward_sum: float,
) -> None:
    lines = [
        "# SparseDrive-v1 HUGSIM Det/BEV/Combine Visualization\n",
        f"- run_dir: `{paths.root}`\n",
        f"- label: {label}\n",
        f"- config: `{config_path}`\n",
        f"- ckpt: `{ckpt_path}`\n",
        f"- sparsedrive_config: `{sparsedrive_config_path}`\n",
        f"- steps: {int(steps)}\n",
        f"- reward_sum: {float(reward_sum):.6f}\n",
        f"- video: `{paths.video}`\n",
        f"- cam_pred_dir: `{paths.cam}`\n",
        f"- bev_pred_dir: `{paths.bev}`\n",
        f"- combine_dir: `{paths.combine}`\n",
        "- gt_boxes: disabled\n",
        "- prediction_sources: SparseDrive-v1 model head post_process outputs\n",
    ]
    paths.manifest.write_text("".join(lines), encoding="utf-8")


def run_online_hugsim_scene(
    *,
    config_path: str | Path,
    ckpt_path: str | Path,
    sparsedrive_config_path: str | Path = _DEFAULT_SPARSEDRIVE_CONFIG,
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
    worker_id: int | None = None,
    render_width: int = 1600,
    render_height: int = 900,
    det_score_thresh: float = 0.3,
    map_score_thresh: float = 0.3,
    draw_map: bool = True,
) -> dict[str, Any]:
    from framework.runner.agent_factory import build_agent
    from framework.runner.env_factory import build_actor_env

    config_resolved = str(resolve_repo_path(str(config_path)))
    ckpt_resolved = str(resolve_repo_path(str(ckpt_path)))
    sd_config_resolved = str(resolve_repo_path(str(sparsedrive_config_path)))
    cfg = v1_detailed._load_yaml(config_resolved)
    v1_detailed._apply_agent_overrides(cfg, ckpt_path=ckpt_resolved, sparsedrive_config_path=sd_config_resolved)
    selected_label = v1_detailed._apply_scenario_path_override(cfg, scenario_path) or None
    selected_label = (
        v1_detailed._apply_hugsim_scene_override(
            cfg,
            argparse.Namespace(hugsim_scene=hugsim_scene, scene_name=None),
        )
        or selected_label
    )
    if selected_label is None:
        selected_label = v1_detailed._apply_scene_index_override(cfg, scene_index)
    label = selected_label or v1_detailed._first_hugsim_label(
        cfg,
        fallback=f"hugsim_scene_{scene_index}" if scene_index is not None else "hugsim_ori",
    )
    ts = timestamp or time.strftime("%Y%m%d-%H%M%S")
    paths = _resolve_output_paths(output_root=output_root, out_dir=out_dir, label=label, timestamp=ts)
    _ensure_dirs(paths)

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
    obs, _reset_info = env.reset()
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
    reward_sum = 0.0
    done = False
    step = 0
    try:
        while (max_steps is None or step < max_steps) and not done:
            obs_for_policy = v1_detailed._obs_for_policy(obs, sim)
            bundle = _forward_sparse_prediction(agent, obs_for_policy)
            candidates = bundle["candidates"]
            action, logp, replay = v1_detailed._select_v1_action_from_candidates(
                candidates,
                mode_idx=int(mode_idx),
                mode_select=str(mode_select),
            )
            selected_mode_idx = int(replay.get("mode_idx", -1))
            if selected_mode_idx < 0 or selected_mode_idx >= int(candidates["traj_lidar_xy"].shape[0]):
                selected_mode_idx = int(np.argmax(np.asarray(candidates["logits"], dtype=np.float32)))

            result = dict(bundle["result"])
            result["final_planning"] = np.asarray(candidates["traj_lidar_xy"][selected_mode_idx], dtype=np.float32)
            result.setdefault("instance_ids", torch.arange(int(result.get("boxes_3d", torch.empty((0,))).shape[0])))

            render_data = _prepare_hugsim_render_data(
                obs_for_policy,
                image_dir=paths.images,
                step=int(step),
                render_size=(int(render_width), int(render_height)),
            )
            render_data["gt_ego_fut_cmd"] = np.asarray(bundle["model_data"]["gt_ego_fut_cmd"][0].detach().cpu().numpy(), dtype=np.float32)

            _cam_path, _bev_path, combine_path = _render_prediction_images(
                policy=agent,
                replay=replay,
                observation=obs_for_policy,
                selected_traj_xyyaw=_to_numpy_float32(replay["traj_xyyaw"]),
                render_data=render_data,
                result=result,
                out_dir=paths.root / "artifacts",
                step=int(step),
                det_score_thresh=float(det_score_thresh),
                map_score_thresh=float(map_score_thresh),
                draw_map=bool(draw_map),
            )
            combined_bgr = cv2.imread(str(combine_path), cv2.IMREAD_COLOR)
            if combined_bgr is None:
                raise RuntimeError(f"Failed to read combined frame: {combine_path}")
            writer.append_data(cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB))

            selected_plan = _to_numpy_float32(replay["traj_xyyaw"])
            setter = getattr(env, "set_external_plan_local_xyyaw", None)
            if callable(setter):
                setter(selected_plan)
            else:
                setattr(sim, "_external_plan_local_xyyaw", selected_plan.astype(np.float64).copy())

            obs, reward_v, terminated, truncated, _info = env.step(action)
            del logp
            done = bool(terminated or truncated)
            reward_sum += float(reward_v)
            step += 1
    finally:
        writer.close()
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    _write_manifest(
        paths=paths,
        config_path=config_resolved,
        ckpt_path=ckpt_resolved,
        sparsedrive_config_path=sd_config_resolved,
        label=str(label),
        steps=int(step),
        reward_sum=float(reward_sum),
    )
    return {
        "run_dir": str(paths.root),
        "video": str(paths.video),
        "cam_pred_dir": str(paths.cam),
        "bev_pred_dir": str(paths.bev),
        "combine_dir": str(paths.combine),
        "steps": int(step),
        "reward_sum": float(reward_sum),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SparseDrive-v1 HUGSIM det/BEV/combine visualizer")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--sparsedrive-config", type=str, default=str(_DEFAULT_SPARSEDRIVE_CONFIG))
    parser.add_argument("--output-root", type=str, default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--out-dir", "--out", dest="out_dir", type=str, default=None)
    parser.add_argument("--hugsim-scene", type=str, default=None)
    parser.add_argument("--scene-name", type=str, default=None)
    parser.add_argument("--scenario-path", type=str, default=None)
    parser.add_argument("--scene-index", type=int, default=None)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--horizon", "--max-steps", dest="horizon", type=int, default=32)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--mode-idx", type=int, default=-1)
    parser.add_argument("--mode-select", type=str, default="greedy", choices=["greedy", "sample"])
    parser.add_argument("--worker-id", type=int, default=None)
    parser.add_argument("--render-width", type=int, default=1600)
    parser.add_argument("--render-height", type=int, default=900)
    parser.add_argument("--det-score-thresh", type=float, default=0.3)
    parser.add_argument("--map-score-thresh", type=float, default=0.3)
    parser.add_argument("--draw-map", dest="draw_map", action="store_true", default=True)
    parser.add_argument("--no-draw-map", dest="draw_map", action="store_false")
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
        worker_id=args.worker_id,
        render_width=int(args.render_width),
        render_height=int(args.render_height),
        det_score_thresh=float(args.det_score_thresh),
        map_score_thresh=float(args.map_score_thresh),
        draw_map=bool(args.draw_map),
    )
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
