"""HUGSIM FIFO evaluation bridge for the ReconDreamer SparseDriveV2 policy.

The bridge is evaluation-only. RL rollout code continues to use
``SparseDriveV2Policy.act()`` from the normal actor path.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from framework.env_wrapper.hugsim_adapter import _hugsim_geometry, build_hugsim_training_obs
from framework.env_wrapper.hugsim_scene_index import HUGSIMFrameMapping
from framework.utils.repo_paths import resolve_ego_ads_subdir

try:
    from .adapter import SparseDriveV2Policy
except ImportError:  # Direct execution from the repository launcher.
    from framework.agent.sparsedrive_v2.adapter import SparseDriveV2Policy


DEFAULT_CHECKPOINT = (
    Path(resolve_ego_ads_subdir("SparseDriveV2")) / "ckpt" / "sparsedrive_navsimv2.ckpt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ReconDreamer SparseDriveV2 on HUGSIM.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("SPARSEDRIVE_CHECKPOINT", str(DEFAULT_CHECKPOINT)),
        type=Path,
    )
    parser.add_argument(
        "--mode-select",
        default=os.environ.get("SPARSEDRIVE_MODE_SELECT", "greedy"),
        choices=("greedy", "argmax", "max", "sample"),
    )
    parser.add_argument("--mode-idx", type=int, default=-1)
    parser.add_argument(
        "--image-preprocess",
        default=os.environ.get("SPARSEDRIVE_IMAGE_PREPROCESS", "legacy_hf"),
        choices=("native", "legacy_hf"),
    )
    parser.add_argument(
        "--camera-geometry",
        default=os.environ.get("SPARSEDRIVE_CAMERA_GEOMETRY", "rendered"),
        choices=("rendered", "navsim_lidar", "vehicle"),
    )
    return parser.parse_args()


def trajectory_to_hugsim_plan(trajectory: np.ndarray, ground_to_model_flu=None) -> np.ndarray:
    """Convert model output with the exact transform used by the HF bridge."""
    points = np.asarray(trajectory, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] <= 0 or points.shape[1] < 2:
        raise ValueError(f"Expected trajectory shape (N, >=2), got {points.shape}")
    if ground_to_model_flu is not None:
        points = _hugsim_geometry().vehicle_trajectory_to_ground(points, ground_to_model_flu)
    plan = points[:, [1, 0]].copy()
    plan[:, 0] *= -1.0
    if not np.isfinite(plan).all():
        raise ValueError("SparseDriveV2 produced a non-finite HUGSIM plan")
    return plan


def _mapping_for_raw_payload(
    raw_data: Any,
    *,
    scene_name: str,
    frame_idx: int,
) -> HUGSIMFrameMapping:
    _obs, info = raw_data
    timestamp = float(info.get("timestamp", frame_idx))
    sample_token = str(info.get("sample_token", info.get("token", f"{scene_name}:{frame_idx}")))
    scene_id = int(info.get("scene_id", info.get("scene", -1)))
    return HUGSIMFrameMapping(
        official_scene_name=str(scene_name),
        recon_scene_id=scene_id,
        sample_token=sample_token,
        frame_idx=int(frame_idx),
        sample_index=int(frame_idx),
        sample_relative_time_s=timestamp,
        hugsim_relative_time_s=timestamp,
    )


def raw_hugsim_to_policy_observation(
    raw_data: Any,
    *,
    scene_name: str,
    frame_idx: int,
    camera_geometry: str = "rendered",
) -> dict[str, Any]:
    """Reuse the canonical HUGSIM -> SparseDrive observation conversion."""
    if not isinstance(raw_data, tuple) or len(raw_data) != 2:
        raise TypeError("HUGSIM bridge expects a raw (observation, info) tuple")
    hugsim_obs, hugsim_info = raw_data
    mapping = _mapping_for_raw_payload(
        raw_data,
        scene_name=scene_name,
        frame_idx=int(frame_idx),
    )
    return build_hugsim_training_obs(
        hugsim_obs=hugsim_obs,
        hugsim_info=hugsim_info,
        mapping=mapping,
        hugsim_step_idx=int(frame_idx),
        camera_geometry=str(camera_geometry),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _save_diagnostic_frame(
    output_dir: Path,
    *,
    frame_idx: int,
    raw_data: Any,
    trajectory: np.ndarray,
    plan: np.ndarray,
) -> None:
    diagnostics_dir = output_dir / "sparsedrive_v2_bridge"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        diagnostics_dir / f"frame_{int(frame_idx):06d}.npz",
        trajectory=np.asarray(trajectory, dtype=np.float32),
        plan=np.asarray(plan, dtype=np.float32),
    )
    _obs, info = raw_data
    (diagnostics_dir / f"frame_{int(frame_idx):06d}.json").write_text(
        json.dumps(_json_safe(info), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_policy(
    checkpoint: Path,
    device: torch.device,
    *,
    image_preprocess: str,
) -> SparseDriveV2Policy:
    policy = SparseDriveV2Policy(
        ckpt_path=str(checkpoint),
        device=str(device),
        image_preprocess=str(image_preprocess),
    )
    policy._model.eval()
    return policy


def run_fifo(
    *,
    output_dir: Path,
    checkpoint: Path,
    mode_idx: int = -1,
    mode_select: str = "greedy",
    image_preprocess: str = "legacy_hf",
    camera_geometry: str = "rendered",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    obs_pipe = output_dir / "obs_pipe"
    plan_pipe = output_dir / "plan_pipe"
    if not obs_pipe.exists():
        os.mkfifo(obs_pipe)
    if not plan_pipe.exists():
        os.mkfifo(plan_pipe)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = _build_policy(
        Path(checkpoint),
        device,
        image_preprocess=str(image_preprocess),
    )
    scene_name = os.environ.get("HUGSIM_SCENE_NAME", "hugsim")

    frame_idx = 0
    while True:
        with obs_pipe.open("rb") as pipe:
            raw_data = pickle.loads(pipe.read())

        if isinstance(raw_data, str) and raw_data == "Done":
            break

        plan: np.ndarray | None = None
        try:
            observation = raw_hugsim_to_policy_observation(
                raw_data,
                scene_name=scene_name,
                frame_idx=frame_idx,
                camera_geometry=str(camera_geometry),
            )
            trajectory = policy.predict_trajectory(
                observation,
                mode_idx=int(mode_idx),
                mode_select=str(mode_select),
                image_preprocess=str(image_preprocess),
            )
            transform = raw_data[1].get("ground_to_model_flu") if camera_geometry == "rendered" else None
            plan = trajectory_to_hugsim_plan(trajectory, transform)
            _save_diagnostic_frame(
                output_dir,
                frame_idx=frame_idx,
                raw_data=raw_data,
                trajectory=trajectory,
                plan=plan,
            )
        except Exception as exc:  # pragma: no cover - runtime safety net
            print(f"[SparseDriveV2 bridge] inference failed: {exc}", flush=True)
            traceback.print_exc()

        with plan_pipe.open("wb") as pipe:
            pipe.write(pickle.dumps(plan))
        if plan is None:
            break
        frame_idx += 1


def main() -> None:
    args = parse_args()
    run_fifo(
        output_dir=Path(args.output),
        checkpoint=Path(args.checkpoint),
        mode_idx=int(args.mode_idx),
        mode_select=str(args.mode_select),
        image_preprocess=str(args.image_preprocess),
        camera_geometry=str(args.camera_geometry),
    )


if __name__ == "__main__":
    main()
