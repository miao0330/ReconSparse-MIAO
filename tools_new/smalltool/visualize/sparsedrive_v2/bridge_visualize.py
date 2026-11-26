"""Small dependency-light visualizer for HUGSIM SparseDriveV2 bridge runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


CAMERA_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


def _as_rgb(image: Any) -> np.ndarray:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"Expected an HWC RGB image, got {array.shape}")
    return array[..., :3]


def build_camera_grid(observation: dict[str, Any], *, tile_size: tuple[int, int] = (320, 180)) -> np.ndarray:
    """Build a stable six-camera grid from either raw or nested RGB observations."""
    rgb = observation.get("rgb", observation)
    width, height = int(tile_size[0]), int(tile_size[1])
    tiles: list[np.ndarray] = []
    for camera_name in CAMERA_ORDER:
        image = rgb.get(camera_name) if isinstance(rgb, dict) else None
        if image is None:
            tile = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            tile = cv2.resize(_as_rgb(image), (width, height), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, camera_name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(tile, camera_name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(tile)
    return np.concatenate(
        [np.concatenate(tiles[:3], axis=1), np.concatenate(tiles[3:], axis=1)],
        axis=0,
    )


def _as_plan_xy(plan_points: Any) -> np.ndarray:
    if plan_points is None:
        return np.empty((0, 2), dtype=np.float32)
    plan = np.asarray(plan_points, dtype=np.float32)
    if plan.ndim != 2 or plan.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float32)
    return plan[:, :2]


def build_bev_image(
    plan_points: Any,
    *,
    obj_boxes: list[Any] | None = None,
    size: int = 720,
    meters_per_pixel: float = 8.0,
) -> np.ndarray:
    canvas = np.full((int(size), int(size), 3), 245, dtype=np.uint8)
    center = np.asarray([size // 2, size - 80], dtype=np.float32)
    scale = float(size) / max(1.0, float(meters_per_pixel))

    cv2.arrowedLine(canvas, tuple(center.astype(int)), (int(center[0]), int(center[1] - scale)), (80, 80, 80), 2, cv2.LINE_AA)
    cv2.putText(canvas, "ego forward", (int(center[0] + 8), int(center[1] - scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(center.astype(int)), 10, (30, 30, 220), -1, cv2.LINE_AA)

    plan = _as_plan_xy(plan_points)
    if len(plan) > 0:
        points = np.stack(
            [center[0] + plan[:, 0] * scale, center[1] - plan[:, 1] * scale],
            axis=1,
        )
        points = np.rint(points).astype(np.int32)
        valid = (
            (points[:, 0] >= 0)
            & (points[:, 0] < size)
            & (points[:, 1] >= 0)
            & (points[:, 1] < size)
        )
        visible = points[valid]
        if len(visible) > 1:
            cv2.polylines(canvas, [visible], isClosed=False, color=(0, 140, 255), thickness=4, lineType=cv2.LINE_AA)
        for point in visible:
            cv2.circle(canvas, tuple(point), 4, (0, 140, 255), -1, cv2.LINE_AA)

    for box in obj_boxes or []:
        values = np.asarray(box, dtype=np.float32).reshape(-1)
        if values.shape[0] < 5:
            continue
        x, y = float(values[0]), float(values[1])
        width, length = float(values[3]), float(values[4])
        corners = np.asarray(
            [[x - width / 2, y - length / 2], [x + width / 2, y - length / 2],
             [x + width / 2, y + length / 2], [x - width / 2, y + length / 2]],
            dtype=np.float32,
        )
        pixels = np.rint(np.stack([center[0] + corners[:, 0] * scale, center[1] - corners[:, 1] * scale], axis=1)).astype(np.int32)
        cv2.polylines(canvas, [pixels], isClosed=True, color=(80, 80, 220), thickness=2, lineType=cv2.LINE_AA)

    return canvas


def save_hugsim_frame(
    prefix: str | Path,
    obs: dict[str, Any],
    info: dict[str, Any],
    plan_points: Any,
    motion_history: dict[str, Any] | None = None,
) -> dict[str, Path]:
    del motion_history
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    camera = build_camera_grid(obs)
    bev = build_bev_image(plan_points, obj_boxes=info.get("obj_boxes", []))
    camera_resized = cv2.resize(camera, (bev.shape[1], camera.shape[0]), interpolation=cv2.INTER_LINEAR)
    combined = np.concatenate([camera_resized, bev], axis=0)
    paths = {
        "camera": prefix.parent / f"{prefix.name}_cam.jpg",
        "bev": prefix.parent / f"{prefix.name}_bev.jpg",
        "combine": prefix.parent / f"{prefix.name}_combine.jpg",
    }
    cv2.imwrite(str(paths["camera"]), camera)
    cv2.imwrite(str(paths["bev"]), bev)
    cv2.imwrite(str(paths["combine"]), combined)
    return paths


save_visualization_frame = save_hugsim_frame


__all__ = ["build_camera_grid", "build_bev_image", "save_hugsim_frame", "save_visualization_frame"]
