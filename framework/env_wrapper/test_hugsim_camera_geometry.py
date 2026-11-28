import numpy as np
import pytest

from framework.env_wrapper.hugsim_adapter import HUGSIM_CAMERA_MAP, build_hugsim_training_obs
from framework.env_wrapper.hugsim_scene_index import HUGSIMFrameMapping


def test_rendered_training_observation_uses_published_camera_poses():
    intrinsic = {"H": 6, "W": 8, "fovx": 1.1, "fovy": 0.7, "cx": 4, "cy": 3}
    rgb = {name: np.zeros((6, 8, 3), dtype=np.uint8) for name in HUGSIM_CAMERA_MAP}
    poses = {}
    for name, (_, idx) in HUGSIM_CAMERA_MAP.items():
        pose = np.array([[0, 0, 1, 0], [-1, 0, 0, idx * 0.5], [0, -1, 0, 1.3], [0, 0, 0, 1]])
        poses[name] = pose
    transform = np.eye(4)
    transform[0, 3] = 1.7
    info = {
        "cam_params": {name: {"intrinsic": intrinsic, "v2c": np.eye(4)} for name in rgb},
        "camera_to_ground_flu": poses,
        "ground_to_model_flu": transform,
    }
    inputs = {
        "hugsim_obs": {"rgb": rgb},
        "hugsim_info": info,
        "mapping": HUGSIMFrameMapping("scene", 0, "token", 0, 0, 0.0, 0.0),
        "hugsim_step_idx": 0,
    }
    obs = build_hugsim_training_obs(**inputs)
    for name, (_, idx) in HUGSIM_CAMERA_MAP.items():
        np.testing.assert_allclose(obs["cam2ego"][idx], poses[name])
    obs_rendered = build_hugsim_training_obs(**inputs, camera_geometry="rendered")
    for name, (_, idx) in HUGSIM_CAMERA_MAP.items():
        np.testing.assert_allclose(obs_rendered["cam2ego"][idx], transform @ poses[name])
    rgb["CAM_FRONT"] = np.zeros((7, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="RGB size or intrinsics are invalid for CAM_FRONT"):
        build_hugsim_training_obs(**inputs)
    del poses["CAM_FRONT"]
    with pytest.raises(ValueError, match="camera_to_ground_flu for CAM_FRONT"):
        build_hugsim_training_obs(**inputs)


def test_rl_and_eval_convert_the_same_vehicle_trajectory():
    from framework.agent.sparsedrive_v2.hugsim_bridge import trajectory_to_hugsim_plan
    from framework.env_wrapper.hugsim_adapter import HUGSIMReconEnv

    transform = np.eye(4)
    transform[0, 3] = 1.7
    plan = np.array([[1.0, 0.0, 0.0], [3.0, 0.5, 0.2]])
    env = HUGSIMReconEnv.__new__(HUGSIMReconEnv)
    env._last_hugsim_info = {"ground_to_model_flu": transform}
    env._external_plan_local_xyyaw = plan.copy()
    env._fifo_plan_lateral_convention = "right_forward"
    expected_ground = trajectory_to_hugsim_plan(plan, None)
    np.testing.assert_allclose(env._plan_from_action(None), expected_ground, atol=1e-6)
    assert env._external_plan_local_xyyaw is None
    np.testing.assert_allclose(env._plan_from_action(tuple(plan[1])), expected_ground[1:], atol=1e-6)

    env.camera_geometry = "rendered"
    env._external_plan_local_xyyaw = plan.copy()
    expected_rendered = trajectory_to_hugsim_plan(plan, transform)
    np.testing.assert_allclose(env._plan_from_action(None), expected_rendered, atol=1e-6)
    np.testing.assert_allclose(env._plan_from_action(tuple(plan[1])), expected_rendered[1:], atol=1e-6)


def test_ground_model_status_converts_native_steering_privately():
    from framework.env_wrapper.hugsim_adapter import _ego_status

    info = {"ego_velo": 4.0, "accelerate": 0.2, "ego_steer": 0.1, "wheelbase": 2.7}
    _, velocity, acceleration = _ego_status(info)
    np.testing.assert_allclose(velocity, [4.0, 0.0])
    np.testing.assert_allclose(acceleration, [0.2, -16.0 * np.tan(0.1) / 2.7])
    assert info["ego_steer"] == 0.1


def test_fifo_lateral_convention_matches_official_hugsim_repo():
    from pathlib import Path
    from framework.agent.sparsedrive_v2.hugsim_bridge import trajectory_to_hugsim_plan
    from framework.env_wrapper.hugsim_adapter import (
        _canonical_local_xy_to_hugsim_fifo_plan,
        _infer_hugsim_fifo_plan_lateral_convention,
    )

    hugsim_repo = Path(__file__).resolve().parents[3] / "HUGSIM-ORI"
    convention = _infer_hugsim_fifo_plan_lateral_convention(hugsim_repo)
    assert convention == "right_forward"
    left_turn_flu = np.array([[2.0, 0.5, 0.1], [4.0, 1.2, 0.25]], dtype=np.float32)
    rl_fifo = _canonical_local_xy_to_hugsim_fifo_plan(left_turn_flu[:, :2], lateral_convention=convention)
    eval_fifo = trajectory_to_hugsim_plan(left_turn_flu, None)
    np.testing.assert_allclose(rl_fifo, eval_fifo, atol=1e-6)
    assert rl_fifo[0, 0] < 0.0 and rl_fifo[0, 1] > 0.0
