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
    info = {
        "cam_params": {name: {"intrinsic": intrinsic, "v2c": np.eye(4)} for name in rgb},
        "camera_to_ground_flu": poses,
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
    expected = trajectory_to_hugsim_plan(plan, transform)
    np.testing.assert_allclose(env._plan_from_action(None), expected, atol=1e-6)
    assert env._external_plan_local_xyyaw is None
    np.testing.assert_allclose(env._plan_from_action(tuple(plan[1])), expected[1:], atol=1e-6)
