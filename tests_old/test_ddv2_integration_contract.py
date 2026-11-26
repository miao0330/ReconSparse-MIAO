from __future__ import annotations

import torch

from framework.agent.diffusiondrive_v2.rl_adapter import (
    DiffusionDriveV2RLAdapterMixin,
    _recompute_pose_yaw_from_xy,
)
from framework.replay_schema import (
    get_env_plan_xyyaw,
    get_policy_action_id,
    get_policy_model_inputs,
)
from script.train_eval_pipeline import (
    ckpt_dir_for_agent_type,
    eval_tool_for_agent_type,
    resolve_agent_type,
)


def test_ddv2_replay_uses_canonical_schema() -> None:
    trajectory = torch.tensor(
        [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    replay = DiffusionDriveV2RLAdapterMixin._make_ddv2_replay(
        camera_feature=torch.zeros((1, 3, 256, 1024)),
        status_feature=torch.zeros((1, 8)),
        diffusion_chain=torch.zeros((1, 80, 8, 2, 3)),
        mode_idx=7,
        traj_xyyaw_raw=trajectory,
        traj_xyyaw=_recompute_pose_yaw_from_xy(trajectory),
        exec_debug={"execute_mode": "first_step"},
        execute_mode="first_step",
    )

    assert replay["schema_version"] == 2
    assert get_policy_action_id(replay)["mode_idx"] == 7
    assert get_policy_model_inputs(replay)["status_feature"].shape == (1, 8)
    assert get_env_plan_xyyaw(replay).shape == (3, 3)


def test_ddv2_yaw_is_consistent_for_single_and_batch_sampling() -> None:
    trajectory = torch.tensor(
        [[1.0, 0.0, 9.0], [1.0, 1.0, 9.0], [2.0, 2.0, 9.0]],
        dtype=torch.float32,
    )
    result = _recompute_pose_yaw_from_xy(trajectory)
    torch.testing.assert_close(
        result[:, 2],
        torch.tensor([0.0, torch.pi / 2, torch.pi / 4]),
    )


def test_train_eval_pipeline_dispatches_ddv2() -> None:
    assert resolve_agent_type({"agent": {"type": "ddv2"}}) == "diffusiondrive_v2"
    assert ckpt_dir_for_agent_type("diffusiondrive_v2").name == "ckpt"
    assert eval_tool_for_agent_type("diffusiondrive_v2").name == (
        "evaluate_existing_diffusiondrive_v2_ckpts.py"
    )
