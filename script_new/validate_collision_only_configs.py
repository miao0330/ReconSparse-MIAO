from pathlib import Path

from framework.runner.config_loader import load_composed_yaml


base = Path(__file__).resolve().parents[1] / "script/configs/sparsedrive_v2/experiments"
for name in (
    "full_nuscenes_88_collision_only_baseline.yaml",
    "full_nuscenes_88_collision_only_static_grpo.yaml",
    "full_nuscenes_88_collision_only_policy_loop_grpo.yaml",
):
    cfg = load_composed_yaml(base / name)
    scenes = cfg["env"]["hugsim"]["scenes"]
    reward = cfg["env"]["reward"]["step_path"]
    assert len(scenes) == 88, (name, len(scenes))
    assert reward["path"]["w_progress"] == 0.0
    assert reward["path"]["w_lateral"] == 0.0
    assert reward["path"]["w_yaw"] == 0.0
    assert reward["comfort"]["w_longitudinal_jerk"] == 0.0
    assert reward["comfort"]["w_yaw_jerk"] == 0.0
    assert reward["collision"]["mode"] == "dense_penalty"
    assert reward["collision"]["w_static"] == 500.0
    assert reward["collision"]["w_dynamic"] == 500.0
    assert reward["safety"]["enable"] is False
    if "policy_loop_grpo" in cfg.get("train", {}):
        branch = cfg["train"]["policy_loop_grpo"]["reward"]["step_path"]
        assert branch["path"]["w_progress"] == 0.0
        assert branch["path"]["w_lateral"] == 0.0
        assert branch["path"]["w_yaw"] == 0.0
        assert branch["comfort"]["w_longitudinal_jerk"] == 0.0
        assert branch["comfort"]["w_yaw_jerk"] == 0.0
        assert branch["collision"]["w_static"] == 500.0
        assert branch["collision"]["w_dynamic"] == 500.0
        assert branch["safety"]["enable"] is False
    print(f"validated {name}: scenes=88 collision_only")
