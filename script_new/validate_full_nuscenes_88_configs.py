from pathlib import Path

from framework.runner.config_loader import load_composed_yaml


base = Path(__file__).resolve().parents[1] / "script/configs/sparsedrive_v2/experiments"
expected = {
    "baseline_collision500": (False, False),
    "static_grpo": (True, False),
    "policy_loop_grpo": (False, True),
}
for name, (grpo_enabled, policy_enabled) in expected.items():
    path = base / f"full_nuscenes_88_{name}.yaml"
    config = load_composed_yaml(path)
    scenes = config["env"]["hugsim"]["scenes"]
    grpo = bool(config.get("train", {}).get("grpo", {}).get("enable", False))
    policy = bool(config.get("train", {}).get("policy_loop_grpo", {}).get("enable", False))
    collision = config["env"]["reward"]["step_path"]["collision"]
    assert len(scenes) == 88, (name, len(scenes))
    assert (grpo, policy) == (grpo_enabled, policy_enabled), (name, grpo, policy)
    assert collision["w_static"] == 500.0 and collision["w_dynamic"] == 500.0, (name, collision)
    print(f"validated {name}: scenes={len(scenes)} grpo={grpo} policy_loop={policy} collision500")
