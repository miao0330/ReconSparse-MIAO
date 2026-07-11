#!/usr/bin/env python3
"""SparseDrive-v1 HUGSIM rollout visualizer.

Writes the rollout mp4 and lightweight artifacts without saving per-step
candidate trajectory screenshots.

Command:
cd /root/clone/ReconDreamer-RL
python tools/smalltool/visualize/sparsedrive_v1/visualize.py \
  --config /root/clone/ReconDreamer-RL/script/configs/sparsedrive/20260630_template_HUGSIM_sparsedrive_v1.yaml \
  --ckpt /root/clone/ReconDreamer-RL/egoADs/SparseDrive/ckpt/sparsedrive_stage2.pth \
  --sparsedrive-config /root/clone/ReconDreamer-RL/egoADs/SparseDrive/projects/configs/sparsedrive_small_stage2.py \
  --hugsim-scene scene-0166-easy-00 \
  --horizon 32 \
  --fps 2 \
  --cuda 0 \
  --mode-select greedy

This is the non-detailed entrypoint paired with `detailed.py`. It keeps a
smaller BEV inset and simpler defaults, while reusing the v1 candidate
extraction and rollout implementation from the detailed script.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_detailed_module() -> Any:
    module_path = Path(__file__).resolve().with_name("detailed.py")
    spec = importlib.util.spec_from_file_location("sparsedrive_v1_detailed", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load detailed SparseDrive-v1 visualizer: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(str(spec.name), module)
    spec.loader.exec_module(module)
    return module


def _build_arg_parser() -> argparse.ArgumentParser:
    detailed = _load_detailed_module()
    parser = argparse.ArgumentParser(description="SparseDrive-v1 HUGSIM rollout visualizer")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--sparsedrive-config", type=str, default=str(detailed._DEFAULT_SPARSEDRIVE_CONFIG))
    parser.add_argument("--output-root", type=str, default=str(detailed._DEFAULT_OUTPUT_ROOT))
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
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--bev-size", type=int, default=420)
    parser.add_argument("--grid-panel-size", type=int, default=260)
    parser.add_argument("--grid-columns", type=int, default=3)
    parser.add_argument("--worker-id", type=int, default=None)
    parser.add_argument("--save-shard", dest="save_shard", action="store_true", default=True)
    parser.add_argument("--no-save-shard", dest="save_shard", action="store_false")
    return parser


def run_online_hugsim_scene(**kwargs: Any) -> dict[str, Any]:
    detailed = _load_detailed_module()
    kwargs.setdefault("detailed_outputs", False)
    return detailed.run_online_hugsim_scene(**kwargs)


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
        detailed_outputs=False,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
