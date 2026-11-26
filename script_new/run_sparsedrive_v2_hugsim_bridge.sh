#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
BRIDGE_PATH="$REPO_ROOT/framework/agent/sparsedrive_v2/hugsim_bridge.py"
PYTHON_BIN="${SPARSEDRIVE_PYTHON_BIN:-/root/miniconda3/envs/recondreamerNew-rl/bin/python}"
CHECKPOINT="${SPARSEDRIVE_CHECKPOINT:-$REPO_ROOT/egoADs/SparseDriveV2/ckpt/sparsedrive_navsimv2.ckpt}"
NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-/root/clone/nuplan-devkit}"

if [ "$#" -lt 2 ]; then
  echo "usage: $0 GPU_ID OUTPUT_DIR" >&2
  exit 2
fi

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/egoADs/SparseDriveV2:$NUPLAN_DEVKIT_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES="$1" "$PYTHON_BIN" "$BRIDGE_PATH" \
  --output "$2" \
  --checkpoint "$CHECKPOINT" \
  --mode-select "${SPARSEDRIVE_MODE_SELECT:-greedy}" \
  --mode-idx "${SPARSEDRIVE_MODE_IDX:--1}"
