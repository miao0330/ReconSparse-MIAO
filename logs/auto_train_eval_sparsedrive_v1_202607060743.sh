#!/usr/bin/env bash
set -uo pipefail

cd /root/clone/ReconDreamer-RL

export PYTHONPATH=/root/clone/ReconDreamer-RL:${PYTHONPATH:-}
export TORCH_EXTENSIONS_DIR=/root/clone/ReconDreamer-RL/.cache/torch_extensions
export HUGSIM_ROOT=/root/clone/HUGSIM-ORI
export HUGSIM_RANDOM_SEED=${HUGSIM_RANDOM_SEED:-0}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

CONFIG=/root/clone/ReconDreamer-RL/script/configs/sparsedrive/202607060743_HUGSM_collision_only_peak1e4_decay5e5_update32.yaml
RUN_TAG=sparsedrive_v1_202607060743_hugsim_collision_update32_20260706_$(date -u +%H%M%S)

echo "[$(date -u +%F_%T)] run_tag=${RUN_TAG}"
echo "[$(date -u +%F_%T)] config=${CONFIG}"
echo "[$(date -u +%F_%T)] stage=train_start"

/root/miniconda3/envs/recondreamerNew-rl/bin/python -u script/train_actor_learner_v2.py --config "${CONFIG}"
train_rc=$?
echo "[$(date -u +%F_%T)] stage=train_exit rc=${train_rc}"
if [[ "${train_rc}" != "0" ]]; then
  exit "${train_rc}"
fi

LATEST_DIR=$(CONFIG="${CONFIG}" /root/miniconda3/envs/recondreamerNew-rl/bin/python -c 'import os,yaml; from pathlib import Path; cfg=yaml.safe_load(Path(os.environ["CONFIG"]).read_text(encoding="utf-8")); base=Path(cfg["train"]["actor_learner"]["buffer_dir"]); candidates=[p for p in base.parent.glob("*_"+base.name) if p.is_dir()]; candidates += ([base] if base.is_dir() else []); assert candidates, f"no candidates for {base}"; print(max(candidates, key=lambda p:p.stat().st_mtime))')
CKPT="${LATEST_DIR}/weights/latest.ckpt"
if [[ ! -f "${CKPT}" ]]; then
  echo "[$(date -u +%F_%T)] missing checkpoint: ${CKPT}" >&2
  exit 1
fi

echo "[$(date -u +%F_%T)] latest_dir=${LATEST_DIR}"
echo "[$(date -u +%F_%T)] eval_ckpt=${CKPT}"
echo "[$(date -u +%F_%T)] stage=eval_start"

/root/miniconda3/envs/recondreamerNew-rl/bin/python -u script/evaluate_existing_sparsedrive_v1_ckpts.py \
  --ckpts "${CKPT}" \
  --sparsedrive-config /root/clone/ReconDreamer-RL/egoADs/SparseDrive/projects/configs/sparsedrive_small_stage2.py \
  --hugsim-template /root/clone/HUGSIM-ORI/configs/sim/nuscenes_eval_sparsedrive_v2_ppo_grpo_ver14.yaml \
  --scenario-dir /root/clone/HUGSIM-ORI/configs/scenarios/nuscenes \
  --eval-output-root /root/clone/HUGSIM-ORI/outputs/evaluate-auto \
  --run-name "${RUN_TAG}_eval" \
  --repeat-evals 2 \
  --slots 0:0 1:1 2:2 3:3 4:4 5:5 6:6 7:7 \
  --max-scenes 88
eval_rc=$?
echo "[$(date -u +%F_%T)] stage=eval_exit rc=${eval_rc}"
exit "${eval_rc}"
