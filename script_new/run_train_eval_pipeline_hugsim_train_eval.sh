#!/usr/bin/env bash

# cd /root/clone/ReconDreamer-RL

# bash script/run_train_eval_pipeline_hugsim_train_eval.sh \
#   --reinforcepp-config /root/clone/ReconDreamer-RL/script/configs/sparsedrive/20260708_0915_HUGSM_collision_only_peak1e4_decay5e5_update32.yaml \
#   --reinforcepp-algo-tag hugsim_sparsedrive_v1_collision_only_reinforcepp_nogrpo_noaux_peak1e4_decay5e5_update32 \
#   --slots 0:0 1:1 2:2 3:3 4:4 5:5 6:6 7:7 \
#   --max-scenes 88 \
#   --repeat-evals 2



set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_REINFORCEPP_CONFIG="$REPO_ROOT/script/configs/sparsedrive/20260708_0915_HUGSM_collision_only_peak1e4_decay5e5_update32.yaml"

DEFAULT_EVAL_SEED=1
HAS_EXPLICIT_REINFORCEPP_CONFIG=0
HAS_EXPLICIT_REINFORCEPP_ALGO_TAG=0
HAS_EXPLICIT_SLOTS=0
HAS_EXPLICIT_MAX_SCENES=0
HAS_EXPLICIT_REPEAT_EVALS=0
for arg in "$@"; do
  if [[ "$arg" == "--no-default-eval-seed" ]]; then
    DEFAULT_EVAL_SEED=0
  elif [[ "$arg" == "--reinforcepp-config" || "$arg" == --reinforcepp-config=* ]]; then
    HAS_EXPLICIT_REINFORCEPP_CONFIG=1
  elif [[ "$arg" == "--reinforcepp-algo-tag" || "$arg" == --reinforcepp-algo-tag=* ]]; then
    HAS_EXPLICIT_REINFORCEPP_ALGO_TAG=1
  elif [[ "$arg" == "--slots" || "$arg" == --slots=* ]]; then
    HAS_EXPLICIT_SLOTS=1
  elif [[ "$arg" == "--max-scenes" || "$arg" == --max-scenes=* ]]; then
    HAS_EXPLICIT_MAX_SCENES=1
  elif [[ "$arg" == "--repeat-evals" || "$arg" == --repeat-evals=* ]]; then
    HAS_EXPLICIT_REPEAT_EVALS=1
  fi
done

if [[ "$DEFAULT_EVAL_SEED" == "1" ]]; then
  export HUGSIM_RANDOM_SEED="${HUGSIM_RANDOM_SEED:-0}"
  export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
  unset HUGSIM_DISABLE_DEFAULT_EVAL_SEED
else
  export HUGSIM_DISABLE_DEFAULT_EVAL_SEED=1
fi

EXTRA_ARGS=()
if [[ "$HAS_EXPLICIT_REINFORCEPP_CONFIG" == "0" ]]; then
  EXTRA_ARGS+=(--reinforcepp-config "$DEFAULT_REINFORCEPP_CONFIG")
fi
if [[ "$HAS_EXPLICIT_REINFORCEPP_ALGO_TAG" == "0" ]]; then
  EXTRA_ARGS+=(--reinforcepp-algo-tag hugsim_sparsedrive_v1_collision_only_reinforcepp_nogrpo_noaux_peak1e4_decay5e5_update32)
fi
if [[ "$HAS_EXPLICIT_SLOTS" == "0" ]]; then
  EXTRA_ARGS+=(--slots 0:0 1:1 2:2 3:3 4:4 5:5 6:6 7:7)
fi
if [[ "$HAS_EXPLICIT_MAX_SCENES" == "0" ]]; then
  EXTRA_ARGS+=(--max-scenes 88)
fi
if [[ "$HAS_EXPLICIT_REPEAT_EVALS" == "0" ]]; then
  EXTRA_ARGS+=(--repeat-evals 2)
fi

exec /root/miniconda3/envs/recondreamerNew-rl/bin/python -u \
  "$REPO_ROOT/script/train_eval_pipeline.py" "$@" "${EXTRA_ARGS[@]}"
