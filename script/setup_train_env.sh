#!/usr/bin/env bash
# Source this file before actor-learner training or gsplat warmup:
#   cd /path/to/ReconSparse
#   source script/setup_train_env.sh
#
# Do not execute it directly; `source` keeps exports in your shell.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "[setup_train_env] Please run: source script/setup_train_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi

_SETUP_TRAIN_ENV_SCRIPT="${BASH_SOURCE[0]}"
while [[ -L "${_SETUP_TRAIN_ENV_SCRIPT}" ]]; do
  _SETUP_TRAIN_ENV_DIR="$(cd -- "$(dirname -- "${_SETUP_TRAIN_ENV_SCRIPT}")" && pwd)"
  _SETUP_TRAIN_ENV_SCRIPT="$(readlink "${_SETUP_TRAIN_ENV_SCRIPT}")"
  [[ "${_SETUP_TRAIN_ENV_SCRIPT}" != /* ]] && _SETUP_TRAIN_ENV_SCRIPT="${_SETUP_TRAIN_ENV_DIR}/${_SETUP_TRAIN_ENV_SCRIPT}"
done
RECONSPARSE_ROOT="$(cd -- "$(dirname -- "${_SETUP_TRAIN_ENV_SCRIPT}")/.." && pwd)"

export RECONSPARSE_ROOT
export PYTHONPATH="${RECONSPARSE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${TORCH_EXTENSIONS_DIR:-}" ]]; then
  export TORCH_EXTENSIONS_DIR="${RECONSPARSE_ROOT}/.cache/torch_extensions"
fi
mkdir -p "${TORCH_EXTENSIONS_DIR}"

_resolve_cuda_home() {
  local candidate=""
  for candidate in "$@"; do
    [[ -z "${candidate}" ]] && continue
    if [[ -x "${candidate}/bin/nvcc" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    if [[ -f "${candidate}/include/cuda_runtime.h" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    if [[ -f "${candidate}/targets/x86_64-linux/include/cuda_runtime.h" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

if ! CUDA_HOME="$(_resolve_cuda_home \
  "${CUDA_HOME:-}" \
  "${CONDA_PREFIX:-}" \
  "/usr/local/cuda" \
  "/usr/local/cuda-12.1" \
  "/usr/local/cuda-12" \
  "/usr/local/cuda-11.8")"; then
  CUDA_HOME="${CONDA_PREFIX:-}"
fi
export CUDA_HOME

if [[ -n "${CUDA_HOME}" ]]; then
  case ":${PATH}:" in
    *":${CUDA_HOME}/bin:"*) ;;
    *) export PATH="${CUDA_HOME}/bin:${PATH}" ;;
  esac
fi

_py_ver="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

_cuda_include_dirs=()
if [[ -n "${CUDA_HOME}" && -f "${CUDA_HOME}/include/cuda_runtime.h" ]]; then
  _cuda_include_dirs+=("${CUDA_HOME}/include")
fi
if [[ -n "${CUDA_HOME}" && -f "${CUDA_HOME}/targets/x86_64-linux/include/cuda_runtime.h" ]]; then
  _cuda_include_dirs+=("${CUDA_HOME}/targets/x86_64-linux/include")
fi
if [[ -n "${CUDA_HOME}" && -f "${CUDA_HOME}/lib/python${_py_ver}/site-packages/nvidia/cuda_runtime/include/cuda_runtime.h" ]]; then
  _cuda_include_dirs+=("${CUDA_HOME}/lib/python${_py_ver}/site-packages/nvidia/cuda_runtime/include")
fi
if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/lib/python${_py_ver}/site-packages/nvidia/cuda_runtime/include/cuda_runtime.h" ]]; then
  _cuda_include_dirs+=("${CONDA_PREFIX}/lib/python${_py_ver}/site-packages/nvidia/cuda_runtime/include")
fi

for _inc in "${_cuda_include_dirs[@]}"; do
  case ":${CPATH:-}:" in
    *":${_inc}:"*) ;;
    *) export CPATH="${_inc}${CPATH:+:${CPATH}}" ;;
  esac
  case ":${CPLUS_INCLUDE_PATH:-}:" in
    *":${_inc}:"*) ;;
    *) export CPLUS_INCLUDE_PATH="${_inc}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}" ;;
  esac
done

_cuda_library_dirs=()
if [[ -n "${CUDA_HOME}" && -d "${CUDA_HOME}/lib64" ]]; then
  _cuda_library_dirs+=("${CUDA_HOME}/lib64")
fi
if [[ -n "${CUDA_HOME}" && -d "${CUDA_HOME}/targets/x86_64-linux/lib" ]]; then
  _cuda_library_dirs+=("${CUDA_HOME}/targets/x86_64-linux/lib")
fi
if [[ -n "${CUDA_HOME}" && -d "${CUDA_HOME}/lib/python${_py_ver}/site-packages/nvidia/cuda_runtime/lib" ]]; then
  _cuda_library_dirs+=("${CUDA_HOME}/lib/python${_py_ver}/site-packages/nvidia/cuda_runtime/lib")
fi
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib/python${_py_ver}/site-packages/nvidia/cuda_runtime/lib" ]]; then
  _cuda_library_dirs+=("${CONDA_PREFIX}/lib/python${_py_ver}/site-packages/nvidia/cuda_runtime/lib")
fi

for _lib in "${_cuda_library_dirs[@]}"; do
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${_lib}:"*) ;;
    *) export LD_LIBRARY_PATH="${_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
  case ":${LIBRARY_PATH:-}:" in
    *":${_lib}:"*) ;;
    *) export LIBRARY_PATH="${_lib}${LIBRARY_PATH:+:${LIBRARY_PATH}}" ;;
  esac
done

echo "[setup_train_env] RECONSPARSE_ROOT=${RECONSPARSE_ROOT}"
echo "[setup_train_env] CUDA_HOME=${CUDA_HOME}"
echo "[setup_train_env] TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
echo "[setup_train_env] nvcc=$(command -v nvcc 2>/dev/null || echo missing)"
if [[ ${#_cuda_include_dirs[@]} -gt 0 ]]; then
  echo "[setup_train_env] cuda_include=${_cuda_include_dirs[0]}"
else
  echo "[setup_train_env] warning: cuda_runtime.h include path not found" >&2
fi

unset _SETUP_TRAIN_ENV_SCRIPT _SETUP_TRAIN_ENV_DIR
unset _py_ver _cuda_include_dirs _cuda_library_dirs _inc _lib candidate
