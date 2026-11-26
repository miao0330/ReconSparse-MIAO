# 配置 HUGSIM-ORI
export HUGSIM_ROOT=/htfs/filesystem_pfs/waytous/lixm/HUGSIM-ORI
cd HUGSIM-ORI
pixi run python --version
ls configs
cd

# 激活环境
conda activate reconsparse

# 进入目录
cd ReconSparse

# 训练前建议设置
source script/setup_train_env.sh
或
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=/htfs/filesystem_pfs/waytous/lixm/ReconSparse/.cache/torch_extensions


# 评估SparseDriveV2_navsimv2.ckpt在HUGSIM上的结果
python tools/evaluate_existing_sparsedrive_v2_ckpts.py \
  --ckpts /htfs/filesystem_pfs/waytous/lixm/ReconSparse/egoADs/SparseDriveV2/ckpt/sparsedrive_navsimv2.ckpt 

#    