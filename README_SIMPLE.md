# ReconDreamer-RL 简单版说明

这个仓库是一个用于自动驾驶闭环强化学习训练的工程。主训练链路是 actor-learner：

```text
actor -> rollout -> shard buffer -> learner -> checkpoint/version publish -> actor reload
```

最常用入口是：

- `script/train_actor_learner_v2.py`：启动 actor-learner 训练。
- `script/train_eval_pipeline.py`：训练结束后自动评估。
- `framework/`：强化学习训练核心代码。

## 1. 推荐目录布局

建议把 ReconDreamer-RL 和 HUGSIM-ORI 放在同一级目录：

```text
/root/clone/ReconDreamer-RL
/root/clone/HUGSIM-ORI
```

其中：

- `ReconDreamer-RL`：训练、策略、RL 算法、actor-learner 调度。
- `HUGSIM-ORI`：闭环仿真后端，通常用自己的 pixi 环境运行。

## 2. 项目结构速览

```text
ReconDreamer-RL/
├── README.md                  # 原始完整说明
├── README_SIMPLE.md           # 当前简单版说明
├── README_FULL.md             # 更完整的结构与部署说明
├── environment.yml            # ReconDreamer-RL 侧 Conda 环境
├── script/                    # 训练、评估、配置入口
│   ├── train_actor_learner_v2.py
│   ├── train_eval_pipeline.py
│   └── configs/
├── framework/                 # RL 训练框架核心
│   ├── runner/                # actor / learner / orchestrator 调度
│   ├── agent/                 # 策略模型适配层
│   ├── env_wrapper/           # HUGSIM / Recon 环境包装
│   ├── rollout/               # actor 侧采样
│   ├── io/                    # shard buffer、权重版本、STOP 文件
│   ├── batch/                 # shard 到训练 batch 的转换
│   ├── algorithms/            # PPO / ReinforcePP / SAC / GRPO 等目标
│   ├── lightning/             # learner 侧 Lightning 训练逻辑
│   ├── rewards/               # reward 计算
│   └── utils/                 # 路径、观测、HUGSIM 工具函数
├── egoADs/                    # DiffusionDriveV2 / SparseDrive / SparseDriveV2 相关代码
├── assets/                    # 数据入口，通常是软链接
├── outputs/                   # 训练和评估输出，通常是软链接
├── tools/                     # 诊断、视频、可视化等辅助工具
├── reconsimulator/            # ReconSimulator 相关环境实现
└── tests/                     # 测试
```

## 3. 下载后部署

### 3.1 创建 Python 环境

```bash
cd /root/clone/ReconDreamer-RL

conda env create -f environment.yml
conda activate recondreamerNew-rl
```

如果环境已经存在，可以更新：

```bash
conda env update -n recondreamerNew-rl -f environment.yml --prune
conda activate recondreamerNew-rl
```

训练前建议设置：

```bash
export PYTHONPATH=/root/clone/ReconDreamer-RL:${PYTHONPATH:-}
```

### 3.2 配置 CUDA 扩展编译环境

本项目通常依赖 CUDA 11.8、PyTorch CUDA 扩展、`gsplat` 和 `nvdiffrast`。

```bash
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=/root/clone/ReconDreamer-RL/.cache/torch_extensions
```

检查 CUDA：

```bash
nvcc --version
python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cpp_extension CUDA_HOME:", CUDA_HOME)
PY
```

### 3.3 配置 HUGSIM-ORI

默认推荐：

```bash
export HUGSIM_ROOT=/root/clone/HUGSIM-ORI
```

检查 HUGSIM-ORI：

```bash
cd "${HUGSIM_ROOT:-/root/clone/HUGSIM-ORI}"
pixi run python --version
ls configs
```

### 3.4 配置数据和输出目录

常见软链接入口如下。下面只是示例，执行前先确认目标路径和本地目录是否已经存在：

```bash
cd /root/clone/ReconDreamer-RL

# 如果 assets 或 outputs 已经存在，先检查内容，不要直接覆盖。
ln -s /OpenDataset/ReconDreamer-RL/assets assets
ln -s /OpenDataset/ReconDreamer-RL/outputs outputs
ln -s /OpenDataset/HUGSIM_data/scenarios /root/clone/HUGSIM-ORI/configs/scenarios
ln -s /OpenDataset/zhaodanqi/HUGSIM_data/outputs /root/clone/HUGSIM-ORI/outputs
```

如果 `assets/` 或 `outputs/` 已经存在，不要直接覆盖。先检查：

```bash
ls -l assets outputs
```

训练 YAML 里的这些字段也要对应到真实路径：

- `env.hugsim.repo`
- `env.hugsim.scenario_dir`
- `env.hugsim.model_base`
- `env.hugsim.recon_data_root`
- `env.hugsim.nuscenes_root`
- `env.hugsim.frame2token_dir`
- `env.hugsim.output_root`

## 4. 启动训练

常用配置文件示例：

```bash
CONFIG=script/configs/sparsedrive_v2/20260616_template_HUGSM_algo-dsl.yaml
```

启动 actor-learner：

```bash
cd /root/clone/ReconDreamer-RL
conda activate recondreamerNew-rl
export PYTHONPATH=/root/clone/ReconDreamer-RL:${PYTHONPATH:-}
export HUGSIM_ROOT=/root/clone/HUGSIM-ORI

python -u script/train_actor_learner_v2.py \
  --role orchestrator \
  --config "$CONFIG"
```

训练输出通常在：

```text
outputs/actor_learner/<timestamp>_<run_name>/
```

关键运行文件：

```text
buffer/shards/       # actor 写入的待训练 shard
buffer/consumed/     # learner 消费后的 shard
weights/latest.ckpt  # learner 发布的最新权重
weights/version.txt  # 权重版本号，actor 用它判断是否 reload
STOP                 # 停止信号
TRAINING_LOCK        # 训练 / 发布权重时的锁文件
```

## 5. 训练后自动评估

```bash
cd /root/clone/ReconDreamer-RL
conda activate recondreamerNew-rl
export PYTHONPATH=/root/clone/ReconDreamer-RL:${PYTHONPATH:-}

python -u script/train_eval_pipeline.py \
  --reinforcepp-config script/configs/sparsedrive_v2/20260616_template_HUGSM_algo-dsl.yaml
```

也可以用脚本：

```bash
bash script/run_train_eval_pipeline_hugsim_train_eval.sh
bash script/run_train_eval_pipeline_recon_train_hugsim_eval.sh
```

## 6. 如果只想看重点

第一次部署时按这个顺序检查：

1. `conda activate recondreamerNew-rl`
2. `export PYTHONPATH=/root/clone/ReconDreamer-RL:${PYTHONPATH:-}`
3. `export HUGSIM_ROOT=/root/clone/HUGSIM-ORI`
4. `pixi run python --version` 能在 HUGSIM-ORI 里运行。
5. `assets/`、`outputs/`、HUGSIM 场景目录和 3DGS 数据路径存在。
6. 当前 YAML 的 `env.hugsim.*` 路径都能访问。
7. 运行 `python -u script/train_actor_learner_v2.py --help`。
8. 用 `--role orchestrator --config <yaml>` 启动训练。

更完整的架构、模块职责和排错说明见 `README_FULL.md`。
