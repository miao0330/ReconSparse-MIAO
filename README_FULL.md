# ReconDreamer-RL 完整说明

ReconDreamer-RL 是一个面向自动驾驶闭环策略优化的强化学习工程。它把策略模型、HUGSIM / Recon 闭环环境、actor-learner 采样训练、shard buffer、PPO / ReinforcePP / SAC / GRPO 类目标、reward shaping、checkpoint 发布和评估流程放在同一个训练系统中。

当前主训练路径是文件缓冲区形式的 actor-learner：

```text
actor -> rollout -> shard buffer -> learner -> checkpoint/version publish -> actor reload
```

核心入口：

- `script/train_actor_learner_v2.py`：主训练入口。
- `script/train_eval_pipeline.py`：训练后自动评估入口。
- `script/configs/`：训练 YAML 配置。
- `framework/`：强化学习训练框架核心。

如果只需要最短部署流程，先看 `README_SIMPLE.md`。

## 1. 整体目录关系

推荐把 `ReconDreamer-RL` 和 `HUGSIM-ORI` 并列放置：

```text
/root/clone/
├── ReconDreamer-RL/
└── HUGSIM-ORI/
```

职责划分：

- `ReconDreamer-RL`：负责 actor-learner 调度、RL 算法、策略 adapter、训练和评估入口。
- `HUGSIM-ORI`：负责闭环仿真后端，通常通过 FIFO / pixi 由 ReconDreamer-RL 启动。

## 2. 仓库结构

```text
ReconDreamer-RL/
├── README.md
├── README_SIMPLE.md
├── README_FULL.md
├── environment.yml
├── script/
│   ├── train_actor_learner_v2.py
│   ├── train_eval_pipeline.py
│   ├── run_train_eval_pipeline_hugsim_train_eval.sh
│   ├── run_train_eval_pipeline_recon_train_hugsim_eval.sh
│   └── configs/
├── framework/
│   ├── runner/
│   ├── agent/
│   ├── env_wrapper/
│   ├── rollout/
│   ├── io/
│   ├── batch/
│   ├── algorithms/
│   ├── lightning/
│   ├── rewards/
│   ├── rewardmodel/
│   └── utils/
├── egoADs/
│   ├── DiffusionDriveV2/
│   ├── SparseDrive/
│   └── SparseDriveV2/
├── policy/
├── reconsimulator/
├── tools/
├── tests/
├── assets/
└── outputs/
```

顶层目录说明：

- `script/`：训练、评估、调试入口，以及所有 YAML 配置。
- `framework/`：RL 主框架，actor-learner、rollout、buffer、batch、objective、Lightning 都在这里。
- `egoADs/`：外部自动驾驶策略后端相关代码，例如 DiffusionDriveV2、SparseDrive、SparseDriveV2。
- `policy/`：策略侧组件和模型相关代码。
- `reconsimulator/`：ReconSimulator 相关环境实现。
- `tools/`：诊断、视频生成、可视化和辅助脚本。
- `tests/`：测试。
- `assets/`：数据入口，通常是指向共享盘的软链接。
- `outputs/`：训练、评估、日志、视频等输出入口，通常是软链接。

## 3. framework 模块职责

`framework/` 是训练链路的核心。每个子目录都有自己的 README，可按需深入查看。

```text
framework/
├── runner/       # 运行时调度和对象装配
├── agent/        # 策略模型 adapter
├── env_wrapper/  # 闭环环境包装
├── rollout/      # actor 侧采样
├── io/           # shard buffer、STOP、权重版本、原子保存
├── batch/        # shard 到训练 batch 的转换
├── algorithms/   # RL objective 和 return / advantage 计算
├── lightning/    # learner 侧 Lightning 训练逻辑
├── rewards/      # reward shaping 和 terminal penalty
├── rewardmodel/  # 结构化 reward / scorer 类型
└── utils/        # 路径、观测、HUGSIM、nuScenes 等工具
```

### 3.1 runner

路径：`framework/runner/`

负责运行时编排，不放模型数学逻辑。主要职责：

- 解析和规范化配置。
- 启动 `orchestrator`、`actor`、`learner` 三类角色。
- 分配 GPU 和运行环境变量。
- 构造 agent、environment、learner、buffer 等对象。

关键文件：

- `orchestrator.py`
- `actor_runtime.py`
- `learner_runtime.py`
- `config_normalization.py`
- `agent_factory.py`
- `env_factory.py`
- `learner_factory.py`

### 3.2 agent

路径：`framework/agent/`

负责把具体策略模型包装成统一 RL 接口。actor 和 learner 都会使用 agent，但 actor / learner 本身不是 agent。

主要职责：

- 动作采样。
- replay 序列化。
- 从 replay 重算 log-prob。
- checkpoint 保存和加载。
- 隔离 DiffusionDriveV2、SparseDrive、SparseDriveV2 等模型后端差异。

### 3.3 env_wrapper

路径：`framework/env_wrapper/`

负责把 HUGSIM / Recon 闭环环境包装成 RL 可调用接口：

- `reset`
- `step`
- 场景采样
- 碰撞和终止信息
- reward 接入
- FIFO / pixi 启动 HUGSIM 后端

### 3.4 rollout

路径：`framework/rollout/`

actor 侧采样逻辑。它调用 agent 和 environment，收集 trajectory，然后打包成 shard。

常见 shard 字段：

- `obs`
- `next_obs`
- `reward`
- `done`
- `terminated`
- `truncated`
- `old_logp`
- `replay`
- `meta`

### 3.5 io

路径：`framework/io/`

actor 和 learner 之间的文件协议层。这里的改动风险比较高，因为 actor-learner 协调依赖固定文件名和目录约定。

关键运行文件：

```text
buffer/shards/       # actor 写入，learner 等待消费
buffer/consumed/     # learner 消费后的 shard
weights/latest.ckpt  # learner 发布的最新权重
weights/version.txt  # 权重版本号
STOP                 # 停止信号
TRAINING_LOCK        # 训练 / 发布期间的互斥标记
```

### 3.6 batch

路径：`framework/batch/`

负责把 shard 转为 learner 可以训练的 batch，包括 return、GAE、advantage normalization 等准备工作。

### 3.7 algorithms

路径：`framework/algorithms/`

负责强化学习目标和训练数学逻辑，包括：

- PPO
- ReinforcePP
- SAC-style objective
- GRPO
- return / advantage 计算
- PDM / CRAFT 等 scorer 或辅助项

### 3.8 lightning

路径：`framework/lightning/`

learner 侧训练生命周期。主要负责：

- LightningDataModule 从 shard buffer 取数据。
- LightningModule 执行 training step。
- optimizer 配置。
- checkpoint 发布。
- WandB logging。

### 3.9 rewards / rewardmodel / utils

- `framework/rewards/`：路径跟踪、碰撞、舒适性、terminal penalty 等 reward。
- `framework/rewardmodel/`：结构化 reward / scorer 类型。
- `framework/utils/`：路径解析、观测处理、gsplat、HUGSIM 执行、nuScenes token 等工具。

## 4. 主训练数据流

主入口：

```text
script/train_actor_learner_v2.py
```

整体调用路径：

```text
script/train_actor_learner_v2.py
  -> framework/runner/orchestrator.py
      -> framework/runner/actor_runtime.py
          -> framework/env_wrapper/
          -> framework/agent/
          -> framework/rollout/collector.py
          -> framework/io/buffer.py
      -> framework/runner/learner_runtime.py
          -> framework/lightning/actor_learner_datamodule.py
          -> framework/batch/actor_learner.py
          -> framework/lightning/actor_learner_module.py
          -> framework/algorithms/trajectory_policy_core.py
          -> framework/io/buffer.py
```

运行角色：

- `orchestrator`：主控进程，负责启动 learner 和多个 actor。
- `actor`：持有一个策略副本，与环境交互，采样 rollout，写 shard。
- `learner`：读取 shard，更新策略，写 `weights/latest.ckpt`，更新 `weights/version.txt`。

## 5. 环境部署

### 5.1 创建 Conda 环境

```bash
cd /root/clone/ReconDreamer-RL

conda env create -f environment.yml
conda activate recondreamerNew-rl
```

如果环境已经存在：

```bash
conda env update -n recondreamerNew-rl -f environment.yml --prune
conda activate recondreamerNew-rl
```

设置 `PYTHONPATH`：

```bash
export PYTHONPATH=/root/clone/ReconDreamer-RL:${PYTHONPATH:-}
```

### 5.2 检查 Python / CUDA / 关键依赖

```bash
python - <<'PY'
import torch
import lightning
import pytorch_lightning

print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("lightning", lightning.__version__)
print("pytorch_lightning", pytorch_lightning.__version__)
PY
```

如果安装了 `gsplat` 和 `nvdiffrast`，可以进一步检查：

```bash
python - <<'PY'
import gsplat
import nvdiffrast.torch as dr
print("gsplat", getattr(gsplat, "__version__", "unknown"))
print("nvdiffrast ok", dr is not None)
PY
```

### 5.3 配置 CUDA 扩展编译

```bash
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=/root/clone/ReconDreamer-RL/.cache/torch_extensions
```

检查：

```bash
nvcc --version
python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME
print("torch cuda:", torch.version.cuda)
print("cpp_extension CUDA_HOME:", CUDA_HOME)
PY
```

如果 actor 第一次启动长时间卡在 CUDA extension 编译，可以尝试清理本地扩展缓存：

```bash
rm -rf /root/clone/ReconDreamer-RL/.cache/torch_extensions/gsplat_cuda_legacy
```

## 6. HUGSIM-ORI 部署

HUGSIM-ORI 是独立仓库，不建议放进 `ReconDreamer-RL/` 内部。

推荐：

```bash
/root/clone/ReconDreamer-RL
/root/clone/HUGSIM-ORI
```

设置：

```bash
export HUGSIM_ROOT=/root/clone/HUGSIM-ORI
```

检查：

```bash
cd "${HUGSIM_ROOT:-/root/clone/HUGSIM-ORI}"
pixi run python --version
ls configs
```

ReconDreamer-RL 侧负责训练、采样、策略更新；HUGSIM-ORI 侧负责仿真进程。通常训练 YAML 里的 `env.hugsim.pixi_cmd` 使用 `pixi`。

## 7. 数据、资产和软链接

大文件数据不应提交到 Git，建议通过软链接接入。

常见入口：

- `assets/`：ReconDreamer-RL 侧数据入口。
- `outputs/`：训练和评估输出入口。
- `HUGSIM-ORI/configs/scenarios`：HUGSIM 场景 YAML。
- `HUGSIM-ORI/outputs`：HUGSIM 评估、渲染、视频输出。

示例。执行前先确认本地目录是否已经存在，不要直接覆盖已有数据：

```bash
ln -s /OpenDataset/HUGSIM_data/scenarios /root/clone/HUGSIM-ORI/configs/scenarios
ln -s /OpenDataset/zhaodanqi/HUGSIM_data/outputs /root/clone/HUGSIM-ORI/outputs
ln -s /OpenDataset/ReconDreamer-RL/outputs /root/clone/ReconDreamer-RL/outputs
ln -s /OpenDataset/ReconDreamer-RL/assets /root/clone/ReconDreamer-RL/assets
```

如果目标目录已存在，先检查，不要直接覆盖：

```bash
ls -l /root/clone/ReconDreamer-RL/assets
ls -l /root/clone/ReconDreamer-RL/outputs
ls -l /root/clone/HUGSIM-ORI/configs/scenarios
ls -l /root/clone/HUGSIM-ORI/outputs
```

常见数据路径：

```text
/root/clone/ReconDreamer-RL/assets/nus/data
/root/clone/ReconDreamer-RL/assets/nuscenes/v1.0-trainval
/root/clone/ReconDreamer-RL/assets/nus/information/frame2token
/OpenDataset/HUGSIM_data/scenes/nuscenes
```

## 8. 策略后端资源

策略后端默认从 `egoADs/` 解析。常见后端：

- `egoADs/DiffusionDriveV2/`
- `egoADs/SparseDrive/`
- `egoADs/SparseDriveV2/`

SparseDriveV2 的典型配置：

```yaml
agent:
  type: sparsedrive_v2
  ckpt: SparseDriveV2/ckpt/sparsedrive_navsimv2.ckpt
```

检查文件：

```bash
ls -l egoADs/SparseDriveV2/ckpt
ls -l egoADs/SparseDriveV2/ckpt/sparsedrive_navsimv2.ckpt
ls -l egoADs/SparseDriveV2/ckpt/resnet34.bin
ls -l egoADs/SparseDriveV2/ckpt/kmeans
```

如果切换策略后端，需要同时检查：

- `agent.type`
- `agent.ckpt`
- `agent.config`
- 对应策略项目需要的额外资源

## 9. 配置文件

训练配置在：

```text
script/configs/
```

启动时传入的 YAML 是运行时配置的主来源：

```bash
--config script/configs/sparsedrive_v2/20260616_template_HUGSM_algo-dsl.yaml
```

常见配置组：

- `env`：环境后端、episode 长度、渲染尺寸、场景采样、reward、HUGSIM 参数。
- `env.hugsim`：HUGSIM 仓库路径、scenario、3DGS 资产、nuScenes 数据、FIFO 启动参数。
- `env.reward`：step reward、terminal penalty、碰撞、舒适性、CRAFT / PDM scorer 等。
- `agent`：策略后端类型、checkpoint、模型配置。
- `train`：算法、学习率、batch/update 设置、WandB、actor-learner 参数。

典型 HUGSIM 配置片段：

```yaml
env:
  backend: hugsim_ori
  hugsim:
    repo: /root/clone/HUGSIM-ORI
    scenario_dir: /root/clone/HUGSIM-ORI/configs/scenarios/nuscenes
    model_base: /OpenDataset/HUGSIM_data/scenes/nuscenes
    recon_data_root: /root/clone/ReconDreamer-RL/assets/nus/data
    nuscenes_root: /root/clone/ReconDreamer-RL/assets/nuscenes/v1.0-trainval
    frame2token_dir: /root/clone/ReconDreamer-RL/assets/nus/information/frame2token
    output_root: outputs/hugsim_rl
    pixi_cmd: pixi
```

如果修改配置字段，通常需要同时检查：

- `script/train_actor_learner_v2.py`
- `framework/runner/config_normalization.py`
- `framework/runner/agent_factory.py`
- `framework/runner/env_factory.py`
- `framework/runner/learner_factory.py`
- `framework/runner/actor_learner.py`
- 代表性的 YAML 文件

## 10. 启动训练

### 10.1 查看帮助

```bash
cd /root/clone/ReconDreamer-RL
conda activate recondreamerNew-rl
export PYTHONPATH=/root/clone/ReconDreamer-RL:${PYTHONPATH:-}

python -u script/train_actor_learner_v2.py --help
```

### 10.2 orchestrator 模式

这是最常用方式。orchestrator 会启动 learner 和多个 actor：

```bash
cd /root/clone/ReconDreamer-RL
conda activate recondreamerNew-rl
export PYTHONPATH=/root/clone/ReconDreamer-RL:${PYTHONPATH:-}
export HUGSIM_ROOT=/root/clone/HUGSIM-ORI

CONFIG=script/configs/sparsedrive_v2/20260616_template_HUGSM_algo-dsl.yaml

python -u script/train_actor_learner_v2.py \
  --role orchestrator \
  --config "$CONFIG"
```

### 10.3 单独 actor / learner

调试时也可以只启动某个角色：

```bash
python -u script/train_actor_learner_v2.py \
  --role learner \
  --config "$CONFIG"
```

```bash
python -u script/train_actor_learner_v2.py \
  --role actor \
  --config "$CONFIG"
```

## 11. 训练输出

默认输出通常位于：

```text
outputs/actor_learner/<timestamp>_<run_name>/
```

关键产物：

```text
buffer/shards/       # actor 产生的训练 shard
buffer/consumed/     # learner 消费后的 shard
weights/latest.ckpt  # learner 最新权重
weights/version.txt  # 权重版本号
STOP                 # 停止信号
TRAINING_LOCK        # 训练 / 权重发布期间的锁
*.yaml               # 当前运行配置快照
```

actor 会通过 `weights/version.txt` 判断是否有新权重，然后加载 `weights/latest.ckpt`。

## 12. 自动训练 + 评估

```bash
cd /root/clone/ReconDreamer-RL
conda activate recondreamerNew-rl
export PYTHONPATH=/root/clone/ReconDreamer-RL:${PYTHONPATH:-}

python -u script/train_eval_pipeline.py \
  --reinforcepp-config script/configs/sparsedrive_v2/20260616_template_HUGSM_algo-dsl.yaml
```

也可以使用 wrapper：

```bash
bash script/run_train_eval_pipeline_hugsim_train_eval.sh
bash script/run_train_eval_pipeline_recon_train_hugsim_eval.sh
```

自动 train + eval 输出通常在：

```text
outputs/TrainEvaluationAuto/<run_id>/
```

评估侧结果通常会写入 HUGSIM-ORI 的：

```text
outputs/evaluate-auto/
```

## 13. 常见问题排查

### 13.1 HUGSIM 找不到

检查：

```bash
echo "$HUGSIM_ROOT"
ls -l /root/clone/HUGSIM-ORI
```

同时检查 YAML：

- `env.hugsim.repo`
- `env.hugsim.pixi_cmd`
- `env.hugsim.scenario_dir`

### 13.2 scenario 或 3DGS 资产找不到

检查：

- `env.hugsim.scenario_dir`
- `env.hugsim.model_base`
- `env.hugsim.recon_data_root`
- `env.hugsim.nuscenes_root`
- `env.hugsim.frame2token_dir`

命令：

```bash
ls -l /root/clone/HUGSIM-ORI/configs/scenarios
ls -l /OpenDataset/HUGSIM_data/scenes/nuscenes
ls -l /root/clone/ReconDreamer-RL/assets/nus/data
ls -l /root/clone/ReconDreamer-RL/assets/nuscenes/v1.0-trainval
ls -l /root/clone/ReconDreamer-RL/assets/nus/information/frame2token
```

### 13.3 CUDA extension 编译卡住

检查 CUDA：

```bash
nvcc --version
echo "$CUDA_HOME"
echo "$TORCH_EXTENSIONS_DIR"
```

清理缓存：

```bash
rm -rf /root/clone/ReconDreamer-RL/.cache/torch_extensions/gsplat_cuda_legacy
```

### 13.4 actor 一直等权重

检查 learner 是否写出了：

```text
weights/latest.ckpt
weights/version.txt
```

也检查是否有异常退出、`TRAINING_LOCK` 是否长时间残留、learner 日志是否报错。

### 13.5 learner 一直等 shard

检查：

```bash
ls -l outputs/actor_learner/*/buffer/shards
```

同时看 actor 日志，重点确认：

- actor 是否成功启动环境。
- HUGSIM FIFO 是否正常。
- 场景路径是否存在。
- policy checkpoint 是否能加载。

### 13.6 log-prob / replay mismatch

优先检查：

- `framework/agent/` 中 replay 序列化和 log-prob 重算逻辑。
- `framework/algorithms/trajectory_policy_core.py`
- `framework/batch/` 中 batch 构建逻辑。

## 14. 修改代码时的注意事项

### 14.1 不要随意改 actor-learner 协议

这些属于协议级约定：

- shard 命名和版本假设
- `weights/latest.ckpt`
- `weights/version.txt`
- `STOP`
- `TRAINING_LOCK`
- shard 字段，如 `obs`、`old_logp`、`reward`、`done`、`replay`

如果必须改 shard schema 或权重发布逻辑，要同时检查 actor、learner、I/O、batch、Lightning 生命周期。

### 14.2 保持目录职责清楚

- runtime 调度放 `framework/runner/`
- 策略模型差异放 `framework/agent/`
- 环境包装放 `framework/env_wrapper/`
- actor 采样放 `framework/rollout/`
- buffer / checkpoint 协议放 `framework/io/`
- batch 准备放 `framework/batch/`
- objective 数学放 `framework/algorithms/`
- learner 生命周期放 `framework/lightning/`
- reward 放 `framework/rewards/`

### 14.3 配置字段要成组修改

如果新增或重命名配置字段，至少检查：

- `script/train_actor_learner_v2.py`
- `framework/runner/config_normalization.py`
- `framework/runner/agent_factory.py`
- `framework/runner/env_factory.py`
- `framework/runner/learner_factory.py`
- `script/configs/` 下的代表性 YAML

## 15. 推荐阅读顺序

如果第一次接手这个仓库，建议按下面顺序看：

1. `README_SIMPLE.md`
2. `README_FULL.md`
3. `script/train_actor_learner_v2.py`
4. `framework/README.md`
5. `framework/runner/README.md`
6. `framework/agent/README.md`
7. `framework/env_wrapper/README.md`
8. `framework/rollout/README.md`
9. `framework/io/README.md`
10. `framework/batch/README.md`
11. `framework/lightning/README.md`
12. `framework/algorithms/README.md`

## 16. 最小部署检查清单

```bash
cd /root/clone/ReconDreamer-RL
conda activate recondreamerNew-rl
export PYTHONPATH=/root/clone/ReconDreamer-RL:${PYTHONPATH:-}
export HUGSIM_ROOT=/root/clone/HUGSIM-ORI

python -u script/train_actor_learner_v2.py --help
```

然后确认：

- `HUGSIM_ROOT` 指向真实 HUGSIM-ORI 仓库。
- HUGSIM-ORI 内 `pixi run python --version` 可运行。
- `assets/` 和 `outputs/` 存在。
- 当前 YAML 的 `env.hugsim.*` 路径都存在。
- 策略 checkpoint 存在。
- CUDA / PyTorch / gsplat / nvdiffrast 可用。

确认后启动：

```bash
CONFIG=script/configs/sparsedrive_v2/20260616_template_HUGSM_algo-dsl.yaml

python -u script/train_actor_learner_v2.py \
  --role orchestrator \
  --config "$CONFIG"
```
