# ReconDreamer-RL

ReconDreamer-RL 是一个用于自动驾驶闭环强化学习训练的工程，主要支持 actor-learner 形式的策略采样、训练、权重发布和评估流程。

当前主训练链路：

```text
actor -> rollout -> shard buffer -> learner -> checkpoint/version publish -> actor reload
```

## 📚 文档入口

如果你是第一次下载或部署这个仓库，建议先看简单版：

- ⚡ [README_SIMPLE.md](README_SIMPLE.md)：快速了解项目结构、下载后部署、数据软链接、启动训练和自动评估。
- 📖 [README_FULL.md](README_FULL.md)：完整说明目录职责、actor-learner 架构、HUGSIM / CUDA / 数据配置、训练评估流程和常见问题。

## 🧭 常用入口

- 🎬 `script/train_actor_learner_v2.py`：actor-learner 主训练入口。
- 🧪 `script/train_eval_pipeline.py`：训练后自动评估入口。
- ⚙️ `script/configs/`：训练 YAML 配置目录。
- 🧠 `framework/`：强化学习训练框架核心代码。
- 🚗 `egoADs/`：DiffusionDriveV2、SparseDrive、SparseDriveV2 等策略后端代码。
- 📦 `assets/`：本地数据入口，通常是软链接。
- 📁 `outputs/`：训练、评估和可视化输出目录，通常是软链接。

## 🚀 最小启动示例

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

部署前请确认 `HUGSIM-ORI`、`assets/`、`outputs/` 和当前 YAML 中的 `env.hugsim.*` 路径都已经配置好。完整步骤见 [README_SIMPLE.md](README_SIMPLE.md) 和 [README_FULL.md](README_FULL.md)。
