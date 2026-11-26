# Script Commands

This directory contains the main training and train-then-evaluate entrypoints for
the actor-learner RL stack.

Run commands from the repo root:

```bash
cd /root/clone/ReconDreamer-RL
```

## Environment And Agent Choices

The environment is selected by the YAML field `env.backend`.

- `hugsim_ori`: train actors in HUGSIM-ORI.
- `recon`: train actors in the ReconDreamer/ReconSimulator environment.

The agent is selected by the YAML field `agent.type`.

- `sparsedrive_v2`: SparseDriveV2 policy adapter.
- `sparsedrive`: SparseDrive v1 policy adapter.
- `ddv2` / `diffusiondrive_v2`: DiffusionDriveV2 policy adapter.

Useful existing configs:

```text
script/configs/sparsedrive_v2/20260616_template_HUGSM_algo-dsl.yaml
script/configs/sparsedrive_v2/20260704_template_RECON_algo-dsl.yaml
script/configs/sparsedrive/20260630_template_HUGSIM_sparsedrive_v1.yaml
script/configs/diffusiondrive_v2/ppo_closed_loop.yaml
```

## Train And Auto-Evaluate

The automatic pipeline is:

```text
train_actor_learner_v2.py -> outputs/TrainEvaluationAuto -> promote latest.ckpt -> HUGSIM evaluation
```

Current auto-evaluation is SparseDriveV2-oriented: it promotes the trained
checkpoint into `egoADs/SparseDriveV2/ckpt` and evaluates with
`tools/evaluate_existing_sparsedrive_v2_ckpts.py`.

### HUGSIM Train + HUGSIM Eval

Default command:

```bash
bash script/run_train_eval_pipeline_hugsim_train_eval.sh
```

Equivalent explicit command:

```bash
bash script/run_train_eval_pipeline_hugsim_train_eval.sh \
  --reinforcepp-config /root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/20260616_template_HUGSM_algo-dsl.yaml \
  --reinforcepp-algo-tag hugsim_ori_reinforcepp_craft_grpo \
  --slots 0:0 1:1 2:2 3:3 4:4 5:5 6:6 7:7 \
  --max-scenes 88 \
  --repeat-evals 2
```

### Recon Train + HUGSIM Eval

Default command:

```bash
bash script/run_train_eval_pipeline_recon_train_hugsim_eval.sh
```

Equivalent explicit command:

```bash
bash script/run_train_eval_pipeline_recon_train_hugsim_eval.sh \
  --reinforcepp-config /root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/20260704_template_RECON_algo-dsl.yaml \
  --reinforcepp-algo-tag recon_train_reinforcepp_nogrpo_noaux \
  --hugsim-template /root/clone/HUGSIM-ORI/configs/sim/nuscenes_eval_sparsedrive_v2_ppo_grpo_ver14.yaml \
  --scenario-dir /root/clone/HUGSIM-ORI/configs/scenarios/nuscenes \
  --slots 0:0 1:1 2:2 3:3 4:4 5:5 6:6 7:7 \
  --max-scenes 88 \
  --repeat-evals 2
```

### Generic Train + Eval Pipeline

Run only the default Reinforce++ branch:

```bash
PYTHONPATH=/root/clone/ReconDreamer-RL \
/root/miniconda3/envs/recondreamerNew-rl/bin/python -u script/train_eval_pipeline.py \
  --reinforcepp-config /root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/20260616_template_HUGSM_algo-dsl.yaml \
  --reinforcepp-algo-tag my_reinforcepp_run
```

Run PPO as well as Reinforce++:

```bash
PYTHONPATH=/root/clone/ReconDreamer-RL \
/root/miniconda3/envs/recondreamerNew-rl/bin/python -u script/train_eval_pipeline.py \
  --ppo \
  --ppo-config /root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/20260616_template_HUGSM_algo-dsl.yaml \
  --ppo-algo-tag my_ppo_run \
  --reinforcepp-config /root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/20260616_template_HUGSM_algo-dsl.yaml \
  --reinforcepp-algo-tag my_reinforcepp_run
```

Run only PPO:

```bash
PYTHONPATH=/root/clone/ReconDreamer-RL \
/root/miniconda3/envs/recondreamerNew-rl/bin/python -u script/train_eval_pipeline.py \
  --ppo \
  --skip-reinforcepp \
  --ppo-config /root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/20260616_template_HUGSM_algo-dsl.yaml \
  --ppo-algo-tag my_ppo_run
```

## Train Only

### Composed SparseDriveV2 templates

SparseDriveV2 experiments can be composed from reusable YAML fragments under
`script/configs/sparsedrive_v2/template`. An experiment file uses `extends`; later
fragments override earlier mappings recursively, while lists are replaced as a
whole. Existing monolithic YAML files remain supported.

For example:

```yaml
extends:
  - ../template/base.yaml
  - ../template/env/base.yaml
  - ../template/env/rewards/step_path.yaml
  - ../template/train/actor_learner.yaml
  - ../template/train/reinforcepp.yaml
  - ../template/agent/sparsedrive_v2.yaml
```

`train/grpo.yaml` enables ordinary candidate GRPO. The separate
`train/policy_loop_grpo.yaml` fragment enables policy-in-the-loop GRPO, and
`env/rewards/policy_loop_dense.yaml` supplies its independent branch reward.
The ready-to-run scene-0411 examples are
`experiments/scene0411_reinforcepp_collision500.yaml` and
`experiments/scene0411_policy_loop_dense_reward.yaml`.
The orchestrator writes the fully resolved configuration into the run directory.

Use `train_actor_learner_v2.py` when you only want to start actor-learner
training and do not want checkpoint promotion or HUGSIM evaluation.

### HUGSIM + SparseDriveV2

```bash
PYTHONPATH=/root/clone/ReconDreamer-RL \
python -u script/train_actor_learner_v2.py \
  --role orchestrator \
  --config /root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/20260616_template_HUGSM_algo-dsl.yaml
```

### Recon + SparseDriveV2

```bash
PYTHONPATH=/root/clone/ReconDreamer-RL \
python -u script/train_actor_learner_v2.py \
  --role orchestrator \
  --config /root/clone/ReconDreamer-RL/script/configs/sparsedrive_v2/20260704_template_RECON_algo-dsl.yaml
```

### HUGSIM + SparseDrive V1

```bash
PYTHONPATH=/root/clone/ReconDreamer-RL \
python -u script/train_actor_learner_v2.py \
  --role orchestrator \
  --config /root/clone/ReconDreamer-RL/script/configs/sparsedrive/20260630_template_HUGSIM_sparsedrive_v1.yaml
```

### DiffusionDriveV2

```bash
PYTHONPATH=/root/clone/ReconDreamer-RL \
python -u script/train_actor_learner_v2.py \
  --role orchestrator \
  --config /root/clone/ReconDreamer-RL/script/configs/diffusiondrive_v2/ppo_closed_loop.yaml
```

For DiffusionDriveV2, confirm the config's `env.backend` before running if you
need a specific environment backend.

## Notes

- Auto-eval logs are written under `/root/clone/HUGSIM-ORI/outputs/evaluate-auto`
  by default.
- Train+eval run manifests and training logs are written under
  `outputs/TrainEvaluationAuto`.
- Train-only actor-learner buffers and checkpoints are controlled by
  `train.actor_learner.buffer_dir` in the selected YAML config.
- If startup stalls during gsplat compilation, remove the cached extension:

```bash
rm -rf /root/clone/ReconDreamer-RL/.cache/torch_extensions/gsplat_cuda_legacy
```
