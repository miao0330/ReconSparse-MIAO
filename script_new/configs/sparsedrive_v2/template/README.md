# SparseDriveV2 Config Templates

Experiment YAML files under `../experiments` compose reusable fragments with
`extends`:

```yaml
extends:
  - ../template/base.yaml
  - ../template/env/base.yaml
  - ../template/env/rewards/step_path.yaml
  - ../template/train/actor_learner.yaml
  - ../template/train/reinforcepp.yaml
  - ../template/agent/sparsedrive_v2.yaml
```

Fragments are loaded in order. Mappings merge recursively; lists and scalar
values from later fragments replace earlier values. Paths in `extends` are
relative to the file that declares them.

Training objective fragments are independent:

- `train/reinforcepp.yaml`, `train/ppo.yaml`, `train/sac.yaml`: primary closed-loop algorithm.
- `train/actor_learner.yaml`: shared actor-learner runtime settings.
- `train/grpo.yaml`: ordinary candidate GRPO using the configured scorer.
- `train/policy_loop_grpo.yaml`: policy-in-the-loop GRPO controls.
- `env/rewards/policy_loop_dense.yaml`: independent branch reward for policy-loop rollouts.

The loader also accepts existing monolithic YAML files without `extends`.
