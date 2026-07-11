from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .base import Algorithm


@dataclass
class TrajectorySACObjective:
    loss: torch.Tensor
    loss_pi: torch.Tensor
    loss_pg: torch.Tensor
    loss_entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_frac: torch.Tensor
    ratio_mean: torch.Tensor
    adv_mean: torch.Tensor
    logp_mean: torch.Tensor
    entropy_coef: torch.Tensor


def compute_sac_objective(
    *,
    new_logp: torch.Tensor,
    old_logp: Optional[torch.Tensor],
    adv: torch.Tensor,
    entropy_coef: float,
    kl_coef: float = 0.0,
    clip_eps: float = 0.2,
) -> TrajectorySACObjective:
    if old_logp is not None and int(old_logp.numel()) == int(adv.numel()):
        log_ratio = new_logp - old_logp
        ratio = torch.exp(log_ratio)
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_frac = ((ratio - 1.0).abs() > float(clip_eps)).to(dtype=torch.float32).mean()
        ratio_mean = ratio.mean()
    else:
        approx_kl = torch.zeros((), device=adv.device, dtype=torch.float32)
        clip_frac = torch.zeros((), device=adv.device, dtype=torch.float32)
        ratio_mean = torch.ones((), device=adv.device, dtype=torch.float32)

    loss_pg = -(adv.detach() * new_logp).mean()
    loss_entropy = torch.as_tensor(float(entropy_coef), device=adv.device, dtype=torch.float32) * new_logp.mean()
    loss = loss_pg + loss_entropy + float(kl_coef) * approx_kl
    return TrajectorySACObjective(
        loss=loss,
        loss_pi=loss,
        loss_pg=loss_pg,
        loss_entropy=loss_entropy,
        approx_kl=approx_kl,
        clip_frac=clip_frac,
        ratio_mean=ratio_mean,
        adv_mean=adv.mean(),
        logp_mean=new_logp.mean(),
        entropy_coef=torch.as_tensor(float(entropy_coef), device=adv.device, dtype=torch.float32),
    )


class SAC(Algorithm):
    """SAC-style policy-gradient configuration for closed-loop replay training.

    The current actor-learner shard contract stores sampled replay, rewards, and
    log-probs, not a standard off-policy SAC replay buffer with differentiable
    actions. This spec therefore exposes an entropy-regularized actor objective
    that can run on the existing closed-loop reward path.
    """

    def __init__(
        self,
        *,
        entropy_coef: float = 0.01,
        kl_coef: float = 0.0,
        epochs: int = 1,
        minibatch_size: int = 64,
        max_grad_norm: float = 0.5,
        grad_accum_steps: int = 1,
        ddp_seed: int = 0,
        eta: float = 1.0,
        use_distributed_sampler: bool = True,
        variant: str = "sac",
        policy_lr: float = 1e-6,
        weight_decay: float = 0.0,
        forward_kl_coef: float = 0.0,
        reverse_kl_coef: float = 0.0,
        distill_temperature: float = 1.0,
        teacher_ckpt: str | None = None,
    ) -> None:
        self.entropy_coef = float(entropy_coef)
        self.kl_coef = float(kl_coef)
        self.epochs = int(epochs)
        self.minibatch_size = int(minibatch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.grad_accum_steps = int(grad_accum_steps)
        self.ddp_seed = int(ddp_seed)
        self.eta = float(eta)
        self.use_distributed_sampler = bool(use_distributed_sampler)
        self.variant = str(variant)
        self.policy_lr = float(policy_lr)
        self.weight_decay = float(weight_decay)
        self.forward_kl_coef = float(forward_kl_coef)
        self.reverse_kl_coef = float(reverse_kl_coef)
        self.distill_temperature = float(distill_temperature)
        self.teacher_ckpt = None if teacher_ckpt is None else str(teacher_ckpt)


__all__ = ["SAC", "TrajectorySACObjective", "compute_sac_objective"]
