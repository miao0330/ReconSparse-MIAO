from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .base import Algorithm


@dataclass
class TrajectoryPPOObjective:
    loss: torch.Tensor
    loss_pi: torch.Tensor
    loss_v: torch.Tensor
    approx_kl: torch.Tensor
    clip_frac: torch.Tensor
    value_clip_frac: torch.Tensor
    ratio_mean: torch.Tensor
    adv_mean: torch.Tensor


def compute_ppo_objective(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    adv: torch.Tensor,
    ret: torch.Tensor,
    value_pred: torch.Tensor,
    old_value: Optional[torch.Tensor],
    clip_eps: float,
    vf_coef: float,
    value_clip_eps: float = 0.0,
    kl_coef: float = 0.0,
    dual_clip: float | None = None,
) -> TrajectoryPPOObjective:
    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps)) * adv
    policy_term = torch.min(surr1, surr2)

    clip_frac = ((ratio - 1.0).abs() > float(clip_eps)).to(dtype=torch.float32).mean()
    if dual_clip is not None:
        clipped = torch.max(policy_term, torch.as_tensor(float(dual_clip), device=adv.device, dtype=adv.dtype) * adv)
        policy_term = torch.where(adv < 0.0, clipped, policy_term)
    loss_pi = -policy_term.mean()

    if float(value_clip_eps) > 0.0 and old_value is not None and int(old_value.numel()) == int(ret.numel()):
        value_delta = value_pred - old_value
        value_pred_clipped = old_value + value_delta.clamp(-float(value_clip_eps), float(value_clip_eps))
        loss_v_unclipped = F.mse_loss(value_pred, ret, reduction="none")
        loss_v_clipped = F.mse_loss(value_pred_clipped, ret, reduction="none")
        loss_v = torch.max(loss_v_unclipped, loss_v_clipped).mean()
        value_clip_frac = (value_delta.abs() > float(value_clip_eps)).to(dtype=torch.float32).mean()
    else:
        loss_v = F.mse_loss(value_pred, ret)
        value_clip_frac = torch.zeros((), device=ret.device, dtype=torch.float32)

    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    loss = loss_pi + float(vf_coef) * loss_v + float(kl_coef) * approx_kl
    return TrajectoryPPOObjective(
        loss=loss,
        loss_pi=loss_pi,
        loss_v=loss_v,
        approx_kl=approx_kl,
        clip_frac=clip_frac,
        value_clip_frac=value_clip_frac,
        ratio_mean=ratio.mean(),
        adv_mean=adv.mean(),
    )


def compute_ppo_metrics(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    adv: torch.Tensor,
    ret: torch.Tensor,
    value_pred: torch.Tensor,
    loss: TrajectoryPPOObjective,
) -> Dict[str, torch.Tensor]:
    del new_logp, old_logp, adv
    var_ret = torch.var(ret, unbiased=False)
    explained_variance = 1.0 - (torch.var(ret - value_pred, unbiased=False) / (var_ret + 1e-8))
    return {
        "loss_pi": loss.loss_pi.detach(),
        "loss_v": loss.loss_v.detach(),
        "approx_kl": loss.approx_kl.detach(),
        "clip_frac": loss.clip_frac.detach(),
        "value_clip_frac": loss.value_clip_frac.detach(),
        "ratio_mean": loss.ratio_mean.detach(),
        "adv_mean": loss.adv_mean.detach(),
        "explained_variance": explained_variance.detach(),
    }


class PPO(Algorithm):
    """PPO configuration/spec container for the learner runtime."""

    def __init__(
        self,
        *,
        value_net: torch.nn.Module,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ppo_epochs: int = 2,
        minibatch_size: int = 64,
        max_grad_norm: float = 0.5,
        grad_accum_steps: int = 1,
        ddp_seed: int = 0,
        eta: float = 1.0,
        use_distributed_sampler: bool = True,
        variant: str = "ppo",
        kl_coef: float = 0.0,
        dual_clip: float | None = None,
        value_clip_eps: float = 0.0,
        policy_lr: float = 1e-5,
        value_lr: float = 1e-4,
        weight_decay: float = 0.0,
        forward_kl_coef: float = 0.0,
        reverse_kl_coef: float = 0.0,
        distill_temperature: float = 1.0,
        teacher_ckpt: str | None = None,
    ):
        self.value_net = value_net
        self.clip_eps = float(clip_eps)
        self.vf_coef = float(vf_coef)
        self.ppo_epochs = int(ppo_epochs)
        self.minibatch_size = int(minibatch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.grad_accum_steps = int(grad_accum_steps)
        self.ddp_seed = int(ddp_seed)
        self.eta = float(eta)
        self.use_distributed_sampler = bool(use_distributed_sampler)
        self.variant = str(variant)
        self.kl_coef = float(kl_coef)
        self.dual_clip = None if dual_clip is None else float(dual_clip)
        self.value_clip_eps = float(value_clip_eps)
        self.policy_lr = float(policy_lr)
        self.value_lr = float(value_lr)
        self.weight_decay = float(weight_decay)
        self.forward_kl_coef = float(forward_kl_coef)
        self.reverse_kl_coef = float(reverse_kl_coef)
        self.distill_temperature = float(distill_temperature)
        self.teacher_ckpt = None if teacher_ckpt is None else str(teacher_ckpt)


__all__ = ["PPO", "TrajectoryPPOObjective", "compute_ppo_objective", "compute_ppo_metrics"]
