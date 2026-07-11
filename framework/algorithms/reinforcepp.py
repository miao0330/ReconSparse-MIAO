from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .base import Algorithm


@dataclass
class TrajectoryReinforceObjective:
    loss: torch.Tensor
    loss_pi: torch.Tensor
    approx_kl: torch.Tensor
    clip_frac: torch.Tensor
    ratio_mean: torch.Tensor
    adv_mean: torch.Tensor


def compute_reinforce_objective(
    *,
    new_logp: torch.Tensor,
    old_logp: Optional[torch.Tensor],
    adv: torch.Tensor,
    clip_eps: float,
    kl_coef: float = 0.0,
) -> TrajectoryReinforceObjective:
    if old_logp is not None and int(old_logp.numel()) == int(adv.numel()):
        log_ratio = new_logp - old_logp
        ratio = torch.exp(log_ratio)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps)) * adv
        loss_pi = -torch.min(surr1, surr2).mean()
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_frac = ((ratio - 1.0).abs() > float(clip_eps)).to(dtype=torch.float32).mean()
        ratio_mean = ratio.mean()
    else:
        loss_pi = -(adv * new_logp).mean()
        approx_kl = torch.zeros((), device=adv.device, dtype=torch.float32)
        clip_frac = torch.zeros((), device=adv.device, dtype=torch.float32)
        ratio_mean = torch.ones((), device=adv.device, dtype=torch.float32)

    loss = loss_pi + float(kl_coef) * approx_kl
    return TrajectoryReinforceObjective(
        loss=loss,
        loss_pi=loss_pi,
        approx_kl=approx_kl,
        clip_frac=clip_frac,
        ratio_mean=ratio_mean,
        adv_mean=adv.mean(),
    )


def compute_reinforce_metrics(
    *,
    new_logp: torch.Tensor,
    old_logp: Optional[torch.Tensor],
    adv: torch.Tensor,
    loss: TrajectoryReinforceObjective,
) -> Dict[str, torch.Tensor]:
    del new_logp, old_logp, adv
    return {
        "loss_pi": loss.loss_pi.detach(),
        "approx_kl": loss.approx_kl.detach(),
        "clip_frac": loss.clip_frac.detach(),
        "ratio_mean": loss.ratio_mean.detach(),
        "adv_mean": loss.adv_mean.detach(),
    }


class ReinforcePP(Algorithm):
    """ReinforcePP configuration/spec container for the learner runtime."""

    def __init__(
        self,
        *,
        clip_eps: float = 0.2,
        kl_coef: float = 0.0,
        epochs: int = 1,
        minibatch_size: int = 64,
        max_grad_norm: float = 0.5,
        grad_accum_steps: int = 1,
        ddp_seed: int = 0,
        eta: float = 1.0,
        use_distributed_sampler: bool = True,
        variant: str = "reinforcepp",
        policy_lr: float = 1e-5,
        weight_decay: float = 0.0,
        forward_kl_coef: float = 0.0,
        reverse_kl_coef: float = 0.0,
        distill_temperature: float = 1.0,
        teacher_ckpt: str | None = None,
        grpo_coef: float = 0.0,
        grpo_num_candidates: int = 0,
        grpo_candidate_select: str = "topk",
        grpo_norm_eps: float = 1e-6,
        grpo_use_rank_adv: bool = False,
        grpo_score_clip: float | None = None,
        grpo_debug_visualize: bool = False,
        grpo_debug_dir: str | None = None,
        grpo_debug_max_batches: int = 0,
        grpo_debug_top_k: int = 4,
    ):
        self.clip_eps = float(clip_eps)
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
        self.grpo_coef = float(grpo_coef)
        self.grpo_num_candidates = int(grpo_num_candidates)
        self.grpo_candidate_select = str(grpo_candidate_select)
        self.grpo_norm_eps = float(grpo_norm_eps)
        self.grpo_use_rank_adv = bool(grpo_use_rank_adv)
        self.grpo_score_clip = None if grpo_score_clip is None else float(grpo_score_clip)
        self.grpo_debug_visualize = bool(grpo_debug_visualize)
        self.grpo_debug_dir = None if grpo_debug_dir is None else str(grpo_debug_dir)
        self.grpo_debug_max_batches = int(grpo_debug_max_batches)
        self.grpo_debug_top_k = int(grpo_debug_top_k)


__all__ = [
    "ReinforcePP",
    "TrajectoryReinforceObjective",
    "compute_reinforce_objective",
    "compute_reinforce_metrics",
]
