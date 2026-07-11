from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TrajectoryGRPOObjective:
    loss: torch.Tensor
    advantages: torch.Tensor
    score_mean: torch.Tensor
    score_std: torch.Tensor
    score_min: torch.Tensor
    score_max: torch.Tensor
    approx_kl: torch.Tensor
    clip_frac: torch.Tensor
    ratio_mean: torch.Tensor


@dataclass
class TrajectoryRiskDecelAuxiliaryObjective:
    loss: torch.Tensor
    active_count: torch.Tensor
    decel_prob_mean: torch.Tensor
    accel_prob_mean: torch.Tensor


def compute_grpo_objective(
    *,
    candidate_log_probs: torch.Tensor,
    old_candidate_log_probs: torch.Tensor | None = None,
    candidate_scores: torch.Tensor,
    candidate_score_logits: torch.Tensor | None = None,
    score_norm_eps: float = 1e-6,
    use_rank_adv: bool = False,
    score_clip: float | None = None,
    objective: str = "grpo",
    temperature: float = 1.0,
    clip_eps: float = 0.2,
) -> TrajectoryGRPOObjective:
    if candidate_log_probs.ndim != 2 or candidate_scores.ndim != 2:
        raise ValueError("candidate_log_probs and candidate_scores must both be 2D tensors (batch, candidates)")
    if tuple(candidate_log_probs.shape) != tuple(candidate_scores.shape):
        raise ValueError(
            "candidate_log_probs and candidate_scores must have identical shapes; "
            f"got log_probs={tuple(candidate_log_probs.shape)} scores={tuple(candidate_scores.shape)}"
        )

    scores = candidate_scores.to(device=candidate_log_probs.device, dtype=torch.float32)
    if score_clip is not None:
        scores = scores.clamp(min=-float(score_clip), max=float(score_clip))

    if use_rank_adv:
        order = torch.argsort(scores, dim=1, descending=False)
        ranks = torch.argsort(order, dim=1, descending=False).to(dtype=torch.float32)
        denom = torch.clamp(torch.as_tensor(scores.shape[1] - 1, device=scores.device, dtype=torch.float32), min=1.0)
        advantages = (ranks / denom) - 0.5
    else:
        score_mean = scores.mean(dim=1, keepdim=True)
        score_std = scores.std(dim=1, keepdim=True, unbiased=False)
        advantages = (scores - score_mean) / (score_std + float(score_norm_eps))

    del candidate_score_logits, temperature
    objective_key = str(objective).strip().lower()
    if objective_key != "grpo":
        raise ValueError(f"Unsupported GRPO objective={objective!r}; expected 'grpo'")
    if old_candidate_log_probs is None:
        raise ValueError("objective='grpo' requires old_candidate_log_probs")
    old_log_probs = old_candidate_log_probs.to(device=candidate_log_probs.device, dtype=torch.float32)
    if tuple(old_log_probs.shape) != tuple(candidate_log_probs.shape):
        raise ValueError(
            "old_candidate_log_probs must match candidate_log_probs shape for GRPO; "
            f"got old={tuple(old_log_probs.shape)} new={tuple(candidate_log_probs.shape)}"
        )
    log_ratio = candidate_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    adv_detached = advantages.detach()
    unclipped = ratio * adv_detached
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    clipped = clipped_ratio * adv_detached
    loss = -torch.min(unclipped, clipped).mean()
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_frac = ((ratio - 1.0).abs() > float(clip_eps)).to(dtype=torch.float32).mean()
    ratio_mean = ratio.mean()
    score_mean = scores.mean(dim=1)
    score_std = scores.std(dim=1, unbiased=False)
    score_min = scores.min(dim=1).values
    score_max = scores.max(dim=1).values
    return TrajectoryGRPOObjective(
        loss=loss,
        advantages=advantages,
        score_mean=score_mean.mean(),
        score_std=score_std.mean(),
        score_min=score_min.mean(),
        score_max=score_max.mean(),
        approx_kl=approx_kl,
        clip_frac=clip_frac,
        ratio_mean=ratio_mean,
    )


def compute_risk_decel_auxiliary_objective(
    *,
    candidate_score_logits: torch.Tensor,
    candidate_traj_xyyaw: torch.Tensor,
    high_risk_mask: torch.Tensor,
    ego_speed_mps: torch.Tensor | None = None,
    dt_s: float = 0.5,
    speed_margin_mps: float = 0.1,
    eps: float = 1.0e-6,
) -> TrajectoryRiskDecelAuxiliaryObjective:
    if candidate_score_logits.ndim != 2:
        raise ValueError(
            "candidate_score_logits must be a 2D tensor (batch, candidates); "
            f"got {tuple(candidate_score_logits.shape)}"
        )
    if candidate_traj_xyyaw.ndim != 4 or int(candidate_traj_xyyaw.shape[-1]) < 2:
        raise ValueError(
            "candidate_traj_xyyaw must have shape (batch, candidates, horizon, dims>=2); "
            f"got {tuple(candidate_traj_xyyaw.shape)}"
        )
    if tuple(candidate_score_logits.shape[:2]) != tuple(candidate_traj_xyyaw.shape[:2]):
        raise ValueError(
            "candidate_score_logits and candidate_traj_xyyaw must agree on batch/candidate dimensions; "
            f"got logits={tuple(candidate_score_logits.shape)} traj={tuple(candidate_traj_xyyaw.shape)}"
        )

    device = candidate_score_logits.device
    dtype = torch.float32
    logits = candidate_score_logits.to(device=device, dtype=dtype)
    traj = candidate_traj_xyyaw.to(device=device, dtype=dtype)
    high_risk = high_risk_mask.to(device=device, dtype=torch.bool).view(-1)
    if int(high_risk.numel()) != int(logits.shape[0]):
        raise ValueError(
            "high_risk_mask must match batch size; "
            f"got mask={int(high_risk.numel())} batch={int(logits.shape[0])}"
        )

    if int(logits.numel()) == 0 or int(traj.shape[2]) <= 0:
        zero = torch.zeros((), device=device, dtype=dtype)
        return TrajectoryRiskDecelAuxiliaryObjective(
            loss=zero,
            active_count=zero,
            decel_prob_mean=zero,
            accel_prob_mean=zero,
        )

    step0_xy = traj[:, :, 0, :2]
    first_window_speed = torch.linalg.norm(step0_xy, dim=-1) / max(1.0e-6, float(dt_s))
    if ego_speed_mps is None:
        if int(traj.shape[2]) > 1:
            step1_speed = torch.linalg.norm(traj[:, :, 1, :2] - step0_xy, dim=-1) / max(1.0e-6, float(dt_s))
            reference_speed = step1_speed
        else:
            reference_speed = first_window_speed.mean(dim=1, keepdim=True).expand_as(first_window_speed)
    else:
        ego_speed = ego_speed_mps.to(device=device, dtype=dtype).view(-1)
        if int(ego_speed.numel()) != int(logits.shape[0]):
            raise ValueError(
                "ego_speed_mps must match batch size; "
                f"got speed={int(ego_speed.numel())} batch={int(logits.shape[0])}"
            )
        reference_speed = ego_speed[:, None].expand_as(first_window_speed)

    margin = max(0.0, float(speed_margin_mps))
    decel_mask = first_window_speed <= (reference_speed - margin)
    accel_mask = first_window_speed >= (reference_speed + margin)
    active = high_risk & decel_mask.any(dim=1)

    probs = torch.softmax(logits, dim=1)
    decel_prob = (probs * decel_mask.to(dtype=dtype)).sum(dim=1)
    accel_prob = (probs * accel_mask.to(dtype=dtype)).sum(dim=1)

    if not bool(active.any()):
        zero = logits.sum() * 0.0
        return TrajectoryRiskDecelAuxiliaryObjective(
            loss=zero,
            active_count=torch.zeros((), device=device, dtype=dtype),
            decel_prob_mean=zero.detach(),
            accel_prob_mean=zero.detach(),
        )

    active_decel_prob = decel_prob[active].clamp(min=float(eps), max=1.0)
    active_accel_prob = accel_prob[active].clamp(min=0.0, max=1.0 - float(eps))
    loss_terms = -torch.log(active_decel_prob) - torch.log1p(-active_accel_prob)
    loss = loss_terms.mean()
    return TrajectoryRiskDecelAuxiliaryObjective(
        loss=loss,
        active_count=active.to(dtype=dtype).sum(),
        decel_prob_mean=active_decel_prob.detach().mean(),
        accel_prob_mean=active_accel_prob.detach().mean(),
    )


__all__ = [
    "TrajectoryGRPOObjective",
    "TrajectoryRiskDecelAuxiliaryObjective",
    "compute_grpo_objective",
    "compute_risk_decel_auxiliary_objective",
]
