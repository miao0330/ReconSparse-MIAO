from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PolicyLoopGRPOObjective:
    loss: torch.Tensor
    advantages: torch.Tensor
    score_mean: torch.Tensor
    score_std: torch.Tensor
    score_min: torch.Tensor
    score_max: torch.Tensor
    approx_kl: torch.Tensor
    clip_frac: torch.Tensor
    ratio_mean: torch.Tensor
    active_groups: torch.Tensor
    active_candidates: torch.Tensor


def _masked_scalar_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def compute_policy_loop_grpo_objective(
    *,
    candidate_log_probs: torch.Tensor,
    old_candidate_log_probs: torch.Tensor,
    candidate_returns: torch.Tensor,
    valid_mask: torch.Tensor,
    score_norm_eps: float = 1.0e-6,
    clip_eps: float = 0.2,
) -> PolicyLoopGRPOObjective:
    tensors = {
        "candidate_log_probs": candidate_log_probs,
        "old_candidate_log_probs": old_candidate_log_probs,
        "candidate_returns": candidate_returns,
        "valid_mask": valid_mask,
    }
    if any(value.ndim != 2 for value in tensors.values()):
        shapes = {key: tuple(value.shape) for key, value in tensors.items()}
        raise ValueError(f"Policy-loop GRPO inputs must be 2D (batch, candidates); got {shapes}")
    expected_shape = tuple(candidate_log_probs.shape)
    for name, value in tensors.items():
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                "Policy-loop GRPO inputs must have identical shapes; "
                f"candidate_log_probs={expected_shape} {name}={tuple(value.shape)}"
            )
    if float(score_norm_eps) <= 0.0:
        raise ValueError("score_norm_eps must be > 0")
    if not 0.0 <= float(clip_eps) < 1.0:
        raise ValueError("clip_eps must be in [0, 1)")

    device = candidate_log_probs.device
    new_log_probs = candidate_log_probs.to(dtype=torch.float32)
    old_log_probs = old_candidate_log_probs.to(device=device, dtype=torch.float32)
    returns = candidate_returns.to(device=device, dtype=torch.float32)
    valid = valid_mask.to(device=device, dtype=torch.bool) & torch.isfinite(returns)
    valid = valid & torch.isfinite(new_log_probs) & torch.isfinite(old_log_probs)

    safe_returns = torch.where(valid, returns, torch.zeros_like(returns))
    valid_count = valid.sum(dim=1, keepdim=True)
    denom = valid_count.to(dtype=torch.float32).clamp_min(1.0)
    group_mean = safe_returns.sum(dim=1, keepdim=True) / denom
    centered = torch.where(valid, returns - group_mean, torch.zeros_like(returns))
    group_var = centered.square().sum(dim=1, keepdim=True) / denom
    group_std = torch.sqrt(group_var)
    advantages = torch.where(
        valid,
        centered / (group_std + float(score_norm_eps)),
        torch.zeros_like(centered),
    )

    active_group_mask = valid_count.view(-1) >= 2
    objective_mask = valid & active_group_mask[:, None]
    log_ratio = new_log_probs - old_log_probs
    #torch.where(condition, input, other)
    safe_log_ratio = torch.where(objective_mask, log_ratio, torch.zeros_like(log_ratio))
    ratio = torch.exp(safe_log_ratio)
    clipped_ratio = ratio.clamp(min=1.0 - float(clip_eps), max=1.0 + float(clip_eps))
    adv_detached = advantages.detach()
    surrogate = torch.minimum(ratio * adv_detached, clipped_ratio * adv_detached)

    if bool(objective_mask.any().item()):
        loss = -_masked_scalar_mean(surrogate, objective_mask)
        approx_kl = _masked_scalar_mean((ratio - 1.0) - safe_log_ratio, objective_mask)
        clip_frac = _masked_scalar_mean(
            ((ratio - 1.0).abs() > float(clip_eps)).to(dtype=torch.float32),
            objective_mask,
        )
        ratio_mean = _masked_scalar_mean(ratio, objective_mask)
    else:
        loss = new_log_probs.sum() * 0.0
        zero = loss.detach()
        approx_kl = zero
        clip_frac = zero
        ratio_mean = torch.ones((), device=device, dtype=torch.float32)

    valid_returns = returns[valid]
    if int(valid_returns.numel()) > 0:
        score_mean = valid_returns.mean()
        score_min = valid_returns.min()
        score_max = valid_returns.max()
        group_has_value = valid_count.view(-1) > 0
        score_std = group_std.view(-1)[group_has_value].mean()
    else:
        zero = loss.detach()
        score_mean = zero
        score_std = zero
        score_min = zero
        score_max = zero

    return PolicyLoopGRPOObjective(
        loss=loss,
        advantages=advantages,
        score_mean=score_mean,
        score_std=score_std,
        score_min=score_min,
        score_max=score_max,
        approx_kl=approx_kl,
        clip_frac=clip_frac,
        ratio_mean=ratio_mean,
        active_groups=active_group_mask.to(dtype=torch.float32).sum(),
        active_candidates=objective_mask.to(dtype=torch.float32).sum(),
    )


__all__ = ["PolicyLoopGRPOObjective", "compute_policy_loop_grpo_objective"]