from __future__ import annotations

from typing import Any

import torch


def compute_gae(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    terminated: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.ndim != 1 or dones.ndim != 1 or values.ndim != 1:
        raise ValueError("compute_gae expects 1D tensors (T,)")
    if rewards.shape[0] != dones.shape[0] or rewards.shape[0] != values.shape[0]:
        raise ValueError("compute_gae expects matching lengths")
    if terminated is None:
        terminated = dones
    if terminated.ndim != 1 or terminated.shape[0] != rewards.shape[0]:
        raise ValueError("terminated must be a 1D tensor matching rewards")

    rewards = rewards.to(dtype=torch.float32)
    dones = dones.to(device=rewards.device, dtype=torch.float32)
    values = values.to(device=rewards.device, dtype=torch.float32)
    terminated = terminated.to(device=rewards.device, dtype=torch.float32)
    last_value = last_value.to(device=rewards.device, dtype=torch.float32).view(())

    horizon = int(rewards.shape[0])
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros((), device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(horizon)):
        not_done = 1.0 - dones[t]
        not_terminated = 1.0 - terminated[t]
        next_value = last_value if t == (horizon - 1) else values[t + 1]
        delta = rewards[t] + float(gamma) * next_value * not_terminated - values[t]
        last_gae = delta + float(gamma) * float(gae_lambda) * not_done * last_gae
        adv[t] = last_gae

    ret = adv + values
    return adv, ret


def compute_returns(*, rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    if rewards.ndim != 1 or dones.ndim != 1:
        raise ValueError("compute_returns expects 1D tensors (T,)")
    if rewards.shape[0] != dones.shape[0]:
        raise ValueError("compute_returns expects matching lengths")

    rewards = rewards.to(dtype=torch.float32)
    dones = dones.to(device=rewards.device, dtype=torch.float32)
    ret = torch.zeros_like(rewards)
    running = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(int(rewards.shape[0]))):
        running = rewards[t] + float(gamma) * running * (1.0 - dones[t])
        ret[t] = running
    return ret


def normalize_advantages(
    adv: torch.Tensor,
    *,
    ddp_enabled: bool,
    dist_module: Any,
    device: torch.device,
    eps: float = 1e-8,
) -> torch.Tensor:
    if adv.numel() == 0:
        return adv

    if ddp_enabled and getattr(dist_module, "is_initialized", lambda: False)():
        stats = torch.stack(
            [
                adv.sum(),
                (adv * adv).sum(),
                torch.tensor(float(adv.numel()), device=device, dtype=adv.dtype),
            ],
            dim=0,
        )
        dist_module.all_reduce(stats, op=dist_module.ReduceOp.SUM)
        mean = stats[0] / stats[2]
        var = (stats[1] / stats[2]) - (mean * mean)
        std = torch.sqrt(torch.clamp(var, min=0.0) + float(eps))
        return (adv - mean) / std

    return (adv - adv.mean()) / (adv.std(unbiased=False) + float(eps))


__all__ = ["compute_gae", "compute_returns", "normalize_advantages"]
