from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def _recompute_pose_yaw_from_xy(traj_xyyaw: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(traj_xyyaw):
        raise TypeError("traj_xyyaw must be a torch.Tensor")
    if traj_xyyaw.shape[-1] < 2:
        raise ValueError(f"traj_xyyaw last dim must be >= 2, got shape={tuple(traj_xyyaw.shape)}")

    out = traj_xyyaw.clone()
    xy = out[..., :2]
    origin = torch.zeros_like(xy[..., :1, :])
    prev_xy = torch.cat([origin, xy[..., :-1, :]], dim=-2)
    delta = xy - prev_xy

    yaw = torch.atan2(delta[..., 1], delta[..., 0])
    if out.shape[-1] >= 3:
        base_yaw = out[..., 2]
    else:
        base_yaw = torch.zeros_like(yaw)
        out = torch.cat([out, base_yaw.unsqueeze(-1)], dim=-1)

    valid = delta.norm(dim=-1) > 1e-6
    out[..., 2] = torch.where(valid, yaw, base_yaw)
    return out


class DiffusionDriveV2RLAdapterMixin:
    ###########################################
    # PPO 采样（无梯度 + 保存 replay）
    ###########################################
    def sample_ddv2rl_with_replay(
        self,
        observation: Dict[str, np.ndarray],
        *,
        eta: float = 1.0,
        mode_idx: int = 0,
        mode_select: str = "greedy",
    ) -> Tuple[Tuple[Any, ...], torch.Tensor, Dict[str, Any]]:
        """Like step_ddv2rl(), but also returns replay info for PPO."""
        if self._agent is None:
            raise RuntimeError("Agent not initialized")

        camera_feature = self._build_camera_feature(observation)
        lidar_feature = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
        status_feature = self._build_status_feature(observation)

        model_device = next(self._agent.parameters()).device if hasattr(self._agent, "parameters") else torch.device("cpu")
        features = {
            "camera_feature": camera_feature.to(model_device),
            "lidar_feature": lidar_feature.to(model_device),
            "status_feature": status_feature.to(model_device),
        }

        self._agent._transfuser_model.eval()
        with torch.inference_mode():
            pred = self._agent._transfuser_model(
                features,
                targets=None,
                eta=float(eta),
                metric_cache=None,
                cal_pdm=False,
                token=None,
            )
        traj = pred.get("trajectory", None)
        log_probs = pred.get("log_probs", None)
        diffusion_chain = pred.get("all_diffusion_output", None)
        if traj is None or log_probs is None or diffusion_chain is None:
            raise RuntimeError("DDV2-RL model did not return trajectory/log_probs/all_diffusion_output")

        traj0 = traj[0]
        if log_probs.dim() == 3:
            mode_logps = log_probs[0].sum(dim=-1)
        elif log_probs.dim() == 2:
            mode_logps = log_probs.sum(dim=-1)
        else:
            mode_logps = log_probs.reshape(-1, log_probs.shape[-1]).sum(dim=-1)

        if int(mode_idx) < 0:
            sel = str(mode_select).strip().lower()
            if sel in {"greedy", "max", "argmax"}:
                mi = int(torch.argmax(mode_logps).item())
            else:
                temperature = 1.0
                probs = torch.softmax(mode_logps / max(1e-6, temperature), dim=0)
                if torch.isfinite(probs).all() and float(probs.sum().item()) > 0:
                    mi = int(torch.distributions.Categorical(probs).sample().item())
                else:
                    mi = int(torch.argmax(mode_logps).item())
        else:
            mi = int(mode_idx)
            mi = max(0, min(mi, int(traj0.shape[0]) - 1))

        traj_sel = traj0[mi]
        action, exec_debug = self._build_env_action_from_traj(traj_sel)

        lp = mode_logps[mi]
        replay = {
            "camera_feature": camera_feature.detach().cpu().clone(),
            "status_feature": status_feature.detach().cpu().clone(),
            "diffusion_chain": diffusion_chain.detach().cpu().clone(),
            "mode_idx": mi,
            "traj_xyyaw_raw": traj0[mi].detach().cpu().clone(),
            "traj_xyyaw": traj_sel.detach().cpu().clone(),
            "first_step_xyyaw": exec_debug.get("first_step_xyyaw"),
            "exec_mode": exec_debug.get("execute_mode", self._execute_mode),
        }
        return action, lp, replay

    def sample_ddv2rl_with_replay_batch(
        self,
        observations: List[Dict[str, np.ndarray]],
        *,
        eta: float = 1.0,
        mode_idx: int = 0,
        mode_select: str = "sample",
    ) -> Tuple[List[Tuple[Any, ...]], List[torch.Tensor], List[Dict[str, Any]]]:
        """Batched variant of sample_ddv2rl_with_replay() for vectorized env rollout."""
        if self._agent is None:
            raise RuntimeError("Agent not initialized")
        if len(observations) == 0:
            return [], [], []

        camera_feature = torch.cat([self._build_camera_feature(obs) for obs in observations], dim=0)
        bsz = int(camera_feature.shape[0])

        lidar_feature = torch.zeros((bsz, 1, 256, 256), dtype=torch.float32)
        status_feature = torch.cat([self._build_status_feature(obs) for obs in observations], dim=0)

        model_device = next(self._agent.parameters()).device if hasattr(self._agent, "parameters") else torch.device("cpu")
        features = {
            "camera_feature": camera_feature.to(model_device),
            "lidar_feature": lidar_feature.to(model_device),
            "status_feature": status_feature.to(model_device),
        }

        self._agent._transfuser_model.eval()
        with torch.inference_mode():
            pred = self._agent._transfuser_model(
                features,
                targets=None,
                eta=float(eta),
                metric_cache=None,
                cal_pdm=False,
                token=None,
            )
        traj = pred.get("trajectory", None)
        log_probs = pred.get("log_probs", None)
        diffusion_chain = pred.get("all_diffusion_output", None)
        if traj is None or log_probs is None or diffusion_chain is None:
            raise RuntimeError("DDV2-RL model did not return trajectory/log_probs/all_diffusion_output")

        if log_probs.dim() == 3:
            mode_logps = log_probs.sum(dim=-1)
        elif log_probs.dim() == 2:
            mode_logps = log_probs
        else:
            mode_logps = log_probs.reshape(log_probs.shape[0], -1, log_probs.shape[-1]).sum(dim=-1)

        sel = str(mode_select).strip().lower()
        mode_idx_list: List[int] = []
        if int(mode_idx) < 0:
            if sel in {"greedy", "max", "argmax"}:
                mode_idx_list = [int(torch.argmax(mode_logps[b]).item()) for b in range(bsz)]
            else:
                temperature = 1.0
                for b in range(bsz):
                    probs = torch.softmax(mode_logps[b] / max(1e-6, temperature), dim=0)
                    if torch.isfinite(probs).all() and float(probs.sum().item()) > 0:
                        mode_idx_list.append(int(torch.distributions.Categorical(probs).sample().item()))
                    else:
                        mode_idx_list.append(int(torch.argmax(mode_logps[b]).item()))
        else:
            n_modes = int(traj.shape[1])
            mi = max(0, min(int(mode_idx), n_modes - 1))
            mode_idx_list = [int(mi) for _ in range(bsz)]

        actions: List[Tuple[Any, ...]] = []
        logps: List[torch.Tensor] = []
        replays: List[Dict[str, Any]] = []

        for b in range(bsz):
            mi = int(mode_idx_list[b])
            traj_sel = _recompute_pose_yaw_from_xy(traj[b, mi])
            action, exec_debug = self._build_env_action_from_traj(traj_sel)
            actions.append(action)
            logps.append(mode_logps[b, mi])
            replays.append(
                {
                    "camera_feature": camera_feature[b : b + 1].detach().cpu().clone(),
                    "status_feature": status_feature[b : b + 1].detach().cpu().clone(),
                    "diffusion_chain": diffusion_chain[b : b + 1].detach().cpu().clone(),
                    "mode_idx": mi,
                    "traj_xyyaw_raw": traj[b, mi].detach().cpu().clone(),
                    "traj_xyyaw": traj_sel.detach().cpu().clone(),
                    "first_step_xyyaw": exec_debug.get("first_step_xyyaw"),
                    "exec_mode": exec_debug.get("execute_mode", self._execute_mode),
                }
            )

        return actions, logps, replays

    def logp_from_replay(self, replay: Dict[str, Any], *, eta: float = 1.0) -> torch.Tensor:
        """Recompute logp for the stored diffusion chain under current params."""
        if self._agent is None:
            raise RuntimeError("Agent not initialized")
        if not hasattr(self._agent, "_transfuser_model"):
            raise RuntimeError("Unexpected DDV2 agent type")
        if not hasattr(self._agent._transfuser_model, "compute_log_probs_from_diffusion_chain"):
            raise RuntimeError("DDV2 model does not expose compute_log_probs_from_diffusion_chain")

        camera_feature = replay["camera_feature"]
        diffusion_chain = replay["diffusion_chain"]
        mode_idx = int(replay.get("mode_idx", 0))

        lidar_feature = torch.zeros((camera_feature.shape[0], 1, 256, 256), dtype=torch.float32)
        status_feature = replay.get("status_feature", None)
        if status_feature is None:
            status_feature = torch.zeros((camera_feature.shape[0], 8), dtype=torch.float32)
        elif not torch.is_tensor(status_feature):
            status_feature = torch.as_tensor(status_feature, dtype=torch.float32)
        if status_feature.ndim == 1:
            status_feature = status_feature.view(1, -1)
        status_feature = status_feature.to(dtype=torch.float32)

        model_device = next(self._agent.parameters()).device if hasattr(self._agent, "parameters") else torch.device("cpu")
        features = {
            "camera_feature": camera_feature.to(model_device),
            "lidar_feature": lidar_feature.to(model_device),
            "status_feature": status_feature.to(model_device),
        }

        chain = diffusion_chain.to(model_device)
        self._agent._transfuser_model.eval()
        all_log_probs = self._agent._transfuser_model.compute_log_probs_from_diffusion_chain(
            features,
            chain,
            eta=float(eta),
        )
        return all_log_probs[0, mode_idx].sum()

    def reinforce_update(self, logp: torch.Tensor, reward: float) -> Dict[str, float]:
        """One-step REINFORCE update for DDV2-RL."""
        if self._ddv2_optimizer is None:
            raise RuntimeError("DDV2 optimizer not initialized (no trainable params?)")
        r = float(reward)
        self._reward_baseline = self._baseline_beta * self._reward_baseline + (1.0 - self._baseline_beta) * r
        adv = r - self._reward_baseline
        loss = -(float(adv) * logp)
        self._ddv2_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._ddv2_optimizer.step()

        return {
            "loss_reinforce": float(loss.detach().cpu().item()),
            "baseline": float(self._reward_baseline),
            "adv": float(adv),
        }


__all__ = ["DiffusionDriveV2RLAdapterMixin"]
