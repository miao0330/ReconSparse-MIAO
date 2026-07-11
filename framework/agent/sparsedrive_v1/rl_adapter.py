from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from framework.replay_schema import (
    get_policy_action_id,
    get_policy_model_inputs,
    make_policy_replay,
    set_env_plan,
    set_grpo_candidates,
    set_grpo_scorer_fields,
)
from framework.utils.nuscenes_token import resolve_sample_token


class SparseDriveRLAdapterMixin:
    @staticmethod
    def _detach_model_inputs(img: torch.Tensor, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "img": img.detach().cpu().clone(),
            "projection_mat": data["projection_mat"].detach().cpu().clone(),
            "image_wh": data["image_wh"].detach().cpu().clone(),
            "T_global": data["T_global"].detach().cpu().clone(),
            "T_global_inv": data["T_global_inv"].detach().cpu().clone(),
            "timestamp": data["timestamp"].detach().cpu().clone(),
            "ego_status": data["ego_status"].detach().cpu().clone(),
            "gt_ego_fut_cmd": data["gt_ego_fut_cmd"].detach().cpu().clone(),
            "img_metas": data["img_metas"],
        }

    @staticmethod
    def _cmd_idx_from_replay(replay: Dict[str, Any]) -> int:
        action_id = get_policy_action_id(replay)
        if "cmd_idx" not in action_id:
            raise RuntimeError("SparseDrive-v1 replay missing policy.action_id.cmd_idx")
        return int(action_id["cmd_idx"])

    @staticmethod
    def _mode_idx_from_replay(replay: Dict[str, Any]) -> int:
        action_id = get_policy_action_id(replay)
        if "mode_idx" not in action_id:
            raise RuntimeError("SparseDrive-v1 replay missing policy.action_id.mode_idx")
        return int(action_id["mode_idx"])

    @staticmethod
    def _update_replay_with_observation_metadata(replay: Dict[str, Any], observation: Dict[str, Any]) -> None:
        token = observation.get("sample_token", observation.get("token", None))
        if token is None:
            token = resolve_sample_token(
                scene_id=observation.get("scene_id", None),
                frame_idx=observation.get("frame_idx", None),
            )
        if token is not None:
            set_grpo_scorer_fields(replay, sample_token=str(token))
        try:
            set_grpo_scorer_fields(replay, scene_id=int(observation.get("scene_id", 0)))
        except Exception:
            pass
        try:
            set_grpo_scorer_fields(replay, frame_idx=int(observation.get("frame_idx", 0)))
        except Exception:
            pass
        try:
            debug = replay.setdefault("debug", {})
            if isinstance(debug, dict):
                debug["timestamp_s"] = float(observation.get("timestamp", 0.0))
        except Exception:
            pass

    def _policy_model_inputs_to_raw_data(self, model_inputs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        required = [
            "img",
            "projection_mat",
            "image_wh",
            "T_global",
            "T_global_inv",
            "timestamp",
            "ego_status",
            "gt_ego_fut_cmd",
            "img_metas",
        ]
        missing = [key for key in required if key not in model_inputs]
        if missing:
            raise RuntimeError(f"SparseDrive-v1 replay policy.model_inputs missing fields: {missing}")
        img = model_inputs["img"]
        if not torch.is_tensor(img):
            raise RuntimeError("SparseDrive-v1 replay policy.model_inputs.img must be a tensor")
        data = {
            "projection_mat": model_inputs["projection_mat"],
            "image_wh": model_inputs["image_wh"],
            "T_global": model_inputs["T_global"],
            "T_global_inv": model_inputs["T_global_inv"],
            "timestamp": model_inputs["timestamp"],
            "ego_status": model_inputs["ego_status"],
            "gt_ego_fut_cmd": model_inputs["gt_ego_fut_cmd"],
            "img_metas": model_inputs["img_metas"],
        }
        return img, data

    def _forward_raw_planning_from_replay(self, replay: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        model_inputs = dict(get_policy_model_inputs(replay))
        img, data = self._policy_model_inputs_to_raw_data(model_inputs)
        model_device = self.device
        img_dev = img.to(model_device)
        data_dev = self._to_model_metas(data, model_device)
        model = self._model.module if isinstance(self._model, DDP) else self._model
        model.eval()
        return self._forward_raw_planning(img=img_dev, data_dev=data_dev)

    # -------------------- sampling / replay / logp --------------------
    def sample_sparsedrive_with_replay(
        self,
        observation: Dict[str, Any],
        *,
        mode_idx: int = -1,
        mode_select: str = "sample",
        anchor_bank: Dict[str, Any] | None = None,
    ) -> Tuple[Tuple[Any, ...], torch.Tensor, Dict[str, Any]]:
        del anchor_bank
        img_t, data = self._build_batch(observation)

        model = self._model.module if isinstance(self._model, DDP) else self._model
        model_device = self.device

        img_t = img_t.to(model_device)
        data_dev = self._to_model_metas(data, model_device)

        model.eval()
        with torch.inference_mode():
            logits_all, traj_all = self._forward_raw_planning(img=img_t, data_dev=data_dev)
        logits_all = logits_all.detach().cpu()
        traj_all = traj_all.detach().cpu()

        cmd = data["gt_ego_fut_cmd"].detach().cpu().reshape(-1)
        if cmd.numel() >= 3:
            cmd_idx = int(torch.argmax(cmd[:3]).item())
        else:
            cmd_idx = 0

        logits = logits_all[cmd_idx].reshape(-1)
        logp_all = torch.log_softmax(logits, dim=0)

        if int(mode_idx) < 0:
            sel = str(mode_select).strip().lower()
            if sel in {"greedy", "max", "argmax"}:
                mi = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits, dim=0)
                if torch.isfinite(probs).all() and float(probs.sum().item()) > 0:
                    mi = int(torch.distributions.Categorical(probs).sample().item())
                else:
                    mi = int(torch.argmax(logits).item())
        else:
            mi = max(0, min(int(mode_idx), int(logits.shape[0]) - 1))

        lp = logp_all[mi]

        traj_xy = traj_all[cmd_idx, mi].reshape(-1, 2).numpy().astype(np.float32)
        traj_xyyaw = self._sparsedrive_lidar_xy_to_hugsim_action_xyyaw(traj_xy)
        traj_xyyaw_t = torch.from_numpy(traj_xyyaw).detach().cpu().clone()
        action, exec_debug = self._build_env_action(traj_xy=traj_xy)

        replay = {
            "schema_version": 2,
            "policy": make_policy_replay(
                backend="sparsedrive_v1",
                model_inputs=self._detach_model_inputs(img_t, data),
                action_id={"cmd_idx": int(cmd_idx), "mode_idx": int(mi)},
            ),
            "debug": {
                "final_planning": torch.from_numpy(traj_xy).detach().cpu().clone(),
                "traj_xyyaw": traj_xyyaw_t.clone(),
                "execute_mode": exec_debug.get("execute_mode", self._execute_mode),
                "candidate_scores": logits.detach().cpu().clone(),
            },
        }
        set_env_plan(replay, plan_xyyaw=traj_xyyaw_t.clone(), plan_frame="ego", action_flag=2)
        grpo_num_candidates = int(getattr(self, "_grpo_num_candidates", 0) or 0)
        if grpo_num_candidates > 0:
            strict_candidates = self._select_strict_grpo_old_policy_candidates(
                score_logits=logits.view(1, -1),
                candidate_trajs=traj_all[cmd_idx : cmd_idx + 1],
                num_candidates=grpo_num_candidates,
            )
            set_grpo_candidates(
                replay,
                mode_indices=strict_candidates["mode_indices"][0].detach().cpu().clone(),
                old_log_probs=strict_candidates["old_log_probs"][0].detach().cpu().clone(),
                traj_xyyaw=strict_candidates["traj_xyyaw"][0].detach().cpu().clone(),
                score_logits=strict_candidates["score_logits"][0].detach().cpu().clone(),
            )
        self._update_replay_with_observation_metadata(replay, observation)

        return action, lp, replay

    def sample_sparsedrive_with_replay_batch(
        self,
        observations: List[Dict[str, Any]],
        *,
        mode_idx: int = -1,
        mode_select: str = "sample",
        anchor_bank: Dict[str, Any] | None = None,
    ) -> Tuple[List[Tuple[Any, ...]], List[torch.Tensor], List[Dict[str, Any]]]:
        if len(observations) == 0:
            return [], [], []

        actions: List[Tuple[Any, ...]] = []
        logps: List[torch.Tensor] = []
        replays: List[Dict[str, Any]] = []
        for obs in observations:
            action, logp, replay = self.sample_sparsedrive_with_replay(
                obs,
                mode_idx=mode_idx,
                mode_select=mode_select,
                anchor_bank=anchor_bank,
            )
            actions.append(action)
            logps.append(logp)
            replays.append(replay)
        return actions, logps, replays

    def logp_from_replay(self, replay: Dict[str, Any], *, eta: float = 1.0) -> torch.Tensor:
        del eta
        mode_idx = self._mode_idx_from_replay(replay)
        cmd_idx = self._cmd_idx_from_replay(replay)
        logits_all, _traj_all = self._forward_raw_planning_from_replay(replay)
        logits = logits_all[cmd_idx].reshape(-1)
        return torch.log_softmax(logits, dim=0)[mode_idx]

    def logp_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
        *,
        eta: float = 1.0,
    ) -> torch.Tensor:
        del eta
        if len(replays) == 0:
            return torch.empty((0,), device=self.device, dtype=torch.float32)
        vals = [self.logp_from_replay(replay, eta=1.0).to(dtype=torch.float32).view(()) for replay in replays]
        return torch.stack(vals, dim=0)

    def replay_is_compatible(self, replay: Dict[str, Any]) -> bool:
        if not isinstance(replay, dict):
            return False
        try:
            model_inputs = get_policy_model_inputs(replay)
            action_id = get_policy_action_id(replay)
        except RuntimeError:
            return False
        return (
            torch.is_tensor(model_inputs.get("img", None))
            and "cmd_idx" in action_id
            and "mode_idx" in action_id
        )


__all__ = ["SparseDriveRLAdapterMixin"]
