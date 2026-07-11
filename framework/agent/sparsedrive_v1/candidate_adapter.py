from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

from reconsimulator.envs import nus_config as nus_cfg
from framework.replay_schema import get_grpo_scorer, require_grpo_candidate_field


def _as_score_tensor(scores: Any, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(scores):
        return scores.to(device=device, dtype=torch.float32)
    if isinstance(scores, (list, tuple)):
        return torch.as_tensor(scores, device=device, dtype=torch.float32)
    try:
        import numpy as np

        if isinstance(scores, np.ndarray):
            return torch.from_numpy(scores).to(device=device, dtype=torch.float32)
    except Exception:
        pass
    raise TypeError(f"Unsupported counterfactual score type: {type(scores)!r}")


class SparseDriveCandidateAdapterMixin:
    def _traj_candidates_to_xyyaw(self, candidate_trajs: torch.Tensor) -> torch.Tensor:
        if candidate_trajs.ndim != 3:
            raise RuntimeError(
                "SparseDrive-v1 candidate trajectories must have shape "
                f"(modes,horizon,dims); got {tuple(candidate_trajs.shape)}"
            )
        converted = [
            torch.from_numpy(
                self._sparsedrive_lidar_xy_to_hugsim_action_xyyaw(
                    candidate_trajs[idx].detach().cpu().numpy()
                )
            ).to(dtype=torch.float32)
            for idx in range(int(candidate_trajs.shape[0]))
        ]
        return torch.stack(converted, dim=0) if converted else torch.empty((0, 0, 3), dtype=torch.float32)

    def _select_counterfactual_candidates_from_policy_outputs(
        self,
        *,
        score_logits: torch.Tensor,
        candidate_trajs: torch.Tensor,
        num_candidates: int,
        candidate_select: str,
        logp_all: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        logits = score_logits.to(dtype=torch.float32)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.ndim != 2 or int(logits.shape[0]) != 1:
            raise RuntimeError(f"SparseDrive-v1 candidate logits must have shape (1,modes), got {tuple(logits.shape)}")
        if candidate_trajs.ndim == 3:
            trajs = candidate_trajs.unsqueeze(0)
        elif candidate_trajs.ndim == 4:
            trajs = candidate_trajs
        else:
            raise RuntimeError(
                "SparseDrive-v1 candidate trajectories must have shape "
                f"(modes,horizon,dims) or (1,modes,horizon,dims); got {tuple(candidate_trajs.shape)}"
            )
        if int(trajs.shape[0]) != 1 or int(trajs.shape[1]) != int(logits.shape[1]):
            raise RuntimeError(
                "SparseDrive-v1 candidate trajectory count must match logits; "
                f"trajs={tuple(trajs.shape)} logits={tuple(logits.shape)}"
            )

        k = max(1, min(int(num_candidates), int(logits.shape[1])))
        select = str(candidate_select).strip().lower()
        if select in {"topk", "all"}:
            selected_indices = torch.topk(logits, k=k, dim=1, largest=True, sorted=True).indices
        elif select in {"sample", "sample_old_policy", "old_policy"}:
            probs = torch.softmax(logits, dim=1)
            if torch.isfinite(probs).all() and torch.all(probs.sum(dim=1) > 0.0):
                selected_indices = torch.multinomial(probs, num_samples=k, replacement=False)
            else:
                selected_indices = torch.topk(logits, k=k, dim=1, largest=True, sorted=True).indices
        else:
            raise ValueError(
                f"Unsupported candidate_select={candidate_select!r}; expected 'topk', 'all', or 'sample_old_policy'"
            )

        gather_traj = selected_indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1,
            -1,
            int(trajs.shape[2]),
            int(trajs.shape[3]),
        )
        selected_logits = torch.gather(logits, dim=1, index=selected_indices)
        selected_trajs = torch.gather(trajs, dim=1, index=gather_traj)
        if logp_all is None:
            logp_all = torch.log_softmax(logits, dim=1)
        selected_log_probs = torch.gather(logp_all, dim=1, index=selected_indices)
        traj_xyyaw = self._traj_candidates_to_xyyaw(selected_trajs[0])

        return {
            "traj_xyyaw": traj_xyyaw.unsqueeze(0).detach(),
            "log_probs": selected_log_probs,
            "mode_indices": selected_indices.to(dtype=torch.long),
            "score_logits": selected_logits,
        }

    def _select_strict_grpo_old_policy_candidates(
        self,
        *,
        score_logits: torch.Tensor,
        candidate_trajs: torch.Tensor,
        num_candidates: int,
    ) -> Dict[str, torch.Tensor]:
        selected = self._select_counterfactual_candidates_from_policy_outputs(
            score_logits=score_logits,
            candidate_trajs=candidate_trajs,
            num_candidates=int(num_candidates),
            candidate_select="sample_old_policy",
        )
        selected["old_log_probs"] = selected["log_probs"].detach()
        return selected

    @staticmethod
    def _stored_grpo_candidate_mode_indices_from_replay(
        replays: Sequence[Dict[str, Any]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        values: List[torch.Tensor] = []
        expected_k: int | None = None
        for idx, replay in enumerate(replays):
            tensor = torch.as_tensor(require_grpo_candidate_field(replay, "mode_indices", index=idx), dtype=torch.long, device=device).view(-1)
            if expected_k is None:
                expected_k = int(tensor.numel())
            elif int(tensor.numel()) != expected_k:
                raise RuntimeError(
                    "SparseDrive-v1 GRPO candidate count mismatch across replay entries: "
                    f"expected={expected_k} got={int(tensor.numel())} index={idx}"
                )
            values.append(tensor)
        return torch.stack(values, dim=0) if values else torch.empty((0, 0), dtype=torch.long, device=device)

    @staticmethod
    def _stored_grpo_candidate_old_log_probs_from_replay(
        replays: Sequence[Dict[str, Any]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        values: List[torch.Tensor] = []
        expected_k: int | None = None
        for idx, replay in enumerate(replays):
            tensor = torch.as_tensor(require_grpo_candidate_field(replay, "old_log_probs", index=idx), dtype=torch.float32, device=device).view(-1)
            if expected_k is None:
                expected_k = int(tensor.numel())
            elif int(tensor.numel()) != expected_k:
                raise RuntimeError(
                    "SparseDrive-v1 GRPO old logp count mismatch across replay entries: "
                    f"expected={expected_k} got={int(tensor.numel())} index={idx}"
                )
            values.append(tensor)
        return torch.stack(values, dim=0) if values else torch.empty((0, 0), dtype=torch.float32, device=device)

    @staticmethod
    def _stored_grpo_candidate_traj_xyyaw_from_replay(
        replays: Sequence[Dict[str, Any]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        values: List[torch.Tensor] = []
        expected_shape: tuple[int, ...] | None = None
        for idx, replay in enumerate(replays):
            tensor = torch.as_tensor(require_grpo_candidate_field(replay, "traj_xyyaw", index=idx), dtype=torch.float32, device=device)
            if tensor.ndim != 3 or int(tensor.shape[-1]) < 3:
                raise RuntimeError(
                    "SparseDrive-v1 GRPO candidate trajectories must have shape (candidates,horizon,3+); "
                    f"got {tuple(tensor.shape)} at index {idx}"
                )
            tensor = tensor[..., :3]
            if expected_shape is None:
                expected_shape = tuple(int(v) for v in tensor.shape)
            elif tuple(int(v) for v in tensor.shape) != expected_shape:
                raise RuntimeError(
                    "SparseDrive-v1 GRPO candidate trajectory shape mismatch across replay entries: "
                    f"expected={expected_shape} got={tuple(tensor.shape)} index={idx}"
                )
            values.append(tensor)
        return torch.stack(values, dim=0) if values else torch.empty((0, 0, 0, 3), dtype=torch.float32, device=device)

    @staticmethod
    def _concat_counterfactual_candidate_batches(
        parts: Sequence[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        if len(parts) == 0:
            return {}
        keys = list(parts[0].keys())
        return {key: torch.cat([part[key] for part in parts], dim=0) for key in keys}

    def sample_counterfactual_trajectories_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
        *,
        num_candidates: int,
        candidate_select: str = "topk",
    ) -> Dict[str, torch.Tensor]:
        if len(replays) == 0:
            return {
                "traj_xyyaw": torch.empty((0, 0, 0, 3), dtype=torch.float32),
                "log_probs": torch.empty((0, 0), device=self.device, dtype=torch.float32),
                "mode_indices": torch.empty((0, 0), device=self.device, dtype=torch.long),
                "score_logits": torch.empty((0, 0), device=self.device, dtype=torch.float32),
            }

        parts: List[Dict[str, torch.Tensor]] = []
        for replay in replays:
            logits_all, traj_all = self._forward_raw_planning_from_replay(replay)
            cmd_idx = self._cmd_idx_from_replay(replay)
            parts.append(
                self._select_counterfactual_candidates_from_policy_outputs(
                    score_logits=logits_all[cmd_idx : cmd_idx + 1],
                    candidate_trajs=traj_all[cmd_idx : cmd_idx + 1],
                    num_candidates=int(num_candidates),
                    candidate_select=str(candidate_select),
                )
            )
        return self._concat_counterfactual_candidate_batches(parts)

    def replay_policy_outputs_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
        *,
        eta: float = 1.0,
        num_candidates: int,
        candidate_select: str = "topk",
    ) -> Dict[str, Any]:
        del eta
        if len(replays) == 0:
            empty = {
                "traj_xyyaw": torch.empty((0, 0, 0, 3), dtype=torch.float32),
                "log_probs": torch.empty((0, 0), device=self.device, dtype=torch.float32),
                "old_log_probs": torch.empty((0, 0), device=self.device, dtype=torch.float32),
                "mode_indices": torch.empty((0, 0), device=self.device, dtype=torch.long),
                "score_logits": torch.empty((0, 0), device=self.device, dtype=torch.float32),
            }
            return {"new_logp": torch.empty((0,), device=self.device, dtype=torch.float32), "counterfactual": empty}

        new_logps: List[torch.Tensor] = []
        candidate_parts: List[Dict[str, torch.Tensor]] = []
        for replay in replays:
            logits_all, traj_all = self._forward_raw_planning_from_replay(replay)
            cmd_idx = self._cmd_idx_from_replay(replay)
            mode_idx = self._mode_idx_from_replay(replay)
            logits = logits_all[cmd_idx : cmd_idx + 1].to(dtype=torch.float32)
            logp_all = torch.log_softmax(logits, dim=1)
            new_logps.append(logp_all[0, mode_idx])

            if isinstance(replay, dict) and isinstance(replay.get("grpo", {}).get("candidates", None), dict):
                mode_indices = self._stored_grpo_candidate_mode_indices_from_replay([replay], device=logits.device)
                if bool(((mode_indices < 0) | (mode_indices >= int(logits.shape[1]))).any().item()):
                    bad = mode_indices.detach().cpu().view(-1).tolist()
                    raise RuntimeError(
                        "SparseDrive-v1 stored GRPO candidate mode_idx is outside current planning modes; "
                        f"modes={int(logits.shape[1])} values={bad[:8]}"
                    )
                gather_idx = mode_indices
                candidate_part = {
                    "traj_xyyaw": self._stored_grpo_candidate_traj_xyyaw_from_replay([replay], device=logits.device),
                    "log_probs": torch.gather(logp_all, dim=1, index=gather_idx),
                    "old_log_probs": self._stored_grpo_candidate_old_log_probs_from_replay([replay], device=logits.device),
                    "mode_indices": mode_indices,
                    "score_logits": torch.gather(logits, dim=1, index=gather_idx),
                }
            else:
                candidate_part = self._select_counterfactual_candidates_from_policy_outputs(
                    score_logits=logits,
                    candidate_trajs=traj_all[cmd_idx : cmd_idx + 1],
                    num_candidates=int(num_candidates),
                    candidate_select=str(candidate_select),
                    logp_all=logp_all,
                )
                candidate_part["old_log_probs"] = candidate_part["log_probs"].detach()
            candidate_parts.append(candidate_part)

        return {
            "new_logp": torch.stack([val.to(dtype=torch.float32).view(()) for val in new_logps], dim=0),
            "counterfactual": self._concat_counterfactual_candidate_batches(candidate_parts),
        }

    def _counterfactual_scorer_backend_name(self) -> str:
        backend = str(getattr(self, "_nuscenes_scorer_config", {}).get("backend", "nuscenes_pdm")).strip().lower()
        if backend in {"craft", "craft_carl", "carl", "nuscenes_craft"}:
            return "craft_carl"
        return "nuscenes_pdm"

    def _ensure_nuscenes_pdm_scorer(self):
        if getattr(self, "_nuscenes_pdm_scorer", None) is None:
            from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

            scorer_kwargs = {
                key: value
                for key, value in getattr(self, "_nuscenes_scorer_config", {}).items()
                if key != "backend"
            }
            self._nuscenes_pdm_scorer = NuScenesPDMScorer(
                token2vad_path=nus_cfg.TOKEN2VAD_FILE,
                **scorer_kwargs,
            )
        return self._nuscenes_pdm_scorer

    def _ensure_nuscenes_craft_scorer(self):
        if getattr(self, "_nuscenes_craft_scorer", None) is None:
            from framework.algorithms.nuscenes_craft_scorer import NuScenesCraftScorer

            scorer_kwargs = {
                key: value
                for key, value in getattr(self, "_nuscenes_scorer_config", {}).items()
                if key != "backend"
            }
            self._nuscenes_craft_scorer = NuScenesCraftScorer(
                token2vad_path=nus_cfg.TOKEN2VAD_FILE,
                **scorer_kwargs,
            )
        return self._nuscenes_craft_scorer

    def _ensure_counterfactual_scorer_backend(self):
        if self._counterfactual_scorer_backend_name() == "craft_carl":
            return self._ensure_nuscenes_craft_scorer()
        return self._ensure_nuscenes_pdm_scorer()

    def pdm_score_counterfactuals_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
        traj_xyyaw: torch.Tensor,
    ) -> torch.Tensor:
        scorer = self._ensure_counterfactual_scorer_backend()
        scorer_replays = [dict(get_grpo_scorer(replay)) for replay in replays]
        return _as_score_tensor(scorer.score(scorer_replays, traj_xyyaw), device=self.device)

    def dump_counterfactual_debug_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
        traj_xyyaw: torch.Tensor,
        candidate_scores: torch.Tensor,
        *,
        out_dir: str,
        step_tag: str,
        top_k: int,
    ) -> None:
        del candidate_scores
        scorer = self._ensure_counterfactual_scorer_backend()
        scorer_replays = [dict(get_grpo_scorer(replay)) for replay in replays]
        scorer.dump_debug_artifacts(
            scorer_replays,
            traj_xyyaw,
            out_dir=out_dir,
            step_tag=step_tag,
            top_k=top_k,
        )


__all__ = ["SparseDriveCandidateAdapterMixin"]
