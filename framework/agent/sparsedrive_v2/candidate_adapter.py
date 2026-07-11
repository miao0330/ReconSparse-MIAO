from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

from reconsimulator.envs import nus_config as nus_cfg
from framework.replay_schema import get_grpo_scorer, get_policy_action_id, require_grpo_candidate_field


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


def _traj_batch_to_xyyaw(trajs: torch.Tensor) -> torch.Tensor:
    arr = trajs.detach().cpu().to(dtype=torch.float32)
    if arr.ndim != 3 or arr.shape[1] < 1:
        raise RuntimeError(f"Invalid batched trajectory output shape: {tuple(arr.shape)}")
    if arr.shape[2] >= 3:
        return arr[:, :, :3].clone()

    xy = arr[:, :, :2]
    out = torch.zeros((xy.shape[0], xy.shape[1], 3), dtype=torch.float32)
    out[:, :, :2] = xy
    prev = torch.cat([torch.zeros((xy.shape[0], 1, 2), dtype=torch.float32), xy[:, :-1, :]], dim=1)
    dxy = xy - prev
    out[:, :, 2] = torch.atan2(dxy[:, :, 1], dxy[:, :, 0])
    return out


class SparseDriveV2CandidateAdapterMixin:
    def _select_counterfactual_candidates_from_policy_outputs(
        self,
        *,
        score_logits: torch.Tensor,
        candidate_trajs: torch.Tensor,
        num_candidates: int,
        candidate_select: str,
        logp_all: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if candidate_trajs.ndim != 4:
            raise RuntimeError(
                "SparseDriveV2 counterfactual candidates require candidate trajectories with shape "
                f"(batch, modes, horizon, dims); got {tuple(candidate_trajs.shape)}"
            )

        batch_size, num_modes, horizon, traj_dim = tuple(candidate_trajs.shape)
        del batch_size, horizon, traj_dim
        k = max(1, min(int(num_candidates), int(num_modes)))
        select = str(candidate_select).strip().lower()
        if select not in {"topk", "all"}:
            raise ValueError(f"Unsupported candidate_select={candidate_select!r}; expected 'topk' or 'all'")

        if select == "all":
            selected_indices = torch.argsort(score_logits, dim=1, descending=True)[:, :k]
        else:
            selected_indices = torch.topk(score_logits, k=k, dim=1, largest=True, sorted=True).indices

        gather_idx_scores = selected_indices
        gather_idx_traj = selected_indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1,
            -1,
            candidate_trajs.shape[2],
            candidate_trajs.shape[3],
        )
        selected_score_logits = torch.gather(score_logits, dim=1, index=gather_idx_scores)
        selected_trajs = torch.gather(candidate_trajs, dim=1, index=gather_idx_traj)

        if logp_all is None:
            logp_all = torch.log_softmax(score_logits, dim=1)
        selected_log_probs = torch.gather(logp_all, dim=1, index=gather_idx_scores)

        selected_trajs_flat = selected_trajs.reshape(-1, selected_trajs.shape[2], selected_trajs.shape[3])
        traj_xyyaw = self._traj_batch_to_xyyaw(selected_trajs_flat).reshape(
            selected_trajs.shape[0],
            selected_trajs.shape[1],
            selected_trajs.shape[2],
            3,
        )

        return {
            "traj_xyyaw": traj_xyyaw.detach(),
            "log_probs": selected_log_probs,
            "mode_indices": selected_indices.to(dtype=torch.long),
            "score_logits": selected_score_logits,
        }

    @staticmethod
    def _select_strict_grpo_old_policy_candidates(
        *,
        score_logits: torch.Tensor,
        candidate_trajs: torch.Tensor,
        candidate_global_indices: torch.Tensor,
        num_candidates: int,
    ) -> Dict[str, torch.Tensor]:
        if score_logits.ndim != 2:
            raise RuntimeError(f"score_logits must be 2D (batch,candidates), got {tuple(score_logits.shape)}")
        if candidate_trajs.ndim != 4:
            raise RuntimeError(f"candidate_trajs must be 4D (batch,candidates,horizon,dims), got {tuple(candidate_trajs.shape)}")
        if tuple(candidate_global_indices.shape) != tuple(score_logits.shape):
            raise RuntimeError(
                "candidate_global_indices must match score_logits shape; "
                f"indices={tuple(candidate_global_indices.shape)} logits={tuple(score_logits.shape)}"
            )
        k = max(1, min(int(num_candidates), int(score_logits.shape[1])))
        probs = torch.softmax(score_logits, dim=1)
        local_indices = torch.multinomial(probs, num_samples=k, replacement=False)
        gather_idx_scores = local_indices
        gather_idx_traj = local_indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1,
            -1,
            int(candidate_trajs.shape[2]),
            int(candidate_trajs.shape[3]),
        )
        selected_trajs = torch.gather(candidate_trajs, dim=1, index=gather_idx_traj)
        selected_logits = torch.gather(score_logits, dim=1, index=gather_idx_scores)
        selected_global_indices = torch.gather(
            candidate_global_indices.to(device=score_logits.device, dtype=torch.long),
            dim=1,
            index=gather_idx_scores,
        )
        old_log_probs = torch.gather(torch.log_softmax(score_logits, dim=1), dim=1, index=gather_idx_scores)
        selected_trajs_flat = selected_trajs.reshape(-1, selected_trajs.shape[2], selected_trajs.shape[3])
        traj_xyyaw = _traj_batch_to_xyyaw(selected_trajs_flat).reshape(
            selected_trajs.shape[0],
            selected_trajs.shape[1],
            selected_trajs.shape[2],
            3,
        )
        return {
            "traj_xyyaw": traj_xyyaw.detach(),
            "old_log_probs": old_log_probs,
            "mode_indices": selected_global_indices,
            "local_indices": local_indices.to(dtype=torch.long),
            "score_logits": selected_logits,
        }

    @staticmethod
    def _stored_grpo_candidate_mode_indices_from_replay(
        replays: Sequence[Dict[str, Any]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        values: List[torch.Tensor] = []
        expected_k: int | None = None
        for idx, replay in enumerate(replays):
            raw = require_grpo_candidate_field(replay, "mode_indices", index=idx)
            tensor = torch.as_tensor(raw, dtype=torch.long, device=device).view(-1)
            if expected_k is None:
                expected_k = int(tensor.numel())
            elif int(tensor.numel()) != expected_k:
                raise RuntimeError(
                    "Strict GRPO candidate count mismatch across replay entries: "
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
            raw = require_grpo_candidate_field(replay, "old_log_probs", index=idx)
            tensor = torch.as_tensor(raw, dtype=torch.float32, device=device).view(-1)
            if expected_k is None:
                expected_k = int(tensor.numel())
            elif int(tensor.numel()) != expected_k:
                raise RuntimeError(
                    "Strict GRPO old logp count mismatch across replay entries: "
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
            raw = require_grpo_candidate_field(replay, "traj_xyyaw", index=idx)
            tensor = torch.as_tensor(raw, dtype=torch.float32, device=device)
            if tensor.ndim != 3 or int(tensor.shape[-1]) < 3:
                raise RuntimeError(
                    "Strict GRPO replay grpo_candidate_traj_xyyaw must have shape (candidates,horizon,3+); "
                    f"got {tuple(tensor.shape)} at index {idx}"
                )
            tensor = tensor[..., :3]
            if expected_shape is None:
                expected_shape = tuple(int(v) for v in tensor.shape)
            elif tuple(int(v) for v in tensor.shape) != expected_shape:
                raise RuntimeError(
                    "Strict GRPO candidate trajectory shape mismatch across replay entries: "
                    f"expected={expected_shape} got={tuple(tensor.shape)} index={idx}"
                )
            values.append(tensor)
        return torch.stack(values, dim=0) if values else torch.empty((0, 0, 0, 3), dtype=torch.float32, device=device)

    @staticmethod
    def _new_log_probs_for_stored_grpo_candidates_from_outputs(
        *,
        score_logits: torch.Tensor,
        candidate_global_indices: torch.Tensor,
        replays: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        target_indices = SparseDriveV2CandidateAdapterMixin._stored_grpo_candidate_mode_indices_from_replay(
            replays,
            device=score_logits.device,
        )
        if int(target_indices.shape[0]) != int(score_logits.shape[0]):
            raise RuntimeError(
                "Strict GRPO target batch mismatch: "
                f"targets={tuple(target_indices.shape)} logits={tuple(score_logits.shape)}"
            )
        logp_all = torch.log_softmax(score_logits, dim=1)
        rows: List[torch.Tensor] = []
        for batch_idx in range(int(score_logits.shape[0])):
            matches = candidate_global_indices[batch_idx].to(device=score_logits.device, dtype=torch.long).view(1, -1) == target_indices[
                batch_idx
            ].view(-1, 1)
            present = matches.any(dim=1)
            if not bool(present.all().item()):
                missing = target_indices[batch_idx][~present].detach().cpu().tolist()
                raise RuntimeError(
                    "Strict GRPO stored candidate indices are absent from current candidate set; "
                    f"batch={batch_idx} missing={missing[:8]}"
                )
            local_indices = matches.to(dtype=torch.long).argmax(dim=1)
            rows.append(logp_all[batch_idx].gather(0, local_indices))
        return torch.stack(rows, dim=0) if rows else torch.empty((0, 0), device=score_logits.device, dtype=torch.float32)

    @staticmethod
    def _score_logits_for_stored_grpo_candidates_from_outputs(
        *,
        score_logits: torch.Tensor,
        candidate_global_indices: torch.Tensor,
        replays: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        target_indices = SparseDriveV2CandidateAdapterMixin._stored_grpo_candidate_mode_indices_from_replay(
            replays,
            device=score_logits.device,
        )
        if int(target_indices.shape[0]) != int(score_logits.shape[0]):
            raise RuntimeError(
                "Strict GRPO target batch mismatch: "
                f"targets={tuple(target_indices.shape)} logits={tuple(score_logits.shape)}"
            )
        rows: List[torch.Tensor] = []
        for batch_idx in range(int(score_logits.shape[0])):
            matches = candidate_global_indices[batch_idx].to(device=score_logits.device, dtype=torch.long).view(1, -1) == target_indices[
                batch_idx
            ].view(-1, 1)
            present = matches.any(dim=1)
            if not bool(present.all().item()):
                missing = target_indices[batch_idx][~present].detach().cpu().tolist()
                raise RuntimeError(
                    "Strict GRPO stored candidate indices are absent from current candidate set; "
                    f"batch={batch_idx} missing={missing[:8]}"
                )
            local_indices = matches.to(dtype=torch.long).argmax(dim=1)
            rows.append(score_logits[batch_idx].gather(0, local_indices))
        return torch.stack(rows, dim=0) if rows else torch.empty((0, 0), device=score_logits.device, dtype=torch.float32)

    def _forced_grpo_logp_and_logit_for_global_index(
        self,
        replay: Dict[str, Any],
        *,
        global_index: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forced_replay = dict(replay)
        forced_policy = dict(get_policy_action_id(replay))
        forced_policy["global_mode_idx"] = int(global_index)
        forced_replay["policy"] = dict(replay["policy"])
        forced_replay["policy"]["action_id"] = forced_policy
        batched_dev = self._batched_replay_features([forced_replay])
        target = torch.as_tensor([int(global_index)], dtype=torch.long, device=self.device)
        out = self._forward_policy_with_forced_global_indices(batched_dev, target)
        score_logits = out["candidate_scores"].to(dtype=torch.float32)
        candidate_identity = self._candidate_identity_from_outputs(out)
        logp, present = self._logp_for_global_modes_or_missing(
            score_logits=score_logits,
            candidate_global_indices=candidate_identity["candidate_global_indices"],
            target_global_indices=target.to(device=score_logits.device),
        )
        if not bool(present.all().item()):
            raise RuntimeError(f"Strict GRPO forced scoring could not recover global_mode_idx={int(global_index)}")
        matches = candidate_identity["candidate_global_indices"].to(device=score_logits.device, dtype=torch.long) == target[
            :, None
        ].to(device=score_logits.device)
        local_index = matches.to(dtype=torch.long).argmax(dim=1)
        forced_logit = score_logits[0].gather(0, local_index.view(-1))[0]
        return logp.view(()).to(device=device), forced_logit.to(device=device)

    def _strict_grpo_log_probs_and_logits_for_stored_candidates(
        self,
        replay: Dict[str, Any],
        *,
        score_logits: torch.Tensor,
        candidate_global_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_indices = self._stored_grpo_candidate_mode_indices_from_replay(
            [replay],
            device=score_logits.device,
        ).view(-1)
        if int(score_logits.shape[0]) != 1:
            raise RuntimeError(
                "Strict GRPO per-replay logp recovery expects single-sample logits; "
                f"got {tuple(score_logits.shape)}"
            )
        logp_all = torch.log_softmax(score_logits, dim=1)
        new_log_probs: List[torch.Tensor] = []
        selected_logits: List[torch.Tensor] = []
        global_indices = candidate_global_indices[0].to(device=score_logits.device, dtype=torch.long).view(-1)
        for target in target_indices:
            matches = global_indices == target.to(device=score_logits.device, dtype=torch.long)
            if bool(matches.any().item()):
                local_index = matches.to(dtype=torch.long).argmax(dim=0).view(())
                new_log_probs.append(logp_all[0].gather(0, local_index.view(1))[0])
                selected_logits.append(score_logits[0].gather(0, local_index.view(1))[0])
                continue
            forced_logp, forced_logit = self._forced_grpo_logp_and_logit_for_global_index(
                replay,
                global_index=int(target.detach().cpu().item()),
                device=score_logits.device,
            )
            new_log_probs.append(forced_logp)
            selected_logits.append(forced_logit)
        if not new_log_probs:
            empty = torch.empty((0,), device=score_logits.device, dtype=torch.float32)
            return empty, empty
        return torch.stack(new_log_probs, dim=0), torch.stack(selected_logits, dim=0)

    @staticmethod
    def _concat_counterfactual_candidate_batches(
        parts: Sequence[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        if len(parts) == 0:
            return {}
        keys = list(parts[0].keys())
        return {
            key: torch.cat([part[key] for part in parts], dim=0)
            for key in keys
        }

    #调用上层函数
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

        if len(replays) > 1:
            parts = [
                self.sample_counterfactual_trajectories_from_replay_batch(
                    [replay],
                    num_candidates=int(num_candidates),
                    candidate_select=str(candidate_select),
                )
                for replay in replays
            ]
            return self._concat_counterfactual_candidate_batches(parts)

        batched_dev = self._batched_replay_features(replays)
        model = self._unwrap_model(self._model)
        model.eval()
        out = self._forward_policy_on_model(model, batched_dev)
        score_logits = out["candidate_scores"].to(dtype=torch.float32)
        candidate_trajs = out["candidate_trajectories"]
        candidate_identity = self._candidate_identity_from_outputs(out)
        return self._select_counterfactual_candidates_from_policy_outputs(
            score_logits=score_logits,
            candidate_trajs=candidate_trajs,
            num_candidates=int(num_candidates),
            candidate_select=str(candidate_select),
        )

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
            empty_candidates = {
                "traj_xyyaw": torch.empty((0, 0, 0, 3), dtype=torch.float32),
                "log_probs": torch.empty((0, 0), device=self.device, dtype=torch.float32),
                "mode_indices": torch.empty((0, 0), device=self.device, dtype=torch.long),
                "score_logits": torch.empty((0, 0), device=self.device, dtype=torch.float32),
            }
            return {
                "new_logp": torch.empty((0,), device=self.device, dtype=torch.float32),
                "counterfactual": empty_candidates,
            }

        if len(replays) > 1:
            parts = [
                self.replay_policy_outputs_from_replay_batch(
                    [replay],
                    eta=1.0,
                    num_candidates=int(num_candidates),
                    candidate_select=str(candidate_select),
                )
                for replay in replays
            ]
            return {
                "new_logp": torch.cat([part["new_logp"].view(-1) for part in parts], dim=0),
                "counterfactual": self._concat_counterfactual_candidate_batches(
                    [part["counterfactual"] for part in parts]
                ),
            }

        batched_dev = self._batched_replay_features(replays)
        model = self._unwrap_model(self._model)
        model.eval()
        out = self._forward_policy_on_model(model, batched_dev)
        score_logits = out["candidate_scores"].to(dtype=torch.float32)
        candidate_trajs = out["candidate_trajectories"]
        candidate_identity = self._candidate_identity_from_outputs(out)
        if candidate_trajs.ndim != 4:
            raise RuntimeError(
                "SparseDriveV2 replay policy outputs require candidate trajectories with shape "
                f"(batch, modes, horizon, dims); got {tuple(candidate_trajs.shape)}"
            )

        target_global_indices = self._global_mode_indices_from_replay(replays, device=score_logits.device)
        new_logp, present = self._logp_for_global_modes_or_missing(
            score_logits=score_logits,
            candidate_global_indices=candidate_identity["candidate_global_indices"],
            target_global_indices=target_global_indices,
        )
        if not bool(present.all().item()):
            missing = (~present).nonzero(as_tuple=False).view(-1)
            if len(replays) == 1:
                new_logp = self._forced_logp_from_replay_batch(replays)
            else:
                for missing_idx_tensor in missing:
                    missing_idx = int(missing_idx_tensor.item())
                    single_outputs = self.replay_policy_outputs_from_replay_batch(
                        [replays[missing_idx]],
                        eta=1.0,
                        num_candidates=int(num_candidates),
                        candidate_select=str(candidate_select),
                    )
                    new_logp[missing_idx] = single_outputs["new_logp"].view(())
        logp_all = torch.log_softmax(score_logits, dim=1)
        if all(isinstance(replay, dict) and isinstance(replay.get("grpo", {}).get("candidates", None), dict) for replay in replays):
            recovered_rows = [
                self._strict_grpo_log_probs_and_logits_for_stored_candidates(
                    replay,
                    score_logits=score_logits[batch_idx : batch_idx + 1],
                    candidate_global_indices=candidate_identity["candidate_global_indices"][batch_idx : batch_idx + 1],
                )
                for batch_idx, replay in enumerate(replays)
            ]
            new_log_probs_row = torch.stack([row[0] for row in recovered_rows], dim=0)
            score_logits_row = torch.stack([row[1] for row in recovered_rows], dim=0)
            old_candidate_log_probs = self._stored_grpo_candidate_old_log_probs_from_replay(
                replays,
                device=score_logits.device,
            )
            traj_xyyaw = self._stored_grpo_candidate_traj_xyyaw_from_replay(
                replays,
                device=score_logits.device,
            )
            mode_indices = self._stored_grpo_candidate_mode_indices_from_replay(
                replays,
                device=score_logits.device,
            )
            counterfactual = {
                "traj_xyyaw": traj_xyyaw.detach(),
                "log_probs": new_log_probs_row,
                "old_log_probs": old_candidate_log_probs,
                "mode_indices": mode_indices,
                "score_logits": score_logits_row,
            }
        else:
            counterfactual = self._select_counterfactual_candidates_from_policy_outputs(
                score_logits=score_logits,
                candidate_trajs=candidate_trajs,
                num_candidates=int(num_candidates),
                candidate_select=str(candidate_select),
                logp_all=logp_all,
            )

        return {
            "new_logp": new_logp,
            "counterfactual": counterfactual,
        }

    def _counterfactual_scorer_backend_name(self) -> str:
        backend = str(self._nuscenes_scorer_config.get("backend", "nuscenes_pdm")).strip().lower()
        if backend in {"craft", "craft_carl", "carl", "nuscenes_craft"}:
            return "craft_carl"
        if backend in {"pdm", "nuscenes_pdm", "token", "nuscenes_token"}:
            return "nuscenes_pdm"
        return "nuscenes_pdm"

    def _ensure_nuscenes_pdm_scorer(self):
        if self._nuscenes_pdm_scorer is None:
            from framework.algorithms.nuscenes_pdm_scorer import NuScenesPDMScorer

            scorer_kwargs = {
                key: value
                for key, value in self._nuscenes_scorer_config.items()
                if key != "backend"
            }
            self._nuscenes_pdm_scorer = NuScenesPDMScorer(
                token2vad_path=nus_cfg.TOKEN2VAD_FILE,
                **scorer_kwargs,
            )
        return self._nuscenes_pdm_scorer

    def _ensure_nuscenes_craft_scorer(self):
        if not hasattr(self, "_nuscenes_craft_scorer"):
            self._nuscenes_craft_scorer = None
        if self._nuscenes_craft_scorer is None:
            from framework.algorithms.nuscenes_craft_scorer import NuScenesCraftScorer

            scorer_kwargs = {
                key: value
                for key, value in self._nuscenes_scorer_config.items()
                if key != "backend"
            }
            self._nuscenes_craft_scorer = NuScenesCraftScorer(
                token2vad_path=nus_cfg.TOKEN2VAD_FILE,
                **scorer_kwargs,
            )
        return self._nuscenes_craft_scorer

    def _ensure_counterfactual_scorer_backend(self):
        backend_name = self._counterfactual_scorer_backend_name()
        if backend_name == "craft_carl":
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



__all__ = ["SparseDriveV2CandidateAdapterMixin"]
