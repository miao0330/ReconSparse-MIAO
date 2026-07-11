from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from framework.replay_schema import get_policy_action_id, get_policy_model_inputs


class SparseDriveV2RLAdapterMixin:
    def logp_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
        *,
        eta: float = 1.0,
    ) -> torch.Tensor:
        del eta
        if len(replays) == 0:
            return torch.empty((0,), device=self.device, dtype=torch.float32)

        bad_indices = [
            idx for idx, replay in enumerate(replays)
            if not self.replay_is_compatible(replay)
        ]
        if len(bad_indices) > 0:
            raise RuntimeError(
                "SparseDriveV2 replay batch contains incompatible replay entries; "
                "this usually means old-format shards are mixed into the current buffer. "
                f"Bad replay indices: {bad_indices[:8]}"
            )
        if len(replays) > 1:
            parts = [self.logp_from_replay_batch([replay], eta=1.0).view(1) for replay in replays]
            return torch.cat(parts, dim=0)

        first_inputs = get_policy_model_inputs(replays[0])
        camera_feature = first_inputs.get("camera_feature", None)
        if not isinstance(camera_feature, dict):
            raise RuntimeError("SparseDriveV2 replay policy.model_inputs requires camera_feature")
        camera_keys = list(camera_feature.keys())
        batched_camera = {
            key: torch.cat([get_policy_model_inputs(rep)["camera_feature"][key] for rep in replays], dim=0)
            for key in camera_keys
        }
        batched_features = {
            "camera_feature": batched_camera,
            "status_feature": torch.cat([get_policy_model_inputs(rep)["status_feature"] for rep in replays], dim=0),
        }
        batched_dev = self._to_device_features(batched_features, self.device)

        model = self._model.module if isinstance(self._model, DDP) else self._model
        model.eval()
        out = self._forward_policy(batched_dev)
        score_logits = out["candidate_scores"].to(dtype=torch.float32)
        candidate_identity = self._candidate_identity_from_outputs(out)
        target_global_indices = self._global_mode_indices_from_replay(replays, device=score_logits.device)
        logp, present = self._logp_for_global_modes_or_missing(
            score_logits=score_logits,
            candidate_global_indices=candidate_identity["candidate_global_indices"],
            target_global_indices=target_global_indices,
        )
        if bool(present.all().item()):
            return logp

        missing = (~present).nonzero(as_tuple=False).view(-1)
        if len(replays) == 1:
            return self._forced_logp_from_replay_batch(replays)

        for missing_idx_tensor in missing:
            missing_idx = int(missing_idx_tensor.item())
            logp[missing_idx] = self.logp_from_replay_batch([replays[missing_idx]], eta=1.0).view(())
        return logp

    def _forced_logp_from_replay_batch(self, replays: Sequence[Dict[str, Any]]) -> torch.Tensor:
        batched_dev = self._batched_replay_features(replays)
        target_global_indices = self._global_mode_indices_from_replay(replays, device=self.device)
        out = self._forward_policy_with_forced_global_indices(batched_dev, target_global_indices)
        score_logits = out["candidate_scores"].to(dtype=torch.float32)
        candidate_identity = self._candidate_identity_from_outputs(out)
        target_global_indices = target_global_indices.to(device=score_logits.device)
        logp, present = self._logp_for_global_modes_or_missing(
            score_logits=score_logits,
            candidate_global_indices=candidate_identity["candidate_global_indices"],
            target_global_indices=target_global_indices,
        )
        if bool(present.all().item()):
            return logp
        missing = (~present).nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
        raise RuntimeError(
            "SparseDriveV2 replay global_mode_idx was not present after forced scoring; "
            f"missing batch indices: {missing[:8]}"
        )

    def init_distillation_teacher(self, *, ckpt_path: str | None = None) -> None:
        teacher_ckpt = str(ckpt_path) if ckpt_path is not None else str(self.ckpt_path)
        teacher = self._SparseDriveModel(self._cfg)
        self._load_state_into_model(teacher, teacher_ckpt)
        teacher.to(self.device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        self._teacher_model = teacher

    def _distill_log_probs_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
        *,
        temperature: float = 1.0,
        model: torch.nn.Module,
        use_inference_mode: bool,
    ) -> torch.Tensor:
        if len(replays) == 0:
            return torch.empty((0, 0), device=self.device, dtype=torch.float32)

        batched_dev = self._batched_replay_features(replays)
        if use_inference_mode:
            model.eval()
            with torch.inference_mode():
                out = self._forward_policy_on_model(model, batched_dev)
        else:
            out = self._forward_policy_on_model(model, batched_dev)
        score_logits = out["candidate_scores"].to(dtype=torch.float32)
        return F.log_softmax(score_logits / float(temperature), dim=1)

    def distill_student_log_probs_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        return self._distill_log_probs_from_replay_batch(
            replays,
            temperature=float(temperature),
            model=self._unwrap_model(self._model),
            use_inference_mode=False,
        )

    def distill_teacher_log_probs_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if self._teacher_model is None:
            raise RuntimeError("SparseDriveV2 distillation teacher is not initialized")
        return self._distill_log_probs_from_replay_batch(
            replays,
            temperature=float(temperature),
            model=self._teacher_model,
            use_inference_mode=True,
        )

    @property
    def value_feature_dim(self) -> int | None:
        model = self._model.module if isinstance(self._model, DDP) else self._model
        backbone_dim = int(getattr(getattr(model, "_backbone", None), "embed_dims", 0))
        status_encoder = getattr(model, "_status_encoding", None)
        status_dim = int(getattr(status_encoder, "out_features", 0)) if status_encoder is not None else 0
        if backbone_dim <= 0 or status_dim <= 0:
            return None
        return int(backbone_dim + status_dim)

    def _batched_replay_features(self, replays: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if len(replays) == 0:
            raise RuntimeError("SparseDriveV2 value replay batch is empty")

        bad_indices = [
            idx for idx, replay in enumerate(replays)
            if not self.replay_is_compatible(replay)
        ]
        if len(bad_indices) > 0:
            raise RuntimeError(
                "SparseDriveV2 value replay batch contains incompatible replay entries. "
                f"Bad replay indices: {bad_indices[:8]}"
            )

        first_inputs = get_policy_model_inputs(replays[0])
        camera_feature = first_inputs.get("camera_feature", None)
        if not isinstance(camera_feature, dict) or "imgs" not in camera_feature:
            raise RuntimeError("SparseDriveV2 value replay batch is missing policy.model_inputs.camera_feature['imgs']")

        batched_camera = {
            key: torch.cat([get_policy_model_inputs(rep)["camera_feature"][key] for rep in replays], dim=0)
            for key in camera_feature.keys()
        }
        try:
            batched_status = torch.cat([get_policy_model_inputs(rep)["status_feature"] for rep in replays], dim=0)
        except KeyError as exc:
            raise RuntimeError("SparseDriveV2 value replay batch requires policy.model_inputs.status_feature") from exc

        return self._to_device_features(
            {
                "camera_feature": batched_camera,
                "status_feature": batched_status,
            },
            self.device,
        )

    def value_features_from_replay_batch(
        self,
        replays: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        model = self._model.module if isinstance(self._model, DDP) else self._model
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = self.device
        if len(replays) == 0:
            feature_dim = int(self.value_feature_dim or 0)
            return torch.empty((0, feature_dim), device=model_device, dtype=torch.float32)

        batched_dev = self._to_device_features(self._batched_replay_features(replays), model_device)
        model.eval()
        with torch.inference_mode():
            feature_maps = model._backbone(batched_dev["camera_feature"]["imgs"])
            last_feature_map = feature_maps[-1].to(dtype=torch.float32)
            pooled_vision = last_feature_map.mean(dim=(1, 3, 4))
            status_encoding = model._status_encoding(batched_dev["status_feature"]).to(dtype=torch.float32)
            features = torch.cat([pooled_vision, status_encoding], dim=1)
        return features.detach().clone()

    def value_features_from_observation(self, observation: Dict[str, Any]) -> torch.Tensor:
        return self.value_features_from_observation_batch([observation]).view(-1)

    def value_features_from_observation_batch(
        self,
        observations: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        if len(observations) == 0:
            feature_dim = int(self.value_feature_dim or 0)
            return torch.empty((0, feature_dim), device=self.device, dtype=torch.float32)

        features_list = [self._build_features(obs) for obs in observations]
        camera_keys = list(features_list[0]["camera_feature"].keys())
        batched_camera = {
            key: torch.cat([feat["camera_feature"][key] for feat in features_list], dim=0)
            for key in camera_keys
        }
        batched_status = torch.cat([feat["status_feature"] for feat in features_list], dim=0)
        batched_dev = self._to_device_features(
            {
                "camera_feature": batched_camera,
                "status_feature": batched_status,
            },
            self.device,
        )

        model = self._model.module if isinstance(self._model, DDP) else self._model
        model.eval()
        with torch.inference_mode():
            feature_maps = model._backbone(batched_dev["camera_feature"]["imgs"])
            last_feature_map = feature_maps[-1].to(dtype=torch.float32)
            pooled_vision = last_feature_map.mean(dim=(1, 3, 4))
            status_encoding = model._status_encoding(batched_dev["status_feature"]).to(dtype=torch.float32)
            features = torch.cat([pooled_vision, status_encoding], dim=1)
        return features.detach().clone()

    def replay_is_compatible(self, replay: Dict[str, Any]) -> bool:
        if not isinstance(replay, dict):
            return False
        try:
            model_inputs = get_policy_model_inputs(replay)
            action_id = get_policy_action_id(replay)
        except RuntimeError:
            return False
        camera_feature = model_inputs.get("camera_feature", None)
        status_feature = model_inputs.get("status_feature", None)
        return isinstance(camera_feature, dict) and len(camera_feature) > 0 and torch.is_tensor(status_feature) and "global_mode_idx" in action_id

    def logp_from_replay(self, replay: Dict[str, Any], *, eta: float = 1.0) -> torch.Tensor:
        return self.logp_from_replay_batch([replay], eta=float(eta)).view(())



__all__ = ["SparseDriveV2RLAdapterMixin"]
