import os
import sys
import uuid
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from torch.nn.parallel import DistributedDataParallel as DDP

from ..base import Agent
from framework.utils.repo_paths import resolve_ego_ads_subdir
from .candidate_adapter import SparseDriveCandidateAdapterMixin
from .rl_adapter import SparseDriveRLAdapterMixin


# Ensure SparseDrive is importable
_SPARSEDRIVE_ROOT = resolve_ego_ads_subdir("SparseDrive")
if _SPARSEDRIVE_ROOT not in sys.path:
    sys.path.append(_SPARSEDRIVE_ROOT)


_HUGSIM_OBS_CAMERA_INDEX = {
    "front": 0,
    "front_left": 1,
    "front_right": 2,
    "back_left": 3,
    "back": 4,
    "back_right": 5,
}

# Keep the camera embedding order used by egoADs/SparseDrive's NuScenes
# converter: CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK,
# CAM_BACK_LEFT, CAM_BACK_RIGHT.
_SPARSEDRIVE_V1_CAMERA_ORDER = (
    "front",
    "front_right",
    "front_left",
    "back",
    "back_left",
    "back_right",
)

_HUGSIM_OBS_MARKER_KEYS = ("hugsim_step_idx", "hugsim_timestamp", "official_scene_name")

# HUGSIM exposes ego_pose/cam2ego in its render vehicle frame. SparseDrive-v1
# temporal anchors and planning trajectories use its original lidar frame:
# x right, y forward, z up. Keep this conversion local to v1 so other HUGSIM
# consumers continue to see the unmodified HUGSIM pose.
_HUGSIM_RENDER_TO_REPO_GLOBAL = np.asarray(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
_HUGSIM_RENDER_VEHICLE_TO_SPARSEDRIVE_V1_LIDAR = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
_SPARSEDRIVE_V1_LIDAR_TO_HUGSIM_RENDER_VEHICLE = np.linalg.inv(
    _HUGSIM_RENDER_VEHICLE_TO_SPARSEDRIVE_V1_LIDAR
).astype(np.float32)


def _as_float_tensor(x: Any, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def _resize_rgb_uint8(img: np.ndarray, *, out_hw: Tuple[int, int]) -> np.ndarray:
    """Resize HWC RGB uint8 to (out_h,out_w).把相机图 resize 到 256*704，保持和 SparseDrive 训练时输入一致。"""
    import cv2

    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if img is None:
        raise ValueError("img is None")
    if img.shape[0] == out_h and img.shape[1] == out_w:
        return img
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def _is_hugsim_observation(observation: Dict[str, Any]) -> bool:
    return any(key in observation for key in _HUGSIM_OBS_MARKER_KEYS)


def _hugsim_render_pose_to_sparsedrive_v1_lidar_global(ego_pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(ego_pose, dtype=np.float32).reshape(4, 4)
    return (
        _HUGSIM_RENDER_TO_REPO_GLOBAL
        @ pose
        @ _SPARSEDRIVE_V1_LIDAR_TO_HUGSIM_RENDER_VEHICLE
    ).astype(np.float32)


def _hugsim_cam2ego_to_sparsedrive_v1_cam2lidar(cam2ego: np.ndarray) -> np.ndarray:
    cam = np.asarray(cam2ego, dtype=np.float32)
    if cam.ndim != 3 or cam.shape[-2:] != (4, 4):
        raise ValueError(f"Expected cam2ego shape (N,4,4), got {cam.shape}")
    return (
        _HUGSIM_RENDER_VEHICLE_TO_SPARSEDRIVE_V1_LIDAR[None, :, :]
        @ cam
    ).astype(np.float32)


def _resolve_plugin_module_path(*, config_path: str, plugin_dir: str | None, sparsedrive_root: str) -> str:
    if plugin_dir:
        raw_plugin_dir = str(plugin_dir)
        if os.path.isdir(raw_plugin_dir) and os.path.exists(os.path.join(raw_plugin_dir, "__init__.py")):
            raw_module_dir = raw_plugin_dir
        else:
            raw_module_dir = os.path.dirname(raw_plugin_dir)
    else:
        raw_module_dir = os.path.dirname(str(config_path))

    candidates: list[str] = []
    if os.path.isabs(raw_module_dir):
        candidates.append(os.path.abspath(raw_module_dir))
        marker = f"{os.sep}SparseDrive{os.sep}"
        if marker in raw_module_dir:
            suffix = raw_module_dir.split(marker, 1)[1]
            candidates.append(os.path.abspath(os.path.join(str(sparsedrive_root), suffix)))
    else:
        candidates.append(os.path.abspath(os.path.join(str(sparsedrive_root), raw_module_dir)))
        candidates.append(os.path.abspath(os.path.join(os.path.dirname(str(config_path)), raw_module_dir)))

    sparse_abs = os.path.abspath(str(sparsedrive_root))
    for candidate in candidates:
        if candidate == sparse_abs or candidate.startswith(sparse_abs + os.sep):
            rel = os.path.relpath(candidate, sparse_abs)
            return rel.replace(os.sep, ".")

    return raw_module_dir.strip(os.sep).replace(os.sep, ".")


def _default_sparse_data_roots() -> list[str]:
    return [
        os.path.join(str(_SPARSEDRIVE_ROOT), "data"),
        "/OpenDataset/zhaodanqi/sparsedrive_data",
    ]


def _resolve_sparse_asset_path(
    path: str,
    *,
    sparsedrive_root: str,
    data_roots: list[str] | None = None,
) -> str:
    text = str(path)
    suffix: str | None = None
    marker = f"{os.sep}SparseDrive{os.sep}"
    if os.path.isabs(text) and marker in text:
        suffix = text.split(marker, 1)[1]
    elif text.startswith("data" + os.sep) or text.startswith("data/"):
        suffix = text.replace("/", os.sep)
    if suffix is None:
        return text

    candidates = [os.path.abspath(os.path.join(str(sparsedrive_root), suffix))]
    data_prefix = "data" + os.sep
    if suffix.startswith(data_prefix):
        data_suffix = suffix[len(data_prefix) :]
        for root in data_roots if data_roots is not None else _default_sparse_data_roots():
            candidates.append(os.path.abspath(os.path.join(str(root), data_suffix)))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return text


def _rewrite_sparse_asset_paths(obj: Any, *, sparsedrive_root: str) -> Any:
    cfg_dict = getattr(obj, "_cfg_dict", None)
    if cfg_dict is not None:
        _rewrite_sparse_asset_paths(cfg_dict, sparsedrive_root=sparsedrive_root)
        return obj
    if isinstance(obj, str):
        return _resolve_sparse_asset_path(obj, sparsedrive_root=sparsedrive_root)
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            obj[key] = _rewrite_sparse_asset_paths(value, sparsedrive_root=sparsedrive_root)
        return obj
    if isinstance(obj, list):
        for idx, value in enumerate(list(obj)):
            obj[idx] = _rewrite_sparse_asset_paths(value, sparsedrive_root=sparsedrive_root)
        return obj
    if isinstance(obj, tuple):
        return tuple(_rewrite_sparse_asset_paths(value, sparsedrive_root=sparsedrive_root) for value in obj)
    return obj


def _checkpoint_has_agent_prefix(path: str) -> bool:
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else None
    if not isinstance(sd, dict):
        return False
    return any(str(key).startswith("agent.") for key in sd.keys())


def _normalize_trainable_prefixes(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def _apply_trainable_prefixes(
    module: torch.nn.Module,
    prefixes: Sequence[str],
    *,
    frozen_prefixes: Sequence[str] | None = None,
) -> tuple[int, int]:
    normalized = [str(prefix).strip() for prefix in prefixes if str(prefix).strip()]
    normalized_frozen = [str(prefix).strip() for prefix in (frozen_prefixes or []) if str(prefix).strip()]
    if len(normalized) == 0:
        if len(normalized_frozen) > 0:
            for name, param in module.named_parameters():
                if any(name.startswith(prefix) for prefix in normalized_frozen):
                    param.requires_grad = False
        total = sum(1 for _ in module.parameters())
        trainable = sum(1 for param in module.parameters() if getattr(param, "requires_grad", False))
        return total, trainable

    for param in module.parameters():
        param.requires_grad = False

    trainable = 0
    total = 0
    for name, param in module.named_parameters():
        total += 1
        if any(name.startswith(prefix) for prefix in normalized):
            param.requires_grad = True
        if any(name.startswith(prefix) for prefix in normalized_frozen):
            param.requires_grad = False
        if getattr(param, "requires_grad", False):
            trainable += 1
    return total, trainable


def _summarize_parameter_status(module: torch.nn.Module) -> tuple[List[str], int, int, int, int]:
    trainable_names: List[str] = []
    total_tensors = 0
    trainable_tensors = 0
    total_params = 0
    trainable_params = 0
    for name, param in module.named_parameters():
        total_tensors += 1
        total_params += int(param.numel())
        if getattr(param, "requires_grad", False):
            trainable_names.append(name)
            trainable_tensors += 1
            trainable_params += int(param.numel())
    return trainable_names, total_tensors, trainable_tensors, total_params, trainable_params


class SparseDrivePolicy(SparseDriveRLAdapterMixin, SparseDriveCandidateAdapterMixin, Agent):
    """RL policy wrapper around SparseDrive planning head.

    This wrapper exposes the same replay/logp semantics used by PPO in this repo.

    Action execution:
    - continuous first-point only: execute (dx, dy, yaw, flag=2)

    Log-prob definition:
    - A categorical policy over planning modes (ego_fut_mode) conditioned on the
      current driving command (cmd_idx).
    - logp = log_softmax(logits)[mode_idx]

    Notes:
    - SparseDrive internally uses multi-view geometry; we construct projection
      matrices from ego_pose + cam2ego + intrinsics provided by the env.
    - PPO updates call `logp_from_replay()` which recomputes logits under current
      parameters with autograd enabled.
    """

    def __init__(
        self,
        *,
        config_path: str,
        ckpt_path: str,
        device: str | None = None,
        rl_lr: float = 1e-5,
        execute_mode: str = "first_step",
        trainable_prefixes: Sequence[str] | None = None,
        frozen_prefixes: Sequence[str] | None = None,
        nuscenes_scorer_config: Dict[str, Any] | None = None,
        grpo_num_candidates: int | None = None,
    ) -> None:
        self.config_path = str(config_path)
        self.ckpt_path = str(ckpt_path)
        self._device_override = device
        self._trainable_prefixes = _normalize_trainable_prefixes(trainable_prefixes)
        self._frozen_prefixes = _normalize_trainable_prefixes(frozen_prefixes)
        self._nuscenes_scorer_config = dict(nuscenes_scorer_config or {})
        self._configured_grpo_num_candidates = grpo_num_candidates
        self._nuscenes_pdm_scorer: Any | None = None
        self._nuscenes_craft_scorer: Any | None = None

        if str(execute_mode).strip().lower() not in {"first_step", "continuous", "step1"}:
            raise ValueError(f"Unsupported execute_mode for SparseDrive-v1: {execute_mode}")
        self._execute_mode = "continuous"

        # Lazy imports (mmcv/mmdet are heavy)
        try:
            from mmcv import Config
            # from mmcv.utils import Config
            # from mmcv.utils import ConfigDict
            from mmcv import DictAction
            import importlib
            from mmdet.models import build_detector
            from mmcv.runner import load_checkpoint
        except Exception as e:
            raise ImportError(
                f"[SparseDrivePolicy] Missing SparseDrive deps (mmcv/mmdet). Import error: {e}"
            )

        # cfg = ConfigDict.fromfile(self.config_path)
        cfg = Config.fromfile(self.config_path)
        _rewrite_sparse_asset_paths(cfg, sparsedrive_root=_SPARSEDRIVE_ROOT)
        if getattr(cfg, "plugin", False):
            plugin_dir = getattr(cfg, "plugin_dir", None)
            _module_path = _resolve_plugin_module_path(
                config_path=self.config_path,
                plugin_dir=(str(plugin_dir) if plugin_dir else None),
                sparsedrive_root=_SPARSEDRIVE_ROOT,
            )
            importlib.import_module(_module_path)

        # Build model   禁用 backbone 的预训练权重
        cfg.model.pretrained = None
        self._model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
        if _checkpoint_has_agent_prefix(self.ckpt_path):
            self.load_from_checkpoint(self.ckpt_path, strict=False)
        else:
            _ = load_checkpoint(self._model, self.ckpt_path, map_location="cpu", strict=False)

        # Find final input size from config (defaults to (256, 704) H,W)
        final_dim = None
        try:
            #读取输入分辨率
            data_aug = getattr(cfg, "data_aug_conf", None)
            if isinstance(data_aug, dict) and data_aug.get("final_dim") is not None:
                fd = data_aug.get("final_dim")
                if isinstance(fd, (list, tuple)) and len(fd) == 2:
                    final_dim = (int(fd[0]), int(fd[1]))
        except Exception:
            final_dim = None
        self._input_hw = final_dim if final_dim is not None else (256, 704)

        # Norm config
        self._img_mean = np.asarray(getattr(cfg, "img_norm_cfg", {}).get("mean", [123.675, 116.28, 103.53]), dtype=np.float32)
        self._img_std = np.asarray(getattr(cfg, "img_norm_cfg", {}).get("std", [58.395, 57.12, 57.375]), dtype=np.float32)

        # Planning modes
        ego_fut_mode = None
        ego_fut_ts = None
        try:
            ego_fut_mode = int(cfg.get("ego_fut_mode"))
            ego_fut_ts = int(cfg.get("ego_fut_ts"))
        except Exception:
            pass
        #TODO: 这里是否考虑了command=3呢
        self._ego_fut_mode = int(ego_fut_mode) if ego_fut_mode is not None else 6
        self._ego_fut_ts = int(ego_fut_ts) if ego_fut_ts is not None else 6
        if self._configured_grpo_num_candidates is None:
            self._grpo_num_candidates = int(self._ego_fut_mode)
        else:
            self._grpo_num_candidates = max(0, int(self._configured_grpo_num_candidates or 0))

        _apply_trainable_prefixes(
            self._model,
            self._trainable_prefixes,
            frozen_prefixes=self._frozen_prefixes,
        )
        trainable_names, total_tensors, trainable_tensors, total_params, trainable_params = _summarize_parameter_status(
            self._model
        )
        if len(self._trainable_prefixes) > 0:
            print(
                "[SparseDrivePolicy] trainable_prefixes="
                f"{self._trainable_prefixes} frozen_prefixes={self._frozen_prefixes} "
                f"-> trainable tensors {trainable_tensors}/{total_tensors}, "
                f"trainable params {trainable_params}/{total_params}"
            )
            for name in trainable_names[-1:]:
                print(f"[SparseDrivePolicy] trainable_param[-1]={name}")
        else:
            print(
                "[SparseDrivePolicy] trainable_prefixes=[] "
                f"frozen_prefixes={self._frozen_prefixes} -> training all currently-enabled "
                "parameters except frozen prefixes: "
                f"trainable tensors {trainable_tensors}/{total_tensors}, trainable params {trainable_params}/{total_params}"
            )

        # Optimizer (optional; PPO update can use external optimizer too, but keep parity with ddv2)
        self._optimizer: torch.optim.Optimizer | None = None
        params = [p for p in self._model.parameters() if getattr(p, "requires_grad", False)]
        if len(params) > 0:
            self._optimizer = torch.optim.Adam(params, lr=float(rl_lr))

        self._ddp_enabled = False
        self.to(self.device)
        print(f"[SparseDrivePolicy] Loaded SparseDrive ckpt: {self.ckpt_path}")

    # -------------------- device / ddp / ckpt --------------------
    @property
    def device(self) -> torch.device:
        if getattr(self, "_device_override", None):
            try:
                return torch.device(str(self._device_override))
            except Exception:
                pass
        try:
            return next(self._model.parameters()).device
        except Exception:
            return torch.device("cpu")

    def to(self, device: str | torch.device) -> "SparseDrivePolicy":
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        self._device_override = str(dev)
        try:
            if isinstance(self._model, DDP):
                self._model.module.to(dev)
            else:
                self._model.to(dev)
        except Exception:
            pass
        return self

    def wrap_ddp(
        self,
        *,
        device_id: int,
        process_group: Any | None = None,
        find_unused_parameters: bool = True,
        rl_lr: float | None = None,
    ) -> None:
        m = self._model
        if isinstance(m, DDP):
            self._ddp_enabled = True
            return

        target_device = torch.device(f"cuda:{int(device_id)}") if torch.cuda.is_available() else torch.device("cpu")
        self.to(target_device)

        self._model = DDP(
            m,
            device_ids=[int(device_id)],
            output_device=int(device_id),
            process_group=process_group,
            find_unused_parameters=bool(find_unused_parameters),
        )
        self._ddp_enabled = True

        if self._optimizer is not None:
            lr = float(rl_lr) if rl_lr is not None else float(self._optimizer.param_groups[0].get("lr", 1e-5))
        else:
            lr = float(rl_lr) if rl_lr is not None else 1e-5

        core = self._model.module
        params = [p for p in core.parameters() if getattr(p, "requires_grad", False)]
        if len(params) == 0:
            raise RuntimeError("SparseDrive DDP wrap found no trainable parameters")
        self._optimizer = torch.optim.Adam(params, lr=lr)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        m = self._model.module if isinstance(self._model, DDP) else self._model
        return m.state_dict()

    @property
    def trainable_module(self):
        return self._model

    def save_checkpoint(self, path: str) -> None:
        sd = {f"agent.{k}": v.detach().cpu() for k, v in self.state_dict().items()}
        out_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.basename(path)
        tmp_path = os.path.join(out_dir, f".{base}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
        try:
            torch.save({"state_dict": sd}, tmp_path)
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def load_from_checkpoint(self, path: str, *, strict: bool = False) -> None:
        ckpt = torch.load(path, map_location=self.device)
        sd = ckpt.get("state_dict", ckpt)
        if not isinstance(sd, dict):
            raise ValueError("Checkpoint does not contain a state_dict")
        sd2: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            kk = str(k)
            if kk.startswith("agent."):
                kk = kk[len("agent.") :]
            if torch.is_tensor(v):
                sd2[kk] = v
        m = self._model.module if isinstance(self._model, DDP) else self._model
        m.load_state_dict(sd2, strict=bool(strict))

    #NOTE-------------------- core: build model inputs --------------------
    def _build_projection_mats(
        self,
        *,
        ego_pose: np.ndarray,
        cam2ego: np.ndarray,
        cam_intrinsics: np.ndarray,
        cam_hw: np.ndarray | None,
        img_hw: Tuple[int, int],
        net_hw: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (projection_mat[6,4,4], image_wh[6,2]) in float32 for net resolution."""
        h_cur, w_cur = int(img_hw[0]), int(img_hw[1])
        h_net, w_net = int(net_hw[0]), int(net_hw[1])

        proj = np.zeros((6, 4, 4), dtype=np.float32)
        image_wh = np.zeros((6, 2), dtype=np.float32)
        image_wh[:, 0] = float(w_net)
        image_wh[:, 1] = float(h_net)

        for i in range(6):
            K0 = np.asarray(cam_intrinsics[i], dtype=np.float32)
            if K0.shape == (4, 4):
                K0 = K0[:3, :3]
            if cam_hw is not None and cam_hw.shape[0] == 6:
                h0 = float(cam_hw[i, 0])
                w0 = float(cam_hw[i, 1])
            else:
                h0 = float(h_cur)
                w0 = float(w_cur)

            # Scale intrinsics to current rendered image size
            sw1 = float(w_cur) / max(1.0, float(w0))
            sh1 = float(h_cur) / max(1.0, float(h0))
            K1 = K0.copy()
            K1[0, 0] *= sw1
            K1[0, 2] *= sw1
            K1[1, 1] *= sh1
            K1[1, 2] *= sh1

            # Scale to network input size
            sw2 = float(w_net) / max(1.0, float(w_cur))
            sh2 = float(h_net) / max(1.0, float(h_cur))
            K = K1.copy()
            K[0, 0] *= sw2
            K[0, 2] *= sw2
            K[1, 1] *= sh2
            K[1, 2] *= sh2

            c2e = np.asarray(cam2ego[i], dtype=np.float32)
            e2c = np.linalg.inv(c2e)

            # lidar == ego; lidar2cam == ego2cam
            P3 = K @ e2c[:3, :]
            P = np.eye(4, dtype=np.float32)
            P[:3, :4] = P3
            proj[i] = P

        return proj, image_wh#shape = (6, 4, 4) projection 6个相机，每个相机4*4 ； image wh:shape = (6, 2) 六个相机，每个相机的图像尺寸wh

    def _normalize_rgb(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32)
        x = (x - self._img_mean[None, None, :]) / self._img_std[None, None, :]
        return x

    def _build_batch(self, observation: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        cam_keys = list(_SPARSEDRIVE_V1_CAMERA_ORDER)
        cam_idx = [_HUGSIM_OBS_CAMERA_INDEX[k] for k in cam_keys]
        imgs = []
        for k in cam_keys:
            img = observation.get(k, None)
            if img is None:
                raise KeyError(f"Missing camera key in observation: {k}")
            imgs.append(img)

        # Resize and normalize
        h_net, w_net = int(self._input_hw[0]), int(self._input_hw[1])
        imgs_rs = [_resize_rgb_uint8(im, out_hw=(h_net, w_net)) for im in imgs]
        imgs_norm = [self._normalize_rgb(im) for im in imgs_rs]
        # (6,3,H,W)
        img_t = torch.from_numpy(np.stack([im.transpose(2, 0, 1) for im in imgs_norm], axis=0)).to(dtype=torch.float32)
        img_t = img_t.unsqueeze(0)  # (1,6,3,H,W)

        ego_pose = np.asarray(observation.get("ego_pose"), dtype=np.float32)
        cam2ego = np.asarray(observation.get("cam2ego"), dtype=np.float32)
        cam_intr = np.asarray(observation.get("cam_intrinsics"), dtype=np.float32)
        cam_hw = observation.get("cam_hw", None)
        if cam2ego.shape[0] >= max(cam_idx) + 1:
            cam2ego = cam2ego[cam_idx]
        if cam_intr.shape[0] >= max(cam_idx) + 1:
            cam_intr = cam_intr[cam_idx]
        if cam_hw is not None:
            cam_hw = np.asarray(cam_hw, dtype=np.float32)
            if cam_hw.shape[0] >= max(cam_idx) + 1:
                cam_hw = cam_hw[cam_idx]

        if _is_hugsim_observation(observation):
            ego_pose = _hugsim_render_pose_to_sparsedrive_v1_lidar_global(ego_pose)
            cam2ego = _hugsim_cam2ego_to_sparsedrive_v1_cam2lidar(cam2ego)

        img0 = imgs[0]
        h_cur, w_cur = int(img0.shape[0]), int(img0.shape[1])
        proj, image_wh = self._build_projection_mats(
            ego_pose=ego_pose,
            cam2ego=cam2ego,
            cam_intrinsics=cam_intr,
            cam_hw=cam_hw,
            img_hw=(h_cur, w_cur),
            net_hw=(h_net, w_net),
        )

        # Command: env/repo uses [left, straight, right, unknown], while
        # SparseDrive-v1 was trained with [right, left, straight].
        cmd4 = np.asarray(observation.get("driving_command", np.zeros((4,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        if cmd4.shape[0] < 3:
            print("💣[attn in SparseDrPolicy] Warning: obs missing driving_command or has insufficient length, attempting to fill with zeros")
        if not np.isfinite(cmd4).all():
            print("💣[attn in SparseDrPolicy] Warning: obs has invalid driving_command, attempting to fill with zeros")
        cmd3 = self._repo_command4_to_sparse_command3(cmd4)

        ego_status = np.asarray(observation.get("ego_status", np.zeros((8,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        if ego_status.shape[0] < 8:
            tmp = np.zeros((8,), dtype=np.float32)
            tmp[: ego_status.shape[0]] = ego_status
            ego_status = tmp

        ts = observation.get("timestamp", 0.0)
        try:
            ts = float(ts)
        except Exception:
            ts = 0.0

        T_global_inv = np.linalg.inv(ego_pose)
        data = {
            "projection_mat": torch.from_numpy(proj).unsqueeze(0),  # (1,6,4,4)
            "image_wh": torch.from_numpy(image_wh).unsqueeze(0),  # (1,6,2)
            "T_global": torch.from_numpy(ego_pose).unsqueeze(0),
            "T_global_inv": torch.from_numpy(T_global_inv).unsqueeze(0),
            "timestamp": torch.tensor([ts], dtype=torch.float32),
            "ego_status": torch.from_numpy(ego_status[None, :]),
            "gt_ego_fut_cmd": torch.from_numpy(cmd3[None, :]),
            # SparseDrive heads access per-sample meta via metas["img_metas"].
            "img_metas": [
                {
                    "T_global": ego_pose.astype(np.float32, copy=True),
                    "T_global_inv": T_global_inv.astype(np.float32, copy=True),
                    "timestamp": np.float32(ts),
                }
            ],
        }

        return img_t, data

    def _to_model_metas(self, data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in data.items():
            if k == "img_metas":
                out[k] = v
            else:
                out[k] = _as_float_tensor(v, device=device)
        return out

    @staticmethod
    def _repo_command4_to_sparse_command3(command: Any) -> np.ndarray:
        """Map repo command [left, straight, right, unknown] to SparseDrive-v1 [right, left, straight]."""
        cmd4 = np.asarray(command, dtype=np.float32).reshape(-1)
        if cmd4.shape[0] < 4:
            padded = np.zeros((4,), dtype=np.float32)
            padded[: cmd4.shape[0]] = cmd4
            cmd4 = padded
        if not np.isfinite(cmd4).all():
            cmd4 = np.zeros((4,), dtype=np.float32)
        cmd3 = np.asarray([cmd4[2], cmd4[0], cmd4[1]], dtype=np.float32)
        if float(np.abs(cmd3).sum()) <= 1e-6:
            cmd3[2] = 1.0
        return cmd3

    #NOTE -------------------- action conversion --------------------
    @staticmethod
    def _sparsedrive_lidar_xy_to_hugsim_action_xyyaw(traj_xy: np.ndarray) -> np.ndarray:
        """Convert SparseDrive-v1 planner xy into the repo canonical ego frame.

        SparseDrive-v1 planning is produced in its lidar/planner frame:
        x points right and y points forward. The RL/env action contract used by
        this repo stores ego-local plans as x forward, y left, yaw. HUGSIMReconEnv
        converts that canonical plan back to the HUGSIM FIFO plan frame.
        """
        raw = np.asarray(traj_xy, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[0] <= 0 or raw.shape[1] < 2:
            raise ValueError(f"Invalid SparseDrive-v1 trajectory shape: {raw.shape}")

        canonical_xy = np.zeros((raw.shape[0], 2), dtype=np.float32)
        canonical_xy[:, 0] = raw[:, 1]
        canonical_xy[:, 1] = -raw[:, 0]

        out = np.zeros((canonical_xy.shape[0], 3), dtype=np.float32)
        out[:, :2] = canonical_xy
        prev_xy = np.concatenate([np.zeros((1, 2), dtype=np.float32), canonical_xy[:-1]], axis=0)
        dxy = canonical_xy - prev_xy
        out[:, 2] = np.arctan2(dxy[:, 1], dxy[:, 0]).astype(np.float32)
        return out

    def _build_env_action(self, *, traj_xy: np.ndarray) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        traj_xyyaw = self._sparsedrive_lidar_xy_to_hugsim_action_xyyaw(traj_xy)
        first = traj_xyyaw[0].astype(np.float32)#只取第一步去执行
        dx = float(first[0])
        dy = float(first[1])
        yaw = float(first[2])
        debug: Dict[str, Any] = {
            "first_step_xy": torch.tensor([dx, dy], dtype=torch.float32),
            "first_step_yaw": torch.tensor([yaw], dtype=torch.float32),
            "execute_mode": self._execute_mode,
        }

        return (dx, dy, yaw, 2), debug

    def _forward_raw_planning(
        self,
        *,
        img: torch.Tensor,
        data_dev: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return raw planning logits/trajs before post-process.

        Returns:
        - logits: (3, M) raw classification logits for 3 commands and M modes.
        - trajs:  (3, M, T, 2) cumulative trajectory in ego frame.
        """
        model = self._model.module if isinstance(self._model, DDP) else self._model
        feature_maps = model.extract_feat(img)
        _det_out, _map_out, _motion_out, planning_out = model.head(feature_maps, data_dev)
        if planning_out is None:
            raise RuntimeError("SparseDrive planning output is None")

        cls_raw = planning_out["classification"][-1]
        reg_raw = planning_out["prediction"][-1]
        if cls_raw.ndim == 2:
            cls_raw = cls_raw.unsqueeze(0)
        if reg_raw.ndim == 4:
            reg_raw = reg_raw.unsqueeze(0)

        bs = int(cls_raw.shape[0])
        cls = cls_raw.reshape(bs, 3, self._ego_fut_mode)
        traj = reg_raw.reshape(bs, 3, self._ego_fut_mode, self._ego_fut_ts, 2).cumsum(dim=-2)
        return cls[0], traj[0]

    #NOTE -------------------- Agent interface adapters --------------------
    def initialize(self) -> None:
        return

    def act(
        self,
        observation: Dict[str, Any],
        *,
        eta: float = 1.0,
        mode_idx: int = -1,
        mode_select: str = "sample",
    ):
        return self.sample_sparsedrive_with_replay(
            observation,
            mode_idx=int(mode_idx),
            mode_select=str(mode_select),
        )

    def act_batch(
        self,
        observations: List[Dict[str, Any]],
        *,
        eta: float = 1.0,
        mode_idx: int = -1,
        mode_select: str = "sample",
    ):
        return self.sample_sparsedrive_with_replay_batch(
            observations,
            mode_idx=int(mode_idx),
            mode_select=str(mode_select),
        )

    def load_checkpoint(self, path: str, *, strict: bool = False) -> None:
        self.load_from_checkpoint(path, strict=bool(strict))

    def parameters(self):
        m = self._model.module if isinstance(self._model, DDP) else self._model
        return m.parameters()
