#!/usr/bin/env python3
"""
Extract and save pose matrices for pose_video_viz.

This script extracts two sets of predictions:
1. PoserV1 model predictions (from the diffusion model)
2. Luma predictions (FoundationPose tracking on luma-generated videos)

And saves them alongside GT poses in .npy format.

Output structure per sample:
    {output_dir}/{sample_name}/
        ├── gt_T_cam_anchor_obj.npy      # (8, 4, 4) GT poses in anchor camera frame
        ├── pred_T_cam_anchor_obj.npy    # (8, 4, 4) PoserV1 model predictions
        └── luma_T_cam_anchor_obj.npy    # (8, 4, 4) Luma/FP tracking predictions

Usage:
    python extract_poses_for_viz.py [--sample SAMPLE_NAME] [--all] [--limit N]
    python extract_poses_for_viz.py --model-only   # Only extract PoserV1 predictions
    python extract_poses_for_viz.py --luma-only    # Only extract luma tracking predictions
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

import torch

# Paths
REPO_ROOT = Path(__file__).parent.absolute()
CSV_PATH = REPO_ROOT / "future_pose_pred" / "outputs" / "viz" / "trajs_for_viz.csv"
LUMA_DIR = REPO_ROOT / "luma_out"
OUTPUT_DIR = REPO_ROOT / "luma_poses_out"

# Existing model prediction directories (from previous runs)
FIG_SAMPLES_SUP = REPO_ROOT / "future_pose_pred" / "outputs" / "fig_samples_sup"
FIG_SAMPLES_SUP2 = REPO_ROOT / "future_pose_pred" / "outputs" / "fig_samples_sup2"


def _load_trajs_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Load trajs_for_viz.csv."""
    rows = []
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return rows
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v if v is not None else "") for k, v in r.items()})
    return rows


def _parse_frame_ids(row: Dict[str, str]) -> List[int]:
    """Parse space-separated frame IDs."""
    try:
        return [int(x) for x in str(row.get("frame_ids", "")).split() if x.strip()]
    except Exception:
        return []


def _luma_name_for_row(row: Dict[str, str]) -> str:
    """Generate luma video filename from CSV row."""
    vid = str(row.get("video_id", "vid"))
    oid = str(row.get("object_id", "obj"))
    oname = str(row.get("object_name", ""))
    fids = _parse_frame_ids(row)
    first_fid = str(fids[0]) if fids else "0"
    base = f"{vid}_{oid}_{oname}_{first_fid}.mp4"
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in base)
    return safe


def _sample_name_for_row(row: Dict[str, str]) -> str:
    """Generate sample directory name from CSV row."""
    vid = str(row.get("video_id", "vid"))
    oid = str(row.get("object_id", "obj"))
    oname = str(row.get("object_name", ""))
    fids = _parse_frame_ids(row)
    first_fid = str(fids[0]) if fids else "0"
    return f"{vid}_{oid}_{oname}_{first_fid}"


def _invert_T(T: np.ndarray) -> np.ndarray:
    """Invert 4x4 transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def _reexpress_object_pose(T_c_o: np.ndarray, E_k: np.ndarray, E_a: np.ndarray, convention: str = "c2w") -> np.ndarray:
    """Re-express object pose from camera k to anchor camera frame."""
    if convention == "c2w":
        E_a_inv = _invert_T(E_a)
        return E_a_inv @ E_k @ T_c_o
    else:
        E_k_inv = _invert_T(E_k)
        return E_a @ E_k_inv @ T_c_o


def _select_extrinsics_convention(T_c_o: np.ndarray, E_c2w: np.ndarray, E_w2c: np.ndarray) -> str:
    """Determine extrinsics convention by checking which one is more consistent."""
    # Heuristic: c2w typically has larger translation (camera position in world)
    t_c2w = np.linalg.norm(E_c2w[:3, 3])
    t_w2c = np.linalg.norm(E_w2c[:3, 3])
    return "c2w" if t_c2w > t_w2c else "w2c"


# ==============================================================================
# Find Existing Model Predictions
# ==============================================================================

def _parse_fig_sample_name(name: str) -> Tuple[str, str, int]:
    """Parse fig_samples directory name to extract video_id, object_id, first_frame.

    Format: P01_105_160_obj4_k057 -> (P01_105, 160, 4, 57)
    Returns: (video_id like 'P01_105', object_id like '4', first_frame like 57)
    """
    # Pattern: P{participant}_{video}_{clip}_obj{obj_id}_k{frame}
    import re
    match = re.match(r'(P\d+_\d+)_(\d+)_obj(\d+)_k0?(\d+)', name)
    if match:
        vid_prefix = match.group(1)  # P01_105
        clip = match.group(2)        # 160
        obj_id = match.group(3)      # 4
        frame = int(match.group(4))  # 57
        video_id = f"{vid_prefix}_{clip}"  # P01_105_160
        return video_id, obj_id, frame
    return "", "", -1


def _parse_luma_sample_info(row: Dict[str, str]) -> Tuple[str, str, int]:
    """Parse luma sample row to get video_id, object_id, first_frame."""
    vid = str(row.get("video_id", ""))
    oid = str(row.get("object_id", ""))
    fids = _parse_frame_ids(row)
    first_frame = fids[0] if fids else -1
    return vid, oid, first_frame


def find_existing_model_predictions(row: Dict[str, str], max_frame_diff: int = 10) -> Optional[Tuple[Path, int]]:
    """Find existing model predictions from fig_samples_sup or fig_samples_sup2.

    Args:
        row: CSV row with sample info
        max_frame_diff: Maximum allowed difference in first frame (default 10)

    Returns:
        Tuple of (path to pred_T_cam_anchor_obj.npy, frame_diff) if found, None otherwise
    """
    luma_vid, luma_oid, luma_frame = _parse_luma_sample_info(row)
    if not luma_vid or luma_frame < 0:
        return None

    best_match = None
    best_diff = float('inf')

    # Search in both fig_samples directories
    for fig_dir in [FIG_SAMPLES_SUP, FIG_SAMPLES_SUP2]:
        if not fig_dir.exists():
            continue

        for sample_dir in fig_dir.iterdir():
            if not sample_dir.is_dir():
                continue

            fig_vid, fig_oid, fig_frame = _parse_fig_sample_name(sample_dir.name)

            # Check if video and object match
            if fig_vid != luma_vid or fig_oid != luma_oid:
                continue

            # Check frame difference
            frame_diff = abs(fig_frame - luma_frame)
            if frame_diff > max_frame_diff:
                continue

            # Check if prediction file exists
            pred_path = sample_dir / "poses" / "pred_T_cam_anchor_obj.npy"
            if not pred_path.exists():
                continue

            # Track best match
            if frame_diff < best_diff:
                best_diff = frame_diff
                best_match = pred_path

    if best_match is not None:
        return best_match, int(best_diff)
    return None



def find_existing_gt_poses(row: Dict[str, str], max_frame_diff: int = 10) -> Optional[Tuple[Path, int]]:
    """Find existing GT poses from fig_samples_sup or fig_samples_sup2."""
    luma_vid, luma_oid, luma_frame = _parse_luma_sample_info(row)
    if not luma_vid or luma_frame < 0:
        return None

    best_match = None
    best_diff = float("inf")

    for fig_dir in [FIG_SAMPLES_SUP, FIG_SAMPLES_SUP2]:
        if not fig_dir.exists():
            continue
        for sample_dir in fig_dir.iterdir():
            if not sample_dir.is_dir():
                continue
            fig_vid, fig_oid, fig_frame = _parse_fig_sample_name(sample_dir.name)
            if fig_vid != luma_vid or fig_oid != luma_oid:
                continue
            frame_diff = abs(fig_frame - luma_frame)
            if frame_diff > max_frame_diff:
                continue
            gt_path = sample_dir / "poses" / "gt_T_cam_anchor_obj.npy"
            if not gt_path.exists():
                continue
            if frame_diff < best_diff:
                best_diff = frame_diff
                best_match = gt_path

    if best_match is not None:
        return best_match, int(best_diff)
    return None


# ==============================================================================
# GT Pose Extraction
# ==============================================================================

def extract_gt_poses(row: Dict[str, str], output_dir: Path, max_frame_diff: int = 10) -> Optional[np.ndarray]:
    """Extract and save GT poses in anchor camera frame."""
    from future_pose_pred.src.data.fpose_io import load_object_poses
    # First, try to find existing GT poses
    existing = find_existing_gt_poses(row, max_frame_diff)
    if existing is not None:
        gt_path, frame_diff = existing
        gt_poses = np.load(gt_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "gt_T_cam_anchor_obj.npy", gt_poses)
        print(f"  [GT] Copied existing GT from {gt_path.parent.parent.name} (frame_diff={frame_diff})")
        return gt_poses


    fids = _parse_frame_ids(row)
    if len(fids) < 11:
        return None

    obj_dir = row.get("object_dir", "")
    sp_npz = row.get("spatrack_npz", "")

    if not obj_dir or not os.path.isdir(obj_dir):
        print(f"  [GT] object_dir not found: {obj_dir}")
        return None

    # Load FP poses
    fp_dir = os.path.join(obj_dir, "foundationpose10", "ob_in_cam")
    if not os.path.isdir(fp_dir):
        print(f"  [GT] foundationpose10/ob_in_cam not found")
        return None

    obj_fp = load_object_poses(fp_dir, verbose=False)
    fp_frame_ids = obj_fp.get("frame_ids", np.array([]))
    fp_T_c_o = obj_fp.get("T_c_o", np.array([]))

    if len(fp_frame_ids) == 0:
        print(f"  [GT] No poses loaded from {fp_dir}")
        return None

    fids_to_idx = {int(f): i for i, f in enumerate(fp_frame_ids.tolist())}

    # Load extrinsics
    if not sp_npz or not os.path.isfile(sp_npz):
        print(f"  [GT] spatrack_npz not found: {sp_npz}")
        return None

    with np.load(sp_npz, allow_pickle=True) as z:
        extrinsics = np.array(z.get("extrinsics", z.get("T_c_w", np.array([]))))

    if extrinsics.size == 0:
        print(f"  [GT] No extrinsics in {sp_npz}")
        return None

    # Frame indices: context (0,1,2), anchor (3), future (3-10)
    anchor_fid = fids[3]
    future_fids = fids[3:11]  # anchor + 7 future = 8 frames

    if anchor_fid not in fids_to_idx:
        print(f"  [GT] anchor_fid {anchor_fid} not in FP poses")
        return None

    # Get anchor extrinsics
    E_anchor = extrinsics[anchor_fid].astype(np.float32)
    E_anchor_inv = _invert_T(E_anchor)

    # Determine convention
    T_anchor = fp_T_c_o[fids_to_idx[anchor_fid]].astype(np.float32)
    conv = _select_extrinsics_convention(T_anchor, E_anchor, E_anchor_inv)

    # Extract and re-express poses
    gt_poses = []
    for fid in future_fids:
        if fid not in fids_to_idx:
            print(f"  [GT] frame {fid} not in FP poses")
            return None
        if fid >= len(extrinsics):
            print(f"  [GT] frame {fid} out of range for extrinsics")
            return None

        T_c_o = fp_T_c_o[fids_to_idx[fid]].astype(np.float32)
        E_k = extrinsics[fid].astype(np.float32)

        T_camA_o = _reexpress_object_pose(T_c_o, E_k, E_anchor, conv)
        gt_poses.append(T_camA_o)

    gt_poses = np.stack(gt_poses, axis=0).astype(np.float32)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "gt_T_cam_anchor_obj.npy", gt_poses)
    print(f"  [GT] Saved {gt_poses.shape}")

    return gt_poses


# ==============================================================================
# PoserV1 Model Predictions
# ==============================================================================

_MODEL_CACHE = {}


def _load_poser_model(cfg_path: str = None):
    """Load PoserV1 model (cached)."""
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["cfg"]

    import hydra
    from omegaconf import OmegaConf

    from future_pose_pred.src.models.poser_v1 import PoserV1
    from future_pose_pred.src.models.poser_v1.utils.checkpoint import (
        pop_lazy_encoder_proj_state,
        resolve_and_load_state_dict,
        restore_lazy_encoder_proj,
    )
    from future_pose_pred.src.utils.config_adapter import apply_config_adapter

    # Default config path
    if cfg_path is None:
        # Use the latest training output config
        cfg_path = str(REPO_ROOT / "future_pose_pred" / "conf" / "debug.yaml")

    # Load config
    with hydra.initialize_config_dir(config_dir=str(REPO_ROOT / "future_pose_pred" / "conf"), version_base=None):
        cfg = hydra.compose(config_name="debug")

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: int(eval(expr, {"__builtins__": {}}, {})))

    cfg = apply_config_adapter(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate model
    model: PoserV1 = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.to(device)

    # Load checkpoint
    ckpt_dir = os.path.join(cfg.train.out_dir, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, "last.pt")

    if os.path.exists(ckpt_path):
        print(f"  [Model] Loading checkpoint: {ckpt_path}")
        state_dict, meta = resolve_and_load_state_dict(ckpt_path, map_location="cpu", prefer_ema=True)
        lazy_state = pop_lazy_encoder_proj_state(state_dict)
        model_ref = model.module if hasattr(model, "module") else model
        model_ref.load_state_dict(state_dict, strict=False)
        restore_lazy_encoder_proj(model_ref, lazy_state)
    else:
        print(f"  [Model] WARNING: No checkpoint found at {ckpt_path}")

    model.eval()
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["cfg"] = cfg

    return model, cfg


def extract_model_predictions(row: Dict[str, str], output_dir: Path, gt_poses: np.ndarray = None,
                               model_state: dict = None, max_frame_diff: int = 10) -> Optional[np.ndarray]:
    """Extract PoserV1 model predictions.

    First tries to find existing predictions from fig_samples_sup/fig_samples_sup2.
    If not found and model_state is provided, runs inference.

    Args:
        row: CSV row with sample info
        output_dir: Output directory
        gt_poses: GT poses (used as fallback)
        model_state: Dict containing 'model', 'cfg', 'device' from hydra setup
        max_frame_diff: Maximum frame difference for matching existing predictions
    """
    # First, try to find existing predictions
    existing = find_existing_model_predictions(row, max_frame_diff)
    if existing is not None:
        pred_path, frame_diff = existing
        pred_poses = np.load(pred_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "pred_T_cam_anchor_obj.npy", pred_poses)
        print(f"  [Model] Copied existing predictions from {pred_path.parent.parent.name} (frame_diff={frame_diff})")
        return pred_poses

    # No existing predictions found
    if model_state is None:
        print("  [Model] No existing predictions found and no model loaded")
        if gt_poses is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_dir / "pred_T_cam_anchor_obj.npy", gt_poses)
            print(f"  [Model] Saved GT as placeholder for pred")
        return None

    try:
        from future_pose_pred.src.geom.canonicalize import canonicalize_preds_to_anchor
        from future_pose_pred.src.temporal.sampling import ddim_sample_tokens
        from future_pose_pred.src.data.epic_dataset import EPICClipsDataset
    except ImportError as e:
        print(f"  [Model] Import error: {e}")
        return None

    model = model_state["model"]
    cfg = model_state["cfg"]
    device = model_state["device"]
    dataset = model_state.get("dataset")

    if dataset is None:
        print("  [Model] No dataset loaded")
        return None

    # Find matching window in dataset
    fids = _parse_frame_ids(row)
    vid = row.get("video_id", "")
    oid = row.get("object_id", "")
    oname = row.get("object_name", "")
    fids_str = " ".join(str(f) for f in fids)

    # Search for matching window
    match_idx = None
    windows = getattr(dataset, "windows", []) or []
    for i, w in enumerate(windows):
        w_vid = str(w.get("video_id", ""))
        w_oid = str(w.get("object_id", ""))
        w_oname = str(w.get("object_name", ""))
        w_fids = w.get("frame_ids", [])
        w_fids_str = " ".join(str(int(f)) for f in w_fids) if w_fids else ""

        if w_vid == vid and w_oid == oid and w_oname == oname and w_fids_str == fids_str:
            match_idx = i
            break

    if match_idx is None:
        print(f"  [Model] No matching window found in dataset")
        return None

    print(f"  [Model] Found window at index {match_idx}")

    # Get sample from dataset
    try:
        sample = dataset[match_idx]
    except Exception as e:
        print(f"  [Model] Failed to load sample: {e}")
        return None

    # Prepare batch (add batch dimension)
    def to_batch(x):
        if isinstance(x, torch.Tensor):
            return x.unsqueeze(0)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).unsqueeze(0)
        return x

    batch = {k: to_batch(v) if isinstance(v, (torch.Tensor, np.ndarray)) else [v] for k, v in sample.items()}

    # Run model
    core = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        cond = core.condition_from_batch(batch)
        scene_pcd = cond["scene_pcd"]
        context_vec = cond["context_vec"]
        T_cam_anchor_obj = cond.get("T_cam_anchor_obj")
        ctx_tokens_9d = batch.get("context_init_9d", None)
        cond_embed = core.encode(scene_pcd, context_vec, T_cam_anchor_obj=T_cam_anchor_obj)

        # Get output format
        ds_ref = dataset.dataset if hasattr(dataset, "dataset") else dataset
        out_fmt = getattr(ds_ref, "output_format", "abs_in_anchor")
        pred_mode = {
            "abs_in_anchor": "abs_in_anchor_cam",
            "delta_from_prev": "deltas_from_prev_cam",
            "delta_from_anchor": "deltas_from_anchor_cam",
        }.get(out_fmt, "abs_in_anchor_cam")

        # Sample
        eval_cfg = getattr(cfg, "eval", {}) or {}
        steps = int(getattr(eval_cfg, "steps", 50))
        eta = float(getattr(eval_cfg, "eta", 0.0))
        H = int(getattr(cfg.data, "H", 8))

        gen = torch.Generator(device=device)
        gen.manual_seed(42)

        y_pred = ddim_sample_tokens(
            poser_model=core,
            cond_embed=cond_embed,
            H=H,
            steps=steps,
            eta=eta,
            generator=gen,
            ctx_tokens_9d=ctx_tokens_9d
        ).to(device=device)

        # Canonicalize to anchor camera
        meta = {
            "K": sample.get("K"),
            "frame_ids": sample.get("frame_ids"),
            "anchor_frame_idx": sample.get("anchor_frame_idx", 0),
            "anchor_local_idx": sample.get("anchor_local_idx", 0),
            "T_c_w": sample.get("T_c_w"),
            "T_c_o": sample.get("T_c_o"),
            "T_cam_anchor_obj": sample.get("T_cam_anchor_obj"),
            "t_mean": [0.0, 0.0, 0.0],
            "t_std": [1.0, 1.0, 1.0],
            "extrinsics_convention": str(sample.get("extrinsics_convention", "c2w")).lower(),
        }

        conv_arrow = "w<-c" if meta["extrinsics_convention"] == "c2w" else "c<-w"
        pred_T, _ = canonicalize_preds_to_anchor(y_pred, meta, pred_mode, conv_arrow, True, False)

        pred_poses = pred_T[0].cpu().numpy().astype(np.float32)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "pred_T_cam_anchor_obj.npy", pred_poses)
    print(f"  [Model] Saved {pred_poses.shape}")

    return pred_poses


# ==============================================================================
# Luma Tracking Predictions (FoundationPose on luma videos)
# ==============================================================================

@dataclass
class LumaReader:
    """Reader for luma video frames with SpaTrack geometry."""
    luma_path: str
    spatrack_npz: str
    first_fid: int

    def __post_init__(self):
        from decord import VideoReader, cpu

        if not os.path.isfile(self.luma_path):
            raise FileNotFoundError(f"Luma video not found: {self.luma_path}")

        cpu_count = os.cpu_count() or 1
        self.vr = VideoReader(self.luma_path, ctx=cpu(0), num_threads=cpu_count)
        self.total_frames = len(self.vr)
        f0 = self.vr[0].asnumpy()
        self.H = int(f0.shape[0])
        self.W = int(f0.shape[1])

        # Load SpaTrack data
        with np.load(self.spatrack_npz, allow_pickle=True) as z:
            K_raw = np.array(z.get("intrinsics"))
            if K_raw.ndim == 3:
                K_raw = K_raw[0]
            self._K_raw = K_raw.astype(np.float64)

            self._depths = None
            for k in ("depth", "depths", "D"):
                if k in z:
                    self._depths = np.asarray(z[k], dtype=np.float32)
                    break

            self._T_c_w = z.get("T_c_w", z.get("extrinsics", None))

        # Scale intrinsics to luma resolution
        if self._depths is not None:
            src_hw = (self._depths.shape[1], self._depths.shape[2])
        else:
            src_hw = (self.H, self.W)

        self.K = self._scale_K(self._K_raw, src_hw, (self.H, self.W))

    def _scale_K(self, K: np.ndarray, src_hw: tuple, dst_hw: tuple) -> np.ndarray:
        """Scale intrinsics."""
        H_src, W_src = src_hw
        H_dst, W_dst = dst_hw
        scale_x = W_dst / W_src
        scale_y = H_dst / H_src
        K_scaled = K.copy()
        K_scaled[0, 0] *= scale_x
        K_scaled[1, 1] *= scale_y
        K_scaled[0, 2] *= scale_x
        K_scaled[1, 2] *= scale_y
        return K_scaled.astype(np.float64)

    def get_color(self, i: int) -> np.ndarray:
        return self.vr[int(i)].asnumpy()

    def get_depth(self, i: int) -> np.ndarray:
        import cv2
        f = self.first_fid + int(i)
        if self._depths is None:
            return np.zeros((self.H, self.W), dtype=np.float32)
        if f >= len(self._depths):
            return np.zeros((self.H, self.W), dtype=np.float32)
        d = self._depths[f].astype(np.float32)
        d = cv2.resize(d, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        d[(d < 0.001) | ~np.isfinite(d)] = 0.0
        return d

    def T_c_w(self, f: int) -> Optional[np.ndarray]:
        if self._T_c_w is None:
            return None
        if f < len(self._T_c_w):
            return np.array(self._T_c_w[f]).astype(np.float32)
        return None



# SpaTrackV2 for luma depth/extrinsics
sys.path.append(str(REPO_ROOT / "SpaTrackerV2"))
try:
    from models.SpaTrackV2.models.predictor import Predictor as _ST_Predictor
    from models.SpaTrackV2.models.utils import get_points_on_a_grid as _st_get_points_on_a_grid
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track as _ST_VGGT4Track
    from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image as _st_preprocess_image
    _SPATRACKV2_AVAILABLE = True
except ImportError:
    _SPATRACKV2_AVAILABLE = False
    print("[WARN] SpaTrackV2 not available - luma tracking will use original extrinsics")

_SPATRACK_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SPATRACK_MODELS = None


def _ensure_spatrack_models():
    """Lazy-load SpaTrackV2 models for Luma depth estimation."""
    global _SPATRACK_MODELS
    if _SPATRACK_MODELS is not None:
        return _SPATRACK_MODELS
    if not _SPATRACKV2_AVAILABLE:
        return None
    if not torch.cuda.is_available():
        print("[Luma] SpaTrack models require CUDA")
        return None
    try:
        vggt_model = _ST_VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
        vggt_model.eval().to(_SPATRACK_DEVICE)
        predictor = _ST_Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
        predictor.spatrack.track_num = 100
        predictor.eval().to(_SPATRACK_DEVICE)
        _SPATRACK_MODELS = (vggt_model, predictor)
        print("[Luma] SpaTrackV2 models loaded")
        return _SPATRACK_MODELS
    except Exception as e:
        print(f"[Luma] Failed to load SpaTrack models: {e}")
        return None


def _spatrack_depth_for_clip(reader, frame_locals):
    """Run SpaTrackV2 on Luma frames to get depth maps and extrinsics (c2w)."""
    import cv2
    models = _ensure_spatrack_models()
    if models is None:
        return None
    vggt_model, predictor = models
    if len(frame_locals) == 0:
        return None
    try:
        frames = [reader.get_color(int(i)) for i in frame_locals]
        frame_arr = np.stack(frames).astype(np.float32)
        video_tensor = torch.from_numpy(frame_arr).permute(0, 3, 1, 2)
        video_tensor_in = _st_preprocess_image(video_tensor, keep_ratio=True)[None]

        dev = _SPATRACK_DEVICE
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            preds = vggt_model((video_tensor_in.to(dev) / 255.0))
            extrinsic = preds["poses_pred"]
            depth_map, depth_conf = preds["points_map"][..., 2], preds["unc_metric"]

        depth_tensor = depth_map.squeeze().detach().cpu().numpy()
        extrs = extrinsic.squeeze().detach().cpu().numpy()
        video_tensor_proc = video_tensor_in.squeeze()
        unc_metric = depth_conf.squeeze().detach().cpu().numpy() > 0.5

        frame_H, frame_W = video_tensor_proc.shape[2:]
        grid_pts = _st_get_points_on_a_grid(10, (frame_H, frame_W), device="cpu")
        query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()

        K_luma = np.asarray(reader.K, dtype=np.float32)
        intrs = np.repeat(K_luma[None, :, :], video_tensor_proc.shape[0], axis=0)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            (c2w_traj, intrs_out, point_map, conf_depth, track3d_pred, track2d_pred,
             vis_pred, conf_pred, video_vis) = predictor.forward(
                video_tensor_proc.to(dev),
                depth=depth_tensor,
                intrs=intrs,
                extrs=extrs,
                queries=query_xyt,
                fps=1,
                full_point=False,
                iters_track=4,
                query_no_BA=True,
                fixed_cam=False,
                stage=1,
                unc_metric=unc_metric,
                support_frame=len(video_tensor_proc) - 1,
                replace_ratio=0.2,
            )

        depth_save = point_map[:, 2, ...].clone()
        depth_save[conf_depth < 0.5] = 0
        depth_np = depth_save.detach().cpu().numpy().astype(np.float32)
        
        H_l, W_l = int(reader.H), int(reader.W)
        depth_resized = np.zeros((depth_np.shape[0], H_l, W_l), dtype=np.float32)
        for t in range(depth_np.shape[0]):
            depth_resized[t] = cv2.resize(depth_np[t], (W_l, H_l), interpolation=cv2.INTER_NEAREST)
        c2w_np = c2w_traj.detach().cpu().numpy().astype(np.float32)
        return depth_resized, c2w_np
    except Exception as e:
        print(f"[Luma] SpaTrack depth run failed: {e}")
        import traceback
        traceback.print_exc()
        return None



def _mesh_diameter(mesh) -> float:
    """Compute mesh diameter using oriented bounds."""
    import trimesh
    try:
        _, extents = trimesh.bounds.oriented_bounds(mesh)
        return float(np.linalg.norm(extents))
    except Exception:
        return 0.0


def extract_luma_predictions(row: Dict[str, str], output_dir: Path) -> Optional[np.ndarray]:
    """Extract FoundationPose tracking predictions on luma video."""
    try:
        import trimesh
        import nvdiffrast.torch as dr
        sys.path.insert(0, str(REPO_ROOT / "FoundationPose"))
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
        from future_pose_pred.src.data.fpose_io import load_object_poses
    except ImportError as e:
        print(f"  [Luma] Import error: {e}")
        return None

    fids = _parse_frame_ids(row)
    if len(fids) < 11:
        print(f"  [Luma] Insufficient frame_ids")
        return None

    obj_dir = row.get("object_dir", "")
    sp_npz = row.get("spatrack_npz", "")
    mesh_path = row.get("mesh_path", "")

    # Check luma video exists
    luma_name = _luma_name_for_row(row)
    luma_path = LUMA_DIR / luma_name
    if not luma_path.exists():
        print(f"  [Luma] Video not found: {luma_path}")
        return None

    if not os.path.isfile(mesh_path):
        print(f"  [Luma] Mesh not found: {mesh_path}")
        return None

    if not os.path.isfile(sp_npz):
        print(f"  [Luma] SpaTrack npz not found: {sp_npz}")
        return None

    # Frame structure
    first_fid = fids[0]
    anchor_fid = fids[3]
    anchor_idx = anchor_fid - first_fid

    # Load mesh and scale
    mesh = trimesh.load(mesh_path, force="mesh")
    raw_diam = _mesh_diameter(mesh)

    # Get target diameter from run_summary.json
    run_summary_path = os.path.join(obj_dir, "foundationpose10", "run_summary.json")
    target_diam = 0.0
    if os.path.isfile(run_summary_path):
        with open(run_summary_path) as f:
            run_summary = json.load(f)
        target_diam = float(run_summary.get("mesh_diameter", 0.0))

    if target_diam > 0 and raw_diam > 0:
        scale = target_diam / raw_diam
        mesh.apply_scale(scale)
        print(f"  [Luma] Mesh scaled: {raw_diam:.4f} -> {target_diam:.4f}")

    # Initialize reader
    try:
        reader = LumaReader(str(luma_path), sp_npz, first_fid)
    except Exception as e:
        print(f"  [Luma] Reader init failed: {e}")
        return None

    # Load initial pose from FP at anchor
    fp_dir = os.path.join(obj_dir, "foundationpose10", "ob_in_cam")
    obj_fp = load_object_poses(fp_dir, verbose=False)
    fp_fids = {int(f): i for i, f in enumerate(obj_fp.get("frame_ids", []))}

    if anchor_fid not in fp_fids:
        print(f"  [Luma] Anchor {anchor_fid} not in FP poses")
        return None

    T_anchor = obj_fp["T_c_o"][fp_fids[anchor_fid]].astype(np.float32)

    # Initialize FoundationPose
    try:
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()

        dbg_dir = str(REPO_ROOT / "tmp")
        os.makedirs(dbg_dir, exist_ok=True)

        est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=0,
            debug_dir=dbg_dir,
        )

        if target_diam > 0:
            est.diameter = target_diam
            est.vox_size = max(est.diameter / 20.0, 0.003)

        # Set initial pose
        tf_to_center = est.get_tf_to_centered_mesh().detach().cpu().numpy().astype(np.float32)
        tf_center_inv = np.linalg.inv(tf_to_center)
        T_centered_init = (T_anchor @ tf_center_inv).astype(np.float32)
        est.pose_last = torch.from_numpy(T_centered_init).float().to("cuda")

    except Exception as e:
        print(f"  [Luma] FoundationPose init failed: {e}")
        return None

    # Check frame range
    if anchor_idx < 0 or (anchor_idx + 7) >= reader.total_frames:
        print(f"  [Luma] Invalid anchor_idx range: {anchor_idx}, total={reader.total_frames}")
        return None

    # Run SpaTrackV2 on the 8-frame Luma clip to get depth and extrinsics
    frame_locals_seq = [anchor_idx + j for j in range(8)]
    spatrack_clip = _spatrack_depth_for_clip(reader, frame_locals_seq)
    depth_clip = None
    extr_clip_c2w = None
    if spatrack_clip is not None:
        depth_clip, extr_clip_c2w = spatrack_clip
        print(f"  [Luma] SpaTrackV2 generated depth and extrinsics for 8 frames")
    else:
        print(f"  [Luma] SpaTrackV2 unavailable, falling back to original extrinsics")

    # Get anchor extrinsics (prefer SpaTrackV2 if available)
    if extr_clip_c2w is not None and extr_clip_c2w.shape[0] >= 1:
        E_anchor = extr_clip_c2w[0]
    else:
        E_anchor = reader.T_c_w(anchor_fid)
    if E_anchor is None:
        print(f"  [Luma] Missing extrinsics for anchor")
        return None

    # Determine convention
    E_anchor_inv = _invert_T(E_anchor)
    conv = _select_extrinsics_convention(T_anchor, E_anchor, E_anchor_inv)

    # Track over 8 frames
    pred_poses = []
    K_fp = np.asarray(reader.K, dtype=np.float32)

    for j in range(8):
        i_local = anchor_idx + j
        f_global = anchor_fid + j

        color = reader.get_color(i_local)
        
        # Use SpaTrackV2 depth if available, else fall back to original
        if depth_clip is not None and j < depth_clip.shape[0]:
            depth = depth_clip[j]
        else:
            depth = reader.get_depth(i_local)

        try:
            T_pred = est.track_one(rgb=color, depth=depth, K=K_fp, iteration=3, ob_mask=None)
        except Exception as e:
            print(f"  [Luma] Tracking failed at frame {j}: {e}")
            return None

        if T_pred is None:
            print(f"  [Luma] Tracking returned None at frame {j}")
            return None

        # Re-express to anchor camera using SpaTrackV2 extrinsics if available
        if extr_clip_c2w is not None and j < extr_clip_c2w.shape[0]:
            E_k = extr_clip_c2w[j]
        else:
            E_k = reader.T_c_w(f_global)
        if E_k is None:
            print(f"  [Luma] Missing extrinsics for frame {f_global}")
            return None

        T_camA = _reexpress_object_pose(T_pred.astype(np.float32), E_k, E_anchor, conv)
        pred_poses.append(T_camA)

    pred_poses = np.stack(pred_poses, axis=0).astype(np.float32)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "luma_T_cam_anchor_obj.npy", pred_poses)
    print(f"  [Luma] Saved {pred_poses.shape}")

    return pred_poses


# ==============================================================================
# Main
# ==============================================================================

def process_sample(row: Dict[str, str], output_base: Path, extract_gt: bool = True,
                   extract_model: bool = True, extract_luma: bool = True,
                   model_state: dict = None, max_frame_diff: int = 10) -> bool:
    """Process a single sample."""
    sample_name = _sample_name_for_row(row)
    output_dir = output_base / sample_name

    print(f"\n[Processing] {sample_name}")

    gt_poses = None

    # Extract GT
    if extract_gt:
        gt_poses = extract_gt_poses(row, output_dir)
        if gt_poses is None:
            print(f"  [SKIP] GT extraction failed")
            return False

    # Extract model predictions
    if extract_model:
        extract_model_predictions(row, output_dir, gt_poses, model_state, max_frame_diff)

    # Extract luma predictions
    if extract_luma:
        luma_poses = extract_luma_predictions(row, output_dir)
        if luma_poses is None:
            print(f"  [WARN] Luma extraction failed")

    return True


def load_checkpoint_compat(model, ckpt_path: str):
    """Load checkpoint with compatibility mapping for old checkpoints."""
    print(f"[Model] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Extract state dict
    if 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    elif 'model' in ckpt:
        sd = ckpt['model']
    else:
        sd = ckpt

    # Strip module. prefix if present
    if any(k.startswith('module.') for k in sd.keys()):
        sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}

    model_sd = model.state_dict()
    ckpt_keys = set(sd.keys())
    model_keys = set(model_sd.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    # Check shape mismatches
    shape_mismatch = {}
    common = ckpt_keys & model_keys
    for k in common:
        if sd[k].shape != model_sd[k].shape:
            shape_mismatch[k] = (sd[k].shape, model_sd[k].shape)

    print(f"  Checkpoint keys: {len(ckpt_keys)}, Model keys: {len(model_keys)}")
    print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}, Shape mismatch: {len(shape_mismatch)}")

    # Filter out shape mismatches from loading
    loadable_sd = {k: v for k, v in sd.items() if k in model_keys and k not in shape_mismatch}

    model.load_state_dict(loadable_sd, strict=False)
    print(f"  Loaded {len(loadable_sd)} keys successfully")


def load_model_and_dataset():
    """Load PoserV0 model with old EPIC checkpoint using epic_eval config."""
    import hydra
    from omegaconf import OmegaConf

    from future_pose_pred.src.models.poser_v0.builder import build_poser_v0
    from future_pose_pred.src.utils.config_adapter import apply_config_adapter
    from future_pose_pred.src.utils.data_utils import get_dataset
    from future_pose_pred.src.eval_main import build_val_subset

    # Initialize hydra with epic_eval config
    config_dir = str(REPO_ROOT / "future_pose_pred" / "conf")

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: int(eval(expr, {"__builtins__": {}}, {})))

    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name="epic_eval")

    cfg = apply_config_adapter(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build PoserV0 model (old architecture for checkpoint compatibility)
    print("[Model] Building PoserV0 model (old architecture)...")
    model = build_poser_v0(**OmegaConf.to_container(cfg.model, resolve=True))
    model.to(device)

    # Load old checkpoint
    ckpt_path = str(REPO_ROOT / "future_pose_pred" / "outputs" / "dit_epic_16gpus_70split_iou_drop_0.1" / "checkpoints" / "best.pt")
    if os.path.exists(ckpt_path):
        load_checkpoint_compat(model, ckpt_path)
    else:
        print(f"[Model] WARNING: Checkpoint not found at {ckpt_path}")

    model.eval()

    # Load dataset
    print("[Model] Loading dataset...")
    dataset = get_dataset(cfg)
    val_dataset = build_val_subset(dataset, cfg, seed=42)

    print(f"[Model] Dataset loaded with {len(val_dataset)} samples")

    return {
        "model": model,
        "cfg": cfg,
        "device": device,
        "dataset": val_dataset,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract poses for pose_video_viz")
    parser.add_argument("--sample", type=str, help="Process specific sample")
    parser.add_argument("--all", action="store_true", help="Process all samples")
    parser.add_argument("--limit", type=int, default=5, help="Limit samples (default: 5)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--model-only", action="store_true", help="Only extract model predictions")
    parser.add_argument("--luma-only", action="store_true", help="Only extract luma predictions")
    parser.add_argument("--gt-only", action="store_true", help="Only extract GT poses")
    parser.add_argument("--with-model", action="store_true", help="Load model for predictions (requires hydra)")
    parser.add_argument("--max-frame-diff", type=int, default=10, help="Max frame diff for existing pred matching (default: 10)")
    args = parser.parse_args()

    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    # Determine what to extract
    extract_gt = not (args.model_only or args.luma_only) or args.gt_only
    extract_model = not (args.gt_only or args.luma_only) or args.model_only
    extract_luma = not (args.gt_only or args.model_only) or args.luma_only

    if args.gt_only:
        extract_gt, extract_model, extract_luma = True, False, False
    if args.model_only:
        extract_gt, extract_model, extract_luma = True, True, False
    if args.luma_only:
        extract_gt, extract_model, extract_luma = True, False, True

    print(f"Extracting: GT={extract_gt}, Model={extract_model}, Luma={extract_luma}")

    # Load model if requested
    model_state = None
    if extract_model and args.with_model:
        print("\n[Loading model and dataset...]")
        try:
            model_state = load_model_and_dataset()
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            print("[WARN] Continuing without model predictions")

    rows = _load_trajs_csv(CSV_PATH)
    if not rows:
        print("[ERROR] No samples in CSV")
        return 1

    print(f"Found {len(rows)} samples")

    if args.sample:
        rows = [r for r in rows if _sample_name_for_row(r) == args.sample]
        if not rows:
            print(f"[ERROR] Sample '{args.sample}' not found")
            return 1
    elif not args.all:
        rows = rows[:args.limit]

    print(f"Processing {len(rows)} samples...")

    success = 0
    for row in rows:
        if process_sample(row, output_base, extract_gt, extract_model, extract_luma, model_state, args.max_frame_diff):
            success += 1

    print(f"\n[Done] {success}/{len(rows)} samples processed")
    print(f"Output: {output_base}")

    return 0


if __name__ == "__main__":
    exit(main())
