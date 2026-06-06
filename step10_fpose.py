#!/usr/bin/env python3
"""FoundationPose 6D tracking over EPIC clips (sharded)."""

import os
import sys
import shutil
import traceback
import argparse
import glob
import hashlib
from pathlib import Path
import csv
import json
from typing import NamedTuple, Optional, List, Tuple

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

sys.path.append("FoundationPose")

import cv2
import imageio
import numpy as np
import nvdiffrast.torch as dr
import pandas as pd
import trimesh
from decord import VideoReader, cpu
import scipy.spatial
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull, distance, cKDTree
from tqdm import tqdm

from Utils import (
    depth2xyzmap,
    set_logging_format,
    set_seed,
)
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
from offscreen_renderer import ModelRendererOffscreen
from utils import rprint as print


def as_float(x, default=float("nan")):
    if x is None:
        return default
    try:
        if hasattr(x, "detach"):
            x = x.detach().cpu()
        return float(x.item()) if hasattr(x, "item") else float(x)
    except Exception:
        return default


def load_config():
    p = argparse.ArgumentParser(description="FoundationPose tracking over EPIC (driver like step7_vas).")

    p.add_argument("--video_root", default="/gpfs/scrubbed/rustin/manip_data", help="Root containing narration_id/*/action.mp4 and objects/")
    p.add_argument("--output_root", default="/gpfs/scrubbed/rustin/manip_data", help="Where narration_id subfolders live (usually same as video_root)")
    p.add_argument("--csv_file", type=str, default="EPIC_100.csv", help="EPIC csv with narration_id,duration_s,no_hands_presence,...")
    p.add_argument("--ext", type=str, default="mp4", help="Video extension (case-insensitive match on 'action.mp4')")

    p.add_argument("--start_video_idx", type=int, default=0, help="Manual slice start (AFTER sharding).")
    p.add_argument("--end_video_idx", type=int, default=-1, help="Manual slice end (AFTER sharding). -1 = no limit.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="Recompute FoundationPose results even if they already exist")

    p.add_argument("--num_shards", type=int, default=1, help="Total number of shards for job array.")
    p.add_argument("--shard_idx", type=int, default=0, help="This shard index [0..num_shards-1].")

    # FoundationPose/reader specific args (unchanged functionality)
    p.add_argument("--depth_unit_scale", type=float, default=1.0)
    p.add_argument("--est_refine_iter", type=int, default=5)
    p.add_argument("--track_refine_iter", type=int, default=1)
    p.add_argument("--mask_erode", type=int, default=0)
    p.add_argument("--mask_dilate", type=int, default=2)
    p.add_argument("--sil_weight", type=float, default=0.25)
    p.add_argument("--overflow_weight", type=float, default=0.10)
    p.add_argument("--iou_min", type=float, default=0.2)
    p.add_argument("--no_mask", action="store_true", help="Disable mask usage during tracking")
    p.add_argument("--debug", type=int, default=2)
    p.add_argument("--init_candidates", type=int, default=5, help="Number of candidate frames to try for initialization")
    p.add_argument("--min_init_iou", type=float, default=0.4, help="Minimum IoU required to accept initial pose and proceed")

    # Deprecated/conf kept for compatibility
    p.add_argument("--init_lock_scale", action="store_true", help="Use a single robust scale for all init candidates")
    p.add_argument("--mesh_dir", default="trellis", help="Folder under each object that contains model.glb (e.g., trellis)")
    p.add_argument("--reinit_max_retries", type=int, default=5, help="Re-register attempts on the same frame before clean-mask fallback")
    p.add_argument("--reinit_search_window", type=int, default=8, help="± window to search for nearest clean-mask frame for re-init")
    p.add_argument("--reinit_cooldown", type=int, default=5, help="Frames to slightly relax IoU gate after a re-init")
    p.add_argument("--reinit_budget", type=int, default=8, help="Abort this object if total successful re-registrations reach this number (forward+backward).")
    p.add_argument("--min_mask_area_px", type=int, default=500, help="Minimum mask area to trust for initialization/re-init")
    p.add_argument("--iou_drop_gate", type=float, default=0.05, help="Reject update if IoU drops more than this and is below threshold")
    p.add_argument("--scale_refine_skip_underfill", type=float, default=0.30, help="Skip silhouette scale refine if underfill ratio (mask_o & ~mask_r)/mask_o exceeds this")
    p.add_argument("--scale_refine_depth_w", type=float, default=0.5, help="Weight of depth term during scale search")
    p.add_argument("--scale_refine_prior_w", type=float, default=0.25, help="Weight of log-scale prior around s=1.0 during scale search")
    p.add_argument("--scale_refine_lo", type=float, default=0.7, help="Lower bound multiplier for silhouette scale search around prior (1.0)")
    p.add_argument("--scale_refine_hi", type=float, default=1.4, help="Upper bound multiplier for silhouette scale search around prior (1.0)")
    p.add_argument("--scale_core_erode_px", type=int, default=8, help="Erode mask by this many px to form a core region for robust scale/pose.")
    p.add_argument("--scale_mask_close_px", type=int, default=3, help="Morphological closing kernel to fill small holes before erosion (0=off).")
    p.add_argument("--scale_neighbors", type=int, default=3, help="Frames on each side of init frame to use for robust scale.")
    p.add_argument("--scale_trim_lo", type=float, default=0.1, help="Lower quantile for trimming radii when fitting scale.")
    p.add_argument("--scale_trim_hi", type=float, default=0.9, help="Upper quantile for trimming radii when fitting scale.")
    p.add_argument("--use_core_mask_for_register", action="store_true", help="Use eroded core mask for initial register() to reduce occlusion bias.")

    # Tracking quality weights (for score_t)
    p.add_argument("--w_underfill", type=float, default=0.7)
    p.add_argument("--w_overflow", type=float, default=0.3)
    p.add_argument("--w_rmse", type=float, default=0.5)
    p.add_argument("--w_cd", type=float, default=1.0)
    p.add_argument("--w_temporal", type=float, default=0.3)
    p.add_argument("--theta0_deg", type=float, default=10.0)  # rotation scale for temporal penalty

    # Gating thresholds
    p.add_argument("--gate_rmse_norm", type=float, default=0.05)  # accept if <= and underfill <= 0.3
    p.add_argument("--gate_underfill", type=float, default=0.30)
    p.add_argument("--gate_dR_deg", type=float, default=15.0)
    p.add_argument("--gate_dt_norm", type=float, default=0.25)
    p.add_argument("--gate_drop", type=float, default=0.2)  # drop in score_t vs prev
    p.add_argument(
        "--accept_mode",
        choices=["lenient", "strict", "off"],
        default="lenient",
        help="Acceptance policy: 'lenient' (relaxed gates), 'strict' (as-is), or 'off' (accept any successful pose).",
    )
    p.add_argument("--accept_relax_factor", type=float, default=1.5, help="When accept_mode='lenient', multiply most gates by this (>1 means less strict).")
    p.set_defaults(use_core_mask_for_register=True)

    return p.parse_args()


def stable_int_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def masked_depth_to_points(depth, mask, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v, u = np.nonzero(mask)
    if v.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth[v, u].astype(np.float32)
    valid = np.isfinite(z) & (z > 0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    u = u[valid]
    v = v[valid]
    z = z[valid]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def hull_diameter(points, max_sample=60000):
    if points.shape[0] < 2:
        return 0.0
    P = points if points.shape[0] <= max_sample else points[np.random.choice(points.shape[0], max_sample, replace=False)]
    try:
        hull = ConvexHull(P)
        H = P[hull.vertices]
        return float(distance.pdist(H, "euclidean").max()) if H.shape[0] >= 2 else 0.0
    except (ValueError, scipy.spatial.QhullError):
        m = min(P.shape[0], 4000)
        A = P[np.random.choice(P.shape[0], m, replace=False)]
        return float(np.linalg.norm(A[:, None, :] - A[None, :, :], axis=-1).max())


def pca_span(points):
    if points.shape[0] < 2:
        return 0.0
    C = points - points.mean(0, keepdims=True)
    proj = C @ np.linalg.svd(C, full_matrices=False)[2][0]
    return float(proj.max() - proj.min())


def visible_diameter(depth, mask, K, trim=0.02):
    P = masked_depth_to_points(depth, mask, K)
    if P.shape[0] < 50:
        return 0.0
    z = P[:, 2]
    lo, hi = np.quantile(z, [trim, 1 - trim])
    P = P[(z >= lo) & (z <= hi)]
    return max(pca_span(P), hull_diameter(P)) if P.shape[0] >= 50 else 0.0


def compute_depth_rmse_on_intersection(mask_r_bool, mask_o_bool, d_r, d_o):
    inter = np.logical_and(mask_r_bool, mask_o_bool)
    if not np.any(inter):
        return np.inf, np.inf
    dz = d_r[inter] - d_o[inter]
    rmse = float(np.sqrt(np.mean(np.square(dz))))
    return rmse, rmse / max(float(np.median(d_o[inter])), 1e-6)


def _voxel_downsample(P, vox=0.005):
    if P.size == 0:
        return P
    _, idx = np.unique(np.floor(P / max(vox, 1e-9)).astype(np.int64), axis=0, return_index=True)
    return P[np.sort(idx)]


def chamfer_cd(Po, Pr, vox=0.005):
    if Po.shape[0] == 0 or Pr.shape[0] == 0:
        return np.inf
    Po, Pr = _voxel_downsample(Po, vox), _voxel_downsample(Pr, vox)
    d_or, _ = cKDTree(Po).query(Pr, k=1, workers=-1)
    d_ro, _ = cKDTree(Pr).query(Po, k=1, workers=-1)
    return float(d_or.mean() + d_ro.mean())


def backproject_masked_points(depth, mask_bool, K):
    return masked_depth_to_points(depth, mask_bool, K)


def pose_deltas(prev4x4, cur4x4, diam):
    R = prev4x4[:3, :3].T @ cur4x4[:3, :3]
    dR_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
    dt_norm = float(np.linalg.norm(cur4x4[:3, 3] - prev4x4[:3, 3]) / max(diam, 1e-9))
    return dR_deg, dt_norm


def tracking_score(iou, overflow, underfill, rmse_norm, cd_norm, dR_deg, dt_norm, cfg):
    temporal_pen = (dR_deg / max(cfg.theta0_deg, 1e-6)) + dt_norm
    return (
        float(iou)
        - cfg.w_underfill * float(underfill)
        - cfg.w_overflow * float(overflow)
        - cfg.w_rmse * float(rmse_norm)
        - cfg.w_cd * float(cd_norm)
        - cfg.w_temporal * float(temporal_pen)
    )


def _dbg_score_parts(iou, overflow, underfill, rmse_norm, cd_norm, dR_deg, dt_norm, cfg, prefix=""):
    if getattr(cfg, "debug", 0) < 3:
        return
    temporal_pen = (dR_deg / max(cfg.theta0_deg, 1e-6)) + dt_norm
    tqdm.write(
        f"{prefix}[score parts] IoU={iou:.3f}, underfill={underfill:.3f}, overflow={overflow:.3f}, rmse_n={rmse_norm:.3f}, cd_n={cd_norm:.3f}, temporal={temporal_pen:.3f}"
    )


class Metrics(NamedTuple):
    iou: float
    overflow: float
    underfill: float
    rmse: float
    rmse_norm: float
    cd: float
    cd_norm: float
    dR_deg: float
    dt_norm: float
    score_net: float
    score_t: float


def mask_metrics(mask_r_bool, mask_o_bool):
    mr, mo = mask_r_bool.astype(bool), mask_o_bool.astype(bool)
    inter, union = np.logical_and(mr, mo).sum(), np.logical_or(mr, mo).sum()
    return float(inter / max(union, 1)), float(np.logical_and(mr, ~mo).sum() / max(mr.sum(), 1)), float(np.logical_and(mo, ~mr).sum() / max(mo.sum(), 1))


def compute_mesh_diameter(mesh):
    try:
        return float(np.linalg.norm(trimesh.bounds.oriented_bounds(mesh)[1]))
    except Exception:
        return 0.0


def _metric_masks_from_depth(rend_depth, ob_mask, mask_dilate, fallback_mode="zeros"):
    mask_r = rend_depth > 0
    if ob_mask is not None:
        mask_o = _close_small_holes(ob_mask.astype(bool), max(0, int(mask_dilate)))
    else:
        mask_o = (rend_depth > 0) if (fallback_mode == "render") else np.zeros_like(mask_r, dtype=bool)
    return mask_r, mask_o


def compute_metrics_common(renderer, mesh, pose, depth_obs, ob_mask, K, mesh_diam, cfg, prev_pose=None, fallback_mode="zeros"):
    rend_rgb, rend_depth = renderer.render(mesh=mesh, ob_in_cvcam=pose)
    mask_r, mask_o = _metric_masks_from_depth(rend_depth, ob_mask, getattr(cfg, "mask_dilate", 0), fallback_mode=fallback_mode)
    iou, overflow, underfill = mask_metrics(mask_r, mask_o)
    rmse, rmse_norm = compute_depth_rmse_on_intersection(mask_r, mask_o, rend_depth, depth_obs)
    Po = backproject_masked_points(depth_obs, mask_o, K)
    Pr = backproject_masked_points(rend_depth, mask_r, K)
    cd = chamfer_cd(Po, Pr, vox=max(0.05 * mesh_diam, 1e-3))
    cd_norm = cd / max(mesh_diam, 1e-9)
    if prev_pose is None:
        dR_deg, dt_norm = 0.0, 0.0
    else:
        dR_deg, dt_norm = pose_deltas(prev_pose, pose, mesh_diam)
    score_t = tracking_score(iou, overflow, underfill, rmse_norm, cd_norm, dR_deg, dt_norm, cfg)
    return rend_rgb, rend_depth, iou, overflow, underfill, rmse, rmse_norm, cd, cd_norm, dR_deg, dt_norm, score_t, mask_r, mask_o


def estimate_scale_multi_frame(reader, mesh, init_frame, max_neighbors=None, n_sample=10000, outlier_ratio=0.25, cfg=None):
    valid_ids = sorted([int(s) for s in reader.id_strs])
    if not valid_ids:
        return None

    center = init_frame if init_frame in valid_ids else min(valid_ids, key=lambda x: abs(x - init_frame))
    max_neighbors = max_neighbors or int(getattr(cfg, "scale_neighbors", 3) if cfg else 3)
    cand = [j for j in valid_ids if abs(j - center) <= max_neighbors]

    model_pts = np.asarray(mesh.vertices)
    d_model = hull_diameter(model_pts)
    if d_model <= 0:
        return None

    erode_px = int(getattr(cfg, "scale_core_erode_px", 8) if cfg else 8)
    close_px = int(getattr(cfg, "scale_mask_close_px", 3) if cfg else 3)
    qlo = float(getattr(cfg, "scale_trim_lo", 0.1) if cfg else 0.1)
    qhi = float(getattr(cfg, "scale_trim_hi", 0.9) if cfg else 0.9)
    if qhi <= qlo:
        qlo, qhi = 0.1, 0.9

    scales, weights = [], []
    for i in cand:
        depth = reader.get_depth(i)
        mask_core = make_core_mask(reader.get_mask(i).astype(bool), erode_px=erode_px, close_px=close_px)
        area = float(mask_core.sum())
        if area < 10:
            continue
        depth_pts = masked_depth_to_points(depth, mask_core, reader.K)
        if depth_pts.shape[0] == 0:
            continue

        X = model_pts if model_pts.shape[0] <= n_sample else model_pts[np.random.choice(model_pts.shape[0], n_sample, replace=False)]
        Y = depth_pts if depth_pts.shape[0] <= n_sample else depth_pts[np.random.choice(depth_pts.shape[0], n_sample, replace=False)]
        rX = np.linalg.norm(X - X.mean(0), axis=1)
        rY = np.linalg.norm(Y - Y.mean(0), axis=1)
        rX = rX[(rX >= np.quantile(rX, qlo)) & (rX <= np.quantile(rX, qhi))]
        rY = rY[(rY >= np.quantile(rY, qlo)) & (rY <= np.quantile(rY, qhi))]
        if rX.size == 0 or rY.size == 0:
            continue
        s = float(np.median(rY) / max(np.median(rX), 1e-12))

        if not np.isfinite(s) or s <= 0:
            d_depth = hull_diameter(depth_pts)
            if d_depth > 0:
                s = float(d_depth / d_model)

        if np.isfinite(s) and s > 0:
            scales.append(s)
            weights.append(area)

    if not scales:
        return None

    scales = np.array(scales, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    order = np.argsort(scales)
    w = weights[order] / max(weights.sum(), 1.0)
    s_med = float(scales[order][np.searchsorted(np.cumsum(w), 0.5, side="left")])
    if not np.isfinite(s_med) or s_med <= 0:
        return None

    keep = np.abs(scales - s_med) <= outlier_ratio * s_med
    if not np.any(keep):
        print(f"Scale estimation: weighted median={s_med:.6f} from {len(scales)} frames; no inliers after trimming")
        return s_med
    s_final = float(np.average(scales[keep], weights=weights[keep]))
    if np.isfinite(s_final) and s_final > 0:
        print(f"Scale estimation: weighted median={s_med:.6f}, trimmed weighted mean={s_final:.6f}, kept {keep.sum()}/{len(scales)} frames")
        return s_final
    print(f"Scale estimation: returning weighted median={s_med:.6f}")
    return s_med


def _scale_objective(log_s, mesh, renderer, pose, mask_obs_bool, depth_obs, s_prior, w_iou, w_depth, w_prior):
    s = float(np.exp(log_s))
    M = mesh.copy()
    M.apply_scale(s)
    _, d_r = renderer.render(mesh=M, ob_in_cvcam=pose)
    m_r = d_r > 0
    inter = np.logical_and(m_r, mask_obs_bool)
    union = np.logical_or(m_r, mask_obs_bool)
    iou = inter.sum() / max(union.sum(), 1)
    depth_l1 = float(np.median(np.abs(d_r[inter] - depth_obs[inter]))) if inter.any() else 1e3
    prior = (np.log(s) - np.log(max(1e-8, s_prior))) ** 2
    return -(w_iou * iou - w_depth * depth_l1 - w_prior * prior)


def scale_object_for_silhouette_depthaware(mesh, renderer, pose, mask_obs_bool, depth_obs, s_prior=1.0, w_iou=1.0, w_depth=0.5, w_prior=0.25, lo_mul=0.7, hi_mul=1.4):
    if float(mask_obs_bool.sum()) < 1:
        return None
    lo, hi = np.log(lo_mul * s_prior), np.log(hi_mul * s_prior)
    res = minimize_scalar(
        lambda log_s: _scale_objective(log_s, mesh, renderer, pose, mask_obs_bool, depth_obs, s_prior, w_iou, w_depth, w_prior),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-3, "maxiter": 60},
    )
    s_opt = float(np.exp(res.x))
    return s_opt if np.isfinite(s_opt) and s_opt > 0 else None


def select_init_candidates(id_list_sorted, k):
    if not id_list_sorted or k >= len(id_list_sorted):
        return id_list_sorted or []
    idxs = sorted({int(round(p)) for p in np.linspace(0, len(id_list_sorted) - 1, num=k)})
    return [id_list_sorted[i] for i in idxs]


def evaluate_init_candidate(reader, base_mesh, frame_idx, scorer, refiner, glctx, debug_dir, cfg, shared_scale=None, renderer_vis=None):
    mesh_cand = base_mesh.copy()
    s_est = float(shared_scale) if shared_scale is not None else estimate_scale_multi_frame(reader, mesh_cand, init_frame=int(frame_idx), cfg=cfg)
    if s_est is not None and np.isfinite(s_est) and s_est > 0:
        mesh_cand.apply_scale(s_est)

    est_cand = FoundationPose(
        model_pts=mesh_cand.vertices,
        model_normals=mesh_cand.vertex_normals,
        mesh=mesh_cand,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(debug_dir),
        debug=int(cfg.debug),
        glctx=glctx,
    )

    color = reader.get_color(frame_idx)
    depth = reader.get_depth(frame_idx)
    ob_mask = None
    if not cfg.no_mask:
        m = reader.get_mask(frame_idx)
        if np.array(m).sum() > 0:
            ob_mask = m.astype(bool)
    ob_mask_reg = ob_mask
    if ob_mask is not None and cfg.use_core_mask_for_register:
        ob_mask_reg = make_core_mask(ob_mask.astype(bool), erode_px=int(cfg.scale_core_erode_px), close_px=int(cfg.scale_mask_close_px))
    pose = est_cand.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask_reg, iteration=int(cfg.est_refine_iter))

    mesh_diam = compute_mesh_diameter(mesh_cand)
    rv = renderer_vis or ModelRendererOffscreen(reader.K, reader.H, reader.W, zfar=10.0)
    _, _, iou, overflow, underfill, rmse, rmse_norm, cd, cd_norm, _, _, score_t, _, _ = compute_metrics_common(
        renderer=rv, mesh=mesh_cand, pose=pose, depth_obs=depth, ob_mask=ob_mask, K=reader.K, mesh_diam=mesh_diam, cfg=cfg, prev_pose=None, fallback_mode="zeros"
    )

    print(f"Init candidate frame {frame_idx}: net={as_float(getattr(est_cand, 'last_score', 0.0)):.4f}, IoU={iou:.3f}, t={score_t:.4f}")
    return float(score_t), float(iou), pose, mesh_cand


def init_debug_dirs(debug_dir):
    shutil.rmtree(debug_dir, ignore_errors=True)
    os.makedirs(debug_dir / "ob_in_cam", exist_ok=True)


def build_estimator(mesh, scorer, refiner, debug_dir, cfg, glctx, iou_min=None):
    return FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(debug_dir),
        debug=int(cfg.debug),
        glctx=glctx,
        **({"iou_min": iou_min} if iou_min is not None else {}),
    )


def get_clean_candidate_frames(reader, id_list_sorted, k, cfg=None):
    clean = getattr(reader, "clean_mask_keys", set())
    pool = [x for x in id_list_sorted if x in clean]
    erode_px = int(getattr(cfg, "scale_core_erode_px", 8) if cfg else 8)
    close_px = int(getattr(cfg, "scale_mask_close_px", 3) if cfg else 3)
    areas = [(idx, core_area_for_frame(reader, idx, erode_px, close_px)) for idx in pool]
    pool = [idx for idx, _ in sorted(areas, key=lambda t: t[1], reverse=True)]
    cand = select_init_candidates(pool, k) if pool else []
    if len(cand) < k:
        fill = [i for i in select_init_candidates(id_list_sorted, k) if i not in cand]
        cand = (cand + fill)[:k]
    return cand


def choose_best_init(reader, mesh, cand_frames, debug_dir, cfg, scorer, refiner, glctx, renderer_vis=None):
    best_by_score = None
    best_by_iou = None
    shared_scale = None
    if cfg.init_lock_scale and cand_frames:
        shared_scale = estimate_scale_multi_frame(reader, mesh, init_frame=int(cand_frames[0]), cfg=cfg)
        if shared_scale is not None and (not np.isfinite(shared_scale) or shared_scale <= 0):
            shared_scale = None
    for fidx in cand_frames:
        try:
            score_t_f, iou_f, pose_f, mesh_f = evaluate_init_candidate(
                reader=reader,
                base_mesh=mesh,
                frame_idx=fidx,
                scorer=scorer,
                refiner=refiner,
                glctx=glctx,
                debug_dir=debug_dir,
                cfg=cfg,
                shared_scale=shared_scale,
                renderer_vis=renderer_vis,
            )
            if best_by_score is None or score_t_f > best_by_score[0]:
                best_by_score = (score_t_f, fidx, pose_f, mesh_f, iou_f, None)
            if best_by_iou is None or iou_f > best_by_iou[0]:
                best_by_iou = (iou_f, fidx, pose_f, mesh_f, iou_f, None)
        except Exception as e:
            traceback.print_exc()
            print(f"Candidate evaluation failed at frame {fidx}: {e}")
    return best_by_score, best_by_iou


def perform_initial_registration(reader, est, frame_idx, cfg):
    color = reader.get_color(frame_idx)
    depth = reader.get_depth(frame_idx)
    ob_mask = None
    if not cfg.no_mask:
        m = reader.get_mask(frame_idx).astype(bool)
        if np.array(m).sum() > 0:
            ob_mask = make_core_mask(m, erode_px=int(cfg.scale_core_erode_px), close_px=int(cfg.scale_mask_close_px)) if cfg.use_core_mask_for_register else m
    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, iteration=int(cfg.est_refine_iter))
    return pose, color, depth, ob_mask


def refine_scale_with_silhouette(reader, renderer_vis, mesh, pose, ob_mask, color, depth, est, scorer, refiner, debug_dir, cfg, glctx):
    _, rend_depth_area = renderer_vis.render(mesh=mesh, ob_in_cvcam=pose)
    raw_obs = ob_mask.astype(bool) if ob_mask is not None else (rend_depth_area > 0)
    mask_obs = make_core_mask(raw_obs, erode_px=int(cfg.scale_core_erode_px), close_px=int(cfg.scale_mask_close_px))
    _, _, underfill0 = mask_metrics(rend_depth_area > 0, mask_obs)
    if underfill0 > float(cfg.scale_refine_skip_underfill):
        print(f"Skip scale refine (underfill={underfill0:.2f} > {float(cfg.scale_refine_skip_underfill):.2f}); keep current scale.")
    else:
        s_opt = scale_object_for_silhouette_depthaware(
            mesh=mesh,
            renderer=renderer_vis,
            pose=pose,
            mask_obs_bool=mask_obs,
            depth_obs=depth,
            s_prior=1.0,
            w_iou=1.0,
            w_depth=float(cfg.scale_refine_depth_w),
            w_prior=float(cfg.scale_refine_prior_w),
            lo_mul=float(cfg.scale_refine_lo),
            hi_mul=float(cfg.scale_refine_hi),
        )
        if s_opt and np.isfinite(s_opt) and s_opt > 0:
            mesh.apply_scale(s_opt)
            print(f"Adjusted mesh scale (depth-aware): factor={s_opt:.6f}")
            est = build_estimator(mesh, scorer, refiner, debug_dir, cfg, glctx, iou_min=cfg.iou_min)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, iteration=int(cfg.est_refine_iter))
    _, extents = trimesh.bounds.oriented_bounds(mesh)
    return mesh, est, pose, float(np.linalg.norm(extents))


def render_overlay(renderer_vis, mesh, pose, color):
    rend_rgb, rend_depth = renderer_vis.render(mesh=mesh, ob_in_cvcam=pose)
    comp = color.copy()
    comp[rend_depth > 0] = rend_rgb[rend_depth > 0]
    return comp


def save_pose_and_vis(debug_dir, i, pose):
    np.savetxt(str(debug_dir / "ob_in_cam" / f"{i}.txt"), pose.reshape(4, 4))


def restart_at_init(reader, est, init_idx, cfg, prev_pose_ref):
    pose, _, _, _ = perform_initial_registration(reader, est, int(init_idx), cfg)
    prev_pose_ref["val"] = pose.copy()
    return pose


class Tracker:
    def __init__(self, est, renderer_vis, mesh, reader, cfg, debug_dir, track_csv_path, mesh_diam, prev_pose_ref, current_init_from_ref, reinit_count_ref, rendered_frames):
        self.est = est
        self.renderer_vis = renderer_vis
        self.mesh = mesh
        self.reader = reader
        self.cfg = cfg
        self.debug_dir = debug_dir
        self.track_csv_path = track_csv_path
        self.mesh_diam = mesh_diam
        self.prev_pose_ref = prev_pose_ref
        self.current_init_from_ref = current_init_from_ref
        self.reinit_count_ref = reinit_count_ref
        self.rendered_frames = rendered_frames
        self._prev_score_t: Optional[float] = None
        self._pass_tag: str = "fwd"
        self._cooldown: int = 0
        self._budget_exhausted: bool = False
        self._last_reject_reason: Optional[str] = None

    def _track_step(self, j: int, color, depth, ob_mask) -> Optional[np.ndarray]:
        return self.est.track_one(rgb=color, depth=depth, K=self.reader.K, ob_mask=ob_mask, iteration=int(self.cfg.track_refine_iter))

    def _compute_metrics(self, j: int, pose, color, depth, ob_mask) -> Metrics:
        _, _, iou, overflow, underfill, rmse, rmse_norm, cd, cd_norm, dR, dt, score_t, _, _ = compute_metrics_common(
            renderer=self.renderer_vis,
            mesh=self.mesh,
            pose=pose,
            depth_obs=depth,
            ob_mask=ob_mask,
            K=self.reader.K,
            mesh_diam=self.mesh_diam,
            cfg=self.cfg,
            prev_pose=self.prev_pose_ref["val"],
            fallback_mode="zeros",
        )
        _dbg_score_parts(iou, overflow, underfill, rmse_norm, cd_norm, dR, dt, self.cfg, prefix=f"{self._pass_tag} j={j} ")
        score_net = as_float(getattr(self.est, "last_score", 0.0), default=0.0)
        return Metrics(iou, overflow, underfill, rmse, rmse_norm, cd, cd_norm, dR, dt, score_net, score_t)

    def _accept(self, metrics: Metrics) -> bool:
        mode = getattr(self.cfg, "accept_mode", "lenient")
        relax = 1.0
        if mode == "lenient":
            relax = max(1.0, float(getattr(self.cfg, "accept_relax_factor", 1.7)))
        elif mode == "off":
            self._last_reject_reason = None
            return True

        iou_min_eff = float(self.cfg.iou_min) / relax
        gate_rmse_norm_eff = float(self.cfg.gate_rmse_norm) * relax
        gate_underfill_eff = float(self.cfg.gate_underfill) * relax
        gate_dR_deg_eff = float(self.cfg.gate_dR_deg) * relax
        gate_dt_norm_eff = float(self.cfg.gate_dt_norm) * relax
        gate_drop_eff = float(self.cfg.gate_drop) * relax

        reasons = []
        geom_ok = (metrics.iou >= iou_min_eff) or ((metrics.rmse_norm <= gate_rmse_norm_eff) and (metrics.underfill <= gate_underfill_eff))
        if not geom_ok:
            reasons.append("geom")

        prev_guard = self._prev_score_t if self._prev_score_t is not None else metrics.score_t
        if self._cooldown <= 0:
            if metrics.score_t < (prev_guard - gate_drop_eff):
                reasons.append("drop")
            if metrics.dR_deg > gate_dR_deg_eff:
                reasons.append("rot")
            if metrics.dt_norm > gate_dt_norm_eff:
                reasons.append("trans")

        ok = len(reasons) == 0
        self._last_reject_reason = None if ok else ",".join(reasons)
        return ok

    def _render_and_cache(self, j: int, pose, color, ob_mask):
        rr, rd = self.renderer_vis.render(mesh=self.mesh, ob_in_cvcam=pose)
        comp = color.copy()
        comp[rd > 0] = rr[rd > 0]
        self.rendered_frames[j] = compose_with_mask_panel(comp, ob_mask.astype(bool) if ob_mask is not None else (rd > 0))

    def _save_pose_txt(self, j: int, pose):
        np.savetxt(str(self.debug_dir / "ob_in_cam" / f"{j}.txt"), pose.reshape(4, 4))

    def _log(self, j: int, metrics: Metrics, status: str):
        log_metrics_row(
            self.track_csv_path,
            int(j),
            metrics.iou,
            metrics.overflow,
            metrics.underfill,
            metrics.rmse,
            metrics.rmse_norm,
            metrics.cd,
            metrics.cd_norm,
            metrics.dR_deg,
            metrics.dt_norm,
            metrics.score_net,
            metrics.score_t,
            self.current_init_from_ref["val"],
            status,
            self.reinit_count_ref["val"],
            self._pass_tag,
        )

    def _postfix(self, iterbar, j: int, metrics: Metrics):
        iterbar.set_postfix({"i": int(j), "IoU": f"{metrics.iou:.3f}", "net": f"{metrics.score_net:.4f}", "t": f"{metrics.score_t:.4f}", "cool": int(self._cooldown)})

    def _log_failed(self, j: int):
        log_metrics_row(
            self.track_csv_path,
            int(j),
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            as_float(getattr(self.est, "last_score", 0.0), default=0.0),
            np.nan,
            self.current_init_from_ref["val"],
            "failed_track",
            self.reinit_count_ref["val"],
            self._pass_tag,
        )

    def _log_reinit(self, j: int, pose, depth, ob_mask, cause: str):
        _, _, iou_r, overflow_r, underfill_r, rmse_r, rmse_r_norm, cd_r, cd_r_norm, _, _, score_t_r, _, _ = compute_metrics_common(
            renderer=self.renderer_vis,
            mesh=self.mesh,
            pose=pose,
            depth_obs=depth,
            ob_mask=ob_mask,
            K=self.reader.K,
            mesh_diam=self.mesh_diam,
            cfg=self.cfg,
            prev_pose=None,
            fallback_mode="render",
        )
        self._log(
            j, Metrics(iou_r, overflow_r, underfill_r, rmse_r, rmse_r_norm, cd_r, cd_r_norm, 0.0, 0.0, as_float(getattr(self.est, "last_score", 0.0)), score_t_r), "reinit"
        )
        tqdm.write(
            f"[reinit] cause={cause} pass={self._pass_tag} at frame {j} (count={self.reinit_count_ref['val']}/{int(self.cfg.reinit_budget)}) IoU={iou_r:.3f}, rmse_n={rmse_r_norm:.3f}, cd_n={cd_r_norm:.3f}"
        )

    def _handle_reinit_success(self, j: int, pose, color, depth, ob_mask, cause: str) -> Tuple[bool, Optional[np.ndarray], bool]:
        self._save_pose_txt(j, pose)
        self._render_and_cache(j, pose, color, ob_mask)
        self.current_init_from_ref["val"] = int(j)
        self.reinit_count_ref["val"] += 1
        if self.reinit_count_ref["val"] >= int(self.cfg.reinit_budget):
            log_metrics_row(
                self.track_csv_path,
                int(j),
                *[np.nan] * 10,
                as_float(getattr(self.est, "last_score", 0.0)),
                np.nan,
                self.current_init_from_ref["val"],
                "budget_exhausted",
                self.reinit_count_ref["val"],
                self._pass_tag,
            )
            tqdm.write(f"Re-init budget ({self.cfg.reinit_budget}) reached in {'forward' if self._pass_tag == 'fwd' else 'backward'} pass at frame {j}. Skipping object.")
            return False, None, True
        self._log_reinit(j, pose, depth, ob_mask, cause)
        self.prev_pose_ref["val"] = pose.copy()
        self._cooldown = int(self.cfg.reinit_cooldown)
        self._prev_score_t = None
        return True, pose, False

    def _reinit_at(self, j: int, cause: Optional[str] = "") -> Tuple[bool, Optional[np.ndarray], bool]:
        for _ in range(int(self.cfg.reinit_max_retries)):
            pose_reg, color_j, depth_j, ob_mask_j = perform_initial_registration(self.reader, self.est, j, self.cfg)
            if pose_reg is not None:
                return self._handle_reinit_success(j, pose_reg, color_j, depth_j, ob_mask_j, cause)
        return False, None, False

    def _fallback_reinit_near(self, j: int, cause: Optional[str] = "") -> Tuple[Optional[int], Optional[np.ndarray], bool]:
        clean_keys = sorted(self.reader.clean_mask_keys)
        cand_fallback = [k for k in clean_keys if abs(int(k) - int(j)) <= int(self.cfg.reinit_search_window)]
        if not cand_fallback:
            return None, None, False
        k = min(cand_fallback, key=lambda t: abs(int(t) - int(j)))
        pose_k, color_k, depth_k, ob_mask_k = perform_initial_registration(self.reader, self.est, int(k), self.cfg)
        if pose_k is None:
            return None, None, False
        success, pose, budget_exhausted = self._handle_reinit_success(k, pose_k, color_k, depth_k, ob_mask_k, f"{cause}-fallback")
        return k, pose, budget_exhausted

    def track_pass(self, direction: str, frames: List[int]) -> bool:
        self._pass_tag = direction
        self._prev_score_t = None
        self._cooldown = 0
        self._budget_exhausted = False

        iterbar = tqdm(frames, desc=("Forward" if direction == "fwd" else "Backward"))
        for j in iterbar:
            color = self.reader.get_color(j)
            depth = self.reader.get_depth(j)
            ob_mask = None
            if not self.cfg.no_mask:
                m = self.reader.get_mask(j)
                if np.array(m).sum() > 0:
                    ob_mask = m

            pose = self._track_step(j, color, depth, ob_mask)

            if pose is not None:
                metrics = self._compute_metrics(j, pose, color, depth, ob_mask)
                if self._accept(metrics):
                    self.prev_pose_ref["val"] = pose.copy()
                    self._prev_score_t = metrics.score_t
                    if self._cooldown > 0:
                        self._cooldown -= 1
                    self._render_and_cache(j, pose, color, ob_mask)
                    self._save_pose_txt(j, pose)
                    self._log(j, metrics, "tracked")
                    self._postfix(iterbar, j, metrics)
                    continue

            # 1) Try re-init at current frame
            cause = None
            if pose is None:
                cause = "track_failed"
            else:
                # we are here only if _accept(metrics) was False
                cause = f"rejected:{self._last_reject_reason or 'unknown'}"
            success, pose_reg, budget_exhausted = self._reinit_at(j, cause=cause)
            if budget_exhausted:
                self._budget_exhausted = True
                break
            if success:
                continue

            # 2) Fallback re-init near j
            k, pose_k, budget_exhausted = self._fallback_reinit_near(j, cause=cause)
            if k is None or pose_k is None:
                self._log_failed(j)
                break
            if budget_exhausted:
                self._budget_exhausted = True
                break

            # 3) Retry j once after fallback
            pose = self._track_step(j, color, depth, ob_mask)
            if pose is None:
                self._log_failed(j)
                break

            metrics = self._compute_metrics(j, pose, color, depth, ob_mask)
            if not self._accept(metrics):
                self._log_failed(j)
                break

            self.prev_pose_ref["val"] = pose.copy()
            self._prev_score_t = metrics.score_t
            if self._cooldown > 0:
                self._cooldown -= 1
            self._render_and_cache(j, pose, color, ob_mask)
            self._save_pose_txt(j, pose)
            self._log(j, metrics, "tracked")
            self._postfix(iterbar, j, metrics)

        return self._budget_exhausted


def finalize_video(output_video_path, reader, id_list_sorted, rendered_frames, frame_rate, downsample_factor=2):
    with imageio.get_writer(output_video_path, fps=frame_rate) as writer:
        for frame_idx in tqdm(id_list_sorted, desc="Writing video frames"):
            frame = rendered_frames.get(frame_idx, reader.get_color(frame_idx))
            fh, fw = frame.shape[:2]
            w_ds, h_ds = max(1, fw // max(1, downsample_factor)), max(1, fh // max(1, downsample_factor))
            writer.append_data(cv2.resize(frame, (w_ds, h_ds)).astype(np.uint8))
    print(f"Saved rendered video to {output_video_path}")


class EpicReader:
    def __init__(self, video_path, downscale=1.0, shorter_side=None, zfar=np.inf, intrinsics_depth_path=None, masks_path=None, depth_unit_scale=1.0, clean_masks_path=None):
        self.zfar = zfar
        self.depth_unit_scale = float(depth_unit_scale)
        self.video_path = video_path
        if not os.path.exists(self.video_path):
            raise FileNotFoundError("--video_path must point to an existing video file")
        self.video_dir = os.path.dirname(self.video_path)
        self.color_files = VideoReader(self.video_path, ctx=cpu(0), num_threads=os.cpu_count() or 1)
        self.total_frames = len(self.color_files)
        self.id_strs = [str(idx) for idx in range(self.total_frames)]
        self.H, self.W = self.color_files[0].shape[:2]

        ds = shorter_side / min(self.H, self.W) if shorter_side is not None else downscale
        self.H, self.W = int(self.H * ds), int(self.W * ds)

        self.K, self.depths = None, None
        if os.path.exists(intrinsics_depth_path):
            data = dict(np.load(intrinsics_depth_path, allow_pickle=True))
            K_val = data.get("intrinsics")
            self.K = K_val[0].astype(np.float64) if isinstance(K_val, np.ndarray) and K_val.ndim == 3 else np.asarray(K_val, dtype=np.float64)
            self.depths = data.get("depths")
        if self.K is None:
            fx = fy = 1.2 * float(max(self.W, self.H))
            self.K = np.array([[fx, 0.0, self.W / 2.0], [0.0, fy, self.H / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            print("Intrinsics not found; using heuristic K. Results may be approximate.")

        H0, W0 = np.array(self.depths[0]).shape[:2] if self.depths is not None else (None, None)
        if H0 and W0:
            sx, sy = float(self.W) / float(W0), float(self.H) / float(H0)
            self.K[0, 0] *= sx
            self.K[0, 2] *= sx
            self.K[1, 1] *= sy
            self.K[1, 2] *= sy
            print(
                f"[EpicReader] Shape-based K scale using depth size {H0}x{W0} -> {self.H}x{self.W}; fx={self.K[0, 0]:.2f}, fy={self.K[1, 1]:.2f}, cx={self.K[0, 2]:.2f}, cy={self.K[1, 2]:.2f}"
            )
        else:
            cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
            if cx > 0 and cy > 0:
                sx, sy = float(self.W) / (2.0 * cx), float(self.H) / (2.0 * cy)
                self.K = np.diag([sx, sy, 1.0]) @ self.K
                print(
                    f"[EpicReader] Fallback center-based K scale -> {self.H}x{self.W}; fx={self.K[0, 0]:.2f}, fy={self.K[1, 1]:.2f}, cx={self.K[0, 2]:.2f}, cy={self.K[1, 2]:.2f}"
                )

        self.masks = {}
        if os.path.exists(masks_path):
            self.masks = {int(k): cv2.resize(v, (self.W, self.H), interpolation=cv2.INTER_NEAREST) for k, v in np.load(masks_path).items()}

        keep = sorted(set(map(int, self.id_strs)) & set(self.masks.keys()))
        self.id_strs = [str(k) for k in keep]
        if not self.id_strs:
            print("No overlapping frames between video and masks.")

        self.clean_masks = {}
        if os.path.exists(clean_masks_path):
            self.clean_masks = {int(k): cv2.resize(v, (self.W, self.H), interpolation=cv2.INTER_NEAREST) for k, v in np.load(clean_masks_path).items()}
        self.clean_mask_keys = set(self.clean_masks.keys())

    def get_video_name(self):
        return os.path.basename(os.path.dirname(self.video_path))

    def __len__(self):
        return len(self.color_files)

    def get_color(self, i):
        frame = self.color_files[i].asnumpy()
        frame = cv2.resize(frame, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return frame

    def get_mask(self, i):
        m = np.asarray(self.masks[int(i)])
        if m.ndim == 3:
            m = (m > 0).any(axis=-1).astype(np.uint8)
        else:
            m = (m > 0).astype(np.uint8)
        return cv2.resize(m, (self.W, self.H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    def get_depth(self, i):
        if self.depths is None:
            depth = np.zeros((self.H, self.W), dtype=np.float32)
        else:
            depth = np.array(self.depths[i]) * self.depth_unit_scale
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        depth[(depth < 0.001) | (depth >= self.zfar)] = 0
        return depth

    def get_xyz_map(self, i):
        depth = self.get_depth(i)
        xyz_map = depth2xyzmap(depth, self.K)
        return xyz_map


def log_row(csv_path, row):
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def log_metrics_row(
    csv_path, frame, iou, overflow, underfill, rmse, rmse_norm, cd, cd_norm, dR_deg, dt_norm, score_net, score_t, init_from_frame, status, reinit_count, pass_tag
):
    log_row(
        csv_path,
        [int(frame), iou, overflow, underfill, rmse, rmse_norm, cd, cd_norm, dR_deg, dt_norm, score_net, score_t, init_from_frame, f"{status}_{pass_tag}", reinit_count],
    )


def write_summary_json(debug_dir, summary_dict):
    with open(debug_dir / "run_summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2, sort_keys=True, default=str)


def compose_with_mask_panel(image, mask_bool):
    mask_img = np.zeros_like(image)
    mask_img[mask_bool] = 255
    return np.concatenate([image, mask_img], axis=1)


def _close_small_holes(mask_bool, ksize):
    return mask_bool if ksize <= 0 else cv2.morphologyEx(mask_bool.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((ksize, ksize), np.uint8)).astype(bool)


def make_core_mask(mask_bool, erode_px=8, close_px=3):
    mb = _close_small_holes(mask_bool.astype(bool), close_px)
    return cv2.erode(mb.astype(np.uint8), np.ones((erode_px, erode_px), np.uint8), iterations=1).astype(bool) if erode_px > 0 else mb


def core_area_for_frame(reader, idx, erode_px, close_px):
    try:
        return float(make_core_mask(reader.get_mask(int(idx)).astype(bool), erode_px=int(erode_px), close_px=int(close_px)).sum())
    except Exception:
        return 0.0


def process_single_object(obj_path, video_path, cfg, out_dir="foundationpose10"):
    seq_dir = os.path.dirname(video_path)
    mesh_file = obj_path / cfg.mesh_dir / "model.glb"
    masks_path = obj_path / "vas_masks.npz"
    clean_masks_path = obj_path / "vas_clean_masks.npz"
    intrinsics_depth_path = os.path.join(seq_dir, "spatracker.npz")

    if not cfg.overwrite and (obj_path / out_dir / "ob_in_cam").exists():
        print(f"Skipping {obj_path.name} (FoundationPose results already exist)")
        return

    if not mesh_file.exists():
        print(f"Missing mesh file for object {obj_path.name}: {mesh_file}")
        return
    if not masks_path.exists() and not cfg.no_mask:
        print(f"Missing masks for object {obj_path.name}: {masks_path}. Use --no_mask to ignore masks.")
        return

    set_logging_format()
    set_seed(cfg.seed)

    mesh = trimesh.load(str(mesh_file), force="mesh")

    reader = EpicReader(
        video_path=video_path,
        shorter_side=None,
        zfar=np.inf,
        intrinsics_depth_path=intrinsics_depth_path,
        masks_path=str(masks_path),
        depth_unit_scale=cfg.depth_unit_scale,
        clean_masks_path=str(clean_masks_path),
    )

    debug_dir = obj_path / out_dir
    init_debug_dirs(debug_dir)

    track_csv_path = debug_dir / "track_log.csv"
    with open(track_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "frame",
                "iou",
                "overflow",
                "underfill",
                "rmse_depth",
                "rmse_norm",
                "chamfer_cd",
                "cd_norm",
                "dR_deg",
                "dt_norm",
                "score_net",
                "score_t",
                "init_from_frame",
                "status",
                "reinit_count",
            ]
        )

    id_list_sorted = sorted([int(s) for s in reader.id_strs]) if len(reader.id_strs) > 0 else list(range(len(reader)))

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    renderer_vis = ModelRendererOffscreen(reader.K, reader.H, reader.W, zfar=100.0)

    cand_frames = get_clean_candidate_frames(reader, id_list_sorted, int(cfg.init_candidates), cfg=cfg)
    if len(cand_frames) == 0:
        print("No clean mask indices available for initialization; skipping object.")
        return
    best, best_iou_tuple = choose_best_init(reader, mesh, cand_frames, debug_dir, cfg, scorer, refiner, glctx, renderer_vis=renderer_vis)

    if best is None:
        print("No valid initialization candidates found; skipping object.")
        return

    score_t_sel, init_frame, pose_init, mesh_best, best_iou, _ = best
    print(f"Best init frame={init_frame} with t={score_t_sel:.4f}, IoU={best_iou:.3f}")

    if not np.isfinite(best_iou) or best_iou < float(cfg.min_init_iou):
        print(f"Best candidate IoU {best_iou:.3f} < min_init_iou {float(cfg.min_init_iou):.3f}; using best-by-IoU candidate if valid")
        if best_iou_tuple is None or not np.isfinite(best_iou_tuple[4]) or best_iou_tuple[4] < float(cfg.min_init_iou):
            print("No candidate meets min_init_iou; skipping object.")
            return
        _, init_frame, pose_init, mesh_best, best_iou, _ = best_iou_tuple
        print(f"Fallback by IoU: frame={init_frame}, IoU={best_iou:.3f}")

    mesh = mesh_best

    video_out = str(debug_dir / "rendered.mp4")
    fps = float(reader.color_files.get_avg_fps())

    est = build_estimator(mesh, scorer, refiner, debug_dir, cfg, glctx)
    print("estimator initialization done")

    est.mask_erode = int(cfg.mask_erode)
    est.mask_dilate = int(cfg.mask_dilate)
    est.sil_weight = float(cfg.sil_weight)
    est.overflow_weight = float(cfg.overflow_weight)
    est.iou_min = float(cfg.iou_min)

    rendered_frames = {}

    i = int(init_frame)
    print(f"Register at init_frame={i}")
    reinit_count_ref = {"val": 0}
    current_init_from_ref = {"val": int(i)}
    pose, color, depth, ob_mask = perform_initial_registration(reader, est, i, cfg)

    mesh, est, pose, mesh_diam = refine_scale_with_silhouette(reader, renderer_vis, mesh, pose, ob_mask, color, depth, est, scorer, refiner, debug_dir, cfg, glctx)

    if not np.isfinite(mesh_diam) or mesh_diam <= 0:
        _, _ext = trimesh.bounds.oriented_bounds(mesh)
        mesh_diam = float(np.linalg.norm(_ext))

    d_vis = visible_diameter(depth=depth, mask=(ob_mask.astype(bool) if ob_mask is not None else np.zeros_like(depth, dtype=bool)), K=reader.K)
    print(f"Frame {i}: visible diameter (lower bound) ≈ {d_vis:.4f} m")

    save_pose_and_vis(debug_dir, i, pose)

    comp_init = render_overlay(renderer_vis, mesh, pose, color)
    mask_init_bool = ob_mask.astype(bool) if ob_mask is not None else np.zeros((reader.H, reader.W), dtype=bool)
    rendered_frames[i] = compose_with_mask_panel(comp_init, mask_init_bool)
    _, _d_r0, iou0, overflow0, underfill0, rmse0, rmse0_norm, cd0, cd0_norm, dR0, dt0, _score_t_unused, _mr0, _mo0 = compute_metrics_common(
        renderer=renderer_vis,
        mesh=mesh,
        pose=pose,
        depth_obs=depth,
        ob_mask=mask_init_bool.astype(bool) if ob_mask is not None else np.zeros((reader.H, reader.W), dtype=bool),
        K=reader.K,
        mesh_diam=mesh_diam,
        cfg=cfg,
        prev_pose=None,
        fallback_mode="zeros",
    )
    print(f"[init] IoU0={iou0:.3f}, underfill0={underfill0:.3f}, overflow0={overflow0:.3f}, rmse_n0={rmse0_norm:.3f}, cd_n0={cd0_norm:.3f}")
    score_net0 = as_float(getattr(est, "last_score", 0.0), default=0.0)
    score_t0 = tracking_score(iou0, overflow0, underfill0, rmse0_norm, cd0_norm, dR0, dt0, cfg)
    log_row(
        track_csv_path,
        [
            int(i),
            iou0,
            overflow0,
            underfill0,
            rmse0,
            rmse0_norm,
            cd0,
            cd0_norm,
            dR0,
            dt0,
            score_net0,
            score_t0,
            current_init_from_ref["val"],
            "registered",
            reinit_count_ref["val"],
        ],
    )
    prev_pose_ref = {"val": pose.copy()}

    tracker = Tracker(
        est=est,
        renderer_vis=renderer_vis,
        mesh=mesh,
        reader=reader,
        cfg=cfg,
        debug_dir=debug_dir,
        track_csv_path=track_csv_path,
        mesh_diam=mesh_diam,
        prev_pose_ref=prev_pose_ref,
        current_init_from_ref=current_init_from_ref,
        reinit_count_ref=reinit_count_ref,
        rendered_frames=rendered_frames,
    )

    restart_at_init(reader, est, i, cfg, prev_pose_ref)
    aborted_bwd = tracker.track_pass("bwd", [x for x in id_list_sorted if x < i][::-1])
    if aborted_bwd:
        print("Backward tracking aborted due to re-init budget (logs and poses still written).")

    restart_at_init(reader, est, i, cfg, prev_pose_ref)
    aborted_fwd = tracker.track_pass("fwd", [x for x in id_list_sorted if x > i])
    if aborted_fwd:
        print("Forward tracking aborted due to re-init budget (logs and poses still written).")

    finalize_video(video_out, reader, id_list_sorted, rendered_frames, fps)
    summary = {
        "mesh_dir": str(cfg.mesh_dir),
        "mesh_path": str(mesh_file),
        "init_frame": int(init_frame),
        "fps": float(fps),
        "H": int(reader.H),
        "W": int(reader.W),
        "iou_min": float(cfg.iou_min),
        "mask_erode": int(cfg.mask_erode),
        "mask_dilate": int(cfg.mask_dilate),
        "num_reinits": int(reinit_count_ref["val"]),
        "frames_total": int(len(id_list_sorted)),
        "mesh_diameter": float(mesh_diam),
    }
    summary.update(
        {
            "aborted_backward": bool(aborted_bwd),
            "aborted_forward": bool(aborted_fwd),
        }
    )
    write_summary_json(debug_dir, summary)


def process_video(objects_path, video_path, cfg):
    objects_list = []
    for obj_path in sorted(list(objects_path.glob("*")), key=lambda p: int(p.name.split("object_")[-1]) if "object_" in p.name else 10**9):
        if (obj_path / cfg.mesh_dir / "model.glb").exists():
            objects_list.append(obj_path)
    print(f"Total objects: {len(objects_list)}")
    for obj_path in objects_list:
        print(f"Processing {obj_path}")
        process_single_object(obj_path, video_path, cfg)


def main():
    cfg = load_config()
    set_logging_format()
    set_seed(cfg.seed)

    df = pd.read_csv(cfg.csv_file)
    filtered_df = df[(df["no_hands_presence"] == 0) & (df["duration_s"] < 10)].copy()

    video_folders_txt = os.path.join(cfg.video_root, "video_folders.txt")
    if os.path.exists(video_folders_txt):
        print("Loading video folders from file")
        with open(video_folders_txt, "r") as f:
            all_videos = [line.strip() for line in f]
    else:
        print("Finding video folders")
        ci_ext = "".join([f"[{c.lower()}{c.upper()}]" for c in cfg.ext])
        all_videos = sorted(vp for vp in glob.glob(os.path.join(cfg.video_root, f"**/*.{ci_ext}"), recursive=True) if os.path.basename(vp).lower() == "action.mp4")
        with open(video_folders_txt, "w") as f:
            f.write("\n".join(all_videos))

    if not all_videos:
        print(f"No action videos found under {cfg.video_root} with extension .{cfg.ext}.")
        return

    print(f"Found {len(all_videos)} candidate videos total.")

    num_shards = max(1, int(cfg.num_shards))
    shard_idx = int(cfg.shard_idx) % num_shards
    sharded_paths = [vp for vp in all_videos if stable_int_hash(os.path.basename(os.path.dirname(vp))) % num_shards == shard_idx]
    print(f"Shard {shard_idx}/{num_shards}: {len(sharded_paths)} videos in this shard.")

    start_idx = max(0, int(cfg.start_video_idx))
    end_idx = int(cfg.end_video_idx)
    candidate_paths = sharded_paths[start_idx:end_idx] if end_idx != -1 else sharded_paths[start_idx:]
    total = len(candidate_paths)
    print(f"After slicing: {total} videos remain for this shard.")

    for local_i, vpath in enumerate(candidate_paths, 1):
        try:
            seq_name = os.path.basename(os.path.dirname(vpath))
            if str(seq_name) not in filtered_df["narration_id"].astype(str).values:
                print(f"Skipping {seq_name}: not in filtered_df.")
                continue

            print(f"\n[{local_i}/{total}] Processing {seq_name}: {vpath}")
            objects_path = Path(cfg.output_root) / seq_name / "objects"
            if not objects_path.exists():
                print(f"No objects/ dir for {seq_name} (looked in {objects_path}). Skipping.")
                continue
            process_video(objects_path, vpath, cfg)
        except Exception as e:
            print(f"Error processing {vpath}: {e}")
            traceback.print_exc()

    print("Done.")


if __name__ == "__main__":
    main()
