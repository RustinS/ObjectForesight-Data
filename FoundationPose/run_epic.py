# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import shutil
import traceback

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

import argparse
import glob
from pathlib import Path

import cv2
import imageio
import numpy as np
import nvdiffrast.torch as dr
import pandas as pd
import trimesh
from decord import VideoReader, cpu
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull, distance

from Utils import (
    depth2xyzmap,
    draw_posed_3d_box,
    draw_xyz_axis,
    erode_dilate_mask,
    set_logging_format,
    set_seed,
)
from estimater import *
from offscreen_renderer import ModelRendererOffscreen
from utils import Logger


def masked_depth_to_points(depth, mask, K):
    """Back-project masked depth into 3D points using intrinsics K."""
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
    """Max pairwise distance on the convex hull; falls back to sampled pairwise distance."""
    if points.shape[0] < 2:
        return 0.0
    P = points
    if P.shape[0] > max_sample:
        idx = np.random.choice(P.shape[0], max_sample, replace=False)
        P = P[idx]
    try:
        hull = ConvexHull(P)
        H = P[hull.vertices]
        if H.shape[0] < 2:
            return 0.0
        return float(distance.pdist(H, "euclidean").max())
    except Exception:
        m = min(P.shape[0], 4000)
        idx = np.random.choice(P.shape[0], m, replace=False)
        A = P[idx]
        D = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=-1)
        return float(D.max())


def pca_span(points):
    """Extent along first principal axis (fast, robust)."""
    if points.shape[0] < 2:
        return 0.0
    C = points - points.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    proj = C @ Vt[0]
    return float(proj.max() - proj.min())


def visible_diameter(depth, mask, K, trim=0.02):
    """Lower-bound physical diameter from a single masked depth map (visible chord)."""
    P = masked_depth_to_points(depth, mask, K)
    if P.shape[0] < 50:
        return 0.0
    z = P[:, 2]
    lo, hi = np.quantile(z, [trim, 1 - trim])
    P = P[(z >= lo) & (z <= hi)]
    if P.shape[0] < 50:
        return 0.0
    d_hull = hull_diameter(P)
    return max(pca_span(P), d_hull)


def estimate_scale_by_radius(model_pts, depth_pts, n_sample=10000):
    """Estimate scale via median radius ratio between model and depth points."""
    X = model_pts if model_pts.shape[0] <= n_sample else model_pts[np.random.choice(model_pts.shape[0], n_sample, replace=False)]
    Y = depth_pts if depth_pts.shape[0] <= n_sample else depth_pts[np.random.choice(depth_pts.shape[0], n_sample, replace=False)]
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return None
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    r_model = np.median(np.linalg.norm(Xc, axis=1))
    r_depth = np.median(np.linalg.norm(Yc, axis=1))
    if r_model <= 1e-12:
        return None
    return float(r_depth / (r_model + 1e-12))


def estimate_scale_multi_frame(reader, mesh, init_frame, max_neighbors=2, n_sample=10000, outlier_ratio=0.25):
    """Estimate a robust metric scale using frames near init_frame.
    Returns scale in meters per mesh unit, or None if unavailable.
    """
    try:
        valid_ids = sorted([int(s) for s in reader.id_strs])
    except Exception:
        return None
    if not valid_ids:
        return None

    center = init_frame if init_frame in valid_ids else min(valid_ids, key=lambda x: abs(x - init_frame))
    cand = [j for j in valid_ids if abs(j - center) <= max_neighbors]

    model_pts = np.asarray(mesh.vertices)
    d_model = hull_diameter(model_pts)
    if d_model <= 0:
        return None

    scales, weights = [], []
    for i in cand:
        depth = reader.get_depth(i)
        mask = reader.get_mask(i).astype(bool)
        area = float(mask.sum())
        if area < 10:
            continue
        depth_pts = masked_depth_to_points(depth, mask, reader.K)
        if depth_pts.shape[0] == 0:
            continue

        X = model_pts if model_pts.shape[0] <= n_sample else model_pts[np.random.choice(model_pts.shape[0], n_sample, replace=False)]
        Y = depth_pts if depth_pts.shape[0] <= n_sample else depth_pts[np.random.choice(depth_pts.shape[0], n_sample, replace=False)]
        Xc = X - X.mean(0)
        Yc = Y - Y.mean(0)
        r_model = np.median(np.linalg.norm(Xc, axis=1))
        r_depth = np.median(np.linalg.norm(Yc, axis=1))
        s = None
        if np.isfinite(r_model) and r_model > 1e-12 and np.isfinite(r_depth) and r_depth > 0:
            s = float(r_depth / (r_model + 1e-12))

        if s is None or not np.isfinite(s) or s <= 0:
            d_depth = hull_diameter(depth_pts)
            if d_depth > 0:
                s = float(d_depth / d_model)

        if s is not None and np.isfinite(s) and s > 0:
            scales.append(s)
            weights.append(area)

    if not scales:
        return None

    scales = np.array(scales, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    order = np.argsort(scales)
    cs = scales[order]
    w = weights[order] / max(weights.sum(), 1.0)
    cdf = np.cumsum(w)
    s_med = float(cs[np.searchsorted(cdf, 0.5, side="left")])
    if not np.isfinite(s_med) or s_med <= 0:
        return None

    keep = np.abs(scales - s_med) <= outlier_ratio * s_med
    if not np.any(keep):
        Logger.info(f"Scale estimation: weighted median={s_med:.6f} from {len(scales)} frames; no inliers after trimming")
        return s_med
    s_final = float(np.average(scales[keep], weights=weights[keep]))
    if np.isfinite(s_final) and s_final > 0:
        Logger.info(f"Scale estimation: weighted median={s_med:.6f}, trimmed weighted mean={s_final:.6f}, kept {keep.sum()}/{len(scales)} frames")
        return s_final
    Logger.info(f"Scale estimation: returning weighted median={s_med:.6f}")
    return s_med


def scale_object_for_silhouette(mesh, renderer, pose, mask, s0=1.0, bounds=(0.2, 5.0)):
    """Find scale maximizing IoU(mesh(pose, s), mask) via Brent search on log-scale."""
    a_ref = float(mask.sum())
    if a_ref < 1:
        return None

    lo, hi = np.log(bounds[0] * s0), np.log(bounds[1] * s0)
    res = minimize_scalar(
        neg_iou_log_s,
        bounds=(lo, hi),
        method="bounded",
        args=(mesh, renderer, pose, mask),
        options={"xatol": 1e-3, "maxiter": 60},
    )
    s_opt = float(np.exp(res.x))
    return s_opt if np.isfinite(s_opt) and s_opt > 0 else None


def neg_iou_log_s(log_s, mesh, renderer, pose, mask):
    s = np.exp(log_s)
    M = mesh.copy()
    M.apply_scale(s)
    _, d = renderer.render(mesh=M, ob_in_cvcam=pose)
    m = d > 0
    inter = float(np.logical_and(m, mask).sum())
    union = float(np.logical_or(m, mask).sum())
    iou = inter / max(union, 1.0)
    return -iou


def select_init_candidates(id_list_sorted, k):
    """Evenly select up to k candidate frame indices from a sorted list of ids."""
    if not id_list_sorted:
        return []
    n = len(id_list_sorted)
    if k >= n:
        return id_list_sorted
    # Even spacing across the list
    positions = np.linspace(0, n - 1, num=k)
    idxs = sorted({int(round(p)) for p in positions})
    return [id_list_sorted[i] for i in idxs]


def evaluate_init_candidate(reader, base_mesh, frame_idx, scorer, refiner, glctx, debug_dir, cfg, shared_scale=None):
    """Estimate scale near frame_idx, register once, and return (score, iou, pose, scaled_mesh)."""
    mesh_cand = base_mesh.copy()
    try:
        if shared_scale is not None:
            s_est = float(shared_scale)
        else:
            s_est = estimate_scale_multi_frame(reader, mesh_cand, init_frame=int(frame_idx), max_neighbors=2)
        if s_est is not None and np.isfinite(s_est) and s_est > 0:
            mesh_cand.apply_scale(s_est)
    except Exception:
        pass

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
        try:
            m = reader.get_mask(frame_idx)
            if np.array(m).sum() > 0:
                ob_mask = m.astype(bool)
        except Exception:
            ob_mask = None
    pose = est_cand.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, iteration=int(cfg.est_refine_iter))
    score = getattr(est_cand, "last_score", None)
    if score is None:
        try:
            scores_t = getattr(est_cand, "scores", None)
            if scores_t is not None and len(scores_t) > 0:
                score = float(scores_t[0].detach().cpu().item())
        except Exception:
            score = None
    if score is None or not np.isfinite(score):
        score = 0.0

    # Compute IoU between rendered silhouette and observed mask
    iou = 0.0
    try:
        renderer_tmp = ModelRendererOffscreen(reader.K, reader.H, reader.W, zfar=100.0)
        _, depth_r = renderer_tmp.render(mesh=mesh_cand, ob_in_cvcam=pose)
        mask_r = depth_r > 0
        mask_o = None
        if ob_mask is not None:
            try:
                mask_o = erode_dilate_mask(ob_mask.astype(bool), erode=1, dilate=2)
            except Exception:
                mask_o = ob_mask.astype(bool)
        if mask_o is not None:
            inter = float(np.logical_and(mask_r, mask_o).sum())
            union = float(np.logical_or(mask_r, mask_o).sum())
            iou = inter / max(union, 1.0)
    except Exception:
        iou = 0.0

    return float(score), float(iou), pose, mesh_cand


def init_debug_dirs(debug_dir):
    try:
        shutil.rmtree(debug_dir, ignore_errors=True)
        os.makedirs(debug_dir / "track_vis", exist_ok=True)
        os.makedirs(debug_dir / "ob_in_cam", exist_ok=True)
    except Exception as e:
        Logger.warning(f"Could not reset debug dir: {e}")


def build_estimator(mesh, scorer, refiner, debug_dir, cfg, glctx, iou_min=None):
    kwargs = dict(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(debug_dir),
        debug=int(cfg.debug),
        glctx=glctx,
    )
    if iou_min is not None:
        kwargs["iou_min"] = iou_min
    return FoundationPose(**kwargs)


def get_clean_candidate_frames(reader, id_list_sorted, k):
    clean = getattr(reader, "clean_mask_keys", set())
    pool = [x for x in id_list_sorted if x in clean]
    return select_init_candidates(pool, k) if len(pool) > 0 else []


def choose_best_init(reader, mesh, cand_frames, debug_dir, cfg, scorer, refiner, glctx):
    best = None
    shared_scale = None
    # Optionally compute one robust scale for all candidates
    if cfg.init_lock_scale and len(cand_frames) > 0:
        try:
            shared_scale = estimate_scale_multi_frame(reader, mesh, init_frame=int(cand_frames[0]), max_neighbors=2)
            if shared_scale is None or not np.isfinite(shared_scale) or shared_scale <= 0:
                shared_scale = None
        except Exception:
            shared_scale = None
    for fidx in cand_frames:
        try:
            score_f, iou_f, pose_f, mesh_f = evaluate_init_candidate(
                reader=reader,
                base_mesh=mesh,
                frame_idx=fidx,
                scorer=scorer,
                refiner=refiner,
                glctx=glctx,
                debug_dir=debug_dir,
                cfg=cfg,
                shared_scale=shared_scale,
            )
            Logger.info(f"Init candidate frame {fidx}: score={score_f:.4f}, IoU={iou_f:.3f}")
            key = score_f
            if best is None or key > best[0]:
                best = (key, fidx, pose_f, mesh_f, iou_f, score_f)
        except Exception as e:
            Logger.info(f"Candidate evaluation failed at frame {fidx}: {e}")
    return best


def perform_initial_registration(reader, est, frame_idx, cfg):
    color = reader.get_color(frame_idx)
    depth = reader.get_depth(frame_idx)
    ob_mask = None
    if not cfg.no_mask:
        m = reader.get_mask(frame_idx)
        if np.array(m).sum() > 0:
            ob_mask = m.astype(bool)
    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, iteration=int(cfg.est_refine_iter))
    return pose, color, depth, ob_mask


def refine_scale_with_silhouette(reader, renderer_vis, mesh, pose, ob_mask, color, depth, est, scorer, refiner, debug_dir, cfg, glctx):
    try:
        _, rend_depth_area = renderer_vis.render(mesh=mesh, ob_in_cvcam=pose)
        a_ref = float((ob_mask.astype(bool) if ob_mask is not None else np.zeros_like(depth, dtype=bool)).sum())
        a_rend = float((rend_depth_area > 0).sum())
        f_area = np.sqrt(a_ref / max(a_rend, 1.0)) if a_ref >= 1 and a_rend >= 1 else 1.0
        s_opt = scale_object_for_silhouette(mesh, renderer_vis, pose, ob_mask.astype(bool) if ob_mask is not None else (rend_depth_area > 0), s0=f_area, bounds=(0.3, 3.5))
        if s_opt:
            mesh.apply_scale(s_opt)
            Logger.info(f"Adjusted mesh scale by silhouette IoU search: factor={s_opt:.6f}")
            est = build_estimator(mesh, scorer, refiner, debug_dir, cfg, glctx, iou_min=cfg.iou_min)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, iteration=int(cfg.est_refine_iter))
    except Exception as e:
        Logger.info(f"Scale IoU search skipped: {e}")
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    return mesh, est, pose, to_origin, bbox


def render_overlay(renderer_vis, mesh, pose, color, fallback):
    try:
        rend_rgb, rend_depth = renderer_vis.render(mesh=mesh, ob_in_cvcam=pose)
        comp = color.copy()
        comp[rend_depth > 0] = rend_rgb[rend_depth > 0]
        return comp
    except Exception as e:
        Logger.info(f"Rendering overlay failed: {e}")
        return fallback


def save_pose_and_vis(debug_dir, i, reader, pose, to_origin, bbox, color, cfg):
    os.makedirs(debug_dir / "ob_in_cam", exist_ok=True)
    np.savetxt(str(debug_dir / "ob_in_cam" / f"{i}.txt"), pose.reshape(4, 4))
    # if int(cfg.debug) >= 1:
    #     center_pose = pose @ np.linalg.inv(to_origin)
    #     vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
    #     vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
    #     if int(cfg.debug) >= 2:
    #         os.makedirs(debug_dir / "track_vis", exist_ok=True)
    #         imageio.imwrite(str(debug_dir / "track_vis" / f"{i}.png"), vis)


def track_forward_only(reader, est, renderer_vis, mesh, debug_dir, to_origin, bbox, start_frame, id_list_sorted, cfg, rendered_frames):
    try:
        from tqdm import tqdm

        forward_iter = tqdm([x for x in id_list_sorted if x > start_frame], desc="Forward")
    except Exception:
        forward_iter = [x for x in id_list_sorted if x > start_frame]
    for j in forward_iter:
        color = reader.get_color(j)
        depth = reader.get_depth(j)
        ob_mask = None
        if not cfg.no_mask:
            m = reader.get_mask(j)
            if np.array(m).sum() > 0:
                ob_mask = m
        pose = est.track_one(rgb=color, depth=depth, K=reader.K, ob_mask=ob_mask, iteration=int(cfg.track_refine_iter))
        if hasattr(forward_iter, "set_postfix"):
            iou_v = getattr(est, "last_iou", None)
            score_v = getattr(est, "last_score", None)
            forward_iter.set_postfix({"i": j, "IoU": f"{iou_v:.3f}" if iou_v is not None else "-", "score": f"{score_v:.4f}" if score_v is not None else "-"})
        os.makedirs(debug_dir / "ob_in_cam", exist_ok=True)
        np.savetxt(str(debug_dir / "ob_in_cam" / f"{j}.txt"), pose.reshape(4, 4))
        # if int(cfg.debug) >= 1:
        #     center_pose = pose @ np.linalg.inv(to_origin)
        #     vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
        #     vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
        #     if int(cfg.debug) >= 2:
        #         os.makedirs(debug_dir / "track_vis", exist_ok=True)
        #         imageio.imwrite(str(debug_dir / "track_vis" / f"{j}.png"), vis)
        try:
            rend_rgb_i, rend_depth_i = renderer_vis.render(mesh=mesh, ob_in_cvcam=pose)
            comp_i = color.copy()
            comp_i[rend_depth_i > 0] = rend_rgb_i[rend_depth_i > 0]
            rendered_frames[j] = comp_i
        except Exception as e:
            Logger.info(f"Per-frame rendering failed at i={j}: {e}")
            rendered_frames[j] = color


def track_backward_only(reader, est, renderer_vis, mesh, debug_dir, to_origin, bbox, start_frame, id_list_sorted, cfg, rendered_frames):
    try:
        from tqdm import tqdm

        backward_iter = tqdm([x for x in id_list_sorted if x < start_frame][::-1], desc="Backward")
    except Exception:
        backward_iter = [x for x in id_list_sorted if x < start_frame][::-1]
    for j in backward_iter:
        color = reader.get_color(j)
        depth = reader.get_depth(j)
        ob_mask = None
        if not cfg.no_mask:
            m = reader.get_mask(j)
            if np.array(m).sum() > 0:
                ob_mask = m
        pose = est.track_one(rgb=color, depth=depth, K=reader.K, ob_mask=ob_mask, iteration=int(cfg.track_refine_iter))
        if hasattr(backward_iter, "set_postfix"):
            iou_v = getattr(est, "last_iou", None)
            score_v = getattr(est, "last_score", None)
            backward_iter.set_postfix({"i": j, "IoU": f"{iou_v:.3f}" if iou_v is not None else "-", "score": f"{score_v:.4f}" if score_v is not None else "-"})
        os.makedirs(debug_dir / "ob_in_cam", exist_ok=True)
        np.savetxt(str(debug_dir / "ob_in_cam" / f"{j}.txt"), pose.reshape(4, 4))
        # if int(cfg.debug) >= 1:
        #     center_pose = pose @ np.linalg.inv(to_origin)
        #     vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
        #     vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
        #     if int(cfg.debug) >= 2:
        #         os.makedirs(debug_dir / "track_vis", exist_ok=True)
        #         imageio.imwrite(str(debug_dir / "track_vis" / f"{j}.png"), vis)
        try:
            rend_rgb_i, rend_depth_i = renderer_vis.render(mesh=mesh, ob_in_cvcam=pose)
            comp_i = color.copy()
            comp_i[rend_depth_i > 0] = rend_rgb_i[rend_depth_i > 0]
            rendered_frames[j] = comp_i
        except Exception as e:
            Logger.info(f"Per-frame rendering failed at i={j}: {e}")
            rendered_frames[j] = color


def finalize_video(video_writer, reader, id_list_sorted, rendered_frames, video_out):
    for j in id_list_sorted:
        frame = rendered_frames.get(j, reader.get_color(j))
        video_writer.write(frame[..., ::-1])
    try:
        video_writer.release()
        Logger.info(f"Saved rendered video to {video_out}")
    except Exception as e:
        Logger.info(f"Failed to finalize video: {e}")


class EpicReader:
    def __init__(self, video_path, downscale=1.00, shorter_side=None, zfar=np.inf, intrinsics_depth_path=None, masks_path=None, depth_unit_scale=1.0, clean_masks_path=None):
        self.downscale = downscale
        self.zfar = zfar
        self.depth_unit_scale = float(depth_unit_scale)

        # Video path and reader
        cpu_count = os.cpu_count() or 1
        self.video_path = video_path
        if not os.path.exists(self.video_path):
            raise FileNotFoundError("--video_path must point to an existing video file")
        self.video_dir = os.path.dirname(self.video_path)
        self.color_files = VideoReader(self.video_path, ctx=cpu(0), num_threads=cpu_count)
        self.total_frames = len(self.color_files)
        self.id_strs = [str(idx) for idx in range(self.total_frames)]

        # Determine frame size from first frame
        self.H, self.W = self.color_files[0].shape[:2]

        # Optional resize factor
        if shorter_side is not None:
            self.downscale = shorter_side / min(self.H, self.W)
        self.H = int(self.H * self.downscale)
        self.W = int(self.W * self.downscale)

        # Load intrinsics and depths
        self.K = None
        self.depths = None
        if os.path.exists(intrinsics_depth_path):
            try:
                data = np.load(intrinsics_depth_path, allow_pickle=True)
                data_dict = dict(data)
                K_val = data_dict.get("intrinsics")
                if isinstance(K_val, np.ndarray) and K_val.ndim == 3:
                    self.K = K_val[0].astype(np.float64)
                else:
                    self.K = np.asarray(K_val, dtype=np.float64)
                cx, cy = self.K[0, 2], self.K[1, 2]
                sx, sy = self.W / (2 * cx), self.H / (2 * cy)
                S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1.0]])
                self.K = S @ self.K
                self.depths = data_dict.get("depths")
            except Exception as e:
                Logger.warning(f"Failed to load intrinsics/depths: {e}")
        if self.K is None:
            Logger.warning("Intrinsics not found; using identity K. Results may be invalid.")
            self.K = np.eye(3, dtype=np.float64)
        # Apply image scale to K
        self.K[:2] *= self.downscale

        # Load masks if available
        self.masks = {}
        if os.path.exists(masks_path):
            try:
                self.masks = {int(k): v for k, v in np.load(masks_path).items()}
            except Exception as e:
                Logger.error(f"Failed to load masks: {e}")

        # Align id_strs safely with mask keys
        mask_keys = set(map(int, self.masks.keys()))
        ids = set(map(int, self.id_strs))
        keep = sorted(ids & mask_keys)
        self.id_strs = [str(k) for k in keep]
        if not self.id_strs:
            Logger.error("No overlapping frames between video and masks.")

        self.clean_masks = {}
        if os.path.exists(clean_masks_path):
            try:
                self.clean_masks = {int(k): v for k, v in np.load(clean_masks_path).items()}
            except Exception as e:
                Logger.error(f"Failed to load clean masks: {e}")

        self.clean_mask_keys = set(map(int, self.clean_masks.keys()))

    def get_video_name(self):
        # Use parent directory name to remain compatible with previous behavior
        return os.path.basename(os.path.dirname(self.video_path))

    def __len__(self):
        return len(self.color_files)

    def get_color(self, i):
        frame = self.color_files[i].asnumpy()  # HWC uint8 RGB
        frame = cv2.resize(frame, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return frame

    def get_mask(self, i):
        mask = self.masks[int(i)]
        if len(mask.shape) == 3:
            for c in range(3):
                if mask[..., c].sum() > 0:
                    mask = mask[..., c]
                    break
        mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
        return mask

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


def load_config():
    parser = argparse.ArgumentParser(description="Run FoundationPose tracking over EPIC-KITCHENS videos/objects, aligned with step6_trellis.py")
    parser.add_argument("--base_dir", default="/gscratch/raivn/rustin/3dmanip/", help="Root directory for the project")
    parser.add_argument("--video_path", default="/gscratch/raivn/rustin/3dmanip/results/", help="Path to a directory containing sequence folders and action videos")
    parser.add_argument("--output_dir", default="/gscratch/raivn/rustin/3dmanip/results/", help="Directory where per-sequence outputs exist (objects folders)")
    parser.add_argument("--csv_file", type=str, default="epic_1000_sample.csv")
    parser.add_argument("--ext", type=str, default="mp4")
    parser.add_argument("--start_video_idx", type=int, default=0)
    parser.add_argument("--end_video_idx", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Recompute FoundationPose results even if they already exist")

    # FoundationPose/reader specific args
    parser.add_argument("--depth_unit_scale", type=float, default=1.0)
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--mask_erode", type=int, default=1)
    parser.add_argument("--mask_dilate", type=int, default=2)
    parser.add_argument("--sil_weight", type=float, default=0.25)
    parser.add_argument("--overflow_weight", type=float, default=0.10)
    parser.add_argument("--iou_min", type=float, default=0.1)
    parser.add_argument("--no_mask", action="store_true", help="Disable mask usage during tracking")
    parser.add_argument("--debug", type=int, default=2)
    parser.add_argument("--init_candidates", type=int, default=5, help="Number of candidate frames to try for initialization")
    parser.add_argument("--min_init_iou", type=float, default=0.2, help="Minimum IoU required to accept initial pose and proceed")
    # Deprecated/conf removed: composite scoring by confidence
    parser.add_argument("--init_lock_scale", action="store_true", help="Use a single robust scale for all init candidates")
    return parser.parse_args()


def ensure_dir(path):
    p = Path(path)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)
    return p


def process_single_object(obj_path, video_path, cfg):
    seq_dir = os.path.dirname(video_path)
    mesh_file = obj_path / "trellis" / "model.glb"
    masks_path = obj_path / "masks.npz"
    clean_masks_path = obj_path / "clean_masks.npz"
    intrinsics_depth_path = os.path.join(seq_dir, "spatrack", "result.npz")

    if not cfg.overwrite and (obj_path / "foundationpose" / "rendered.mp4").exists():
        Logger.info(f"Skipping {obj_path.name} (FoundationPose results already exist)")
        return

    if not mesh_file.exists():
        Logger.info(f"Missing mesh file for object {obj_path.name}: {mesh_file}")
        return
    if not masks_path.exists() and not cfg.no_mask:
        Logger.info(f"Missing masks for object {obj_path.name}: {masks_path}. Use --no_mask to ignore masks.")
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

    debug_dir = obj_path / "foundationpose"
    init_debug_dirs(debug_dir)

    id_list_sorted = sorted([int(s) for s in reader.id_strs]) if len(reader.id_strs) > 0 else list(range(len(reader)))

    # Initialize model components for evaluation
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    # Try multiple initialization candidates and pick the best by score
    cand_frames = get_clean_candidate_frames(reader, id_list_sorted, int(cfg.init_candidates))
    if len(cand_frames) == 0:
        Logger.info("No clean mask indices available for initialization; skipping object.")
        return
    best = choose_best_init(reader, mesh, cand_frames, debug_dir, cfg, scorer, refiner, glctx)

    if best is None:
        Logger.info("No valid initialization candidates found; skipping object.")
        return

    sel_key, init_frame, pose_init, mesh_best, best_iou, best_raw_score = best
    Logger.info(f"Best init frame={init_frame} with score={best_raw_score:.4f}, IoU={best_iou:.3f}")

    # Enforce minimum IoU threshold; if violated, try fallback to best IoU overall
    if not np.isfinite(best_iou) or best_iou < float(cfg.min_init_iou):
        Logger.info(f"Best candidate IoU {best_iou:.3f} < min_init_iou {float(cfg.min_init_iou):.3f}; searching fallback by IoU")
        # Recompute selection purely by IoU
        best_iou_tuple = None
        for fidx in cand_frames:
            try:
                _, iou_f, pose_f, mesh_f = evaluate_init_candidate(
                    reader=reader,
                    base_mesh=mesh,
                    frame_idx=fidx,
                    scorer=scorer,
                    refiner=refiner,
                    glctx=glctx,
                    debug_dir=debug_dir,
                    cfg=cfg,
                    shared_scale=None,
                )
                if best_iou_tuple is None or iou_f > best_iou_tuple[0]:
                    best_iou_tuple = (iou_f, fidx, pose_f, mesh_f)
            except Exception:
                pass
        if best_iou_tuple is None or best_iou_tuple[0] < float(cfg.min_init_iou):
            Logger.info("No candidate meets min_init_iou; skipping object.")
            return
        best_iou, init_frame, pose_init, mesh_best = best_iou_tuple
        Logger.info(f"Fallback by IoU: frame={init_frame}, IoU={best_iou:.3f}")

    # Use the best scaled mesh going forward
    mesh = mesh_best

    renderer_vis = ModelRendererOffscreen(reader.K, reader.H, reader.W, zfar=100.0)

    # Initialize video writer for per-frame rendered overlays
    video_out = str(debug_dir / "rendered.mp4")
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_out, fourcc, fps, (reader.W, reader.H))

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    est = build_estimator(mesh, scorer, refiner, debug_dir, cfg, glctx)
    Logger.info("estimator initialization done")

    est.mask_erode = int(cfg.mask_erode)
    est.mask_dilate = int(cfg.mask_dilate)
    est.sil_weight = float(cfg.sil_weight)
    est.overflow_weight = float(cfg.overflow_weight)
    est.iou_min = float(cfg.iou_min)

    rendered_frames = {}

    i = int(init_frame)
    Logger.info(f"Register at init_frame={i}")
    pose, color, depth, ob_mask = perform_initial_registration(reader, est, i, cfg)

    # Silhouette IoU-based scale refinement and re-registration
    mesh, est, pose, to_origin, bbox = refine_scale_with_silhouette(reader, renderer_vis, mesh, pose, ob_mask, color, depth, est, scorer, refiner, debug_dir, cfg, glctx)

    try:
        d_vis = visible_diameter(depth=depth, mask=(ob_mask.astype(bool) if ob_mask is not None else np.zeros_like(depth, dtype=bool)), K=reader.K)
        Logger.info(f"Frame {i}: visible diameter (lower bound) ≈ {d_vis:.4f} m")
    except Exception as e:
        Logger.info(f"Visible diameter computation failed: {e}")

    save_pose_and_vis(debug_dir, i, reader, pose, to_origin, bbox, color, cfg)

    comp_init = render_overlay(renderer_vis, mesh, pose, color, color)
    # imageio.imwrite(str(debug_dir / "rendered.png"), comp_init)
    rendered_frames[i] = comp_init

    # Backward tracking from the selected init frame
    track_backward_only(reader, est, renderer_vis, mesh, debug_dir, to_origin, bbox, i, id_list_sorted, cfg, rendered_frames)

    # Reset estimator at init frame before forward tracking
    pose, _, _, _ = perform_initial_registration(reader, est, i, cfg)

    # Forward tracking from the selected init frame
    track_forward_only(reader, est, renderer_vis, mesh, debug_dir, to_origin, bbox, i, id_list_sorted, cfg, rendered_frames)

    finalize_video(video_writer, reader, id_list_sorted, rendered_frames, video_out)


def process_video(objects_path, video_path, cfg):
    objects_list = list(objects_path.glob("*"))
    Logger.info(f"Total objects: {len(objects_list)}")
    for obj_path in objects_list:
        Logger.info(f"Processing {obj_path}")
        process_single_object(obj_path, video_path, cfg)


def main():
    cfg = load_config()
    set_logging_format()
    set_seed(cfg.seed)

    ci_ext = "".join([f"[{c.lower()}{c.upper()}]" for c in cfg.ext])
    video_paths = sorted(glob.glob(os.path.join(cfg.video_path, f"**/*.{ci_ext}"), recursive=True))
    video_paths = [vp for vp in video_paths if os.path.basename(vp).lower() == "action.mp4"]
    if len(video_paths) == 0:
        Logger.error(f"No videos found under {cfg.video_path} with extension .{cfg.ext}.")
        return

    Logger.info(f"Found {len(video_paths)} videos under {cfg.video_path} with extension .{cfg.ext}.")

    csv_path = os.path.join(cfg.video_path, cfg.csv_file)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        filtered_df = df[(df.get("both_hands", 1) == 1) & (df.get("duration_s", 0) < 10)]
        valid_ids = set(filtered_df.get("narration_id", []).astype(str).tolist())
    else:
        Logger.warning(f"CSV file not found: {csv_path}. Proceeding without filtering.")
        valid_ids = None

    total = len(video_paths)
    for i, vpath in enumerate(video_paths, 1):
        if i < cfg.start_video_idx:
            continue
        if i >= cfg.end_video_idx and cfg.end_video_idx != -1:
            break
        try:
            seq_name = os.path.basename(os.path.dirname(vpath))
            seq_dir = os.path.dirname(vpath)
            Logger.init(f"{seq_dir}/opt_log.txt")
            Logger.info("\n")
            Logger.info(f"[{i}/{total}] Processing {seq_name}: {vpath}")
            if valid_ids is not None and seq_name not in valid_ids:
                Logger.info(f"Skipping {seq_name} because seq_name not in filtered_df.")
                continue

            video_output_dir = Path(cfg.output_dir) / seq_name
            objects_path = video_output_dir / "objects"
            if not objects_path.exists():
                Logger.info(f"No 'objects' folder found in {video_output_dir}. Nothing to process.")
                continue

            process_video(objects_path, vpath, cfg)
        except Exception as e:
            Logger.error(traceback.format_exc())
            Logger.error(f"Error processing {vpath}: {e}")
            continue


if __name__ == "__main__":
    main()
