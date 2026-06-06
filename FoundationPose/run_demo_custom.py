# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import shutil

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["HYDRA_FULL_ERROR"] = "1"

from estimater import *
from Utils import set_logging_format, set_seed, depth2xyzmap, draw_posed_3d_box, draw_xyz_axis
from datareader import *
import argparse
from decord import VideoReader, cpu
import numpy as np
import logging
import imageio
from offscreen_renderer import ModelRendererOffscreen
import cv2
import trimesh
import nvdiffrast.torch as dr
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull, distance


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
    """Max Euclidean distance between any two points on the convex hull (visible set)."""
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
        # Fallback: pairwise on a small random subset
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
    try:
        d_hull = hull_diameter(P)
    except Exception:
        d_hull = 0.0
    return max(pca_span(P), d_hull)


def estimate_scale_by_radius(model_pts, depth_pts, n_sample=10000):
    if model_pts.shape[0] > n_sample:
        idx = np.random.choice(model_pts.shape[0], n_sample, replace=False)
        X = model_pts[idx]
    else:
        X = model_pts

    if depth_pts.shape[0] > n_sample:
        jdx = np.random.choice(depth_pts.shape[0], n_sample, replace=False)
        Y = depth_pts[jdx]
    else:
        Y = depth_pts

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

    # choose neighborhood
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

        # primary: radius ratio (median distance to centroid)
        X = model_pts if model_pts.shape[0] <= n_sample else model_pts[np.random.choice(model_pts.shape[0], n_sample, replace=False)]
        Y = depth_pts if depth_pts.shape[0] <= n_sample else depth_pts[np.random.choice(depth_pts.shape[0], n_sample, replace=False)]
        Xc = X - X.mean(0)
        Yc = Y - Y.mean(0)
        r_model = np.median(np.linalg.norm(Xc, axis=1))
        r_depth = np.median(np.linalg.norm(Yc, axis=1))
        s = None
        if np.isfinite(r_model) and r_model > 1e-12 and np.isfinite(r_depth) and r_depth > 0:
            s = float(r_depth / (r_model + 1e-12))

        # fallback: diameter ratio on hulls
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
    # weighted median
    order = np.argsort(scales)
    cs = scales[order]
    w = weights[order] / max(weights.sum(), 1.0)
    cdf = np.cumsum(w)
    s_med = float(cs[np.searchsorted(cdf, 0.5, side="left")])
    if not np.isfinite(s_med) or s_med <= 0:
        return None

    keep = np.abs(scales - s_med) <= outlier_ratio * s_med
    if not np.any(keep):
        logging.info(f"Scale estimation: weighted median={s_med:.6f} from {len(scales)} frames; no inliers after trimming")
        return s_med
    s_final = float(np.average(scales[keep], weights=weights[keep]))
    if np.isfinite(s_final) and s_final > 0:
        logging.info(f"Scale estimation: weighted median={s_med:.6f}, trimmed weighted mean={s_final:.6f}, kept {keep.sum()}/{len(scales)} frames")
        return s_final
    logging.info(f"Scale estimation: returning weighted median={s_med:.6f}")
    return s_med


def scale_object_for_silhouette(mesh, renderer, pose, mask, s0=1.0, bounds=(0.2, 5.0)):
    """Find scale s maximizing IoU(mesh(pose, s), mask) by Brent search on log-scale."""
    a_ref = float(mask.sum())
    if a_ref < 1:
        return None

    def neg_iou(log_s):
        s = np.exp(log_s)
        M = mesh.copy()
        M.apply_scale(s)
        _, d = renderer.render(mesh=M, ob_in_cvcam=pose)
        m = d > 0
        inter = float(np.logical_and(m, mask).sum())
        union = float(np.logical_or(m, mask).sum())
        iou = inter / max(union, 1.0)
        return -iou

    lo, hi = np.log(bounds[0] * s0), np.log(bounds[1] * s0)
    res = minimize_scalar(neg_iou, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3, "maxiter": 60})
    s_opt = float(np.exp(res.x))
    return s_opt if np.isfinite(s_opt) and s_opt > 0 else None


class EpicReader:
    def __init__(self, video_path, downscale=1.00, shorter_side=None, zfar=np.inf, intrinsics_depth_path=None, masks_path=None, depth_unit_scale=1.0):
        self.downscale = downscale
        self.zfar = zfar
        self.depth_unit_scale = float(depth_unit_scale)

        # Video path and reader
        cpu_count = os.cpu_count() or 1
        self.video_path = video_path
        if self.video_path is None or not os.path.exists(self.video_path):
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
        if intrinsics_depth_path is None:
            cand = os.path.join(self.video_dir, "result.npz")
            intrinsics_depth_path = cand if os.path.exists(cand) else None
        if intrinsics_depth_path is not None and os.path.exists(intrinsics_depth_path):
            try:
                data = np.load(intrinsics_depth_path, allow_pickle=True)
                data_dict = dict(data)
                K_val = data_dict.get("intrinsics")
                if isinstance(K_val, np.ndarray) and K_val.ndim == 3:
                    self.K = K_val[0].astype(np.float64)
                else:
                    self.K = np.asarray(K_val, dtype=np.float64)
                self.depths = data_dict.get("depths")
            except Exception as e:
                logging.warning(f"Failed to load intrinsics/depths: {e}")
        if self.K is None:
            logging.warning("Intrinsics not found; using identity K. Results may be invalid.")
            self.K = np.eye(3, dtype=np.float64)
        # Apply image scale to K
        self.K[:2] *= self.downscale

        # Load masks if available
        self.masks = {}
        if masks_path is None:
            cand_m = os.path.join(self.video_dir, "masks.npz")
            masks_path = cand_m if os.path.exists(cand_m) else None
        if masks_path is not None and os.path.exists(masks_path):
            try:
                self.masks = {int(k): v for k, v in np.load(masks_path).items()}
            except Exception as e:
                logging.warning(f"Failed to load masks: {e}")

        # Align id_strs safely with mask keys
        mask_keys = set(map(int, getattr(self.masks, "keys", lambda: [])()))
        ids = set(map(int, self.id_strs))
        keep = sorted(ids & mask_keys)
        self.id_strs = [str(k) for k in keep]
        if not self.id_strs:
            logging.warning("No overlapping frames between video and masks.")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument("--mesh_file", type=str, default="/gscratch/raivn/rustin/3dmanip/test3_results/objects/0+knife/trellis/model.glb")
    parser.add_argument("--video_path", type=str, default="/gscratch/raivn/rustin/3dmanip/test3.mp4")
    parser.add_argument("--intrinsics_depth_path", type=str, default="/gscratch/raivn/rustin/3dmanip/SpaTrackerV2/testing3/results/result.npz")
    parser.add_argument("--masks_path", type=str, default="/gscratch/raivn/rustin/3dmanip/test3_results/objects/0+knife/masks.npz")
    parser.add_argument("--depth_unit_scale", type=float, default=1.0)
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--mask_erode", type=int, default=1)
    parser.add_argument("--mask_dilate", type=int, default=2)
    parser.add_argument("--sil_weight", type=float, default=0.25)
    parser.add_argument("--overflow_weight", type=float, default=0.10)
    parser.add_argument("--iou_min", type=float, default=0.2)
    parser.add_argument("--no_mask", action="store_true", help="Disable mask usage during tracking")
    parser.add_argument("--debug", type=int, default=2)
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug3")
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    # mesh
    if args.mesh_file is None:
        raise ValueError("--mesh_file must be provided")
    mesh = trimesh.load(args.mesh_file, force="mesh")
    # mesh.apply_scale(1000.0)
    # mesh.apply_scale(0.4)

    # Prepare reader and estimate metric scale before building the estimator
    reader = EpicReader(
        video_path=args.video_path,
        shorter_side=None,
        zfar=np.inf,
        intrinsics_depth_path=args.intrinsics_depth_path,
        masks_path=args.masks_path,
        depth_unit_scale=args.depth_unit_scale,
    )

    init_frame = 0
    id_list_sorted = sorted([int(s) for s in reader.id_strs]) if len(reader.id_strs) > 0 else list(range(len(reader)))
    id_set = set(id_list_sorted)
    if init_frame not in id_set:
        init_frame = min(id_list_sorted, key=lambda x: abs(x - init_frame))
        logging.info(f"init_frame not in available ids; using closest {init_frame}")

    renderer_vis = ModelRendererOffscreen(reader.K, reader.H, reader.W, zfar=100.0)
    if len(reader.id_strs) > 0:
        try:
            valid_ids = [int(s) for s in reader.id_strs]
            s_est = estimate_scale_multi_frame(reader, mesh, init_frame=init_frame, max_neighbors=2)
            if s_est is not None and np.isfinite(s_est) and s_est > 0:
                mesh.apply_scale(s_est)
                logging.info(f"Applied metric scale (robust multi-frame): s = {s_est:.6f} (m/unit)")
            else:
                logging.info("Scale estimate unavailable; proceeding without mesh scaling.")
        except Exception as e:
            logging.info(f"Scale estimation failed; proceeding without mesh scaling. Error: {e}")

    debug = args.debug
    debug_dir = args.debug_dir
    try:
        shutil.rmtree(debug_dir, ignore_errors=True)
        os.makedirs(f"{debug_dir}/track_vis", exist_ok=True)
        os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
    except Exception as e:
        logging.warning(f"Could not reset debug dir: {e}")

    # Initialize video writer for per-frame rendered overlays
    video_out = f"{debug_dir}/rendered.mp4"
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_out, fourcc, fps, (reader.W, reader.H))

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")
    # Configure mask-aware hyperparameters (non-breaking defaults already set in class)
    est.mask_erode = int(args.mask_erode)
    est.mask_dilate = int(args.mask_dilate)
    est.sil_weight = float(args.sil_weight)
    est.overflow_weight = float(args.overflow_weight)
    est.iou_min = float(args.iou_min)

    # Build index lists around init frame

    rendered_frames = {}

    # 1) Registration at init_frame
    i = int(init_frame)
    logging.info(f"Register at init_frame={i}")
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    mask = reader.get_mask(i).astype(bool)
    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

    # Silhouette IoU-based scale refinement and re-registration
    try:
        _, rend_depth_area = renderer_vis.render(mesh=mesh, ob_in_cvcam=pose)
        a_ref = float(mask.sum())
        a_rend = float((rend_depth_area > 0).sum())
        f_area = np.sqrt(a_ref / max(a_rend, 1.0)) if a_ref >= 1 and a_rend >= 1 else 1.0
        s_opt = scale_object_for_silhouette(mesh, renderer_vis, pose, mask, s0=f_area, bounds=(0.3, 3.5))
        if s_opt:
            mesh.apply_scale(s_opt)
            logging.info(f"Adjusted mesh scale by silhouette IoU search: factor={s_opt:.6f}")
            # refresh bounds, estimator, and pose
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
            est = FoundationPose(
                model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx
            )
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
    except Exception as e:
        logging.info(f"Scale IoU search skipped: {e}")

    # Log visible diameter lower bound for diagnostics
    try:
        d_vis = visible_diameter(depth=depth, mask=mask.astype(bool), K=reader.K)
        logging.info(f"Frame {i}: visible diameter (lower bound) ≈ {d_vis:.4f} m")
    except Exception as e:
        logging.info(f"Visible diameter computation failed: {e}")

    # Save pose/vis
    os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
    np.savetxt(f"{debug_dir}/ob_in_cam/{i}.txt", pose.reshape(4, 4))
    if debug >= 1:
        center_pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
    if debug >= 2:
        os.makedirs(f"{debug_dir}/track_vis", exist_ok=True)
        imageio.imwrite(f"{debug_dir}/track_vis/{i}.png", vis)

    # Render overlay and store for video
    try:
        rend_rgb_init, rend_depth_init = renderer_vis.render(mesh=mesh, ob_in_cvcam=pose)
        comp_init = color.copy()
        comp_init[rend_depth_init > 0] = rend_rgb_init[rend_depth_init > 0]
        imageio.imwrite(f"{debug_dir}/rendered.png", comp_init)
        rendered_frames[i] = comp_init
    except Exception as e:
        logging.info(f"Rendering overlay failed at i={i}: {e}")
        rendered_frames[i] = color

    # Keep internal last pose to reseed for backward pass if needed
    try:
        pose_last_init = est.pose_last.detach().clone()
    except Exception:
        pose_last_init = None

    # 2) Forward tracking
    for j in [x for x in id_list_sorted if x > i]:
        logging.info(f"Forward track i={j}")
        color = reader.get_color(j)
        depth = reader.get_depth(j)
        ob_mask = None
        if not args.no_mask:
            try:
                m = reader.get_mask(j)
                if m is not None and np.array(m).sum() > 0:
                    ob_mask = m
            except Exception:
                ob_mask = None
        pose = est.track_one(rgb=color, depth=depth, K=reader.K, ob_mask=ob_mask, iteration=args.track_refine_iter)
        os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
        np.savetxt(f"{debug_dir}/ob_in_cam/{j}.txt", pose.reshape(4, 4))
        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
        if debug >= 2:
            os.makedirs(f"{debug_dir}/track_vis", exist_ok=True)
            imageio.imwrite(f"{debug_dir}/track_vis/{j}.png", vis)
        try:
            rend_rgb_i, rend_depth_i = renderer_vis.render(mesh=mesh, ob_in_cvcam=pose)
            comp_i = color.copy()
            comp_i[rend_depth_i > 0] = rend_rgb_i[rend_depth_i > 0]
            rendered_frames[j] = comp_i
        except Exception as e:
            logging.info(f"Per-frame rendering failed at i={j}: {e}")
            rendered_frames[j] = color

    # 3) Backward tracking
    if pose_last_init is not None:
        try:
            est.pose_last = pose_last_init
        except Exception:
            pass
    for j in [x for x in reversed(id_list_sorted) if x < i]:
        logging.info(f"Backward track i={j}")
        color = reader.get_color(j)
        depth = reader.get_depth(j)
        ob_mask = None
        if not args.no_mask:
            try:
                m = reader.get_mask(j)
                if m is not None and np.array(m).sum() > 0:
                    ob_mask = m
            except Exception:
                ob_mask = None
        pose = est.track_one(rgb=color, depth=depth, K=reader.K, ob_mask=ob_mask, iteration=args.track_refine_iter)
        os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
        np.savetxt(f"{debug_dir}/ob_in_cam/{j}.txt", pose.reshape(4, 4))
        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
        if debug >= 2:
            os.makedirs(f"{debug_dir}/track_vis", exist_ok=True)
            imageio.imwrite(f"{debug_dir}/track_vis/{j}.png", vis)
        try:
            rend_rgb_i, rend_depth_i = renderer_vis.render(mesh=mesh, ob_in_cvcam=pose)
            comp_i = color.copy()
            comp_i[rend_depth_i > 0] = rend_rgb_i[rend_depth_i > 0]
            rendered_frames[j] = comp_i
        except Exception as e:
            logging.info(f"Per-frame rendering failed at i={j}: {e}")
            rendered_frames[j] = color

    # Write video in index order
    for j in id_list_sorted:
        frame = rendered_frames.get(j, reader.get_color(j))
        video_writer.write(frame[..., ::-1])

    try:
        video_writer.release()
        logging.info(f"Saved rendered video to {video_out}")
    except Exception as e:
        logging.info(f"Failed to finalize video: {e}")
