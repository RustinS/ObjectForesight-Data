# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["HYDRA_FULL_ERROR"] = "1"

from estimater import *
from datareader import *
import argparse
from decord import VideoReader, cpu
import numpy as np
import logging
import imageio
from offscreen_renderer import ModelRendererOffscreen
import cv2
import trimesh


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


def approximate_diameter(points, n_sample=2000, block_size=1024):
    if points.shape[0] == 0:
        return 0.0
    if points.shape[0] > n_sample:
        idx = np.random.choice(points.shape[0], n_sample, replace=False)
        pts = points[idx]
    else:
        pts = points

    max_dist = 0.0
    N = pts.shape[0]
    for start in range(0, N, block_size):
        A = pts[start : start + block_size]
        d = np.linalg.norm(A[:, None, :] - pts[None, :, :], axis=-1)
        local_max = float(d.max())
        if local_max > max_dist:
            max_dist = local_max
    return max_dist


def erode_mask(mask, iterations=1):
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)
    return (eroded > 0).astype(np.uint8)


def binary_iou(a, b):
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    inter = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def refine_scale_by_silhouette(renderer, mesh, pose, ref_mask, scale_factors, min_coverage=0.8):
    best_iou = -1.0
    best_factor = 1.0
    ref_area = float(ref_mask.sum() + 1e-9)
    for f in scale_factors:
        m = mesh.copy()
        try:
            m.apply_scale(f)
            _, rend_depth = renderer.render(mesh=m, ob_in_cvcam=pose)
            rend_mask = (rend_depth > 0).astype(np.uint8)
            inter = np.logical_and(rend_mask.astype(bool), ref_mask.astype(bool)).sum()
            coverage = float(inter) / ref_area
            if coverage < min_coverage:
                continue
            iou = binary_iou(rend_mask, ref_mask)
            if iou > best_iou:
                best_iou = iou
                best_factor = f
        except Exception:
            continue
    return best_factor, best_iou


def refine_scale_by_depth(renderer, mesh, pose, sensor_depth, ref_mask, scale_factors, min_coverage=0.8):
    best_err = float("inf")
    best_factor = 1.0
    mask_bool = ref_mask.astype(bool)
    ref_area = float(ref_mask.sum() + 1e-9)
    for f in scale_factors:
        m = mesh.copy()
        try:
            m.apply_scale(f)
            _, rend_depth = renderer.render(mesh=m, ob_in_cvcam=pose)
            rend_mask = (rend_depth > 0).astype(np.uint8)
            inter = np.logical_and(rend_mask.astype(bool), mask_bool).sum()
            coverage = float(inter) / ref_area
            if coverage < min_coverage:
                continue
            valid = (rend_depth > 0) & (sensor_depth > 0) & mask_bool
            if not np.any(valid):
                continue
            diff = np.abs(rend_depth[valid] - sensor_depth[valid])
            err = float(np.median(diff))
            if err < best_err:
                best_err = err
                best_factor = f
        except Exception:
            continue
    return best_factor, best_err


def compute_area_scale_factor(renderer, mesh, pose, ref_mask, min_factor=0.7, max_factor=1.4):
    try:
        _, rend_depth = renderer.render(mesh=mesh, ob_in_cvcam=pose)
        rend_mask = (rend_depth > 0).astype(np.uint8)
        a_ref = float(ref_mask.sum())
        a_rend = float(rend_mask.sum())
        if a_ref < 1 or a_rend < 1:
            return 1.0, a_rend, a_ref
        factor = float(np.sqrt(a_ref / (a_rend + 1e-9)))
        factor = float(np.clip(factor, min_factor, max_factor))
        return factor, a_rend, a_ref
    except Exception:
        return 1.0, 0.0, float(ref_mask.sum())


class EpicReader:
    def __init__(self, video_dir, downscale=1, shorter_side=None, zfar=np.inf):
        self.video_dir = video_dir
        self.downscale = downscale
        self.zfar = zfar
        data = np.load("/gscratch/raivn/rustin/3dmanip/SpaTrackerV2/testing/results/result.npz", allow_pickle=True)
        data_dict = dict(data)
        self.K = data_dict["intrinsics"][0].astype(np.float64)
        self.depths = data_dict["depths"]
        cpu_count = os.cpu_count() or 1
        self.color_files = VideoReader("/gscratch/raivn/rustin/3dmanip/test.mp4", ctx=cpu(0), num_threads=cpu_count)
        self.total_frames = len(self.color_files)
        self.id_strs = []
        for idx in range(len(self.color_files)):
            self.id_strs.append(str(idx))
        # self.H, self.W = cv2.imread(self.color_files[0]).shape[:2]
        self.H, self.W = self.color_files[0].shape[:2]

        if shorter_side is not None:
            self.downscale = shorter_side / min(self.H, self.W)

        self.H = int(self.H * self.downscale)
        self.W = int(self.W * self.downscale)
        self.K[:2] *= self.downscale

        # self.mask = np.load("/gscratch/raivn/rustin/3dmanip/test_results/objects/0+pot/masks.npz", allow_pickle=True)
        # self.mask_dict = dict(self.mask)

        self.masks = {int(k): v for k, v in np.load("/gscratch/raivn/rustin/3dmanip/test_results/objects/0+pot/masks.npz").items()}

        # Align id_strs with available mask keys without mutating during iteration
        mask_keys = set(self.masks.keys())
        self.id_strs = [s for s in self.id_strs if int(s) in mask_keys]
        # Ensure the sets match (ignore type/order differences)
        assert set(int(s) for s in self.id_strs) == mask_keys

        self.videoname_to_object = {
            "bleach0": "021_bleach_cleanser",
            "bleach_hard_00_03_chaitanya": "021_bleach_cleanser",
            "cracker_box_reorient": "003_cracker_box",
            "cracker_box_yalehand0": "003_cracker_box",
            "mustard0": "006_mustard_bottle",
            "mustard_easy_00_02": "006_mustard_bottle",
            "sugar_box1": "004_sugar_box",
            "sugar_box_yalehand0": "004_sugar_box",
            "tomato_soup_can_yalehand0": "005_tomato_soup_can",
        }

    def get_video_name(self):
        return self.video_dir.split("/")[-1]

    def __len__(self):
        return len(self.color_files)

    def get_color(self, i):
        color = self.color_files[i].asnumpy()
        return color

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
        depth = np.array(self.depths[i])
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        depth[(depth < 0.001) | (depth >= self.zfar)] = 0
        return depth

    def get_xyz_map(self, i):
        depth = self.get_depth(i)
        xyz_map = depth2xyzmap(depth, self.K)
        return xyz_map

    def get_occ_mask(self, i):
        hand_mask_file = self.color_files[i].replace("rgb", "masks_hand")
        occ_mask = np.zeros((self.H, self.W), dtype=bool)
        if os.path.exists(hand_mask_file):
            occ_mask = occ_mask | (cv2.imread(hand_mask_file, -1) > 0)

        right_hand_mask_file = self.color_files[i].replace("rgb", "masks_hand_right")
        if os.path.exists(right_hand_mask_file):
            occ_mask = occ_mask | (cv2.imread(right_hand_mask_file, -1) > 0)

        occ_mask = cv2.resize(occ_mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

        return occ_mask.astype(np.uint8)

    def get_gt_mesh(self):
        ob_name = self.videoname_to_object[self.get_video_name()]
        YCB_VIDEO_DIR = os.getenv("YCB_VIDEO_DIR")
        mesh = trimesh.load(f"{YCB_VIDEO_DIR}/models/{ob_name}/textured_simple.obj")
        return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument("--mesh_file", type=str, default="/gscratch/raivn/rustin/3dmanip/test_results/objects/0+pot/trellis/model.glb")
    parser.add_argument("--test_scene_dir", type=str, default=f"{code_dir}/demo_data/mustard0")
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=2)
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    # mesh = trimesh.load(args.mesh_file)
    mesh = trimesh.load(args.mesh_file, force="mesh")
    # mesh.apply_scale(1000.0)
    # mesh.apply_scale(0.4)

    init_frame = 145

    # Prepare reader and estimate metric scale before building the estimator
    reader = EpicReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
    renderer_vis = ModelRendererOffscreen(reader.K, reader.H, reader.W, zfar=100.0)
    if len(reader.id_strs) > 0:
        try:
            valid_ids = [int(s) for s in reader.id_strs]
            # Use the first frame that will be processed in the loop (>=145) if available
            # cand = [ii for ii in valid_ids if ii >= 145]
            # sel_id = min(cand) if len(cand) > 0 else min(valid_ids)
            sel_id = min(valid_ids)
            depth_0 = reader.get_depth(sel_id)
            mask_0 = reader.get_mask(sel_id).astype(bool)
            depth_pts_0 = masked_depth_to_points(depth_0, mask_0, reader.K)
            model_pts_0 = np.asarray(mesh.vertices)
            s_est_0 = estimate_scale_by_radius(model_pts_0, depth_pts_0)
            if s_est_0 is not None and np.isfinite(s_est_0) and s_est_0 > 0:
                mesh.apply_scale(s_est_0)
                logging.info(f"Applied metric scale to mesh before registering: s = {s_est_0:.6f} (m/unit)")
            else:
                logging.info("Scale estimate unavailable; proceeding without mesh scaling.")
        except Exception as e:
            logging.info(f"Scale estimation failed; proceeding without mesh scaling. Error: {e}")

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f"rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam")

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

    # Build index lists around init frame
    id_list_sorted = sorted([int(s) for s in reader.id_strs])
    id_set = set(id_list_sorted)
    if init_frame not in id_set:
        # Fallback to closest available index
        init_frame = min(id_list_sorted, key=lambda x: abs(x - init_frame))
        logging.info(f"init_frame not in available ids; using closest {init_frame}")

    rendered_frames = {}

    # 1) Registration at init_frame
    i = int(init_frame)
    logging.info(f"Register at init_frame={i}")
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    mask = reader.get_mask(i).astype(bool)
    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
    # Area-based quick scale correction to align silhouette area before fine refinements
    try:
        area_factor, area_rend, area_ref = compute_area_scale_factor(renderer_vis, mesh, pose, mask, min_factor=0.7, max_factor=1.4)
        if np.isfinite(area_factor) and area_factor > 0 and abs(area_factor - 1.0) > 1e-3:
            mesh.apply_scale(area_factor)
            logging.info(f"Pre-adjust mesh scale by area ratio: factor={area_factor:.4f} (rend={area_rend:.0f}, ref={area_ref:.0f})")
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
            est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
    except Exception as e:
        logging.info(f"Area-based scale pre-adjustment failed: {e}")
    # Render posed mesh on top of the RGB frame and save once for the first frame
    try:
        rend_rgb, rend_depth = renderer_vis.render(mesh=mesh, ob_in_cvcam=pose)
        comp = color.copy()
        fg_mask = rend_depth > 0
        comp[fg_mask] = rend_rgb[fg_mask]
        imageio.imwrite(f"{debug_dir}/rendered.png", comp)
    except Exception as e:
        logging.info(f"Rendering overlay failed: {e}")

    # Silhouette-based scale refinement around current scale
    try:
        mask_sil = erode_mask(mask, iterations=1).astype(np.uint8)
        scale_grid = np.linspace(0.92, 1.08, 17)
        best_factor, best_iou = refine_scale_by_silhouette(renderer_vis, mesh, pose, mask_sil, scale_grid)
        if np.isfinite(best_factor) and best_factor > 0 and abs(best_factor - 1.0) > 5e-3 and best_iou > 0.0:
            mesh.apply_scale(best_factor)
            logging.info(f"Refined mesh scale by silhouette IoU: factor={best_factor:.4f}, IoU={best_iou:.4f}")
            # Recompute bounds and reinitialize estimator with the updated mesh
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
            est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
            # Re-register with the refined scale
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
            try:
                rend_rgb, rend_depth = renderer_vis.render(mesh=mesh, ob_in_cvcam=pose)
                comp = color.copy()
                fg_mask = rend_depth > 0
                comp[fg_mask] = rend_rgb[fg_mask]
                imageio.imwrite(f"{debug_dir}/rendered.png", comp)
            except Exception as e:
                logging.info(f"Rendering overlay after scale refine failed: {e}")
    except Exception as e:
        logging.info(f"Silhouette scale refinement failed: {e}")

    # Depth-based scale refinement (rendered depth vs sensor depth inside mask)
    try:
        mask_sil = erode_mask(mask, iterations=1).astype(np.uint8)
        scale_grid_dep = np.linspace(0.9, 1.1, 21)
        best_factor_dep, best_err = refine_scale_by_depth(renderer_vis, mesh, pose, depth, mask_sil, scale_grid_dep)
        if np.isfinite(best_factor_dep) and best_factor_dep > 0 and abs(best_factor_dep - 1.0) > 5e-3:
            mesh.apply_scale(best_factor_dep)
            logging.info(f"Refined mesh scale by depth residual: factor={best_factor_dep:.4f}, med_err={best_err:.4f}m")
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
            est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
            try:
                rend_rgb, rend_depth = renderer_vis.render(mesh=mesh, ob_in_cvcam=pose)
                comp = color.copy()
                fg_mask = rend_depth > 0
                comp[fg_mask] = rend_rgb[fg_mask]
                imageio.imwrite(f"{debug_dir}/rendered.png", comp)
            except Exception as e:
                logging.info(f"Rendering overlay after depth refine failed: {e}")
    except Exception as e:
        logging.info(f"Depth scale refinement failed: {e}")

    # Save pose and vis for init_frame
    os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
    np.savetxt(f"{debug_dir}/ob_in_cam/{i}.txt", pose.reshape(4, 4))
    if debug >= 1:
        center_pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
    if debug >= 2:
        os.makedirs(f"{debug_dir}/track_vis", exist_ok=True)
        imageio.imwrite(f"{debug_dir}/track_vis/{i}.png", vis)

    # Store overlay frame for video
    try:
        rend_rgb_init, rend_depth_init = renderer_vis.render(mesh=mesh, ob_in_cvcam=pose)
        comp_init = color.copy()
        comp_init[rend_depth_init > 0] = rend_rgb_init[rend_depth_init > 0]
        rendered_frames[i] = comp_init
    except Exception as e:
        logging.info(f"Per-frame rendering failed at i={i}: {e}")
        rendered_frames[i] = color

    # Keep a copy of the internal last pose for backward tracking seed
    try:
        pose_last_init = est.pose_last.detach().clone()
    except Exception:
        pose_last_init = None

    # 2) Forward pass from init_frame+1 to end
    for j in [x for x in id_list_sorted if x > i]:
        logging.info(f"Forward track i={j}")
        color = reader.get_color(j)
        depth = reader.get_depth(j)
        pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
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

    # 3) Backward pass from init_frame-1 to start
    if pose_last_init is not None:
        try:
            est.pose_last = pose_last_init
        except Exception:
            pass
    for j in [x for x in reversed(id_list_sorted) if x < i]:
        logging.info(f"Backward track i={j}")
        color = reader.get_color(j)
        depth = reader.get_depth(j)
        pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
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

    # After collecting all frames in rendered_frames, write them sequentially by index order
    for j in id_list_sorted:
        if j in rendered_frames:
            frame = rendered_frames[j]
        else:
            # fall back to raw color if something went wrong earlier
            frame = reader.get_color(j)
        video_writer.write(frame[..., ::-1])

    # Finalize video
    try:
        video_writer.release()
        logging.info(f"Saved rendered video to {video_out}")
    except Exception as e:
        logging.info(f"Failed to finalize video: {e}")
