# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
try:
    from utils import Logger
except Exception:
    import sys as _sys
    from pathlib import Path as _Path
    # Project root containing utils.py
    _root = str(_Path(__file__).resolve().parents[1])
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from utils import Logger
from datareader import *
from typing import Optional, Union
import itertools
from learning.training.predict_score import *
from learning.training.predict_pose_refine import *
import yaml
import numpy as np


class FoundationPose:
    def __init__(
        self,
        model_pts,
        model_normals,
        symmetry_tfs=None,
        mesh=None,
        scorer: ScorePredictor = None,
        refiner: PoseRefinePredictor = None,
        glctx=None,
        debug=0,
        debug_dir="/home/bowen/debug/novel_pose_debug/",
        iou_min=0.1,
        allow_internal_reregister=False,
    ):
        self.gt_pose = None
        self.ignore_normal_flip = True
        self.debug = debug
        self.debug_dir = debug_dir
        os.makedirs(debug_dir, exist_ok=True)

        self.reset_object(model_pts, model_normals, symmetry_tfs=symmetry_tfs, mesh=mesh)
        self.make_rotation_grid(min_n_views=90, inplane_step=15)

        self.glctx = glctx

        if scorer is not None:
            self.scorer = scorer
        else:
            self.scorer = ScorePredictor()

        if refiner is not None:
            self.refiner = refiner
        else:
            self.refiner = PoseRefinePredictor()

        self.pose_last = None  # Used for tracking; per the centered mesh

        # Mask-aware tracking hyperparameters (safe defaults)
        self.mask_erode = 1
        self.mask_dilate = 2
        self.sil_weight = 0.25
        self.overflow_weight = 0.10
        self.iou_min = iou_min
        self.iou_min = iou_min
        self.allow_internal_reregister = allow_internal_reregister
        self.reinit_event = None
        self.last_score = 0.0

    def reset_object(self, model_pts, model_normals, symmetry_tfs=None, mesh=None):
        max_xyz = mesh.vertices.max(axis=0)
        min_xyz = mesh.vertices.min(axis=0)
        self.model_center = (min_xyz + max_xyz) / 2
        if mesh is not None:
            self.mesh_ori = mesh.copy()
            mesh = mesh.copy()
            mesh.vertices = mesh.vertices - self.model_center.reshape(1, 3)

        model_pts = mesh.vertices
        self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
        self.vox_size = max(self.diameter / 20.0, 0.003)
        Logger.info(f"self.diameter:{self.diameter}, vox_size:{self.vox_size}")
        self.dist_bin = self.vox_size / 2
        self.angle_bin = 20  # Deg
        pcd = toOpen3dCloud(model_pts, normals=model_normals)
        pcd = pcd.voxel_down_sample(self.vox_size)
        self.max_xyz = np.asarray(pcd.points).max(axis=0)
        self.min_xyz = np.asarray(pcd.points).min(axis=0)
        self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device="cuda")
        self.normals = F.normalize(torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device="cuda"), dim=-1)
        self.mesh_path = None
        self.mesh = mesh
        if self.mesh is not None:
            self.mesh_path = f"/tmp/{uuid.uuid4()}.obj"
            self.mesh.export(self.mesh_path)
        self.mesh_tensors = make_mesh_tensors(self.mesh)

        if symmetry_tfs is None:
            self.symmetry_tfs = torch.eye(4).float().cuda()[None]
        else:
            self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device="cuda", dtype=torch.float)

    def get_tf_to_centered_mesh(self):
        tf_to_center = torch.eye(4, dtype=torch.float, device="cuda")
        tf_to_center[:3, 3] = -torch.as_tensor(self.model_center, device="cuda", dtype=torch.float)
        return tf_to_center

    def to_device(self, s="cuda:0"):
        for k in self.__dict__:
            self.__dict__[k] = self.__dict__[k]
            if torch.is_tensor(self.__dict__[k]) or isinstance(self.__dict__[k], nn.Module):
                self.__dict__[k] = self.__dict__[k].to(s)
        for k in self.mesh_tensors:
            self.mesh_tensors[k] = self.mesh_tensors[k].to(s)
        if self.refiner is not None:
            self.refiner.model.to(s)
        if self.scorer is not None:
            self.scorer.model.to(s)
        if self.glctx is not None:
            self.glctx = dr.RasterizeCudaContext(s)

    def make_rotation_grid(self, min_n_views=60, inplane_step=45):
        cam_in_obs = sample_views_icosphere(n_views=min_n_views)
        rot_grid = []
        for i in range(len(cam_in_obs)):
            for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
                cam_in_ob = cam_in_obs[i]
                R_inplane = euler_matrix(0, 0, inplane_rot)
                cam_in_ob = cam_in_ob @ R_inplane
                ob_in_cam = np.linalg.inv(cam_in_ob)
                rot_grid.append(ob_in_cam)

        rot_grid = np.asarray(rot_grid)
        rot_grid = mycpp.cluster_poses(20, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy())
        rot_grid = np.asarray(rot_grid)
        self.rot_grid = torch.as_tensor(rot_grid, device="cuda", dtype=torch.float)

    def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None):
        """
        @scene_pts: torch tensor (N,3)
        """
        ob_in_cams = self.rot_grid.clone()
        center = self.guess_translation(depth=depth, mask=mask, K=K)
        ob_in_cams[:, :3, 3] = torch.tensor(center, device="cuda", dtype=torch.float).reshape(1, 3)
        return ob_in_cams

    def guess_translation(self, depth, mask, K):
        vs, us = np.where(mask > 0)
        if len(us) == 0:
            return np.zeros((3))
        uc = (us.min() + us.max()) / 2.0
        vc = (vs.min() + vs.max()) / 2.0
        valid = mask.astype(bool) & (depth >= 0.001)
        if not valid.any():
            return np.zeros((3))

        zc = np.median(depth[valid])
        center = (np.linalg.inv(K) @ np.asarray([uc, vc, 1]).reshape(3, 1)) * zc

        # if self.debug >= 2:
        #     pcd = toOpen3dCloud(center.reshape(1, 3))
        #     o3d.io.write_point_cloud(f"{self.debug_dir}/init_center.ply", pcd)

        return center.reshape(3)

    def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5):
        """Copmute pose from given pts to self.pcd
        @pts: (N,3) np array, downsampled scene points
        """
        

        if self.glctx is None:
            if glctx is None:
                self.glctx = dr.RasterizeCudaContext()
                # self.glctx = dr.RasterizeGLContext()
            else:
                self.glctx = glctx

        depth = erode_depth(depth, radius=2, device="cuda")
        depth = bilateral_filter_depth(depth, radius=2, device="cuda")

        # if self.debug >= 2:
        #     xyz_map = depth2xyzmap(depth, K)
        #     valid = xyz_map[..., 2] >= 0.001
        #     pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
        #     o3d.io.write_point_cloud(f"{self.debug_dir}/scene_raw.ply", pcd)
        #     cv2.imwrite(f"{self.debug_dir}/ob_mask.png", (ob_mask * 255.0).clip(0, 255))

        normal_map = None
        valid = (depth >= 0.001) & (ob_mask > 0)
        if valid.sum() < 4:
            pose = np.eye(4)
            pose[:3, 3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
            return pose

        # if self.debug >= 2:
        #     imageio.imwrite(f"{self.debug_dir}/color.png", rgb)
        #     cv2.imwrite(f"{self.debug_dir}/depth.png", (depth * 1000).astype(np.uint16))
        #     valid = xyz_map[..., 2] >= 0.001
        #     pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
        #     o3d.io.write_point_cloud(f"{self.debug_dir}/scene_complete.ply", pcd)

        self.H, self.W = depth.shape[:2]
        self.K = K
        self.ob_id = ob_id
        self.ob_mask = ob_mask

        poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
        poses = poses.data.cpu().numpy()
        center = self.guess_translation(depth=depth, mask=ob_mask, K=K)

        poses = torch.as_tensor(poses, device="cuda", dtype=torch.float)
        poses[:, :3, 3] = torch.as_tensor(center.reshape(1, 3), device="cuda")

        add_errs = self.compute_add_err_to_gt_pose(poses)

        xyz_map = depth2xyzmap(depth, K)
        poses, vis = self.refiner.predict(
            mesh=self.mesh,
            mesh_tensors=self.mesh_tensors,
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=poses.data.cpu().numpy(),
            normal_map=normal_map,
            xyz_map=xyz_map,
            glctx=self.glctx,
            mesh_diameter=self.diameter,
            iteration=iteration,
            get_vis=self.debug >= 2,
        )
        # if vis is not None:
        #     imageio.imwrite(f"{self.debug_dir}/vis_refiner.png", vis)

        scores, vis = self.scorer.predict(
            mesh=self.mesh,
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=poses.data.cpu().numpy(),
            normal_map=normal_map,
            mesh_tensors=self.mesh_tensors,
            glctx=self.glctx,
            mesh_diameter=self.diameter,
            get_vis=self.debug >= 2,
        )
        # if vis is not None:
        #     imageio.imwrite(f"{self.debug_dir}/vis_score.png", vis)

        add_errs = self.compute_add_err_to_gt_pose(poses)

        ids = torch.as_tensor(scores).argsort(descending=True)
        scores = scores[ids]
        poses = poses[ids]

        best_pose = poses[0] @ self.get_tf_to_centered_mesh()
        self.pose_last = poses[0]
        self.best_id = ids[0]

        self.poses = poses
        self.scores = scores

        # Expose a finite net score for external logging (e.g., init candidate "net=")
        try:
            s0 = float(scores[0])
            self.last_score = s0 if np.isfinite(s0) else 0.0
        except Exception:
            self.last_score = 0.0

        return best_pose.data.cpu().numpy()

    def compute_add_err_to_gt_pose(self, poses):
        """
        @poses: wrt. the centered mesh
        """
        return -torch.ones(len(poses), device="cuda", dtype=torch.float)

    def render_silhouette(self, pose: torch.Tensor, K, H: int, W: int) -> torch.Tensor:
        """Render a binary silhouette mask using nvdiffrast for the given pose.

        Args:
            pose: Pose tensor of shape (4, 4) on CUDA.
            K: Intrinsic matrix as numpy array (3, 3).
            H: Image height.
            W: Image width.

        Returns:
            torch.Tensor: Predicted silhouette mask of shape (1, 1, H, W), float in [0, 1].
        """
        if pose.ndim == 2:
            ob_in_cams = pose.reshape(1, 4, 4)
        else:
            ob_in_cams = pose
        color_r, depth_r, _ = nvdiffrast_render(
            K=K, H=H, W=W, ob_in_cams=ob_in_cams, context="cuda", get_normal=False, glctx=self.glctx, mesh_tensors=self.mesh_tensors, output_size=[H, W], use_light=False
        )
        depth_r = depth_r.reshape(1, H, W)
        pred_mask = (depth_r > 0).float().unsqueeze(1)  # (1,1,H,W)
        return pred_mask

    def track_one(self, rgb, depth, K, iteration, extra={}, ob_mask: Optional[Union[np.ndarray, torch.Tensor]] = None):
        """Track pose for a single frame with optional mask-aware stabilization.

        If ob_mask is provided, residuals are computed inside the mask and a
        silhouette IoU-based gating is applied per refinement step.

        Args:
            rgb: HxWx3 uint8 array (RGB).
            depth: HxW float array or tensor in meters.
            K: 3x3 numpy intrinsics.
            iteration: Number of refine iterations.
            extra: Dict for returning visualization when debug>=2.
            ob_mask: Optional HxW binary mask (numpy or torch). When None, behavior
                     matches the original tracker.

        Returns:
            np.ndarray: 4x4 pose matrix in the original (centered-mesh) coordinate.
        """
        if self.pose_last is None:
            raise RuntimeError

        H, W = int(depth.shape[0]), int(depth.shape[1])

        # Depth pre-processing (as before)
        depth_t = torch.as_tensor(depth, device="cuda", dtype=torch.float)
        depth_t = erode_depth(depth_t, radius=2, device="cuda")
        depth_t = bilateral_filter_depth(depth_t, radius=2, device="cuda")

        use_mask = False
        gt_mask_f = None
        rgb_for_refine = rgb
        depth_for_refine = depth_t
        if ob_mask is not None:
            try:
                # Morphological refinement for stability
                if isinstance(ob_mask, np.ndarray):
                    m_np = erode_dilate_mask(ob_mask, erode=self.mask_erode, dilate=self.mask_dilate)
                else:
                    m_np = erode_dilate_mask(ob_mask.detach().cpu().numpy(), erode=self.mask_erode, dilate=self.mask_dilate)
                if m_np.sum() > 0:
                    use_mask = True
                    gt_mask_f = to_torch_mask(m_np, H, W, device="cuda")  # (1,1,H,W)
                    # Zero-out RGB/Depth outside mask for the refiner
                    rgb_masked = np.array(rgb).copy()
                    rgb_masked[m_np == 0] = 0
                    rgb_for_refine = rgb_masked
                    depth_for_refine = depth_t * torch.as_tensor(m_np, device="cuda", dtype=torch.float)
                else:
                    pass
            except Exception as e:
                Logger.info(f"Mask processing failed; proceeding without mask. Err: {e}")

        # xyz map for the refiner
        xyz_map = depth2xyzmap_batch(depth_for_refine[None], torch.as_tensor(K, dtype=torch.float, device="cuda")[None], zfar=np.inf)[0]

        # Non-masked path: preserve original behavior
        if not use_mask:
            pose, vis = self.refiner.predict(
                mesh=self.mesh,
                mesh_tensors=self.mesh_tensors,
                rgb=rgb,
                depth=depth_t,
                K=K,
                ob_in_cams=self.pose_last.reshape(1, 4, 4).data.cpu().numpy(),
                normal_map=None,
                xyz_map=xyz_map,
                mesh_diameter=self.diameter,
                glctx=self.glctx,
                iteration=iteration,
                get_vis=self.debug >= 2,
            )
            if self.debug >= 2:
                extra["vis"] = vis
            self.pose_last = pose
            # Expose last metrics for external progress reporting
            self.last_iou = None
            # Try to produce a meaningful score here too (best-effort).
            try:
                _scores, _ = self.scorer.predict(
                    mesh=self.mesh,
                    rgb=rgb,
                    depth=depth if isinstance(depth, np.ndarray) else depth_t.detach().cpu().numpy(),
                    K=K,
                    ob_in_cams=pose.data.cpu().numpy(),
                    normal_map=None,
                    mesh_tensors=self.mesh_tensors,
                    glctx=self.glctx,
                    mesh_diameter=self.diameter,
                    get_vis=False,
                )
                s0 = float(_scores[0]) if hasattr(_scores, "__len__") else float(_scores)
                self.last_score = s0 if np.isfinite(s0) else 0.0
            except Exception:
                self.last_score = 0.0
            return (pose @ self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4, 4)

        # Mask-aware iterative refinement: call one step at a time to enable IoU gating
        current_pose = self.pose_last.clone()
        iou_prev = None
        for it in range(int(max(1, iteration))):
            pose_step, vis = self.refiner.predict(
                mesh=self.mesh,
                mesh_tensors=self.mesh_tensors,
                rgb=rgb_for_refine,
                depth=depth_for_refine,
                K=K,
                ob_in_cams=current_pose.reshape(1, 4, 4).data.cpu().numpy(),
                normal_map=None,
                xyz_map=xyz_map,
                mesh_diameter=self.diameter,
                glctx=self.glctx,
                iteration=1,
                get_vis=False,
            )
            candidate_pose = pose_step  # (1,4,4)

            # Compute silhouette IoU and masked residuals
            pred_mask_f = self.render_silhouette(candidate_pose, K, H, W)  # (1,1,H,W)
            iou_val = soft_iou(pred_mask_f, gt_mask_f)

            if float(iou_val) < float(self.iou_min):
                if self.debug >= 1:
                    Logger.info(f"IoU {float(iou_val):.3f} < threshold {self.iou_min:.3f}; signaling re-init to outer tracker")
                try:
                    self.last_iou = float(iou_val)
                except Exception:
                    self.last_iou = None
                self.last_score = 0.0
                self.reinit_event = {"cause": "iou_gate", "iou": float(iou_val)}
                return None   # <<< let outer tracker re-init & log it

            # Build weights for in-mask and overflow rim
            w_in, w_out = mask_weights(gt_mask_f, rim_width_px=6)

            # Render full-res RGB/Depth at candidate pose for residuals
            rgb_r, depth_r, _ = nvdiffrast_render(
                K=K,
                H=H,
                W=W,
                ob_in_cams=candidate_pose,
                context="cuda",
                get_normal=False,
                glctx=self.glctx,
                mesh_tensors=self.mesh_tensors,
                output_size=[H, W],
                use_light=True,
            )
            rgb_r = rgb_r.permute(0, 3, 1, 2)  # (1,3,H,W), in [0,1]
            depth_r = depth_r.unsqueeze(1)  # (1,1,H,W)

            # Observations as tensors
            rgb_obs = torch.as_tensor(rgb_for_refine, device="cuda", dtype=torch.float).permute(2, 0, 1).unsqueeze(0) / 255.0
            depth_obs = depth_for_refine.unsqueeze(0).unsqueeze(0)

            # In-mask residuals
            eps = 1e-6
            L_photo = (w_in * (rgb_r - rgb_obs).abs()).sum() / (w_in.sum() * 3 + eps)
            valid_depth = (depth_obs > 0).float()
            L_depth = (w_in * valid_depth * (depth_r - depth_obs).abs()).sum() / (w_in.sum() + eps)

            # Silhouette loss and overflow penalty
            L_sil = 1.0 - iou_val
            overflow = torch.clamp(pred_mask_f - gt_mask_f, min=0.0)
            L_overflow = (w_out * overflow).mean()

            # Score (for logging)
            score = -(L_photo + L_depth + self.sil_weight * L_sil + self.overflow_weight * L_overflow)

            if self.debug >= 3:
                Logger.info(
                    f"track iter {it}: IoU={float(iou_val):.3f}, L_photo={float(L_photo):.4f}, L_depth={float(L_depth):.4f}, overflow={float(L_overflow):.4f}, score={float(score):.4f}"
                )

            # Early rejection on catastrophic updates
            if iou_prev is not None and (float(iou_prev) - float(iou_val)) > 0.05 and float(iou_val) < float(self.iou_min):
                if self.debug >= 1:
                    Logger.info(f"Rejecting update at iter {it}: IoU drop {float(iou_prev) - float(iou_val):.3f} below min {self.iou_min}")
                break  # keep current_pose unchanged and stop early

            # Accept update
            current_pose = candidate_pose.detach()
            iou_prev = iou_val.detach()

        # Finalize
        self.pose_last = current_pose
        # Expose last metrics for external progress reporting
        try:
            self.last_iou = float(iou_prev) if iou_prev is not None else None
        except Exception:
            self.last_iou = None
        try:
            s = float(score)
            self.last_score = s if np.isfinite(s) else 0.0
        except Exception:
            self.last_score = 0.0
        return (current_pose @ self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4, 4)
