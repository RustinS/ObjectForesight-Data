#!/usr/bin/env python3
"""SpaTrackerV2 over EPIC action videos (sharded) -> <narration_id>/spatracker.npz."""

import argparse
import glob
import hashlib
import os
import random
import sys
import traceback

import decord
import numpy as np
import pandas as pd
import torch

sys.path.append("SpaTrackerV2")

from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.models.utils import get_points_on_a_grid
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from utils import rprint as print

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def stable_int_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def parse_args():
    p = argparse.ArgumentParser(description="SpaTrackerV2 over EPIC videos (sharded).")
    p.add_argument("--video_root", type=str, default="./manip_data")
    p.add_argument("--csv_file", type=str, default="EPIC_100.csv")
    p.add_argument("--start_video_idx", type=int, default=0)
    p.add_argument("--end_video_idx", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--grid-size", type=int, default=10)
    p.add_argument("--vo-points", type=int, default=100)
    p.add_argument("--stride", type=int, default=-1)
    return p.parse_args()


@torch.no_grad()
def process_single_video(args, seq_dir: str, video_path: str, models):
    vggt4track_model, model = models

    vr = decord.VideoReader(video_path)
    video_tensor = torch.from_numpy(vr.get_batch(range(len(vr))).asnumpy()).permute(0, 3, 1, 2).float()

    if args.stride > 0:
        video_tensor = video_tensor[:: args.stride]

    print(f"Video tensor shape: {tuple(video_tensor.shape)}")

    video_tensor = preprocess_image(video_tensor, keep_ratio=True)[None]  # [1, T, C, H, W]

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        preds = vggt4track_model(video_tensor.to(DEVICE) / 255.0)

    depth_tensor = preds["points_map"][..., 2].squeeze().detach().cpu().numpy()
    extrs = preds["poses_pred"].squeeze().detach().cpu().numpy()
    intrs = preds["intrs"].squeeze().detach().cpu().numpy()
    unc_metric = preds["unc_metric"].squeeze().detach().cpu().numpy() > 0.5
    video_tensor = video_tensor.squeeze()  # [T, C, H, W]

    grid_pts = get_points_on_a_grid(args.grid_size, video_tensor.shape[2:], device="cpu")
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        c2w_traj, intrs_out, point_map, conf_depth, track3d_pred, _, vis_pred, *_ = model.forward(
            video_tensor,
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
            support_frame=len(video_tensor) - 1,
            replace_ratio=0.2,
        )

    depth_save = point_map[:, 2, ...].clone()
    depth_save[conf_depth < 0.5] = 0

    output = {
        "coords": (torch.einsum("tij,tnj->tni", c2w_traj[:, :3, :3], track3d_pred[:, :, :3].cpu()) + c2w_traj[:, :3, 3][:, None, :]).numpy(),
        "extrinsics": torch.inverse(c2w_traj).cpu().numpy(),
        "intrinsics": intrs_out.cpu().numpy(),
        "depths": depth_save.cpu().numpy(),
        "visibs": vis_pred.cpu().numpy(),
        "unc_metric": conf_depth.cpu().numpy(),
    }

    out_path = os.path.join(seq_dir, "spatracker.npz")
    np.savez_compressed(out_path, **output)
    print(f"Saved: {out_path}")

    torch.cuda.empty_cache()


def load_video_paths(video_root: str) -> list[str]:
    cache_file = os.path.join(video_root, "video_folders.txt")
    if os.path.exists(cache_file):
        print("Loading video folders from file")
        with open(cache_file) as f:
            return [line.strip() for line in f if line.strip()]

    print("Finding video folders")
    all_videos = sorted(vp for vp in glob.glob(os.path.join(video_root, "**/*.[mM][pP]4"), recursive=True) if os.path.basename(vp).lower() == "action.mp4")
    with open(cache_file, "w") as f:
        f.write("\n".join(all_videos))
    return all_videos


def main():
    args = parse_args()
    set_seed(args.seed)
    print(f"Using device: {DEVICE}")

    df = pd.read_csv(args.csv_file)
    for col in ["narration_id", "no_hands_presence", "duration_s"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column '{col}'")
    valid_ids = set(df[(df["no_hands_presence"] == 0) & (df["duration_s"] < 10)]["narration_id"].astype(str))

    all_videos = load_video_paths(args.video_root)
    if not all_videos:
        print(f"No action videos found under {args.video_root}")
        return
    print(f"Found {len(all_videos)} candidate videos total.")

    num_shards = max(1, args.num_shards)
    shard_idx = args.shard_idx % num_shards
    sharded_paths = [vp for vp in all_videos if stable_int_hash(os.path.basename(os.path.dirname(vp))) % num_shards == shard_idx]
    print(f"Shard {shard_idx}/{num_shards}: {len(sharded_paths)} videos in this shard.")

    end_idx = args.end_video_idx if args.end_video_idx != -1 else None
    candidate_paths = sharded_paths[args.start_video_idx : end_idx]
    print(f"After slicing: {len(candidate_paths)} videos remain for this shard.")

    # Load models once
    vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().to(DEVICE)
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    model.spatrack.track_num = args.vo_points
    model.eval().to(DEVICE)
    models = (vggt4track_model, model)

    # Process videos
    for i, vpath in enumerate(candidate_paths, 1):
        seq_dir = os.path.dirname(vpath)
        seq_name = os.path.basename(seq_dir)

        # Skip if not in filtered set, already processed, or missing objects folder
        if seq_name not in valid_ids or os.path.exists(os.path.join(seq_dir, "spatracker.npz")) or not os.path.exists(os.path.join(seq_dir, "objects")):
            continue

        try:
            print(f"\n[{i}/{len(candidate_paths)}] Processing {seq_name}: {vpath}")
            process_single_video(args, seq_dir, vpath, models)
        except Exception as e:
            print(f"Error processing {vpath}: {e}\n{traceback.format_exc()}")

    print("Done.")


if __name__ == "__main__":
    main()
