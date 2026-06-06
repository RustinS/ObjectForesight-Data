"""Batched EgoHOS inference and refinement over action clips (sharded)."""

import sys

sys.path.append("EgoHOS/mmsegmentation")

import argparse
import glob
import hashlib
import os
import random
import traceback
import warnings

import numpy as np
import pandas as pd
import segmentation_refinement as refine
import torch
from decord import VideoReader, cpu
from mmseg.apis import inference_segmentor, init_segmentor
from segmentation_refinement.eval_helper import process_high_res_im, process_im_single_pass
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from utils import rprint as print

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ------------------------- CLI & Utils -------------------------


def parse_args():
    p = argparse.ArgumentParser(description="EgoHOS batched inference (sharded for Slurm arrays)")
    p.add_argument("--data_root", type=str, default="./manip_data")
    p.add_argument("--csv_file", type=str, default="EPIC_100.csv")
    p.add_argument("--start_video_idx", type=int, default=0)
    p.add_argument("--end_video_idx", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--ext", type=str, default="mp4", help="Video extension to match (case-insensitive)")
    p.add_argument("--decord_threads", type=int, default=2)

    # Model configs/checkpoints
    p.add_argument("--twohands_config_file", type=str, default="./EgoHOS/mmsegmentation/work_dirs/seg_twohands_ccda/seg_twohands_ccda.py")
    p.add_argument("--twohands_checkpoint_file", type=str, default="./EgoHOS/mmsegmentation/work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth")
    p.add_argument("--cb_config_file", type=str, default="./EgoHOS/mmsegmentation/work_dirs/twohands_to_cb_ccda/twohands_to_cb_ccda.py")
    p.add_argument("--cb_checkpoint_file", type=str, default="./EgoHOS/mmsegmentation/work_dirs/twohands_to_cb_ccda/best_mIoU_iter_76000.pth")
    p.add_argument("--obj1_config_file", type=str, default="./EgoHOS/mmsegmentation/work_dirs/twohands_cb_to_obj1_ccda/twohands_cb_to_obj1_ccda.py")
    p.add_argument("--obj1_checkpoint_file", type=str, default="./EgoHOS/mmsegmentation/work_dirs/twohands_cb_to_obj1_ccda/best_mIoU_iter_34000.pth")

    # Refinement
    p.add_argument("--refine_L", type=int, default=900, help="Seg-refine stride (L)")

    # Sharding for multi-node scaling
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_idx", type=int, default=0)

    # Skipping
    p.add_argument("--force", action="store_true", help="Force recompute even if outputs exist")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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


def load_action_videos(root: str, ext: str) -> list[str]:
    """Load action videos with a cached listing."""
    video_folders_file = os.path.join(root, "video_folders.txt")
    if os.path.exists(video_folders_file):
        print("Loading video folders from file")
        with open(video_folders_file, "r") as f:
            videos = [line.strip() for line in f if line.strip()]
    else:
        print("Finding video folders")
        ci_ext = "".join(f"[{c.lower()}{c.upper()}]" for c in ext)
        videos = [
            vp
            for vp in sorted(glob.glob(os.path.join(root, f"**/*.{ci_ext}"), recursive=True))
            if os.path.basename(vp).lower() == "action.mp4"
        ]
        with open(video_folders_file, "w") as f:
            f.write("\n".join(videos))
    return videos


def outputs_exist(out_dir: str) -> bool:
    """Check if output mask files already exist."""
    two = os.path.join(out_dir, "twohands_masks.npz")
    obj = os.path.join(out_dir, "obj_masks.npz")
    return os.path.exists(two) and os.path.exists(obj) and os.path.getsize(two) > 0 and os.path.getsize(obj) > 0


# ------------------------- Refinement -------------------------


def batch_refine_masks(refiner: refine.Refiner, imgs_rgb: np.ndarray, masks: np.ndarray, fast: bool = True, L: int = 900) -> np.ndarray:
    device = next(refiner.model.parameters()).device if hasattr(refiner.model, "parameters") else refiner.device
    with torch.no_grad():
        imgs = torch.from_numpy(imgs_rgb).permute(0, 3, 1, 2).float().div_(255.0).to(device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        imgs = (imgs - mean) / std

        seg = torch.from_numpy((masks > 127).astype(np.float32)).unsqueeze(1).to(device)
        seg = seg * 2.0 - 1.0  # match refiner normalization

        out = process_im_single_pass(refiner.model, imgs, seg, L) if fast else process_high_res_im(refiner.model, imgs, seg, L)
        return (out[:, 0].clamp(0, 1).cpu().numpy() * 255).astype("uint8")


# ------------------------- Pipeline -------------------------


class EgoHOSPipeline:
    def __init__(self, two_cfg: str, two_ckpt: str, cb_cfg: str, cb_ckpt: str, obj_cfg: str, obj_ckpt: str):
        print("Loading models on cuda:0 ...")
        self.two_model = init_segmentor(two_cfg, two_ckpt, device="cuda:0")
        self.cb_model = init_segmentor(cb_cfg, cb_ckpt, device="cuda:0")
        self.obj_model = init_segmentor(obj_cfg, obj_ckpt, device="cuda:0")
        self.refiner = refine.Refiner(device="cuda:0")

    @torch.no_grad()
    def predict_video(self, vr: VideoReader, model, batch_size: int, refine_L: int, aux_twohands: np.ndarray = None, aux_cb: np.ndarray = None) -> np.ndarray:
        total_frames = len(vr)
        result_frames = []

        for start in tqdm(range(0, total_frames, batch_size)):
            end = min(start + batch_size, total_frames)
            indices = list(range(start, end))

            batch_rgb = vr.get_batch(indices).asnumpy()  # (B,H,W,3) RGB
            batch_bgr = batch_rgb[..., ::-1].copy()  # mmseg wants BGR

            if aux_twohands is not None or aux_cb is not None:
                batch_th = [aux_twohands[i] for i in indices] if aux_twohands is not None else None
                batch_cb = [aux_cb[i] for i in indices] if aux_cb is not None else None
                seg_results = inference_segmentor(model, batch_bgr, twohands_list=batch_th, cb_list=batch_cb)
            else:
                seg_results = inference_segmentor(model, batch_bgr)

            if isinstance(seg_results, np.ndarray):
                seg_results = list(seg_results)

            h, w = batch_rgb.shape[1:3]
            batch_seg_final = np.zeros((len(batch_rgb), h, w), dtype=np.uint8)

            present_labels = [lbl for lbl in range(1, 6) if any((seg == lbl).any() for seg in seg_results)]
            for label in present_labels:
                bin_masks = np.stack([(seg == label).astype(np.uint8) * 255 for seg in seg_results])
                refined = batch_refine_masks(self.refiner, batch_rgb, bin_masks, fast=True, L=refine_L)
                for i in range(refined.shape[0]):
                    batch_seg_final[i][refined[i] > 0] = label

            result_frames.extend(list(batch_seg_final))

        return np.array(result_frames)

    def run_full_video(self, vpath: str, out_dir: str, batch_size: int, refine_L: int, decord_threads: int, force: bool) -> tuple[bool, str]:
        """Full pipeline on one video: twohands -> cb -> obj1 -> save."""
        if not force and outputs_exist(out_dir):
            return True, "skipped (exists)"

        os.makedirs(out_dir, exist_ok=True)
        vr = VideoReader(vpath, ctx=cpu(0), num_threads=max(1, decord_threads))

        print("Predicting twohands ...")
        twohands = self.predict_video(vr, self.two_model, batch_size, refine_L)

        print("Predicting contact boundary ...")
        cb = self.predict_video(vr, self.cb_model, batch_size, refine_L, aux_twohands=twohands)

        print("Predicting 1st-order interacting objects ...")
        obj1 = self.predict_video(vr, self.obj_model, batch_size, refine_L, aux_twohands=twohands, aux_cb=cb)

        np.savez_compressed(os.path.join(out_dir, "twohands_masks.npz"), masks=twohands.astype(np.uint8))
        np.savez_compressed(os.path.join(out_dir, "obj_masks.npz"), masks=obj1.astype(np.uint8))

        del twohands, cb, obj1, vr
        torch.cuda.empty_cache()
        return True, "done"


# ------------------------- Main -------------------------


def main():
    args = parse_args()
    set_seed(args.seed)

    df = pd.read_csv(args.csv_file)
    if "duration_s" not in df.columns and {"start_timestamp", "stop_timestamp"} <= set(df.columns):
        st = pd.to_timedelta(df["start_timestamp"], errors="coerce")
        en = pd.to_timedelta(df["stop_timestamp"], errors="coerce")
        df["duration_s"] = (en - st).dt.total_seconds().fillna(0).clip(lower=0)
        df.to_csv(args.csv_file, index=False)
        print(f"Added duration_s to {args.csv_file}")
    allow_nids = set(df.loc[df["duration_s"] < 10, "narration_id"].astype(str)) if {"narration_id", "duration_s"} <= set(df.columns) else set()

    all_video_paths = load_action_videos(args.data_root, args.ext)
    if not all_video_paths:
        print(f"No videos found under {args.data_root}")
        return

    num_shards = max(1, args.num_shards)
    shard_idx = args.shard_idx % num_shards
    sharded = [vp for vp in all_video_paths if stable_int_hash(os.path.dirname(vp)) % num_shards == shard_idx]
    end_idx = args.end_video_idx if args.end_video_idx != -1 else None
    candidate_paths = sharded[args.start_video_idx : end_idx]
    video_paths = [vp for vp in candidate_paths if os.path.basename(os.path.dirname(vp)) in allow_nids] if allow_nids else candidate_paths
    print(f"Shard {shard_idx}/{num_shards}: {len(video_paths)} videos after filtering.")

    # Init models once
    pipe = EgoHOSPipeline(
        args.twohands_config_file,
        args.twohands_checkpoint_file,
        args.cb_config_file,
        args.cb_checkpoint_file,
        args.obj1_config_file,
        args.obj1_checkpoint_file,
    )

    for i, vpath in enumerate(video_paths, 1):
        seq_name = os.path.basename(os.path.dirname(vpath))
        out_dir = os.path.join(os.path.dirname(vpath), "egohos")

        print(f"\n[{i}/{len(sharded)}] Processing {seq_name}")

        try:
            ok, status = pipe.run_full_video(
                vpath=vpath,
                out_dir=out_dir,
                batch_size=args.batch_size,
                refine_L=args.refine_L,
                decord_threads=args.decord_threads,
                force=args.force,
            )
            print(f"Status: {status}")
        except Exception as e:
            print(f"Error processing {vpath}: {e}")
            print(traceback.format_exc())


if __name__ == "__main__":
    main()
