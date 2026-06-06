#!/usr/bin/env python3
"""Amodal mask completion with Diffusion-VAS over EPIC (sharded)."""

import argparse
import glob
import hashlib
import multiprocessing as mp
import os
import sys
import traceback
import warnings
from pathlib import Path

sys.path.append("diffusion-vas")

import cv2
import numpy as np
import pandas as pd
import torch
from decord import VideoReader
from models.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
from models.diffusion_vas.pipeline_diffusion_vas import DiffusionVASPipeline
from PIL import Image
from torchvision import transforms

from utils import rprint as print

warnings.filterwarnings("ignore")
mp.set_start_method("spawn", force=True)
os.environ.update({"SPCONV_ALGO": "native", "OMP_NUM_THREADS": "1", "OMP_WAIT_POLICY": "ACTIVE", "OMP_PROC_BIND": "false", "ORT_DISABLE_THREAD_AFFINITY": "1"})

try:
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass

PRED_RESOLUTION = (256, 512)
DEPTH_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def init_amodal_segmentation_model(model_path: str):
    pipe = DiffusionVASPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=False)
    return pipe


def init_depth_model(ckpt_path: str, encoder: str):
    model = DepthAnythingV2(**DEPTH_CONFIGS[encoder]).to("cuda")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model


def load_and_transform_masks(masks: dict, resolution):
    mask_transform = transforms.Compose(
        [
            transforms.Resize(resolution, antialias=True),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )
    sorted_masks = [masks[k] for k in sorted(masks)]
    original_size = sorted_masks[0].shape[:2]
    processed = [mask_transform(Image.fromarray(((m > 0).astype(np.uint8) * 255))) for m in sorted_masks]
    return torch.stack(processed).unsqueeze(0), original_size


def load_and_transform_rgbs(video_path: str, idxs, resolution):
    rgb_transform = transforms.Compose(
        [
            transforms.Resize(resolution, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )
    vr = VideoReader(video_path)
    sorted_idxs = sorted(idxs)
    frames = [rgb_transform(Image.fromarray(vr[i].asnumpy()).convert("RGB")) for i in sorted_idxs]
    return torch.stack(frames).unsqueeze(0), len(sorted_idxs)


def rgb_to_depth(rgb_tensor: torch.Tensor, depth_model):
    rgb_images = ((rgb_tensor.squeeze(0) + 1.0) / 2.0) * 255.0
    depth_maps = np.stack([depth_model.infer_image(rgb_images[i].cpu().numpy().astype(np.uint8).transpose(1, 2, 0)) for i in range(rgb_images.shape[0])])
    mn, mx = depth_maps.min(), depth_maps.max()
    depth_maps = (depth_maps - mn) / (mx - mn) if mx > mn else np.zeros_like(depth_maps)
    depth = torch.from_numpy(depth_maps * 2.0 - 1.0).float().unsqueeze(1).repeat(1, 3, 1, 1)
    return depth.unsqueeze(0)


def stable_int_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def load_config():
    p = argparse.ArgumentParser(description="Diffusion-VAS: amodal mask completion over EPIC (sharded).")
    p.add_argument("--video_root", default="/gpfs/scrubbed/rustin/manip_data")
    p.add_argument("--output_root", default="/gpfs/scrubbed/rustin/manip_data")
    p.add_argument("--csv_file", default="EPIC_100.csv")
    p.add_argument("--ext", default="mp4")
    p.add_argument("--start_video_idx", type=int, default=0)
    p.add_argument("--end_video_idx", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--model_path_mask", default="diffusion-vas/checkpoints/diffusion-vas-amodal-segmentation")
    p.add_argument("--depth_encoder", choices=["vits", "vitb", "vitl", "vitg"], default="vitl")
    p.add_argument("--model_path_depth_dir", default="diffusion-vas/checkpoints/")
    p.add_argument("--pred_h", type=int, default=PRED_RESOLUTION[0])
    p.add_argument("--pred_w", type=int, default=PRED_RESOLUTION[1])
    return p.parse_args()


def process_single_object(vpath: str, obj_path: Path, generator, pipeline_mask, depth_model, cfg):
    obj_name = obj_path.stem
    vas_masks_path = obj_path / "vas_masks.npz"
    vas_clean_masks_path = obj_path / "vas_clean_masks.npz"

    if not cfg.overwrite and vas_clean_masks_path.exists() and vas_masks_path.exists():
        print(f"Skipping {obj_name} (VAS outputs exist)")
        return

    clean_masks_path, all_masks_path = obj_path / "clean_masks.npz", obj_path / "masks.npz"
    if not clean_masks_path.exists() or not all_masks_path.exists():
        print(f"Missing clean/masks npz for {obj_name}. Skipping…")
        return

    clean_masks_dict = {int(k): v for k, v in np.load(clean_masks_path, allow_pickle=False).items()}
    all_masks_dict = {int(k): v for k, v in np.load(all_masks_path, allow_pickle=False).items()}
    mask_keys = sorted(all_masks_dict.keys())
    clean_frame_keys = set(clean_masks_dict.keys())
    print(f"Loaded {len(clean_masks_dict)} clean masks for {obj_name} (total: {len(all_masks_dict)})")

    pred_res = (cfg.pred_h, cfg.pred_w)
    modal_pixels, ori_shape = load_and_transform_masks(all_masks_dict, resolution=pred_res)
    rgb_pixels, num_frames = load_and_transform_rgbs(vpath, idxs=mask_keys, resolution=pred_res)
    depth_pixels = rgb_to_depth(rgb_pixels, depth_model)

    print("Amodal segmentation with Diffusion-VAS …")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        result = pipeline_mask(
            modal_pixels,
            depth_pixels,
            height=pred_res[0],
            width=pred_res[1],
            num_frames=num_frames,
            decode_chunk_size=8,
            motion_bucket_id=127,
            fps=60,
            noise_aug_strength=0.02,
            min_guidance_scale=1.5,
            max_guidance_scale=1.5,
            generator=generator,
        )

    pred_amodal = np.stack([np.array(im) for im in result.frames[0]], axis=0)
    pred_amodal = (pred_amodal.sum(axis=-1) > 600).astype(np.uint8)
    modal_mask_union = (modal_pixels[0].sum(dim=1) > 0).cpu().numpy().astype(np.uint8)
    pred_amodal = np.logical_or(pred_amodal, modal_mask_union).astype(np.uint8)
    pred_amodal_resized = [cv2.resize(m, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_NEAREST) for m in pred_amodal]

    vas_masks_dict = {idx: am.astype(np.uint8) for idx, am in zip(mask_keys, pred_amodal_resized)}
    vas_clean_masks_dict = {idx: vas_masks_dict[idx] for idx in mask_keys if idx in clean_frame_keys}

    np.savez_compressed(vas_masks_path, **{str(k): v for k, v in vas_masks_dict.items()})
    np.savez_compressed(vas_clean_masks_path, **{str(k): v for k, v in vas_clean_masks_dict.items()})
    print(f"[OK] VAS saved for {obj_name}")


def process_video(vpath: str, objects_path: Path, generator, pipeline_mask, depth_model, cfg):
    objects_list = [p for p in sorted(objects_path.glob("*")) if p.is_dir() and (p / "clean_cropped_frames.npz").exists() and (p / "clean_masks.npz").exists()]
    print(f"Total objects eligible for VAS: {len(objects_list)}")
    for obj_path in objects_list:
        try:
            process_single_object(vpath, obj_path, generator, pipeline_mask, depth_model, cfg)
        except Exception as e:
            print(f"Error processing object {obj_path}: {e}\n{traceback.format_exc()}")


def main():
    cfg = load_config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    print("Using device: cuda")

    if not os.path.exists(cfg.csv_file):
        raise FileNotFoundError(f"CSV not found: {cfg.csv_file}")
    df = pd.read_csv(cfg.csv_file)
    for col in ["narration_id", "no_hands_presence", "duration_s"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column '{col}'")
    valid_narrations = set(df[(df["no_hands_presence"] == 0) & (df["duration_s"] < 10)]["narration_id"].astype(str))

    video_folders_txt = os.path.join(cfg.video_root, "video_folders.txt")
    if os.path.exists(video_folders_txt):
        print("Loading video folders from file")
        with open(video_folders_txt) as f:
            all_videos = [line.strip() for line in f]
    else:
        print("Finding video folders")
        ci_ext = "".join(f"[{c.lower()}{c.upper()}]" for c in cfg.ext)
        all_videos = sorted(vp for vp in glob.glob(os.path.join(cfg.video_root, f"**/*.{ci_ext}"), recursive=True) if os.path.basename(vp).lower() == "action.mp4")
        with open(video_folders_txt, "w") as f:
            f.write("\n".join(all_videos))

    if not all_videos:
        print(f"No action videos found under {cfg.video_root}")
        return
    print(f"Found {len(all_videos)} candidate videos total.")

    shard_idx = cfg.shard_idx % max(1, cfg.num_shards)
    sharded_paths = [vp for vp in all_videos if stable_int_hash(os.path.basename(os.path.dirname(vp))) % cfg.num_shards == shard_idx]
    print(f"Shard {shard_idx}/{cfg.num_shards}: {len(sharded_paths)} videos in this shard.")

    candidates = sharded_paths[cfg.start_video_idx : None if cfg.end_video_idx == -1 else cfg.end_video_idx]
    print(f"After slicing: {len(candidates)} videos remain for this shard.")

    print("Loading Diffusion-VAS and DepthAnythingV2 …")
    pipeline_mask = init_amodal_segmentation_model(cfg.model_path_mask)
    depth_model = init_depth_model(os.path.join(cfg.model_path_depth_dir, f"depth_anything_v2_{cfg.depth_encoder}.pth"), cfg.depth_encoder)
    generator = torch.Generator(device="cuda").manual_seed(cfg.seed)
    print("Models ready!")

    for i, vpath in enumerate(candidates, 1):
        try:
            seq_name = os.path.basename(os.path.dirname(vpath))
            if seq_name not in valid_narrations:
                print(f"Skipping {seq_name}: not in filtered_df.")
                continue

            print(f"\n[{i}/{len(candidates)}] Processing {seq_name}: {vpath}")
            objects_path = Path(cfg.output_root) / seq_name / "objects"
            if not objects_path.exists():
                print(f"No objects/ dir for {seq_name}. Skipping.")
                continue

            process_video(vpath, objects_path, generator, pipeline_mask, depth_model, cfg)
        except Exception as e:
            print(f"Error processing {vpath}: {e}\n{traceback.format_exc()}")

    print("Done.")


if __name__ == "__main__":
    main()
