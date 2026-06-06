#!/usr/bin/env python3
"""TRELLIS image-to-3D on filtered per-object crops (sharded)."""

import os
import shutil
import warnings
import multiprocessing
import argparse
from pathlib import Path
import traceback
import glob
import hashlib
import sys

sys.path.append("trellis")

import imageio
import numpy as np
from PIL import Image
import torch
import cv2
import pandas as pd
from utils import rprint as print
from decord import VideoReader, cpu

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils, render_utils
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline

os.environ.update(
    {
        "SPCONV_ALGO": "native",
        "OMP_NUM_THREADS": "1",
        "OMP_WAIT_POLICY": "ACTIVE",
        "OMP_PROC_BIND": "false",
        "ORT_DISABLE_THREAD_AFFINITY": "1",
    }
)
warnings.filterwarnings("ignore")
multiprocessing.set_start_method("spawn", force=True)

try:
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass


def load_config():
    p = argparse.ArgumentParser(description="Run TRELLIS image-to-3D over EPIC (sharded).")

    p.add_argument("--video_root", default="/gpfs/scrubbed/rustin/manip_data", help="Root containing narration_id/*/action.mp4 and objects/")
    p.add_argument("--output_root", default="/gpfs/scrubbed/rustin/manip_data", help="Where narration_id subfolders live (usually same as video_root)")
    p.add_argument("--csv_file", type=str, default="EPIC_100.csv", help="EPIC csv with narration_id,duration_s,no_hands_presence,...")
    p.add_argument("--ext", type=str, default="mp4", help="Video extension (case-insensitive match on 'action.mp4')")

    p.add_argument("--start_video_idx", type=int, default=0, help="Manual slice start (AFTER sharding).")
    p.add_argument("--end_video_idx", type=int, default=-1, help="Manual slice end (AFTER sharding). -1 = no limit.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num_shards", type=int, default=1, help="Total number of shards for job array.")
    p.add_argument("--shard_idx", type=int, default=0, help="This shard index [0..num_shards-1].")

    p.add_argument("--per_object_image_num", type=int, default=10, help="Number of best clean images to feed TRELLIS.")
    p.add_argument("--viz", action="store_true", help="Render side-by-side GS+mesh videos.")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if trellis outputs exist.")

    p.add_argument("--flux_sr_enable", action="store_true", default=True, help="Enable Flux ControlNet Upscaler (diffusers).")
    p.add_argument("--flux_sr_repo_controlnet", type=str, default="jasperai/Flux.1-dev-Controlnet-Upscaler")
    p.add_argument("--flux_sr_repo_base", type=str, default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--flux_sr_scale", type=int, default=2, help="Pre-upscale factor for control image (e.g., 2 or 4)")
    p.add_argument("--flux_sr_steps", type=int, default=28)
    p.add_argument("--flux_sr_guidance", type=float, default=3.5)
    p.add_argument("--flux_sr_cond_scale", type=float, default=0.6)
    p.add_argument("--flux_sr_prompt", type=str, default="")

    return p.parse_args()


def stable_int_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def ensure_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_video(video_data, path: Path, fps=30):
    imageio.mimsave(str(path), video_data, fps=fps)


def render_and_save_videos(outputs, obj_name: str, video_dir: Path):
    ensure_dir(video_dir)
    gs_video = render_utils.render_video(outputs["gaussian"][0], verbose=False)["color"]
    mesh_video = render_utils.render_video(outputs["mesh"][0], verbose=False)["normal"]
    combined = [np.concatenate([g, m], axis=1) for g, m in zip(gs_video, mesh_video)]
    save_video(combined, video_dir / f"{obj_name}_final.mp4")


def save_3d_models(outputs, result_dir: Path):
    glb_mesh = postprocessing_utils.to_glb(outputs["gaussian"][0], outputs["mesh"][0], simplify=0.7, texture_size=2048)
    (result_dir / "model.glb").parent.mkdir(parents=True, exist_ok=True)
    glb_mesh.export(str(result_dir / "model.glb"))
    outputs["gaussian"][0].save_ply(str(result_dir / "gaussian.ply"))


def init_flux_upscaler(cfg):
    try:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        controlnet = FluxControlNetModel.from_pretrained(cfg.flux_sr_repo_controlnet, torch_dtype=dtype)
        pipe = FluxControlNetPipeline.from_pretrained(cfg.flux_sr_repo_base, controlnet=controlnet, torch_dtype=dtype)
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        pipe.set_progress_bar_config(disable=True)
        return pipe
    except Exception as e:
        print(f"Flux SR unavailable ({e}); proceeding without Flux SR.")
        return None


def find_clean_pair(obj_path: Path):
    crops, masks = obj_path / "clean_cropped_frames.npz", obj_path / "clean_masks.npz"
    return (crops, masks) if crops.exists() and masks.exists() else (None, None)


def process_single_object(vpath: str, obj_path: Path, trellis_pipeline, flux_pipe, cfg):
    obj_name = obj_path.stem.split("+", 1)[-1] if "+" in obj_path.stem else obj_path.stem

    result_dir = obj_path / "trellis"
    if not cfg.overwrite and (result_dir / "model.glb").exists() and (result_dir / "gaussian.ply").exists():
        print(f"Skipping {obj_name} (trellis results already exist)")
        return

    crops_path, masks_path = find_clean_pair(obj_path)
    all_masks_path = obj_path / "masks.npz"
    if crops_path is None or masks_path is None or (not all_masks_path.exists()):
        print(f"Missing clean npz or masks for {obj_name}. Skipping...")
        return

    vr = VideoReader(vpath, ctx=cpu(0), num_threads=1)
    clean_masks_dict = {int(k): v for k, v in np.load(masks_path, allow_pickle=False).items()}
    all_masks_dict = {int(k): v for k, v in np.load(all_masks_path, allow_pickle=False).items()}

    print(f"Loaded {len(clean_masks_dict)} clean masks for {obj_name} (Total masks: {len(all_masks_dict)})")

    if len(clean_masks_dict) < max(3, len(all_masks_dict) // 5):
        print(f"Insufficient clean frames for {obj_name}. Skipping…")
        return

    proc_masks, areas_per_key = {}, {}
    kernel = np.ones((3, 3), np.uint8)
    for k, m in clean_masks_dict.items():
        mb = (m > 0.5).astype(np.uint8) if m.dtype != bool else m.astype(np.uint8)
        mb = cv2.morphologyEx(mb, cv2.MORPH_OPEN, kernel, iterations=1)
        area = np.count_nonzero(mb)
        if area > 0:
            proc_masks[k] = mb
            areas_per_key[k] = area

    if not proc_masks:
        print(f"No valid masks after preprocessing for {obj_name}. Skipping…")
        return

    areas = list(areas_per_key.values())
    if len(areas) >= 3:
        q1, q3 = np.percentile(areas, 25), np.percentile(areas, 75)
        cap = q3 + 1.5 * (q3 - q1)
        filtered_keys = [k for k, a in areas_per_key.items() if a <= cap]
    else:
        filtered_keys = list(proc_masks.keys())

    top_keys = sorted(filtered_keys or list(proc_masks.keys()), key=lambda x: areas_per_key[x], reverse=True)
    top_keys = top_keys[: cfg.per_object_image_num]

    pil_imgs = []
    for k in top_keys:
        mask = proc_masks[k]
        try:
            frame_rgb = vr[k].asnumpy()
        except Exception:
            continue

        if mask.shape[:2] != frame_rgb.shape[:2]:
            mask = cv2.resize(mask, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            pil_imgs.append(Image.fromarray(frame_rgb, mode="RGB"))
            continue

        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        x, y, x2, y2 = max(0, x), max(0, y), min(frame_rgb.shape[1], x + w), min(frame_rgb.shape[0], y + h)
        if x2 <= x or y2 <= y:
            pil_imgs.append(Image.fromarray(frame_rgb, mode="RGB"))
            continue

        roi = frame_rgb[y:y2, x:x2]
        cropped_mask = mask[y:y2, x:x2]
        cm3 = np.repeat(cropped_mask[:, :, None].astype(bool), 3, axis=2) if cropped_mask.ndim == 2 else cropped_mask.astype(bool)
        masked_img = np.where(cm3, roi, np.zeros_like(roi))
        pil_imgs.append(Image.fromarray(masked_img, mode="RGB"))

    target_max = 770
    resample = Image.LANCZOS
    processed_imgs = []
    for im in pil_imgs:
        w, h = im.size
        if flux_pipe is not None:
            try:
                s = max(1, cfg.flux_sr_scale)
                control_image = im.resize((w * s, h * s), resample=resample)
                gen = torch.Generator(device="cuda").manual_seed(cfg.seed)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out = flux_pipe(
                        prompt=cfg.flux_sr_prompt,
                        control_image=control_image,
                        controlnet_conditioning_scale=cfg.flux_sr_cond_scale,
                        num_inference_steps=cfg.flux_sr_steps,
                        guidance_scale=cfg.flux_sr_guidance,
                        height=control_image.size[1],
                        width=control_image.size[0],
                        generator=gen,
                    )
                im = out.images[0]
                w, h = im.size
            except Exception:
                pass

        if max(w, h) != target_max:
            scale = target_max / max(w, h)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), resample=resample)
        processed_imgs.append(im)
    pil_imgs = processed_imgs
    result_dir = ensure_dir(result_dir)
    print(f"Running TRELLIS for {obj_name} with {len(pil_imgs)} images…")
    torch.manual_seed(cfg.seed)
    outputs = trellis_pipeline.run_multi_image(
        pil_imgs,
        seed=cfg.seed,
        sparse_structure_sampler_params={"steps": cfg.per_object_image_num * 4, "cfg_strength": 10},
        slat_sampler_params={"steps": cfg.per_object_image_num * 4, "cfg_strength": 10},
    )

    if cfg.viz:
        render_and_save_videos(outputs, obj_name, result_dir / "videos")
    save_3d_models(outputs, result_dir)
    print(f"[OK] Saved TRELLIS outputs for {obj_name} -> {result_dir}")


def process_video_objects(vpath: str, objects_path: Path, trellis_pipeline, flux_pipe, cfg):
    """Iterate over each object dir; only those with clean_* pairs are processed."""
    objects_list = []
    for p in sorted(objects_path.glob("*")):
        if not p.is_dir():
            continue
        crops_path, masks_path = find_clean_pair(p)
        if crops_path and masks_path:
            objects_list.append(p)

    print(f"Total objects with clean pairs: {len(objects_list)}")
    for obj_path in objects_list:
        try:
            process_single_object(vpath, obj_path, trellis_pipeline, flux_pipe, cfg)
        except Exception as e:
            print(f"Error processing object {obj_path}: {e}")
            print(traceback.format_exc())


# -------------------------
# Main driver (mirrors step6)
# -------------------------
def main():
    cfg = load_config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # CSV filter
    if not os.path.exists(cfg.csv_file):
        raise FileNotFoundError(f"CSV not found: {cfg.csv_file}")
    df = pd.read_csv(cfg.csv_file)
    for col in ["narration_id", "no_hands_presence", "duration_s"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column '{col}'")
    filtered_df = df[(df["no_hands_presence"] == 0) & (df["duration_s"] < 10)]

    # Cache list of action.mp4
    video_folders_txt = os.path.join(cfg.video_root, "video_folders.txt")
    if os.path.exists(video_folders_txt):
        print("Loading video folders from file")
        with open(video_folders_txt, "r") as f:
            all_videos = [line.strip() for line in f]
    else:
        print("Finding video folders")
        ci_ext = "".join([f"[{c.lower()}{c.upper()}]" for c in cfg.ext])
        pattern = os.path.join(cfg.video_root, f"**/*.{ci_ext}")
        all_videos = sorted(glob.glob(pattern, recursive=True))
        all_videos = [vp for vp in all_videos if os.path.basename(vp).lower() == "action.mp4"]
        with open(video_folders_txt, "w") as f:
            for vp in all_videos:
                f.write(vp + "\n")

    if len(all_videos) == 0:
        print(f"No action videos found under {cfg.video_root} with extension .{cfg.ext}.")
        return

    print(f"Found {len(all_videos)} candidate videos total.")

    # Shard & slice
    num_shards = max(1, cfg.num_shards)
    shard_idx = cfg.shard_idx % num_shards
    sharded_paths = [vp for vp in all_videos if stable_int_hash(os.path.basename(os.path.dirname(vp))) % num_shards == shard_idx]
    print(f"Shard {shard_idx}/{num_shards}: {len(sharded_paths)} videos in this shard.")

    start_idx = max(0, cfg.start_video_idx)
    end_idx = cfg.end_video_idx
    candidate_paths = sharded_paths[start_idx:end_idx] if end_idx != -1 else sharded_paths[start_idx:]

    total = len(candidate_paths)
    print(f"After slicing: {total} videos remain for this shard.")

    # Load TRELLIS once per shard
    print("Loading TRELLIS pipeline…")
    trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    trellis_pipeline.cuda()
    flux_pipe = init_flux_upscaler(cfg)
    print("TRELLIS loaded!")

    # Iterate
    for local_i, vpath in enumerate(candidate_paths, 1):
        try:
            seq_dir = os.path.dirname(vpath)
            seq_name = os.path.basename(seq_dir)

            if str(seq_name) not in filtered_df["narration_id"].astype(str).values:
                print(f"Skipping {seq_name}: not in filtered_df.")
                continue

            print("\n")
            print(f"[{local_i}/{total}] Processing {seq_name}: {vpath}")

            objects_path = Path(cfg.output_root) / seq_name / "objects"
            if not objects_path.exists():
                print(f"No objects/ dir for {seq_name} (looked in {objects_path}). Skipping.")
                continue

            process_video_objects(vpath, objects_path, trellis_pipeline, flux_pipe, cfg)

        except Exception as e:
            print(f"Error processing {vpath}: {e}")
            print(traceback.format_exc())
            print("Recovering TRELLIS pipeline...")
            del trellis_pipeline
            trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
            trellis_pipeline.cuda()
            continue

    print("Done.")


if __name__ == "__main__":
    main()
