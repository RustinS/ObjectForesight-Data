#!/usr/bin/env python3
"""Verify per-object tracks from step4_sam.py with InternVL and write moved_by_hand.txt."""

import argparse
import glob
import hashlib
import os
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from utils import rprint as print

os.environ.update({"OMP_NUM_THREADS": "1", "OMP_WAIT_POLICY": "ACTIVE", "OMP_PROC_BIND": "false", "ORT_DISABLE_THREAD_AFFINITY": "1"})
warnings.filterwarnings("ignore")

try:
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_config():
    parser = argparse.ArgumentParser(description="Verify SAM2 tracks with InternVL (sharded).")
    parser.add_argument("--video_root", default="./manip_data", help="Root containing narration_id/*/action.mp4 and objects/")
    parser.add_argument("--output_root", default="./manip_data", help="Where narration_id subfolders live")
    parser.add_argument("--csv_file", type=str, default="EPIC_100.csv", help="EPIC csv with narration_id,duration_s,no_hands_presence")
    parser.add_argument("--ext", type=str, default="mp4", help="Video extension")
    parser.add_argument("--start_video_idx", type=int, default=0, help="Manual slice start (AFTER sharding)")
    parser.add_argument("--end_video_idx", type=int, default=-1, help="Manual slice end (AFTER sharding). -1 = no limit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--internvl_model_id", type=str, default="OpenGVLab/InternVL3-78B", help="HF model id for InternVL3")
    parser.add_argument("--internvl_dtype", type=str, choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--internvl_input_size", type=int, default=448, help="InternVL tile input size")
    parser.add_argument("--internvl_max_tiles", type=int, default=1, help="Max tiles for dynamic preprocess")
    parser.add_argument("--internvl_use_thumbnail", action="store_true", default=True)
    parser.add_argument("--internvl_max_new_tokens", type=int, default=2)
    parser.add_argument("--internvl_non_blocking", action="store_true", help="Pinned memory + non-blocking H2D copies")
    parser.add_argument("--num_segments", type=int, default=32, help="How many temporal samples per object clip")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards for job array")
    parser.add_argument("--shard_idx", type=int, default=0, help="This shard index [0..num_shards-1]")
    return parser.parse_args()


def stable_int_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def build_transform(input_size):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff, best_ratio = float("inf"), (1, 1)
    area = width * height
    for ratio in target_ratios:
        ratio_diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if ratio_diff < best_ratio_diff or (
            ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
        ):
            best_ratio_diff, best_ratio = ratio_diff, ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / max(1e-6, orig_height)
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    best_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width, target_height = image_size * best_ratio[0], image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]

    resized_img = image.resize((target_width, target_height))
    cols = target_width // image_size
    processed_images = [
        resized_img.crop(((i % cols) * image_size, (i // cols) * image_size, ((i % cols) + 1) * image_size, ((i // cols) + 1) * image_size)) for i in range(blocks)
    ]

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def get_bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not np.any(rows):
        return None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax + 1, ymax + 1


def apply_mask_overlay(image_array: np.ndarray, mask_array: np.ndarray, color=(255, 0, 0), alpha=0.4) -> np.ndarray:
    mask_bool = mask_array.astype(bool)
    result = image_array.copy()
    result[mask_bool] = ((1.0 - alpha) * result[mask_bool].astype(np.float32) + alpha * np.array(color, dtype=np.float32)).astype(image_array.dtype)
    return result


def build_object_frames(obj_dir: Path, video_path: str, input_size=448, max_num=12, num_segments=8, use_thumbnail=True):
    mask_npz = obj_dir / "masks.npz"
    if not mask_npz.exists():
        raise FileNotFoundError(f"masks.npz not found in {obj_dir}")

    data = np.load(mask_npz)
    available_keys = sorted(data.files, key=int)
    if not available_keys:
        raise RuntimeError(f"No masks inside {mask_npz}")

    # Sample num_segments frames uniformly
    idxs = np.linspace(0, len(available_keys) - 1, num_segments, dtype=int)
    chosen = [(available_keys[i], int(available_keys[i])) for i in idxs]

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    frame_h, frame_w = vr[0].asnumpy().shape[:2]
    transform = build_transform(input_size)
    pixel_values_list, num_patches_list = [], []

    for key, frame_idx in chosen:
        frame_array = vr[min(frame_idx, max_frame)].asnumpy()
        mask_array = data[key]
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]

        mask_resized = np.array(Image.fromarray(mask_array.astype(np.uint8)).resize((frame_w, frame_h), resample=Image.NEAREST))
        bbox = get_bbox_from_mask(mask_resized)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        pad_w, pad_h = max(1, x2 - x1), max(1, y2 - y1)
        x1_pad, y1_pad = max(0, x1 - pad_w), max(0, y1 - pad_h)
        x2_pad, y2_pad = min(frame_w, x2 + pad_w), min(frame_h, y2 + pad_h)
        if x2_pad <= x1_pad or y2_pad <= y1_pad:
            continue

        crop = frame_array[y1_pad:y2_pad, x1_pad:x2_pad]
        if crop.size == 0:
            continue

        mask_crop = mask_resized[y1_pad:y2_pad, x1_pad:x2_pad]
        highlighted = apply_mask_overlay(crop, mask_crop, color=(255, 0, 0), alpha=0.7)
        img = Image.fromarray(highlighted).convert("RGB")

        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num)
        pv = torch.stack([transform(t) for t in tiles])
        pixel_values_list.append(pv)
        num_patches_list.append(pv.shape[0])

    if not pixel_values_list:
        raise RuntimeError(f"No valid crops from {mask_npz}")

    return torch.cat(pixel_values_list, dim=0), num_patches_list


def ask_internvl_yes_no(model, tokenizer, pixel_values, num_patches_list, question, dtype=torch.bfloat16, non_blocking=False, max_new_tokens=2):
    pixel_values = pixel_values.to(device="cuda", dtype=dtype, non_blocking=non_blocking)
    video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)

    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=dtype):
        response = model.chat(tokenizer, pixel_values, video_prefix + question, generation_config, num_patches_list=num_patches_list, history=None, return_history=False)

    txt = str(response).strip().lower()
    if txt.startswith("yes"):
        return "yes", txt
    if txt.startswith("no"):
        return "no", txt
    if ("yes" in txt) != ("no" in txt):
        return ("yes" if "yes" in txt else "no"), txt
    if "yes" in txt and "no" in txt:
        return ("yes" if txt.index("yes") < txt.index("no") else "no"), txt
    return "no", txt


def init_internvl_model(cfg):
    print(f"Loading {cfg.internvl_model_id}...")
    dtype = torch.bfloat16 if cfg.internvl_dtype == "bf16" else torch.float16
    model = AutoModel.from_pretrained(
        cfg.internvl_model_id,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto",
        dtype=dtype,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.internvl_model_id, trust_remote_code=True, use_fast=False)
    print(f"{cfg.internvl_model_id} loaded successfully!")
    return model, tokenizer


QUESTION = (
    "You are a video analyst. Your task is to determine if the masked object (highlighted in bright red) "
    "is being **actively moved** *by a human hand*. This is an egocentric video, so the camera will be moving. "
    "Answer with only the word 'yes' or 'no'.\n\n"
    "**Answer 'yes' ONLY IF:**\n"
    "1. The mask is on an external object (NOT the hand itself).\n"
    "2. A human hand is clearly **causing the object's motion** (e.g., grasping, pushing, pulling, picking up, setting down).\n\n"
    "**Answer 'no' IF:**\n"
    "1. The mask is on the hand itself.\n"
    "2. The masked object is stationary *relative to the scene* (e.g., sitting on a counter), "
    "even if it appears to move because the camera is moving.\n"
    "3. A hand is **only touching** the object but **not moving it** (e.g., a hand resting on a stationary object).\n\n"
    "Question: Is the masked object being moved by a human hand?\n\n"
)


def process_single_object(obj_path: Path, video_path: str, cfg, model, tokenizer):
    parts = obj_path.stem.split("+", 1)
    obj_id, obj_name = (parts[0], parts[1]) if len(parts) == 2 else (obj_path.stem, obj_path.stem)
    print(f"Video QA for object {obj_id}: {obj_name} in {obj_path}")

    if (obj_path / "moved_by_hand.txt").exists():
        print(f"Skipping {obj_path}: moved_by_hand.txt already exists")
        return

    pv, npl = build_object_frames(
        obj_path, video_path, input_size=cfg.internvl_input_size, max_num=cfg.internvl_max_tiles, num_segments=cfg.num_segments, use_thumbnail=cfg.internvl_use_thumbnail
    )
    dtype = torch.bfloat16 if cfg.internvl_dtype == "bf16" else torch.float16
    cls, raw = ask_internvl_yes_no(model, tokenizer, pv, npl, QUESTION, dtype=dtype, non_blocking=cfg.internvl_non_blocking, max_new_tokens=cfg.internvl_max_new_tokens)

    print(f"[{obj_name}] moved_by_hand: {cls} | raw: {raw}")
    (obj_path / "moved_by_hand.txt").write_text(f"{cls}\n{raw}")


def process_video_objects(objects_path: Path, video_path: str, cfg, model, tokenizer):
    objects_list = [p for p in objects_path.glob("*") if p.is_dir()]
    print(f"Total objects: {len(objects_list)}")
    for obj_path in objects_list:
        try:
            process_single_object(obj_path, video_path, cfg, model, tokenizer)
        except Exception as e:
            print(f"Error processing {obj_path}: {e}\n{traceback.format_exc()}")


def main():
    cfg = load_config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    print(f"Using device: cuda" if torch.cuda.is_available() else "Using device: cpu")

    if not os.path.exists(cfg.csv_file):
        raise FileNotFoundError(f"CSV not found: {cfg.csv_file}")

    df = pd.read_csv(cfg.csv_file)
    for col in ["narration_id", "no_hands_presence", "duration_s"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column '{col}'")
    valid_narration_ids = set(df[(df["no_hands_presence"] == 0) & (df["duration_s"] < 10)]["narration_id"].astype(str))

    # Load or discover video paths
    video_folders_txt = os.path.join(cfg.video_root, "video_folders.txt")
    if os.path.exists(video_folders_txt):
        print("Loading video folders from file")
        with open(video_folders_txt) as f:
            all_videos = [line.strip() for line in f]
    else:
        print("Finding video folders")
        ci_ext = "".join([f"[{c.lower()}{c.upper()}]" for c in cfg.ext])
        all_videos = sorted([vp for vp in glob.glob(os.path.join(cfg.video_root, f"**/*.{ci_ext}"), recursive=True) if os.path.basename(vp).lower() == "action.mp4"])
        with open(video_folders_txt, "w") as f:
            f.write("\n".join(all_videos))

    if not all_videos:
        print(f"No action videos found under {cfg.video_root}")
        return

    print(f"Found {len(all_videos)} candidate videos total.")

    num_shards = max(1, cfg.num_shards)
    shard_idx = cfg.shard_idx % num_shards
    sharded_paths = [vp for vp in all_videos if stable_int_hash(os.path.basename(os.path.dirname(vp))) % num_shards == shard_idx]
    print(f"Shard {shard_idx}/{num_shards}: {len(sharded_paths)} videos in this shard.")

    end_idx = cfg.end_video_idx if cfg.end_video_idx != -1 else None
    candidate_paths = sharded_paths[cfg.start_video_idx : end_idx]
    print(f"After slicing: {len(candidate_paths)} videos remain for this shard.")

    model, tokenizer = init_internvl_model(cfg)

    for local_i, vpath in enumerate(candidate_paths, 1):
        seq_name = os.path.basename(os.path.dirname(vpath))
        if seq_name not in valid_narration_ids:
            print(f"Skipping {seq_name}: not in filtered_df.")
            continue

        print(f"\n[{local_i}/{len(candidate_paths)}] Processing {seq_name}: {vpath}")
        objects_path = Path(cfg.output_root) / seq_name / "objects"
        if not objects_path.exists():
            print(f"No objects/ dir for {seq_name}. Skipping.")
            continue

        try:
            process_video_objects(objects_path, vpath, cfg, model, tokenizer)
        except Exception as e:
            print(f"Error processing {seq_name}: {e}\n{traceback.format_exc()}")

    print("Done.")


if __name__ == "__main__":
    main()
