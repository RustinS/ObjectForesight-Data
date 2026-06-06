#!/usr/bin/env python3
"""Filter per-object crops/masks using InternVL visibility labels (yes/partial/no)."""

import argparse
import glob
import hashlib
import multiprocessing
import os
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode
from tqdm.rich import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from utils import rprint as print

# Env setup
os.environ.update({"SPCONV_ALGO": "native", "OMP_NUM_THREADS": "1", "OMP_WAIT_POLICY": "ACTIVE", "OMP_PROC_BIND": "false", "ORT_DISABLE_THREAD_AFFINITY": "1"})
warnings.filterwarnings("ignore")
multiprocessing.set_start_method("spawn", force=True)
try:
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BATCH_VIS_PROMPT = """<image>
You are filtering frames for single-view image-to-3D (TRELLIS).
Context: EPIC-KITCHENS egocentric video. The image is a CROP of one target object; black/blurred areas are outside the mask—IGNORE them. Hands may appear; IGNORE them unless they block the object.
Return EXACTLY one lowercase word: yes | partial | no
Definitions:
yes = object is mostly visible (≈≥70%), in focus, not severely motion-blurred, clear edges/shape.
partial = object present but heavily occluded (e.g., by hands), cut off, very small, or strong motion blur/glare.
no = object absent/unrecognizable or crop mostly padding/background.
Answer:"""


def load_config():
    p = argparse.ArgumentParser(description="Filter per-object crops/masks by visibility using InternVL (sharded).")
    p.add_argument("--video_root", default="./manip_data")
    p.add_argument("--output_root", default="./manip_data")
    p.add_argument("--csv_file", default="EPIC_100.csv")
    p.add_argument("--ext", default="mp4")
    p.add_argument("--start_video_idx", type=int, default=0)
    p.add_argument("--end_video_idx", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--internvl_model_id", default="OpenGVLab/InternVL3-38B")
    p.add_argument("--internvl_dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--internvl_input_size", type=int, default=448)
    p.add_argument("--internvl_max_tiles", type=int, default=4)
    p.add_argument("--internvl_use_thumbnail", action="store_true", default=True)
    p.add_argument("--internvl_max_new_tokens", type=int, default=2)
    p.add_argument("--internvl_non_blocking", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_idx", type=int, default=0)
    return p.parse_args()


def stable_int_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def _parse_visibility_label(txt: str) -> str:
    t = (txt or "").strip().lower()
    if "partial" in t:
        return "partial"
    if "yes" in t:
        return "yes"
    if "no" in t or t.startswith("n"):
        return "no"
    if t.startswith("y"):
        return "yes"
    if t.startswith("p"):
        return "partial"
    return "no"


def build_transform(input_size):
    return T.Compose(
        [
            T.Lambda(lambda img: img if img.mode == "RGB" else img.convert("RGB")),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def dynamic_preprocess(image, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / max(1e-6, orig_height)
    target_ratios = sorted({(i, j) for n in range(1, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if 1 <= i * j <= max_num}, key=lambda x: x[0] * x[1])

    # Find closest aspect ratio
    best, best_diff = (1, 1), float("inf")
    for r in target_ratios:
        diff = abs(aspect_ratio - (r[0] / r[1]))
        if diff < best_diff:
            best_diff, best = diff, r
    tiles_rc = best

    target_w, target_h = image_size * tiles_rc[0], image_size * tiles_rc[1]
    resized = image.resize((target_w, target_h))
    cols = target_w // image_size
    processed = [
        resized.crop(
            (
                (i % cols) * image_size,
                (i // cols) * image_size,
                ((i % cols) + 1) * image_size,
                ((i // cols) + 1) * image_size,
            )
        )
        for i in range(tiles_rc[0] * tiles_rc[1])
    ]

    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed


class FrameDataset(Dataset):
    def __init__(self, frames_pil, cfg):
        self.frames = frames_pil
        self.cfg = cfg
        self.transform = build_transform(cfg.internvl_input_size)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        tiles = dynamic_preprocess(
            self.frames[idx], image_size=self.cfg.internvl_input_size, use_thumbnail=self.cfg.internvl_use_thumbnail, max_num=self.cfg.internvl_max_tiles
        )
        return torch.stack([self.transform(ti) for ti in tiles])


def collate_by_frames(batch):
    """Collate tiles from multiple frames into a single batch."""
    npl = [t.size(0) for t in batch]
    return torch.cat(batch, dim=0).contiguous(), npl


class GPUPrefetcher:
    def __init__(self, loader, dtype=torch.bfloat16, use_pin=True):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.dtype = dtype
        self.use_pin = use_pin
        self.next = None
        self._preload()

    def _preload(self):
        try:
            pv, npl = next(self.loader)
        except StopIteration:
            self.next = None
            return
        with torch.cuda.stream(self.stream):
            if self.use_pin:
                try:
                    pv = pv.pin_memory()
                except Exception:
                    pass
            pv = pv.to("cuda", dtype=self.dtype, non_blocking=True)
        self.next = (pv, npl)

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next is None:
            raise StopIteration
        batch = self.next
        self._preload()
        return batch


def init_internvl_model(cfg):
    print(f"Loading {cfg.internvl_model_id} for image verification...")

    dtype = torch.bfloat16 if cfg.internvl_dtype == "bf16" else torch.float16
    load_kwargs = dict(low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True, device_map="auto", dtype=dtype)

    if cfg.internvl_model_id == "OpenGVLab/InternVL3-78B":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModel.from_pretrained(cfg.internvl_model_id, **load_kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.internvl_model_id, trust_remote_code=True, use_fast=False)
    print(f"{cfg.internvl_model_id} loaded successfully!")
    return model, tokenizer


def classify_visibility_per_frame_batched(frames_pil, model, tokenizer, cfg):
    if not frames_pil:
        return []

    dtype = torch.bfloat16 if cfg.internvl_dtype == "bf16" else torch.float16
    ds = FrameDataset(frames_pil, cfg)
    dl_kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.internvl_non_blocking,
        persistent_workers=(cfg.num_workers > 0),
        collate_fn=collate_by_frames,
    )
    if cfg.num_workers > 0:
        dl_kwargs["prefetch_factor"] = cfg.prefetch_factor
    dl = DataLoader(ds, **dl_kwargs)

    prefetch = GPUPrefetcher(dl, dtype=dtype, use_pin=cfg.internvl_non_blocking)
    gen_cfg = dict(max_new_tokens=cfg.internvl_max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    results = []

    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=dtype):
        for pixel_values, npl in tqdm(prefetch, total=len(dl), desc="Classifying visibility"):
            responses = model.batch_chat(tokenizer, pixel_values, num_patches_list=npl, questions=[BATCH_VIS_PROMPT] * len(npl), generation_config=gen_cfg)
            results.extend(_parse_visibility_label(str(resp)) for resp in responses)
    return results


def process_single_object(obj_path: Path, cfg, internvl_model, internvl_tokenizer):
    obj_name = obj_path.stem
    crops_path, masks_path = obj_path / "cropped_frames.npz", obj_path / "masks.npz"
    clean_crops_path, clean_masks_path = obj_path / "clean_cropped_frames.npz", obj_path / "clean_masks.npz"

    if not cfg.overwrite and clean_crops_path.exists() and clean_masks_path.exists():
        print(f"Skipping {obj_name} (clean outputs already exist)")
        return

    if not crops_path.exists() or not masks_path.exists():
        print(f"Missing crops/masks for {obj_path}")
        return

    crops_dict = {int(k): v for k, v in np.load(crops_path, allow_pickle=False).items()}
    masks_dict = {int(k): v for k, v in np.load(masks_path, allow_pickle=False).items()}
    print(f"Loaded {len(crops_dict)} images and {len(masks_dict)} masks for {obj_name}")

    items = sorted(crops_dict.items())
    frame_keys = [k for k, _ in items]
    frames_pil = [Image.fromarray(arr) for _, arr in items]
    labels = classify_visibility_per_frame_batched(frames_pil, internvl_model, internvl_tokenizer, cfg)

    clean_crops_dict, clean_masks_dict = {}, {}
    counts = {"yes": 0, "partial": 0, "no": 0}
    for k, lab in zip(frame_keys, labels):
        counts[lab] += 1
        if lab == "yes":
            clean_crops_dict[k] = crops_dict[k]
            if k in masks_dict:
                clean_masks_dict[k] = masks_dict[k]

    print(f"Results: yes={counts['yes']}, partial={counts['partial']}, no={counts['no']}")
    np.savez_compressed(clean_crops_path, **{str(k): v for k, v in clean_crops_dict.items()})
    np.savez_compressed(clean_masks_path, **{str(k): v for k, v in clean_masks_dict.items()})


def process_video_objects(objects_path: Path, cfg, internvl_model, internvl_tokenizer):
    """Process all objects with moved_by_hand=yes in the objects directory."""
    objects_list = [p for p in objects_path.glob("*") if p.is_dir() and (p / "moved_by_hand.txt").exists() and "yes" in (p / "moved_by_hand.txt").read_text().lower()]
    print(f"Objects with moved_by_hand=yes: {len(objects_list)}")

    for obj_path in objects_list:
        try:
            process_single_object(obj_path, cfg, internvl_model, internvl_tokenizer)
        except Exception as e:
            print(f"Error processing {obj_path}: {e}\n{traceback.format_exc()}")


def main():
    cfg = load_config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if not os.path.exists(cfg.csv_file):
        raise FileNotFoundError(f"CSV not found: {cfg.csv_file}")
    df = pd.read_csv(cfg.csv_file)
    for col in ["narration_id", "no_hands_presence", "duration_s"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column '{col}'")
    valid_narrations = set(df[(df["no_hands_presence"] == 0) & (df["duration_s"] < 10)]["narration_id"].astype(str))

    # Load or find video paths
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
    print(f"Found {len(all_videos)} candidate videos")

    # Sharding
    shard_idx = cfg.shard_idx % max(1, cfg.num_shards)
    sharded = [vp for vp in all_videos if stable_int_hash(os.path.basename(os.path.dirname(vp))) % cfg.num_shards == shard_idx]
    print(f"Shard {shard_idx}/{cfg.num_shards}: {len(sharded)} videos")

    # Slicing
    candidates = sharded[cfg.start_video_idx : None if cfg.end_video_idx == -1 else cfg.end_video_idx]
    print(f"After slicing: {len(candidates)} videos")

    internvl_model, internvl_tokenizer = init_internvl_model(cfg)

    for i, vpath in enumerate(candidates, 1):
        seq_name = os.path.basename(os.path.dirname(vpath))
        if seq_name not in valid_narrations:
            continue
        print(f"\n[{i}/{len(candidates)}] Processing {seq_name}")

        objects_path = Path(cfg.output_root) / seq_name / "objects"
        if not objects_path.exists():
            print(f"No objects/ dir for {seq_name}")
            continue
        try:
            process_video_objects(objects_path, cfg, internvl_model, internvl_tokenizer)
        except Exception as e:
            print(f"Error processing {vpath}: {e}\n{traceback.format_exc()}")
    print("Done.")


if __name__ == "__main__":
    main()
