#!/usr/bin/env python3
"""SAM3D mesh generation for fig_samples_sup2 samples.

This script processes samples from the future pose prediction outputs,
using SAM 3D Objects to generate 3D meshes from single anchor frame images.
"""

import os
import sys
import argparse
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

# Add SAM3D notebook directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sam-3d-objects", "notebook"))

from inference import Inference


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 3D meshes using SAM3D for fig_samples_sup2 samples.")
    parser.add_argument(
        "--samples_dir",
        type=str,
        default="future_pose_pred/outputs/fig_samples_sup2",
        help="Directory containing sample subdirectories",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="future_pose_pred/outputs/fig_samples_sup2/selected_samples.csv",
        help="CSV file with sample metadata",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="sam-3d-objects/checkpoints/hf/pipeline.yaml",
        help="SAM3D pipeline config file",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing mesh files")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index in CSV (for resuming)")
    parser.add_argument("--end_idx", type=int, default=-1, help="End index in CSV (-1 for all)")
    return parser.parse_args()


def extract_frame_idx_from_anchor_png(anchor_png_path: str) -> int:
    """Extract frame index from anchor_png filename like 'anchor_101.png' -> 101."""
    basename = os.path.basename(anchor_png_path)
    match = re.search(r"anchor_(\d+)\.png", basename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract frame index from: {basename}")


def load_mask_for_frame(object_dir: str, frame_idx: int) -> np.ndarray:
    """Load binary mask for a specific frame from masks.npz."""
    masks_path = os.path.join(object_dir, "masks.npz")
    if not os.path.exists(masks_path):
        raise FileNotFoundError(f"Masks file not found: {masks_path}")

    masks_data = np.load(masks_path, allow_pickle=False)
    frame_key = str(frame_idx)

    if frame_key not in masks_data:
        available_keys = list(masks_data.keys())[:10]
        raise KeyError(f"Frame {frame_idx} not found in masks.npz. Available keys (first 10): {available_keys}")

    mask = masks_data[frame_key]
    # Ensure binary mask
    mask = (mask > 0.5).astype(np.uint8)
    return mask


def load_image(image_path: str) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = Image.open(image_path)
    img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def process_sample(row: pd.Series, samples_dir: str, inference_model, seed: int, overwrite: bool) -> bool:
    """Process a single sample and generate mesh.

    Returns True if successful, False otherwise.
    """
    slug = row["slug"]
    anchor_png_rel = row["anchor_png"]
    object_dir = row["object_dir"]

    # Determine output path
    sample_dir = os.path.join(samples_dir, slug)
    output_mesh_path = os.path.join(sample_dir, "mesh", "sam3d_model.glb")

    # Check if already exists
    if os.path.exists(output_mesh_path) and not overwrite:
        print(f"  [SKIP] Mesh already exists: {output_mesh_path}")
        return True

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_mesh_path), exist_ok=True)

    # Get anchor image path (could be relative or absolute in CSV)
    if os.path.isabs(anchor_png_rel):
        anchor_png_path = anchor_png_rel
    else:
        # Relative paths in CSV are relative to repo root
        anchor_png_path = anchor_png_rel

    if not os.path.exists(anchor_png_path):
        raise FileNotFoundError(f"Anchor image not found: {anchor_png_path}")

    # Extract frame index and load mask
    frame_idx = extract_frame_idx_from_anchor_png(anchor_png_path)
    mask = load_mask_for_frame(object_dir, frame_idx)

    # Load image
    image = load_image(anchor_png_path)

    # Handle size mismatch between image and mask
    if mask.shape[:2] != image.shape[:2]:
        # Resize mask to match image
        from PIL import Image as PILImage

        mask_pil = PILImage.fromarray(mask)
        mask_pil = mask_pil.resize((image.shape[1], image.shape[0]), PILImage.NEAREST)
        mask = np.array(mask_pil)

    print(f"  Image shape: {image.shape}, Mask shape: {mask.shape}")

    # Run SAM3D inference
    output = inference_model(image, mask, seed=seed)

    # Export GLB mesh
    if "glb" in output and output["glb"] is not None:
        output["glb"].export(output_mesh_path)
        print(f"  [OK] Saved mesh: {output_mesh_path}")
        return True
    else:
        print(f"  [WARN] No GLB output from SAM3D")
        return False


def main():
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading CSV: {args.csv_file}")
    if not os.path.exists(args.csv_file):
        raise FileNotFoundError(f"CSV file not found: {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    print(f"Loaded {len(df)} samples")

    # Slice if needed
    end_idx = args.end_idx if args.end_idx != -1 else len(df)
    df = df.iloc[args.start_idx : end_idx]
    print(f"Processing samples [{args.start_idx}:{end_idx}] ({len(df)} samples)")

    print(f"\nLoading SAM3D model from: {args.config}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    inference_model = Inference(args.config, compile=False)
    print("SAM3D model loaded!\n")

    # Process each sample
    success_count = 0
    fail_count = 0
    skip_count = 0

    for idx, (_, row) in enumerate(df.iterrows()):
        slug = row["slug"]
        print(f"[{idx + 1}/{len(df)}] Processing: {slug}")

        try:
            result = process_sample(row, args.samples_dir, inference_model, args.seed, args.overwrite)
            if result:
                success_count += 1
            else:
                fail_count += 1
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            fail_count += 1
        except KeyError as e:
            print(f"  [ERROR] {e}")
            fail_count += 1
        except Exception as e:
            print(f"  [ERROR] Unexpected error: {e}")
            traceback.print_exc()
            fail_count += 1

    print(f"\n{'=' * 50}")
    print(f"Summary:")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Total:   {len(df)}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
