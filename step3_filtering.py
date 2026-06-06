import argparse
import glob
import hashlib
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from utils import rprint as print

# Presence flag columns used throughout the script
PRESENCE_COLUMNS = [
    "left_hand_presence",
    "right_hand_presence",
    "both_hands_presence",
    "no_hands_presence",
    "no_object_presence",
    "left_object_presence",
    "right_object_presence",
    "twohands_object_presence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute hand/object presence flags from EgoHOS masks.")
    parser.add_argument("--data_root", type=str, default="/gpfs/scrubbed/rustin/manip_data", help="Root directory containing narration_id folders with action.mp4 and egohos/")
    parser.add_argument("--csv_file", type=str, default="EPIC_100.csv", help="CSV manifest to update (should contain narration_id column)")
    parser.add_argument("--threshold_ratio", type=float, default=0.1, help="Ratio of total frames used as smoothing threshold for presence.")
    parser.add_argument("--start_video_idx", type=int, default=0, help="Optional manual slice start index into the list of videos (after sharding).")
    parser.add_argument("--end_video_idx", type=int, default=-1, help="Optional manual slice end index into the list of videos (after sharding). -1 = no limit.")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards for multi-node / Slurm array execution.")
    parser.add_argument("--shard_idx", type=int, default=0, help="This shard's index [0..num_shards-1].")
    parser.add_argument("--save_every", type=int, default=100, help="If > 0, periodically save accumulated updates every N sequences.")
    parser.add_argument("--ext", type=str, default="mp4", help="Video extension to match action clips (case-insensitive).")
    return parser.parse_args()


def stable_int_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def compute_runs(series: np.ndarray) -> List[Tuple[bool, int, int]]:
    if series.size == 0:
        return []
    runs = []
    current, start = bool(series[0]), 0
    for idx in range(1, series.size):
        if bool(series[idx]) != current:
            runs.append((current, start, idx - start))
            current, start = bool(series[idx]), idx
    runs.append((current, start, series.size - start))
    return runs


def fill_short_false_runs(series: np.ndarray, threshold: int) -> np.ndarray:
    if series.size == 0 or threshold <= 0:
        return series
    filled = series.copy()
    for val, start, length in compute_runs(series):
        if not val and length < threshold:
            filled[start : start + length] = True
    return filled


def remove_short_true_runs(series: np.ndarray, threshold: int) -> np.ndarray:
    if series.size == 0 or threshold <= 0:
        return series
    return ~fill_short_false_runs(~series, threshold)


def clean(series: np.ndarray, threshold: int) -> np.ndarray:
    if series.size == 0 or threshold <= 0:
        return series
    return remove_short_true_runs(fill_short_false_runs(series, threshold), threshold)


def frame_presence(mask: np.ndarray, value: int, min_pixels: int = 10) -> np.ndarray:
    if mask.size == 0:
        return np.array([], dtype=bool)
    return np.count_nonzero(mask == value, axis=(1, 2)) > min_pixels


def compute_hand_presence_flags(twohands_mask: np.ndarray, threshold_ratio: float) -> Tuple[int, int, int, int]:
    thr = max(1, int(round(twohands_mask.shape[0] * threshold_ratio)))
    left_c = clean(frame_presence(twohands_mask, value=1), thr)
    right_c = clean(frame_presence(twohands_mask, value=2), thr)
    return int(left_c.all()), int(right_c.all()), int((left_c & right_c).all()), int((~(left_c | right_c)).all())


def compute_object_presence_flag(obj_mask: np.ndarray, threshold_ratio: float) -> Tuple[int, int, int, int]:
    thr = max(1, int(round(obj_mask.shape[0] * threshold_ratio)))
    left_c = clean(frame_presence(obj_mask, value=1), thr)
    right_c = clean(frame_presence(obj_mask, value=2), thr)
    both_c = clean(frame_presence(obj_mask, value=3), thr)
    any_obj = left_c | right_c | both_c
    return int((~any_obj).all()), int(left_c.all()), int(right_c.all()), int(both_c.all())


def list_action_videos(data_root: str, ext: str) -> List[str]:
    video_folders_file = os.path.join(data_root, "video_folders.txt")
    if os.path.exists(video_folders_file):
        print("Loading video folders from file")
        with open(video_folders_file, "r") as f:
            videos = [line.strip() for line in f if line.strip()]
    else:
        print("Finding video folders")
        ci_ext = "".join(f"[{c.lower()}{c.upper()}]" for c in ext)
        videos = [
            vp for vp in sorted(glob.glob(os.path.join(data_root, f"**/*.{ci_ext}"), recursive=True)) if os.path.basename(vp).lower() == "action.mp4"
        ]
        with open(video_folders_file, "w") as f:
            f.write("\n".join(videos))
    return videos


def ensure_presence_columns(df: pd.DataFrame) -> None:
    for col in PRESENCE_COLUMNS:
        if col not in df:
            df[col] = -1


def write_shard_chunk(shard_csv_path: str, chunk_updates: List[dict]) -> None:
    if not chunk_updates:
        return
    header = not os.path.exists(shard_csv_path) or os.path.getsize(shard_csv_path) == 0
    pd.DataFrame(chunk_updates).to_csv(shard_csv_path, index=False, mode="a", header=header)


def load_sequence_masks(seq_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    egohos_dir = os.path.join(seq_dir, "egohos")
    twohands_mask = np.load(os.path.join(egohos_dir, "twohands_masks.npz"))["masks"]
    obj_mask = np.load(os.path.join(egohos_dir, "obj_masks.npz"))["masks"]
    return twohands_mask, obj_mask


def already_processed_row(row: pd.Series) -> bool:
    return all(c in row and int(row[c]) != -1 for c in PRESENCE_COLUMNS)


def main() -> None:
    args = parse_args()

    csv_path = os.path.abspath(args.csv_file)
    df = pd.read_csv(csv_path)
    if "narration_id" not in df.columns:
        print("CSV has no 'narration_id' column; cannot update rows.")
        return
    ensure_presence_columns(df)

    all_video_paths = list_action_videos(args.data_root, args.ext)
    if not all_video_paths:
        print(f"No action.mp4 videos found under {args.data_root}.")
        return
    print(f"Found {len(all_video_paths)} videos under {args.data_root}.")

    num_shards = max(1, args.num_shards)
    shard_idx = args.shard_idx % num_shards

    shards_dir = os.path.join(os.path.dirname(csv_path), "csv_shards")
    os.makedirs(shards_dir, exist_ok=True)
    shard_csv_path = os.path.join(shards_dir, f"shard_{shard_idx}.csv")
    print(f"Shard {shard_idx} writing updates to {shard_csv_path}")

    sharded_video_paths = [vp for vp in all_video_paths if stable_int_hash(os.path.basename(os.path.dirname(vp))) % num_shards == shard_idx]
    print(f"Shard {shard_idx}/{num_shards}: {len(sharded_video_paths)} videos assigned.")

    end_idx = args.end_video_idx if args.end_video_idx != -1 else None
    candidate_paths = sharded_video_paths[args.start_video_idx : end_idx]
    total = len(candidate_paths)
    print(f"After slicing: {total} videos to process.")

    updates: List[dict] = []

    for local_i, video_path in enumerate(candidate_paths, 1):
        seq_dir = os.path.dirname(video_path)
        seq_name = os.path.basename(seq_dir)
        print(f"\n[{local_i}/{total}] {seq_name}: {video_path}")

        selection = df["narration_id"].astype(str) == str(seq_name)
        if not selection.any():
            print(f"No CSV row matched narration_id={seq_name}. Skipping.")
            continue
        if df[selection].apply(already_processed_row, axis=1).all():
            print(f"Skipping {seq_name} (already processed).")
            continue

        twohands_mask, obj_mask = load_sequence_masks(seq_dir)
        left_flag, right_flag, both_flag, no_hands_flag = compute_hand_presence_flags(twohands_mask, args.threshold_ratio)
        no_object_flag, left_object_flag, right_object_flag, twohands_object_flag = compute_object_presence_flag(obj_mask, args.threshold_ratio)

        df.loc[selection, "left_hand_presence"] = left_flag
        df.loc[selection, "right_hand_presence"] = right_flag
        df.loc[selection, "both_hands_presence"] = both_flag
        df.loc[selection, "no_hands_presence"] = no_hands_flag
        df.loc[selection, "left_object_presence"] = left_object_flag
        df.loc[selection, "right_object_presence"] = right_object_flag
        df.loc[selection, "twohands_object_presence"] = twohands_object_flag
        df.loc[selection, "no_object_presence"] = no_object_flag

        updates.append(
            {
                "narration_id": seq_name,
                "left_hand_presence": left_flag,
                "right_hand_presence": right_flag,
                "both_hands_presence": both_flag,
                "no_hands_presence": no_hands_flag,
                "no_object_presence": no_object_flag,
                "left_object_presence": left_object_flag,
                "right_object_presence": right_object_flag,
                "twohands_object_presence": twohands_object_flag,
            }
        )

        if args.save_every > 0 and len(updates) >= args.save_every:
            write_shard_chunk(shard_csv_path, updates)
            print(f"Saved {len(updates)} updates to {shard_csv_path} (periodic)")
            updates.clear()

        print(
            f"Updated {seq_name}: L={left_flag}, R={right_flag}, B={both_flag}, none_hands={no_hands_flag}, "
            f"obj_none={no_object_flag}, obj_L={left_object_flag}, obj_R={right_object_flag}, obj_both={twohands_object_flag}"
        )

    if updates:
        write_shard_chunk(shard_csv_path, updates)
        print(f"Saved {len(updates)} updates to {shard_csv_path}")
    else:
        print("No updates to write for this shard.")


if __name__ == "__main__":
    main()
