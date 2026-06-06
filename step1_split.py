import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("DECORD_FFMPEG_LOG_LEVEL", "error")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
import pandas as pd
from decord import VideoReader, cpu
from tqdm import tqdm

from utils import rprint as print


# ------------------------ Utilities ------------------------


def build_video_mapping(root: str) -> dict[str, Path]:
    return {p.stem: p for p in sorted(Path(root).rglob("*.[mM][pP]4"))}


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ranges.sort(key=lambda x: x[0])
    merged = []
    cur_start, cur_stop = ranges[0]
    for s, e in ranges[1:]:
        if s <= cur_stop:
            cur_stop = max(cur_stop, e)
        else:
            merged.append((cur_start, cur_stop))
            cur_start, cur_stop = s, e
    merged.append((cur_start, cur_stop))
    return merged


def stable_int_hash(s: str) -> int:
    """Deterministic hash for sharding by ID."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def write_meta(out_dir: str, meta: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "action.meta.json.tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f)
    os.replace(tmp, os.path.join(out_dir, "action.meta.json"))


def should_skip(out_dir: str, out_file: str, expected: dict, skip_existing: bool) -> bool:
    """Skip only if existing file's metadata matches expectations."""
    if not skip_existing or not os.path.exists(out_file):
        return False
    mpath = os.path.join(out_dir, "action.meta.json")
    if not os.path.exists(mpath):
        try:
            return os.path.getsize(out_file) > 0
        except Exception:
            return False
    try:
        with open(mpath, "r") as f:
            m = json.load(f)
        if abs(float(m.get("fps", -1.0)) - float(expected["fps"])) > 1e-2:
            return False
        return all(m.get(k) == expected[k] for k in ["start_frame", "stop_frame", "pad", "width", "height"])
    except Exception:
        return False


def open_cv2_capture(video_path: str) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(video_path)
    return cap if cap.isOpened() else None


def read_frame_cv2(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok and bgr is not None else None
    except Exception:
        return None


def remux_with_ffmpeg(src: str) -> str | None:
    """Try to remux bitstream errors away without re-encoding."""
    try:
        tmp_out = src + ".remux.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-err_detect", "ignore_err", "-fflags", "+genpts", "-i", src, "-c", "copy", "-an", tmp_out],
            check=True,
        )
        return tmp_out if os.path.exists(tmp_out) and os.path.getsize(tmp_out) > 0 else None
    except Exception:
        return None


def get_frame_safe(vr: VideoReader, cv_cap: cv2.VideoCapture | None, idx: int) -> np.ndarray | None:
    try:
        return vr[idx].asnumpy()
    except Exception:
        return read_frame_cv2(cv_cap, idx) if cv_cap else None


class SegmentWriter:
    MAX_CONSEC_FAIL = 24

    def __init__(self, out_root: str, expected_meta_by_nid: dict, fps: float, width: int, height: int):
        self.out_root = out_root
        self.expected_meta_by_nid = expected_meta_by_nid
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.fps = fps
        self.width = width
        self.height = height
        self.open_writers: dict[str, cv2.VideoWriter] = {}
        self.written_segments: dict[str, bool] = {}
        self.nid_tmpfile: dict[str, str] = {}
        self.nid_finalfile: dict[str, str] = {}
        self.nid_finaldir: dict[str, str] = {}
        self.last_good_rgb: np.ndarray | None = None
        self.consec_fail = 0

    def start(self, frame_idx: int, starts_at: dict[int, list[str]]):
        for nid in starts_at.get(frame_idx, []):
            if nid in self.open_writers:
                continue
            out_dir = os.path.join(self.out_root, nid)
            os.makedirs(out_dir, exist_ok=True)
            tmp_file = os.path.join(out_dir, "action.tmp.mp4")
            self.open_writers[nid] = cv2.VideoWriter(tmp_file, self.fourcc, self.fps, (self.width, self.height))
            self.written_segments[nid] = False
            self.nid_tmpfile[nid] = tmp_file
            self.nid_finalfile[nid] = os.path.join(out_dir, "action.mp4")
            self.nid_finaldir[nid] = out_dir

    def _promote(self, nid: str):
        writer = self.open_writers.pop(nid, None)
        if writer:
            writer.release()
        tmpf, finalf, finald = self.nid_tmpfile.get(nid), self.nid_finalfile.get(nid), self.nid_finaldir.get(nid)
        if self.written_segments.get(nid) and tmpf and finalf:
            os.makedirs(os.path.dirname(finalf), exist_ok=True)
            try:
                os.replace(tmpf, finalf)
            except Exception:
                if tmpf != finalf and os.path.exists(tmpf):
                    shutil.copy2(tmpf, finalf)
                    os.remove(tmpf)
            if finald and nid in self.expected_meta_by_nid:
                write_meta(finald, self.expected_meta_by_nid[nid])
        elif tmpf and os.path.exists(tmpf):
            try:
                os.remove(tmpf)
            except Exception:
                pass

    def finish_segments(self, frame_idx: int, stops_at_minus_one: dict[int, list[str]], nid_to_stop_m1: dict[str, int]):
        for nid in stops_at_minus_one.get(frame_idx, []):
            self._promote(nid)
        for nid, stop_idx in list(nid_to_stop_m1.items()):
            if frame_idx > stop_idx:
                self._promote(nid)

    def handle_missing_frame(self):
        self.consec_fail += 1
        if self.open_writers and self.last_good_rgb is not None:
            bgr_dup = cv2.cvtColor(self.last_good_rgb, cv2.COLOR_RGB2BGR)
            for writer in self.open_writers.values():
                writer.write(bgr_dup)
            for nid in self.open_writers:
                self.written_segments[nid] = True
        if self.consec_fail >= self.MAX_CONSEC_FAIL:
            for nid in list(self.open_writers.keys()):
                self._promote(nid)
            self.consec_fail = 0

    def handle_frame(self, frame_idx: int, rgb: np.ndarray | None, starts_at: dict[int, list[str]], stops_at_minus_one: dict[int, list[str]], nid_to_stop_m1: dict[str, int]):
        if rgb is None or getattr(rgb, "size", 0) == 0:
            self.handle_missing_frame()
            return

        self.consec_fail = 0
        self.last_good_rgb = rgb

        self.start(frame_idx, starts_at)

        if self.open_writers:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for nid, writer in self.open_writers.items():
                writer.write(bgr)
                self.written_segments[nid] = True

        self.finish_segments(frame_idx, stops_at_minus_one, nid_to_stop_m1)

    def close_all(self):
        for nid in list(self.open_writers.keys()):
            self._promote(nid)


# ------------------------ Worker ------------------------


def process_video(
    video_id: str,
    video_path: str,
    video_rows: pd.DataFrame,
    out_root: str,
    pad: int,
    chunk_size: int,
    decord_threads: int,
    skip_existing: bool,
    mem_per_worker_mb: int,
) -> tuple[str, int, int]:
    try:
        skipped_existing = 0
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=max(1, decord_threads))
    except Exception:
        remuxed = remux_with_ffmpeg(video_path)
        if not remuxed:
            raise RuntimeError(f"Video {video_id} failed:\n{traceback.format_exc()}")
        vr = VideoReader(remuxed, ctx=cpu(0), num_threads=max(1, decord_threads))

    cv_cap = open_cv2_capture(video_path)
    max_frame = len(vr) - 1

    fps = float(vr.get_avg_fps())
    if not (0 < fps < 240):
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        cap.release()

    probe = vr[0].asnumpy() if len(vr) else vr[max(0, max_frame // 2)].asnumpy()
    height, width = probe.shape[:2]

    segments = []
    expected_meta_by_nid = {}
    for _, row in video_rows.iterrows():
        narration_id = str(row["narration_id"])
        start_secs = pd.to_timedelta(row["start_timestamp"]).total_seconds()
        stop_secs = pd.to_timedelta(row["stop_timestamp"]).total_seconds()
        ts_start = max(0, min(int(round(start_secs * fps)), max_frame))
        ts_stop = max(0, min(int(round(stop_secs * fps)), max_frame))
        start_frame = max(0, ts_start - pad)
        stop_frame = min(max_frame, ts_stop + pad)
        if stop_frame <= start_frame:
            print(f"SKIP reason=invalid_range_after_pad | start={start_frame} stop={stop_frame} | max_frame={max_frame} | nid={narration_id}")
            continue

        out_dir = os.path.join(out_root, narration_id)
        out_file = os.path.join(out_dir, "action.mp4")
        expected_meta = {"start_frame": start_frame, "stop_frame": stop_frame, "pad": pad, "fps": fps, "width": width, "height": height}
        if should_skip(out_dir, out_file, expected_meta, skip_existing):
            skipped_existing += 1
            continue
        segments.append((start_frame, stop_frame, narration_id))
        expected_meta_by_nid[narration_id] = expected_meta

    if not segments:
        if cv_cap:
            cv_cap.release()
        return video_id, 0, skipped_existing

    starts_at, stops_at_minus_one, nid_to_stop_m1 = defaultdict(list), defaultdict(list), {}
    for s, e, nid in segments:
        starts_at[s].append(nid)
        stops_at_minus_one[e - 1].append(nid)
        nid_to_stop_m1[nid] = e - 1

    decode_ranges = merge_ranges([(s, e) for s, e, _ in segments])
    bytes_per_frame = height * width * 3
    budget_bytes = max(64 * 1024 * 1024, mem_per_worker_mb * 1024 * 1024)
    local_chunk = max(1, min(chunk_size, budget_bytes // (bytes_per_frame * 3)))

    writer = SegmentWriter(out_root, expected_meta_by_nid, fps, width, height)

    try:
        for a, b in decode_ranges:
            chunk_start = a
            while chunk_start < b:
                chunk_stop = min(b, chunk_start + local_chunk)
                indices = list(range(chunk_start, chunk_stop))
                batch = None
                for _ in range(3):
                    try:
                        batch = vr.get_batch(indices).asnumpy()
                        break
                    except Exception:
                        if local_chunk > 1:
                            local_chunk = max(1, local_chunk // 2)
                            chunk_stop = min(b, chunk_start + local_chunk)
                            indices = list(range(chunk_start, chunk_stop))
                        else:
                            break

                if batch is None:
                    for frame_idx in indices:
                        writer.handle_frame(frame_idx, get_frame_safe(vr, cv_cap, frame_idx), starts_at, stops_at_minus_one, nid_to_stop_m1)
                else:
                    for i, rgb in enumerate(batch):
                        writer.handle_frame(indices[i], rgb, starts_at, stops_at_minus_one, nid_to_stop_m1)
                    del batch

                chunk_start = chunk_stop
    finally:
        writer.close_all()
        if cv_cap:
            cv_cap.release()

    return video_id, sum(writer.written_segments.values()), skipped_existing


# ------------------------ Main ------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split EPIC-Kitchens videos into action segments.")
    parser.add_argument("--epic_csv", default="EPIC_100.csv", help="Path to EPIC annotations CSV")
    parser.add_argument("--video_root", default="/gpfs/datasets/epic_kitchens", help="Root directory containing source videos")
    parser.add_argument("--out_root", default="/gpfs/scrubbed/rustin/manip_data", help="Output directory for segments")
    parser.add_argument("--pad_frames", type=int, default=5, help="Pad each segment by this many frames on both sides")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes")
    parser.add_argument("--chunk", type=int, default=256, help="Upper bound on frames decoded per batch")
    parser.add_argument("--decord_threads", type=int, default=2, help="Decord internal threads per video reader")
    parser.add_argument("--no_skip_existing", action="store_true", help="Do not skip existing segments")
    parser.add_argument("--mem_per_worker_mb", type=int, default=768, help="Memory budget per worker (MB)")
    parser.add_argument("--cv2_threads", type=int, default=0, help="OpenCV thread count (0 disables)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total shards for distributed processing")
    parser.add_argument("--shard_idx", type=int, default=0, help="Shard index [0..num_shards-1]")
    args = parser.parse_args()

    try:
        cv2.setNumThreads(args.cv2_threads)
    except Exception:
        pass

    print("\nSplitting videos into action segments.")

    # Ensure duration_s exists in CSV
    actions_df_full = pd.read_csv(args.epic_csv)
    if "duration_s" not in actions_df_full.columns:
        start_td = pd.to_timedelta(actions_df_full["start_timestamp"], errors="coerce")
        stop_td = pd.to_timedelta(actions_df_full["stop_timestamp"], errors="coerce")
        actions_df_full["duration_s"] = (stop_td - start_td).dt.total_seconds().fillna(0).clip(lower=0)
        actions_df_full.to_csv(args.epic_csv, index=False)
        print(f"Added duration_s to {args.epic_csv}")

    # Filter to required columns
    cols = ["video_id", "start_frame", "stop_frame", "narration_id", "start_timestamp", "stop_timestamp"]
    actions_df = actions_df_full[cols].copy()
    actions_df[["start_frame", "stop_frame"]] = actions_df[["start_frame", "stop_frame"]].astype(int)

    vidid_to_path = build_video_mapping(args.video_root)
    available_ids = set(vidid_to_path.keys())

    # Log missing videos
    missing = actions_df[~actions_df["video_id"].isin(available_ids)]
    for _, r in missing.iterrows():
        print(f"SKIP reason=missing_video | video={r['video_id']} | narration_id={r['narration_id']}")

    actions_df = actions_df[actions_df["video_id"].isin(available_ids)]
    if actions_df.empty:
        print("No matching videos found.")
        raise SystemExit(0)

    # Shard by video_id
    groups = [(vid, df) for vid, df in actions_df.groupby("video_id") if stable_int_hash(str(vid)) % args.num_shards == args.shard_idx % args.num_shards]
    print(f"Shard {args.shard_idx}/{args.num_shards}: {len(groups)} videos selected.")

    os.makedirs(args.out_root, exist_ok=True)
    tasks = [(vid, str(vidid_to_path[vid]), df) for vid, df in groups]
    skip_existing = not args.no_skip_existing

    print(f"Processing {len(tasks)} videos | workers={args.workers}, chunk={args.chunk}, pad={args.pad_frames}\n")

    written_total, skipped_total = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        futures = {ex.submit(process_video, vid, path, df, args.out_root, args.pad_frames, args.chunk, args.decord_threads, skip_existing, args.mem_per_worker_mb): vid for vid, path, df in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                _, w, s = fut.result()
                written_total += w
                skipped_total += s
            except Exception as e:
                print(f"Error processing {futures[fut]}: {e}")

    print(f"\nDone. Segments written: {written_total} | skipped: {skipped_total}")
