#!/usr/bin/env python3
"""SAM2 propagation of EgoHOS-initialized objects over action clips (sharded)."""

import argparse
import glob
import hashlib
import os
import pathlib
import sys
import traceback
from pathlib import Path

sys.path.append(str(pathlib.Path(__file__).resolve().parent / "sam2"))

import cv2
import numpy as np
import pandas as pd
import supervision as sv
import torch
from decord import VideoReader, cpu
from tqdm.rich import tqdm

from sam2.build_sam import build_sam2_video_predictor
from utils import create_video_from_images
from utils import rprint as print

torch.set_float32_matmul_precision("high")

SAM2_CHECKPOINT = "sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 video propagation from EgoHOS masks (sharded).")

    # Core paths
    parser.add_argument("--video_root", default="./manip_data", help="Root directory containing narration_id/*/action.mp4 and egohos/")
    parser.add_argument("--output_root", default="./manip_data", help="Root directory to write per-sequence outputs (usually same as video_root)")
    parser.add_argument("--csv_file", type=str, default="EPIC_100.csv", help="CSV manifest (same one updated in step3_filtering)")
    parser.add_argument("--ext", type=str, default="mp4", help="Extension to match action videos (default mp4)")
    parser.add_argument("--start_video_idx", type=int, default=0, help="Manual slice start (AFTER sharding).")
    parser.add_argument("--end_video_idx", type=int, default=-1, help="Manual slice end (AFTER sharding). -1 = no limit.")
    parser.add_argument("--seed", type=int, default=42)

    # Mask-based init (object and hand stacks saved by step2_egohos)
    parser.add_argument("--obj_mask_relpath", type=str, default="egohos/obj_masks.npz", help="Relative path under <seq_dir> to object mask stack.")
    parser.add_argument("--hand_mask_relpath", type=str, default="egohos/twohands_masks.npz", help="Relative path under <seq_dir> to hand mask stack (0,1,2).")

    parser.add_argument("--init_mode", choices=["points", "box"], default="points", help="Initialize SAM2 with sampled points or tight bbox.")
    parser.add_argument("--pos_points", type=int, default=16, help="Num positive points from object core.")
    parser.add_argument("--neg_points", type=int, default=8, help="Num negative points from hands.")
    parser.add_argument("--neg_points_obj", type=int, default=8, help="Num negative points from OTHER objects.")
    parser.add_argument("--min_area", type=int, default=1000, help="Minimum area (px) for a valid connected component / track seed.")
    parser.add_argument("--start_frame_policy", choices=["first", "largest"], default="first", help="Choose first-valid frame or largest-area frame for init.")
    parser.add_argument("--pos_temporal_window", type=int, default=20, help="±window around seed frame for consensus positives.")
    parser.add_argument("--pos_consensus_frac", type=float, default=0.8, help="Consensus fraction threshold for positives.")
    parser.add_argument("--pos_erode_px", type=int, default=2, help="Fallback erode(px) if consensus is empty.")
    parser.add_argument("--neg_exclude_dilate_px", type=int, default=3, help="Exclude band around object mask for negative sampling.")
    parser.add_argument("--neg_points_bg", type=int, default=6, help="How many negatives from background ring near object.")
    parser.add_argument("--bg_ring_inner_px", type=int, default=45, help="Inner radius(px) of near-background ring.")
    parser.add_argument("--bg_ring_outer_px", type=int, default=65, help="Outer radius(px) of near-background ring.")
    parser.add_argument("--bg_far_points", type=int, default=4, help="How many negatives from far background.")
    parser.add_argument("--bg_far_dilate_px", type=int, default=85, help="Radius(px) to define far background zone.")

    # Simple tracking / linking
    parser.add_argument("--iou_thresh", type=float, default=0.3, help="IoU threshold to continue a track across frames.")
    parser.add_argument("--max_skip", type=int, default=3, help="Max consecutive unmatched frames before track ends.")
    parser.add_argument("--track_min_len", type=int, default=2, help="Tracks shorter than this are dropped.")
    parser.add_argument("--track_temporal_window", type=int, default=20, help="±window for per-track temporal consensus prefilter.")
    parser.add_argument("--track_consensus_frac", type=float, default=0.2, help="Consensus fraction for track prefilter mask.")
    parser.add_argument("--track_open_px", type=int, default=1, help="Morphological open(px) before components.")
    parser.add_argument("--track_close_px", type=int, default=0, help="Morphological close(px) before components.")

    # Track dedup within a single video
    parser.add_argument("--dup_iou_thresh", type=float, default=0.3, help="IoU threshold to consider two propagated tracks duplicates.")
    parser.add_argument("--dup_time_window", type=int, default=5, help="Max |Δt| when comparing for duplication.")
    parser.add_argument("--dedup_sort", choices=["length", "area"], default="area", help="Sort tracks before dedup by total area or length.")

    # Bidirectional propagation
    parser.add_argument("--bidir", action="store_true", help="If set, also propagate backward from init frame.")
    parser.set_defaults(bidir=True)  # default ON

    # Debug viz
    parser.add_argument("--debug_viz", action="store_true", help="If set, save visualizations / annotated videos / init point plots.")

    # Sharding
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards for job array.")
    parser.add_argument("--shard_idx", type=int, default=0, help="This shard index [0..num_shards-1].")

    return parser.parse_args()


def stable_int_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def _to_np(frame):
    if hasattr(frame, "asnumpy"):
        return frame.asnumpy()
    if torch.is_tensor(frame):
        return frame.detach().cpu().numpy()
    return np.asarray(frame)


def _morph_op(mask_bool: np.ndarray, px: int, op: int) -> np.ndarray:
    """Unified morphological operation. op: cv2.MORPH_ERODE, MORPH_DILATE, MORPH_OPEN, MORPH_CLOSE."""
    if px <= 0:
        return mask_bool.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    if op in (cv2.MORPH_ERODE, cv2.MORPH_DILATE):
        fn = cv2.erode if op == cv2.MORPH_ERODE else cv2.dilate
        return fn(mask_bool.astype(np.uint8), k, iterations=1).astype(bool)
    return cv2.morphologyEx(mask_bool.astype(np.uint8), op, k, iterations=1).astype(bool)


def _erode(mask_bool: np.ndarray, px: int) -> np.ndarray:
    return _morph_op(mask_bool, px, cv2.MORPH_ERODE)


def _dilate(mask_bool: np.ndarray, px: int) -> np.ndarray:
    return _morph_op(mask_bool, px, cv2.MORPH_DILATE)


def _mask_iou(a, b) -> float:
    inter = np.logical_and(a, b).sum()
    return float(inter) / float(np.logical_or(a, b).sum()) if inter > 0 else 0.0


def _bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.int32)


def _sample_points(mask_bool, k, rng):
    ys, xs = np.where(mask_bool > 0)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    idx = rng.choice(ys.size, size=min(k, ys.size), replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.int32)  # (x,y)


def _temporal_consensus_fast(bin_stack: np.ndarray, window: int, frac: float) -> np.ndarray:
    """Temporal consensus filter: (T,H,W) uint8 -> (T,H,W) uint8."""
    T = bin_stack.shape[0]
    if T == 0:
        return bin_stack
    w = max(0, window)
    csum = np.concatenate([np.zeros((1, *bin_stack.shape[1:]), dtype=np.int32), np.cumsum(bin_stack.astype(np.int32), axis=0)], axis=0)
    out = np.zeros_like(bin_stack, dtype=np.uint8)
    for t in range(T):
        a, b = max(0, t - w), min(T - 1, t + w)
        votes = csum[b + 1] - csum[a]
        out[t] = (votes >= max(1, int(np.ceil(frac * (b - a + 1))))).astype(np.uint8)
    return out


def _pre_filter_stack_for_tracks(obj_stack: np.ndarray, window: int, frac: float, open_px: int, close_px: int) -> np.ndarray:
    """Apply temporal consensus + morphology to object stack. Returns (T,H,W) uint8."""
    bin_stack = (obj_stack > 0).astype(np.uint8)
    core = _temporal_consensus_fast(bin_stack, window, frac)
    filtered = np.zeros_like(core, dtype=np.uint8)
    for t in range(core.shape[0]):
        m = core[t].astype(bool)
        if open_px > 0:
            m = _morph_op(m, open_px, cv2.MORPH_OPEN)
        if close_px > 0:
            m = _morph_op(m, close_px, cv2.MORPH_CLOSE)
        filtered[t] = m.astype(np.uint8)
        if filtered[t].sum() == 0 and bin_stack[t].sum() > 0:
            filtered[t] = _morph_op(bin_stack[t].astype(bool), max(1, open_px), cv2.MORPH_OPEN).astype(np.uint8)
    return filtered


def _components_from_mask(frame_mask, min_area=50):
    num, labels = cv2.connectedComponents((frame_mask > 0).astype(np.uint8), connectivity=8)
    comps = []
    for lab in range(1, num):
        comp = labels == lab
        area = int(comp.sum())
        if area >= min_area:
            ys, xs = np.where(comp)
            comps.append({"mask": comp, "area": area, "bbox": np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.int32)})
    return comps


def _build_tracks_from_stack(
    obj_stack, min_area=100, iou_thresh=0.3, max_skip=3, min_len=2, track_temporal_window=20, track_consensus_frac=0.2, track_open_px=1, track_close_px=0
):
    """
    Link per-frame components into tracks using greedy IoU matching.
    Returns a list of dicts {id, frames[], masks[], areas[]}.
    """
    filtered_stack = _pre_filter_stack_for_tracks(
        obj_stack=obj_stack,
        window=track_temporal_window,
        frac=track_consensus_frac,
        open_px=track_open_px,
        close_px=track_close_px,
    )

    T = filtered_stack.shape[0]
    per_frame = [_components_from_mask(filtered_stack[t], min_area=min_area) for t in range(T)]

    # prune extremely huge comps per-frame using local median
    MED_WIN = 5
    for t, comps in enumerate(per_frame):
        if not comps:
            continue
        lo = max(0, t - MED_WIN)
        hi = min(T - 1, t + MED_WIN)
        neigh = []
        for u in range(lo, hi + 1):
            if per_frame[u]:
                neigh.extend([c["area"] for c in per_frame[u]])
        if not neigh:
            continue
        med_area = max(1, int(np.median(neigh)))
        cap = 6 * med_area
        per_frame[t] = [c for c in comps if c["area"] <= cap]

    tracks = []
    active = []
    next_id = 0

    for t in range(T):
        comps = per_frame[t]
        used = set()

        # try to extend active tracks
        for tr in active:
            best_j, best_iou = -1, 0.0
            for j, c in enumerate(comps):
                if j in used:
                    continue
                iou = _mask_iou(tr["last_mask"], c["mask"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= iou_thresh:
                c = comps[best_j]
                used.add(best_j)
                tr["last_frame"] = t
                tr["last_mask"] = c["mask"]
                tr["skip"] = 0
                tr["frames"].append(t)
                tr["masks"].append(c["mask"])
                tr["areas"].append(c["area"])
            else:
                tr["skip"] += 1

        # start new track for each unused comp
        for j, c in enumerate(comps):
            if j in used:
                continue
            active.append(
                {
                    "id": next_id,
                    "last_frame": t,
                    "last_mask": c["mask"],
                    "skip": 0,
                    "frames": [t],
                    "masks": [c["mask"]],
                    "areas": [c["area"]],
                }
            )
            next_id += 1

        # retire tracks that skipped too long
        still_active = []
        for tr in active:
            if tr["skip"] > max_skip:
                if len(tr["frames"]) >= min_len:
                    tracks.append(
                        {
                            "id": tr["id"],
                            "frames": tr["frames"],
                            "masks": tr["masks"],
                            "areas": tr["areas"],
                        }
                    )
            else:
                still_active.append(tr)
        active = still_active

    # flush
    for tr in active:
        if len(tr["frames"]) >= min_len:
            tracks.append(
                {
                    "id": tr["id"],
                    "frames": tr["frames"],
                    "masks": tr["masks"],
                    "areas": tr["areas"],
                }
            )

    return tracks


def _pick_track_start(tr, policy="first"):
    idx = int(np.argmax(tr["areas"])) if policy == "largest" else 0
    return tr["frames"][idx], tr["masks"][idx]


def _track_stats(tr):
    return int(np.sum(tr["areas"])), len(tr["frames"])


def _temporal_consensus_from_track(tr, center_t, window, min_frac):
    sel = [m for t, m in zip(tr["frames"], tr["masks"]) if abs(t - center_t) <= window]
    if not sel:
        return None
    stack = np.stack([m.astype(np.uint8) for m in sel], axis=0)
    thr = max(1, int(np.ceil(min_frac * len(sel))))
    core = stack.sum(axis=0) >= thr
    return core if core.any() else None


def _infer_track_label(frame_obj_labels, comp_mask):
    vals = frame_obj_labels[comp_mask]
    vals = vals[vals > 0]
    if vals.size == 0:
        return None
    counts = np.bincount(vals.astype(np.int32))
    if counts.size < 4:
        counts = np.pad(counts, (0, 4 - counts.size))
    return int(np.argmax(counts[1:]) + 1)


def _restrict_to_label(mask_bool, frame_obj_labels, label):
    return mask_bool.astype(bool) if label is None else (mask_bool.astype(bool) & (frame_obj_labels == label))


def _sample_other_object_negatives(frame_obj_labels, comp_mask, track_label, k, rng, exclude_mask=None):
    if track_label is None or k <= 0:
        return np.zeros((0, 2), dtype=np.int32)
    cand = (frame_obj_labels > 0) & (frame_obj_labels != track_label) & ~comp_mask.astype(bool)
    if exclude_mask is not None:
        cand &= ~exclude_mask.astype(bool)
    return _sample_points(cand, k, rng)


def _sample_background_ring_negatives(core_mask, frame_obj_labels, hand_mask, k, inner_px, outer_px, rng):
    if k <= 0:
        return np.zeros((0, 2), dtype=np.int32)
    inner = _dilate(core_mask, max(0, inner_px))
    outer = _dilate(core_mask, max(outer_px, inner_px + 1))
    cand = (outer & ~inner) & (frame_obj_labels == 0) & (hand_mask == 0)
    if cand.sum() < k:
        outer2 = _dilate(core_mask, int(1.5 * max(outer_px, inner_px + 1)))
        cand |= (outer2 & ~inner) & (frame_obj_labels == 0) & (hand_mask == 0)
    return _sample_points(cand, k, rng)


def _sample_far_background_negatives(core_mask, frame_obj_labels, hand_mask, k, far_px, rng):
    if k <= 0:
        return np.zeros((0, 2), dtype=np.int32)
    cand = ~_dilate(core_mask, max(1, far_px)) & (frame_obj_labels == 0) & (hand_mask == 0)
    return _sample_points(cand, k, rng)


def _build_pos_core_from(tr, start_idx, start_mask, cfg):
    core = _temporal_consensus_from_track(tr, center_t=start_idx, window=cfg.pos_temporal_window, min_frac=cfg.pos_consensus_frac)
    if core is None or core.sum() < max(1, cfg.pos_points):
        core = _erode(start_mask, cfg.pos_erode_px)
        if core.sum() == 0:
            core = start_mask.astype(bool)
    return core


def _points_labels_for(core, start_idx, obj_stack, hand_stack, cfg, rng):
    frame_labels = obj_stack[start_idx]  # {0,1,2,3}
    hand_mask = hand_stack[start_idx]  # {0,1,2}
    track_label = _infer_track_label(frame_labels, core)

    # label-consistent positive region
    pos_region = _restrict_to_label(core, frame_labels, track_label)
    if pos_region.sum() < max(1, cfg.pos_points):
        eroded_core = _erode(core, cfg.pos_erode_px)
        pos_region = _restrict_to_label(eroded_core, frame_labels, track_label)
    if pos_region.sum() < max(1, cfg.pos_points):
        pos_region = _restrict_to_label(core, frame_labels, track_label)
    if (track_label is not None) and (pos_region.sum() < max(1, cfg.pos_points)):
        pos_region = frame_labels == int(track_label)

    pos = _sample_points(pos_region, cfg.pos_points, rng)

    exclude_band = _dilate(core, cfg.neg_exclude_dilate_px)
    neg_hands = _sample_points((hand_mask > 0) & (~exclude_band), cfg.neg_points, rng)
    neg_other_objs = _sample_other_object_negatives(
        frame_obj_labels=frame_labels,
        comp_mask=core,
        track_label=track_label,
        k=cfg.neg_points_obj,
        rng=rng,
        exclude_mask=exclude_band,
    )
    neg_bg_ring = _sample_background_ring_negatives(
        core_mask=pos_region,
        frame_obj_labels=frame_labels,
        hand_mask=hand_mask,
        k=cfg.neg_points_bg,
        inner_px=cfg.bg_ring_inner_px,
        outer_px=cfg.bg_ring_outer_px,
        rng=rng,
    )
    neg_bg_far = _sample_far_background_negatives(
        core_mask=pos_region,
        frame_obj_labels=frame_labels,
        hand_mask=hand_mask,
        k=cfg.bg_far_points,
        far_px=cfg.bg_far_dilate_px,
        rng=rng,
    )

    neg_parts = [neg_hands, neg_other_objs, neg_bg_ring, neg_bg_far]
    neg = np.vstack([p for p in neg_parts if p.size]) if any(p.size for p in neg_parts) else np.zeros((0, 2), np.int32)

    if pos.size + neg.size == 0:
        return None, None

    pts = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate([np.ones(len(pos), np.int32), np.zeros(len(neg), np.int32)], axis=0)
    return pts, labels


def _new_state(video_predictor, video_path: str):
    st = video_predictor.init_state(
        video_path=str(video_path),
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    )
    video_predictor.reset_state(st)
    return st


def _collect_propagation(predictor, inference_state, backward=False):
    video_segments = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        direction = {"reverse": backward}
        try:
            it = predictor.propagate_in_video(inference_state, **direction)
        except TypeError:
            if backward:
                return {}
            it = predictor.propagate_in_video(inference_state)
        for fr_idx, seg_obj_ids, seg_mask_logits in it:
            video_segments[fr_idx] = {seg_obj_id: (seg_mask_logits[i] > 0.0).detach().cpu().numpy() for i, seg_obj_id in enumerate(seg_obj_ids)}
    return video_segments


def _merge_segments(seg_a, seg_b):
    out = dict(seg_a)
    for k, v in seg_b.items():
        out.setdefault(k, {}).update(v)
    return out


def _segments_to_trackdict(video_segments, obj_id):
    frames, masks, areas = [], [], []
    for t in sorted(video_segments):
        m = video_segments[t].get(obj_id) or (next(iter(video_segments[t].values())) if len(video_segments[t]) == 1 else None)
        if m is not None:
            mb = m.astype(bool)
            frames.append(t)
            masks.append(mb)
            areas.append(int(mb.sum()))
    return {"frames": frames, "masks": masks, "areas": areas}


def _max_pairwise_iou_between_segments(seg_a, seg_b, time_window=5):
    if not seg_a or not seg_b:
        return 0.0
    seg_b_keys = set(seg_b.keys())
    best = 0.0
    for ta in seg_a:
        ma = seg_a[ta].astype(bool)
        for tb in range(ta - time_window, ta + time_window + 1):
            if tb in seg_b_keys:
                iou = _mask_iou(ma, seg_b[tb].astype(bool))
                best = max(best, iou)
    return best


def _save_init_points_image(img_rgb, pos_pts, neg_pts, out_path):
    im = img_rgb.copy()
    for x, y in pos_pts:
        cv2.circle(im, (int(x), int(y)), 3, (0, 255, 0), -1)
    for x, y in neg_pts:
        cv2.circle(im, (int(x), int(y)), 3, (0, 0, 255), -1)
    cv2.imwrite(str(out_path), im)


def _already_processed(seq_output_dir: Path) -> bool:
    objects_dir = seq_output_dir / "objects"
    if not objects_dir.exists():
        return False
    return any(child.is_dir() and (child / "masks.npz").exists() and (child / "cropped_frames.npz").exists() for child in objects_dir.iterdir())


class VideoProcessor:
    def __init__(self, cfg):
        self.cfg = cfg

    def _load_models(self):
        return build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT)

    def _save_object_data(self, masks_dict, cropped_frames, object_output_dir: Path):
        if masks_dict:
            np.savez_compressed(object_output_dir / "masks.npz", **{str(k): v for k, v in masks_dict.items()})
        if cropped_frames:
            np.savez_compressed(object_output_dir / "cropped_frames.npz", **{str(k): v for k, v in cropped_frames.items()})

    def _track_object(self, video_reader, segments, obj_id: int, label: str, out_dir: Path):
        object_output_dir = out_dir / "objects" / f"{obj_id}+{label.replace(' ', '_')}"
        object_output_dir.mkdir(parents=True, exist_ok=True)

        debug = bool(getattr(self.cfg, "debug_viz", False))
        annotated_frames = {} if debug else None
        masks_dict = {}
        cropped_frames = {}

        for frame_idx, segdict in tqdm(segments.items(), desc="Processing frames"):
            img_bgr = _to_np(video_reader[frame_idx])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            mask_list = [segdict[k] for k in segdict]
            masks = np.concatenate(mask_list, axis=0)

            if debug:
                det = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks),
                    mask=masks,
                    class_id=np.fromiter(segdict.keys(), dtype=np.int32),
                )
                ann = sv.MaskAnnotator().annotate(
                    sv.LabelAnnotator().annotate(
                        sv.BoxAnnotator().annotate(img_rgb.copy(), det),
                        det,
                        labels=[label],
                    ),
                    det,
                )
                annotated_frames[frame_idx] = ann

            # Save primary mask & crop
            mask0 = masks[0].astype(np.uint8)
            masks_dict[frame_idx] = mask0

            mimg = cv2.bitwise_and(img_rgb, img_rgb, mask=mask0 * 255)
            cnts, _ = cv2.findContours(mask0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                x, y, w, h = cv2.boundingRect(cnts[0])
                cropped_frames[frame_idx] = mimg[y : y + h, x : x + w]

        self._save_object_data(masks_dict, cropped_frames, object_output_dir)

        if debug and annotated_frames:
            create_video_from_images(annotated_frames, object_output_dir)

    def _init_and_propagate(self, video_path: str, frame_idx: int, track_id: int, *, box=None, points=None, labels=None, backward=False):
        video_predictor = self._load_models()
        st = _new_state(video_predictor, video_path)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            req = {
                "inference_state": st,
                "frame_idx": int(frame_idx),
                "obj_id": int(track_id),
            }
            if box is not None:
                req["box"] = box.astype(np.float32)
            if points is not None and labels is not None:
                req["points"] = points.astype(np.float32)
                req["labels"] = labels.astype(np.int32)
            video_predictor.add_new_points_or_box(**req)

        return _collect_propagation(video_predictor, st, backward=bool(backward))

    def process_single_sequence(self, seq_name: str, video_path: str, seq_out_dir: Path):
        """
        Do full pipeline for one narration_id.
        """
        seq_out_dir.mkdir(parents=True, exist_ok=True)

        # Fast skip if already processed
        if _already_processed(seq_out_dir):
            print(f"Skipping {seq_name} (already has objects/* with masks.npz & cropped_frames.npz).")
            return

        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=5)
        total_frames = len(vr)
        print(f"[{seq_name}] Video frames: {total_frames}")

        obj_path = seq_out_dir / self.cfg.obj_mask_relpath
        hand_path = seq_out_dir / self.cfg.hand_mask_relpath

        obj_stack = self._load_mask_stack(obj_path)
        hand_stack = self._load_mask_stack(hand_path)

        # sync length
        T = min(total_frames, obj_stack.shape[0], hand_stack.shape[0])
        obj_stack = obj_stack[:T]
        hand_stack = hand_stack[:T]

        base_tracks = _build_tracks_from_stack(
            obj_stack=obj_stack,
            min_area=self.cfg.min_area,
            iou_thresh=self.cfg.iou_thresh,
            max_skip=self.cfg.max_skip,
            min_len=self.cfg.track_min_len,
            track_temporal_window=self.cfg.track_temporal_window,
            track_consensus_frac=self.cfg.track_consensus_frac,
            track_open_px=self.cfg.track_open_px,
            track_close_px=self.cfg.track_close_px,
        )

        if not base_tracks:
            print(f"[{seq_name}] No valid tracks found.")
            del vr
            return

        print(f"[{seq_name}] Found {len(base_tracks)} raw tracks.")

        # sort before dedup (area or length)
        sort_key = 1 if self.cfg.dedup_sort == "length" else 0
        base_tracks = sorted(base_tracks, key=lambda tr: _track_stats(tr)[sort_key], reverse=True)

        kept_sam_tracks = []

        for tr in base_tracks:
            start_frame_idx, start_mask = _pick_track_start(tr, self.cfg.start_frame_policy)
            rng = np.random.default_rng(self.cfg.seed + int(tr["id"]))

            # build init prompt
            forward_segments = {}
            backward_segments = {}

            if self.cfg.init_mode == "box":
                box = _bbox_from_mask(start_mask)
                if box is None:
                    print(f"[{seq_name}] track {tr['id']}: no bbox, skip.")
                    continue

                forward_segments = self._init_and_propagate(
                    str(video_path),
                    start_frame_idx,
                    int(tr["id"]),
                    box=box,
                    backward=False,
                )

                if getattr(self.cfg, "bidir", False) and start_frame_idx > 0:
                    backward_segments = self._init_and_propagate(
                        str(video_path),
                        start_frame_idx,
                        int(tr["id"]),
                        box=box,
                        backward=True,
                    )

            else:
                core = _build_pos_core_from(tr, start_frame_idx, start_mask, self.cfg)
                pts, labs = _points_labels_for(core, start_frame_idx, obj_stack, hand_stack, self.cfg, rng)
                if pts is None:
                    print(f"[{seq_name}] track {tr['id']}: no init points, skip.")
                    continue

                # debug plot for init points
                if getattr(self.cfg, "debug_viz", False):
                    img_rgb = cv2.cvtColor(_to_np(vr[start_frame_idx]), cv2.COLOR_BGR2RGB)
                    pos_pts = pts[labs.astype(bool)]
                    neg_pts = pts[~labs.astype(bool)]
                    obj_label = f"object_{tr['id']}"
                    obj_dir = seq_out_dir / "objects" / f"{int(tr['id'])}+{obj_label}"
                    obj_dir.mkdir(parents=True, exist_ok=True)
                    out_png = obj_dir / f"init_points_track{tr['id']}_t{start_frame_idx}.png"
                    _save_init_points_image(img_rgb, pos_pts, neg_pts, out_png)

                forward_segments = self._init_and_propagate(
                    str(video_path),
                    start_frame_idx,
                    int(tr["id"]),
                    points=pts,
                    labels=labs,
                    backward=False,
                )

                if getattr(self.cfg, "bidir", False) and start_frame_idx > 0:
                    core_b = _build_pos_core_from(tr, start_frame_idx, start_mask, self.cfg)
                    pts_b, labs_b = _points_labels_for(core_b, start_frame_idx, obj_stack, hand_stack, self.cfg, rng)
                    if pts_b is not None:
                        if getattr(self.cfg, "debug_viz", False):
                            img_rgb = cv2.cvtColor(_to_np(vr[start_frame_idx]), cv2.COLOR_BGR2RGB)
                            pos_pts_b = pts_b[labs_b.astype(bool)]
                            neg_pts_b = pts_b[~labs_b.astype(bool)]
                            obj_label = f"object_{tr['id']}"
                            obj_dir = seq_out_dir / "objects" / f"{int(tr['id'])}+{obj_label}"
                            obj_dir.mkdir(parents=True, exist_ok=True)
                            out_png_b = obj_dir / f"init_points_track{tr['id']}_t{start_frame_idx}_back.png"
                            _save_init_points_image(img_rgb, pos_pts_b, neg_pts_b, out_png_b)

                        backward_segments = self._init_and_propagate(
                            str(video_path),
                            start_frame_idx,
                            int(tr["id"]),
                            points=pts_b,
                            labels=labs_b,
                            backward=True,
                        )

            # merge fwd/bwd SAM propagation
            merged_segments = _merge_segments(backward_segments, forward_segments)
            cand_track = _segments_to_trackdict(merged_segments, obj_id=int(tr["id"]))

            # dedup against already kept SAM tracks
            is_dup = False
            for kept in kept_sam_tracks:
                max_iou = _max_pairwise_iou_between_segments(
                    {f: m for f, m in zip(cand_track["frames"], cand_track["masks"])},
                    kept["segments"],
                    time_window=self.cfg.dup_time_window,
                )
                if max_iou >= self.cfg.dup_iou_thresh:
                    is_dup = True
                    print(f"[{seq_name}] track {tr['id']} dup of {kept['obj_id']} (IoU {max_iou:.2f}).")
                    break
            if is_dup:
                continue

            kept_sam_tracks.append(
                {
                    "obj_id": int(tr["id"]),
                    "segments": {f: m for f, m in zip(cand_track["frames"], cand_track["masks"])},
                }
            )

            obj_label = f"object_{tr['id']}"
            self._track_object(
                video_reader=vr,
                segments=merged_segments,
                obj_id=int(tr["id"]),
                label=obj_label,
                out_dir=seq_out_dir,
            )

        del vr
        torch.cuda.empty_cache()

    def _load_mask_stack(self, p: Path) -> np.ndarray:
        if not p.exists():
            raise FileNotFoundError(f"Mask stack not found: {p}")

        if p.suffix.lower() == ".npz":
            data = np.load(p, allow_pickle=False)

            numeric_keys = [k for k in data.files if str(k).isdigit()]
            if len(numeric_keys) == len(data.files) and len(numeric_keys) > 0:
                keys = sorted(numeric_keys, key=lambda k: int(k))
                stack = np.stack([data[k] for k in keys], axis=0)

            elif len(data.files) == 1:
                stack = data[data.files[0]]

            else:
                raise ValueError(f"Unexpected npz structure at {p}: keys={data.files}")

        else:
            stack = np.load(p, allow_pickle=False)

        # sanity check
        if stack.ndim != 3:
            raise ValueError(f"Expected (T,H,W) mask stack, got {stack.shape} at {p}")

        return stack.astype(np.uint8)


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # load CSV and apply filter
    df = pd.read_csv(cfg.csv_file)
    required_cols = {"no_hands_presence", "duration_s", "narration_id"}
    if not required_cols.issubset(df.columns):
        print(f"CSV missing required columns: {required_cols - set(df.columns)}")
        return

    filtered_df = df[(df["no_hands_presence"] == 0) & (df["duration_s"] < 10)]

    video_folders_file = os.path.join(cfg.video_root, "video_folders.txt")
    if os.path.exists(video_folders_file):
        print("Loading video folders from file")
        all_videos = [line.strip() for line in open(video_folders_file)]
    else:
        print("Finding video folders")
        ci_ext = "".join(f"[{c.lower()}{c.upper()}]" for c in cfg.ext)
        all_videos = [vp for vp in sorted(glob.glob(os.path.join(cfg.video_root, f"**/*.{ci_ext}"), recursive=True)) if os.path.basename(vp).lower() == "action.mp4"]
        with open(video_folders_file, "w") as f:
            f.write("\n".join(all_videos))

    if len(all_videos) == 0:
        print(f"No videos found under {cfg.video_root} with extension .{cfg.ext}.")
        return

    print(f"Found {len(all_videos)} candidate videos total.")

    # shard by narration_id (parent dir of action.mp4)
    num_shards = max(1, cfg.num_shards)
    shard_idx = cfg.shard_idx % num_shards

    sharded_paths = [vp for vp in all_videos if stable_int_hash(os.path.basename(os.path.dirname(vp))) % num_shards == shard_idx]

    print(f"Shard {shard_idx}/{num_shards}: {len(sharded_paths)} videos in this shard.")

    # optional post-shard slicing
    start_idx = max(0, cfg.start_video_idx)
    candidate_paths = sharded_paths[start_idx : cfg.end_video_idx] if cfg.end_video_idx != -1 else sharded_paths[start_idx:]
    print(f"After slicing: {len(candidate_paths)} videos remain.")

    processor = VideoProcessor(cfg)
    narration_ids = set(filtered_df["narration_id"].astype(str))

    for local_i, vpath in enumerate(candidate_paths, 1):
        seq_name = os.path.basename(os.path.dirname(vpath))

        if seq_name not in narration_ids:
            print(f"Skipping {seq_name} (not in filtered_df).")
            continue

        print(f"\n[{local_i}/{len(candidate_paths)}] {seq_name}: {vpath}")

        try:
            processor.process_single_sequence(seq_name, vpath, Path(cfg.output_root) / seq_name)
        except Exception as e:
            print(f"Error processing {seq_name}: {e}\n{traceback.format_exc()}")

    print("Done.")


if __name__ == "__main__":
    main()
