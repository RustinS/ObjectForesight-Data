import argparse
import datetime
import glob
import inspect
import os
import random
import sys
import traceback

import decord
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.models.utils import get_points_on_a_grid
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from rich import print


class Logger(object):
    """
    Static logging class
    """

    log_file = None
    COLORS = {
        "DEBUG": "\x1b[36m",  # cyan
        "INFO": "\x1b[32m",  # green for info
        "SUCCESS": "\x1b[32m",  # green
        "WARNING": "\x1b[33m",  # yellow
        "ERROR": "\x1b[31m",  # red
    }
    RESET = "\x1b[0m"

    @staticmethod
    def init(log_path):
        Logger.log_file = log_path

    @staticmethod
    def log(write_str, to_stdout=True, level="INFO"):
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        # Find first non-internal caller frame
        caller_file = "unknown"
        caller_line = 0
        try:
            for frame_info in inspect.stack()[1:15]:
                base = os.path.basename(frame_info.filename)
                func = frame_info.function
                if func in ("log", "write", "debug", "info", "warning", "error", "success"):
                    continue
                caller_file = base
                caller_line = frame_info.lineno or 0
                break
        except Exception:
            pass
        prefix = f"[{time_str} {caller_file}:{caller_line}] "
        level_upper = str(level).upper()
        message = str(write_str)
        if to_stdout:
            if Logger._should_color():
                color = Logger.COLORS.get(level_upper, "")
                if color:
                    sys.__stdout__.write(color + prefix + Logger.RESET + message + "\n")
                else:
                    sys.__stdout__.write(prefix + message + "\n")
            else:
                sys.__stdout__.write(prefix + message + "\n")
        if not Logger.log_file:
            return
        with open(Logger.log_file, "a") as f:
            f.write(prefix + message + "\n")

    @staticmethod
    def _should_color():
        if os.environ.get("NO_COLOR"):
            return False
        try:
            return sys.__stdout__.isatty()
        except Exception:
            return False

    @staticmethod
    def debug(write_str, to_stdout=True):
        Logger.log(write_str, to_stdout=to_stdout, level="DEBUG")

    @staticmethod
    def info(write_str, to_stdout=True):
        Logger.log(write_str, to_stdout=to_stdout, level="INFO")

    @staticmethod
    def warning(write_str, to_stdout=True):
        Logger.log(write_str, to_stdout=to_stdout, level="WARNING")

    @staticmethod
    def error(write_str, to_stdout=True):
        Logger.log(write_str, to_stdout=to_stdout, level="ERROR")

    @staticmethod
    def success(write_str, to_stdout=True):
        Logger.log(write_str, to_stdout=to_stdout, level="SUCCESS")


def log_cur_stats(stats_dict, iter=None, to_stdout=True):
    loss = stats_dict.pop("total", 0)
    Logger.log(f"LOSS: {loss:.04f}", to_stdout=to_stdout)
    for k, v in stats_dict.items():
        Logger.log(f"{k}: {v:.04f}", to_stdout=to_stdout)
    if to_stdout:
        if iter is not None:
            print("======= iter %d =======" % iter)
        else:
            print("========")


class LoggerWriter:
    def __init__(self, level="INFO"):
        self.level = level
        self._buffer = ""
        self._real = sys.__stdout__

    def write(self, message):
        if not message:
            return 0
        if "\r" in message and "\n" not in message:
            self._real.write(message)
            self._real.flush()
            return len(message)
        self._buffer += str(message)
        lines = self._buffer.split("\n")
        self._buffer = lines.pop()
        for line in lines:
            if line.strip():
                Logger.log(line, to_stdout=False, level=self.level)
                self._real.write(line + "\n")
        self._real.flush()
        return len(message)

    def flush(self):
        if self._buffer:
            Logger.log(self._buffer, to_stdout=False, level=self.level)
            self._real.write(self._buffer + "\n")
            self._real.flush()
            self._buffer = ""

    def isatty(self):
        return getattr(self._real, "isatty", lambda: False)()

    def fileno(self):
        return self._real.fileno()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/gscratch/raivn/rustin/3dmanip/results")
    parser.add_argument("--csv_file", type=str, default="epic_1000_sample.csv")
    parser.add_argument("--ext", type=str, default="MP4")
    parser.add_argument("--track_mode", type=str, default="offline")
    parser.add_argument("--data_type", type=str, default="RGBD")
    parser.add_argument("--data_dir", type=str, default="assets/example0")
    parser.add_argument("--video_name", type=str, default="action")
    parser.add_argument("--grid_size", type=int, default=50)
    parser.add_argument("--vo_points", type=int, default=756)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--start_video_idx", type=int, default=0)
    parser.add_argument("--end_video_idx", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def process_video(args, out_dir, vggt4track_model, model):
    vid_dir = os.path.join(args.data_dir, "action.mp4")
    video_reader = decord.VideoReader(vid_dir, ctx=decord.cpu(0), num_threads=5)
    video_tensor = torch.from_numpy(video_reader.get_batch(range(len(video_reader))).asnumpy()).permute(0, 3, 1, 2)
    video_tensor = video_tensor[::args.fps].float()

    video_tensor = preprocess_image(video_tensor, keep_ratio=True)[None]
    Logger.info(f"Video tensor shape: {video_tensor.shape}")
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = vggt4track_model((video_tensor.cuda() / 255).to(dtype=torch.bfloat16))
            extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
            depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]

    depth_tensor = depth_map.squeeze().cpu().numpy()
    extrs = np.eye(4)[None].repeat(len(depth_tensor), axis=0)
    extrs = extrinsic.squeeze().cpu().numpy()
    intrs = intrinsic.squeeze().cpu().numpy()
    video_tensor = video_tensor.squeeze()
    # NOTE: 20% of the depth is not reliable
    # threshold = depth_conf.squeeze()[0].view(-1).quantile(0.6).item()
    unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5

    data_npz_load = {}

    grid_size = args.grid_size
    frame_H, frame_W = video_tensor.shape[2:]

    Logger.info(f"Frame H W: {frame_H}, {frame_W}")

    grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()

    # Run model inference
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (c2w_traj, intrs, point_map, conf_depth, track3d_pred, track2d_pred, vis_pred, conf_pred, video) = model.forward(
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

        data_npz_load["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:, :3, :3], track3d_pred[:, :, :3].cpu()) + c2w_traj[:, :3, 3][:, None, :]).numpy()
        data_npz_load["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
        data_npz_load["intrinsics"] = intrs.cpu().numpy()
        depth_save = point_map[:, 2, ...]
        depth_save[conf_depth < 0.5] = 0
        data_npz_load["depths"] = depth_save.cpu().numpy()
        data_npz_load["visibs"] = vis_pred.cpu().numpy()
        data_npz_load["unc_metric"] = conf_depth.cpu().numpy()
        np.savez(os.path.join(out_dir, "result.npz"), **data_npz_load)

        Logger.info(f"Results saved to {out_dir}.")

        del video_tensor, depth_tensor, intrs, extrs, video, point_map, conf_depth, track3d_pred, track2d_pred, vis_pred, conf_pred
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    set_seed(args.seed)

    sys.stdout = LoggerWriter("INFO")
    sys.stderr = LoggerWriter("ERROR")
    
    vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    vggt4track_model.eval()
    vggt4track_model = vggt4track_model.to("cuda")

    if args.track_mode == "offline":
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")

    # config the model; the track_num is the number of points in the grid
    model.spatrack.track_num = args.vo_points

    model.eval()
    model.to("cuda")

    # Discover videos to process
    ci_ext = "".join([f"[{c.lower()}{c.upper()}]" for c in args.ext])
    epic_df = pd.read_csv(os.path.join(args.data_root, args.csv_file))
    video_paths = sorted(glob.glob(os.path.join(args.data_root, f"**/*.{ci_ext}"), recursive=True))
    video_paths = [vp for vp in video_paths if os.path.basename(vp).lower() == "action.mp4"]

    if len(video_paths) == 0:
        Logger.error(f"No videos found under {args.data_root} with extension .{args.ext}.")
        exit()

    Logger.info(f"Found {len(video_paths)} videos under {args.data_root} with extension .{args.ext}.")
    Logger.info(f"Start video index: {args.start_video_idx}, End video index: {args.end_video_idx}")

    total = len(video_paths)
    for i, vpath in enumerate(video_paths, 1):
        if i < args.start_video_idx:
            continue
        if i >= args.end_video_idx and args.end_video_idx != -1:
            break
        try:
            # Use parent directory name as sequence id (video filename is always action.mp4)
            seq_name = os.path.basename(os.path.dirname(vpath))
            seq_dir = os.path.dirname(vpath)

            if epic_df[epic_df["narration_id"] == seq_name].iloc[0]["duration_s"] > 10:
                Logger.info(f"Skipping {seq_name} because duration is greater than 10 seconds.")
                continue

            # Per-video output directory: <SEQ_DIR>/dynhamr
            out_dir = os.path.join(seq_dir, "spatrack")
            os.makedirs(out_dir, exist_ok=True)
            if os.path.exists(os.path.join(out_dir, "result.npz")):
                continue
            Logger.init(f"{out_dir}/opt_log.txt")
            Logger.info("\n")
            Logger.info(f"[{i}/{total}] Processing {seq_name}: {vpath}")

            args.data_dir = seq_dir

            process_video(args, out_dir, vggt4track_model, model)
        except Exception as e:
            print(f"Error processing {vpath}: {e}")
            print(traceback.format_exc())
            exit()


if __name__ == "__main__":
    main()
