from __future__ import annotations

import datetime
import inspect
import os
import sys
import time
from pathlib import Path
import cv2
import imageio
import numpy as np

from decord import VideoReader, cpu
from PIL import Image
from rich.console import Console
from tqdm.rich import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))  # project root (contains 'src')
console = Console()

def rprint(*args, stack_level: int = 1, print_location: bool = False, no_extra: bool = False, **kwargs) -> None:
    """Rich print with 'path:line  YYYY-MM-DD HH:MM:SS  message' ordering.

    Adds padding after 'path:line' so messages align.
    """
    # Resolve caller frame
    frame = inspect.currentframe()
    for _ in range(max(int(stack_level), 1)):
        if frame is None:
            break
        frame = frame.f_back
    path_disp = "?"
    lineno = 0
    if frame is not None and print_location:
        info = inspect.getframeinfo(frame)
        path_disp = os.path.relpath(info.filename, _ROOT)
        lineno = info.lineno
    prefix = f"{path_disp}:{lineno}"
    # Pad prefix to fixed width for alignment
    PREFIX_WIDTH = 25
    if len(prefix) < PREFIX_WIDTH:
        prefix = prefix.ljust(PREFIX_WIDTH)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if no_extra:
        prefix = ""
    elif print_location:
        prefix = f"[dim]{prefix}[/dim] [[bold green]{ts}[/bold green]] "
    else:
        prefix = f"[[bold green]{ts}[/bold green]] "

    if len(args) > 0 and all(isinstance(a, (str, bytes)) for a in args):
        msg = " ".join(a.decode() if isinstance(a, (bytes,)) else a for a in args)

        console.print(f"{prefix}{msg}", **kwargs)
    else:
        console.print(f"{prefix}")
        for obj in args:
            console.print(obj, **kwargs)

text_colors = {
    "logs": "\033[34m",  # 033 is the escape code and 34 is the color code
    "info": "\033[32m",
    "warning": "\033[33m",
    "error": "\033[31m",
    "bold": "\033[1m",
    "end_color": "\033[0m",
}


def get_curr_time_stamp():
    return time.strftime("|%Y-%m-%d|%H:%M:%S|")


def error_print(message, empty_line=False):
    time_stamp = get_curr_time_stamp()
    error_str = "[" + text_colors["error"] + text_colors["bold"] + "ERROR" + text_colors["end_color"] + "]"
    if not empty_line:
        print("{} - {} - {}".format(time_stamp, error_str, message))
        print("{} - {} - {}".format(time_stamp, error_str, "Exiting!!!"))
    else:
        print("\n{} - {} - {}".format(time_stamp, error_str, message))
        print("{} - {} - {}".format(time_stamp, error_str, "Exiting!!!"))
    sys.exit(-1)


def log_print(message, empty_line=False, end=None):
    time_stamp = get_curr_time_stamp()
    log_str = "[" + text_colors["logs"] + text_colors["bold"] + "LOGS" + text_colors["end_color"] + "]"
    if not empty_line:
        if end:
            print("{} - {} - {}".format(time_stamp, log_str, message), end=end)
        else:
            print("{} - {} - {}".format(time_stamp, log_str, message))
    else:
        if end:
            print("\n{} - {} - {}".format(time_stamp, log_str, message), end=end)
        else:
            print("\n{} - {} - {}".format(time_stamp, log_str, message))


def warning_print(message, empty_line=False):
    time_stamp = get_curr_time_stamp()
    warn_str = "[" + text_colors["warning"] + text_colors["bold"] + "WARNING" + text_colors["end_color"] + "]"
    if not empty_line:
        print("{} - {} - {}".format(time_stamp, warn_str, message))
    else:
        print("\n{} - {} - {}".format(time_stamp, warn_str, message))


def info_print(message, empty_line=False):
    time_stamp = get_curr_time_stamp()
    info_str = "[" + text_colors["info"] + text_colors["bold"] + "INFO" + text_colors["end_color"] + "]"
    if not empty_line:
        print("{} - {} - {}".format(time_stamp, info_str, message))
    else:
        print("\n{} - {} - {}".format(time_stamp, info_str, message))


def load_local_video(video_path, frames_dir) -> list[Path]:
    if type(video_path) is str:
        video_path = Path(video_path)

    if type(frames_dir) is str:
        frames_dir = Path(frames_dir)

    frames_dir.mkdir(parents=True, exist_ok=True)

    if not any(frames_dir.iterdir()):
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=6)
        num_frames = len(vr)
        print(f"Video has {num_frames} frames @ {vr.get_avg_fps():.2f} fps")

        for idx in tqdm(range(num_frames), desc="Saving frames"):
            frame = vr[idx].asnumpy()
            img = Image.fromarray(frame)
            img.save(str(frames_dir / f"{idx:06d}.jpg"))

        del vr

    # collect all saved frame paths, sorted by numeric stem
    frame_paths = sorted(frames_dir.iterdir(), key=lambda p: int(p.stem))

    return frame_paths


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
        caller_file = "unknown"
        caller_line = 0
        try:
            for frame_info in inspect.stack()[1:15]:
                base = os.path.basename(frame_info.filename)
                func = frame_info.function
                if func in ("log","write","debug","info","warning","error","success"):
                    continue
                caller_file = base
                caller_line = frame_info.lineno or 0
                break
        except Exception:
            pass
        prefix = f"[{time_str} {caller_file}:{caller_line}] "
        level_upper = str(level).upper()
        message = str(write_str)
        out_line = prefix + message + "\n"

        if to_stdout:
            # Robust write to job stdout (Slurm-captured), with flush and fallback
            try:
                stream = getattr(sys, "__stdout__", None) or sys.stdout
                if Logger._should_color():
                    color = Logger.COLORS.get(level_upper, "")
                    if color:
                        stream.write(color + prefix + Logger.RESET + message + "\n")
                    else:
                        stream.write(out_line)
                else:
                    stream.write(out_line)
                stream.flush()
            except Exception:
                # Last-resort write directly to FD 1
                try:
                    os.write(1, out_line.encode("utf-8", "ignore"))
                except Exception:
                    pass

        if not Logger.log_file:
            return
        try:
            with open(Logger.log_file, "a") as f:
                f.write(out_line)
        except Exception:
            pass

    @staticmethod
    def _should_color():
        if os.environ.get("NO_COLOR"):
            return False
        try:
            stream = getattr(sys, "__stdout__", None) or sys.stdout
            return hasattr(stream, "isatty") and stream.isatty()
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
        # Write to the same robust stream Slurm captures
        self._real = getattr(sys, "__stdout__", None) or sys.stdout

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
                # Keep file-only write for opt_log
                Logger.log(line, to_stdout=False, level=self.level)
                # ...and also mirror to Slurm stdout
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

def create_video_from_images(frames_dict_or_folder, output_dir, frame_rate=30, downsample_factor=2):
    output_video_path = os.path.join(output_dir, "masked_video.mp4")
    
    # Check if input is a dictionary or a folder path
    if isinstance(frames_dict_or_folder, dict):
        # Handle dictionary input
        print(f"Creating video from {len(frames_dict_or_folder)} frames dictionary...")
        
        if not frames_dict_or_folder:
            raise ValueError("No frames found in the dictionary.")
        
        # Sort frames by frame_idx
        sorted_frame_indices = sorted(frames_dict_or_folder.keys())
        first_frame = frames_dict_or_folder[sorted_frame_indices[0]]
        height, width = first_frame.shape[:2]
        
        with imageio.get_writer(output_video_path, fps=frame_rate) as writer:
            for frame_idx in tqdm(sorted_frame_indices, desc="Writing video frames"):
                frame = frames_dict_or_folder[frame_idx]
                # Convert BGR to RGB if needed (OpenCV uses BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                resized_frame = cv2.resize(frame, (width // downsample_factor, height // downsample_factor))
                writer.append_data(resized_frame.astype(np.uint8))
        
        print(f"Video saved at {output_video_path}")
        
    else:
        # Handle folder path input (original functionality)
        image_folder = frames_dict_or_folder
        print(f"Creating video from images in {image_folder} ...")
        valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]

        image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[1] in valid_extensions]
        image_files.sort()
        if not image_files:
            raise ValueError("No valid image files found in the specified folder.")

        first_image_path = os.path.join(image_folder, image_files[0])
        first_image = cv2.imread(first_image_path)
        height, width, _ = first_image.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width // downsample_factor, height // downsample_factor))

        for image_file in tqdm(image_files):
            image_path = os.path.join(image_folder, image_file)
            image = cv2.resize(cv2.imread(image_path), (width // downsample_factor, height // downsample_factor))
            video_writer.write(image)

        video_writer.release()
        print(f"Video saved at {output_video_path}")