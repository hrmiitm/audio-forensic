"""Highlight in video: track a clicked or boxed object and build SAM-Audio's visual prompt.

The masked video follows Meta's example (`processor.mask_videos`): frames * (mask == 0), i.e. the
**highlighted object is blacked out** and the rest of the scene stays visible.

Tracking uses SAM 3 when this Hugging Face account has access to it (gated), otherwise SAM 2.1
(open). Both run through `transformers` with the same API.

Memory: frames are decoded by FFmpeg at TRACK_FPS / TRACK_WIDTH and streamed through the tracker one
at a time (forward from the highlighted frame, then backward), so GPU memory doesn't grow with video
length. Frames and masks live in disk-backed memmaps, not RAM.
"""

import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
from functools import cache
from pathlib import Path

import numpy as np

from . import backends
from .backends import check
from .config import ROOT

SEGMENTERS = [("facebook/sam3", "SAM 3"), ("facebook/sam2.1-hiera-small", "SAM 2.1")]
TRACK_FPS, TRACK_WIDTH = 10, 640
PRUNE_AFTER = 24  # keep tracker outputs for the last N frames only (memory stays flat)


def _token() -> str | None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    return os.environ.get("HF_ACCESS_TOKEN")


@cache
def segmenter() -> tuple[str, str]:
    """(repo, label) of the best tracker this account can download."""
    from huggingface_hub import HfApi

    for repo, label in SEGMENTERS:
        try:
            HfApi().auth_check(repo, token=_token())
            return repo, label
        except Exception:
            continue
    return SEGMENTERS[-1]


class Segmenter:
    note = None

    def __init__(self, repo: str, device: str):
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor, Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

        sam3 = "sam3" in repo
        model_cls, proc_cls = (Sam3TrackerVideoModel, Sam3TrackerVideoProcessor) if sam3 else (Sam2VideoModel, Sam2VideoProcessor)
        self.torch, self.device = torch, torch.device(device)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = model_cls.from_pretrained(repo, token=_token(), dtype=self.dtype).to(self.device).eval()
        self.processor = proc_cls.from_pretrained(repo, token=_token())

    def session(self):
        return self.processor.init_video_session(inference_device=self.device, dtype=self.dtype)

    def step(self, session, frame: np.ndarray, idx: int, prompt: dict | None = None) -> np.ndarray:
        """Feed one RGB frame (H, W, 3); returns the object's mask (H, W) bool."""
        torch = self.torch
        h, w = frame.shape[:2]
        inputs = self.processor(images=frame, device=self.device, return_tensors="pt")
        if prompt is not None:
            kw = {}
            if prompt.get("points"):
                kw["input_points"] = [[[[x * w, y * h] for x, y, _ in prompt["points"]]]]
                kw["input_labels"] = [[[int(lab) for _, _, lab in prompt["points"]]]]
            if prompt.get("box"):
                x0, y0, x1, y1 = prompt["box"]
                kw["input_boxes"] = [[[x0 * w, y0 * h, x1 * w, y1 * h]]]
            self.processor.add_inputs_to_inference_session(
                inference_session=session, frame_idx=idx, obj_ids=1, original_size=inputs.original_sizes[0], **kw)
        with torch.inference_mode():
            out = self.model(inference_session=session, frame=inputs.pixel_values[0].to(self.dtype))
        mask = self.processor.post_process_masks([out.pred_masks], original_sizes=inputs.original_sizes, binarize=True)[0]
        _prune(session, idx)
        return mask[0, 0].cpu().numpy().astype(bool)

    def release(self):
        del self.model
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()


def _prune(session, idx: int) -> None:
    """Keep memory flat while streaming: drop old processed frames (1024² each, on the GPU) and
    per-frame tracker outputs older than PRUNE_AFTER frames (the tracker only looks back ~7)."""
    # The session numbers new frames by len(processed_frames): keep the keys, free the pixels.
    frames = session.processed_frames
    for f in range(1, idx):  # frame 0 is the highlighted frame
        if frames.get(f) is not None and frames[f].numel():
            frames[f] = frames[f].new_empty(0)
    for store in (getattr(session, "output_dict_per_obj", None) or {}).values():
        frames = store.get("non_cond_frame_outputs", {})
        for f in [f for f in frames if abs(f - idx) > PRUNE_AFTER]:
            del frames[f]


def get_segmenter() -> Segmenter:
    import torch

    repo, _ = segmenter()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return backends.acquire(("segmenter", repo, device), lambda: Segmenter(repo, device))


# ───────── Frames via FFmpeg ─────────

def _ff_input(src: str, headers: dict | None, start: float) -> list[str]:
    head = ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())] if headers else []
    return [*head, "-ss", f"{max(0.0, start):.3f}", "-i", str(src)]


SCALE = f"scale='min({TRACK_WIDTH},iw)':-2"


def grab_frame(src: str, headers: dict | None, t: float) -> np.ndarray:
    """One RGB frame at time t (seconds in the file), scaled like the tracking frames."""
    from PIL import Image

    p = subprocess.run(["ffmpeg", "-v", "error", *_ff_input(src, headers, t), "-frames:v", "1", "-vf", SCALE,
                        "-f", "image2pipe", "-vcodec", "png", "pipe:1"], capture_output=True)
    if p.returncode or not p.stdout:
        raise RuntimeError("Couldn't read a video frame at that time")
    return np.asarray(Image.open(io.BytesIO(p.stdout)).convert("RGB"))


def preview(src: str, headers: dict | None, t: float, prompt: dict) -> bytes:
    """PNG (RGBA) of the object's mask on the frame at t, for the UI to overlay."""
    from PIL import Image

    frame = grab_frame(src, headers, t)
    seg = get_segmenter()
    mask = seg.step(seg.session(), frame, 0, prompt)
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[mask] = (233, 162, 59, 150)  # accent, semi-transparent
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return buf.getvalue()


def track(src: str, headers: dict | None, start: float, duration: float, prompt: dict, out: Path,
          progress=None, cancel=None) -> Path:
    """Track the object across [start, start+duration) and write the masked video (object blacked out)."""
    first = grab_frame(src, headers, start)
    h, w = first.shape[:2]
    n = max(1, math.ceil(duration * TRACK_FPS))
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        frames = np.lib.format.open_memmap(Path(tmp) / "frames.npy", "w+", np.uint8, (n, h, w, 3))
        masks = np.lib.format.open_memmap(Path(tmp) / "masks.npy", "w+", np.bool_, (n, h, w))
        dec = subprocess.Popen(["ffmpeg", "-v", "error", *_ff_input(src, headers, start), "-t", f"{duration:.3f}",
                                "-vf", f"fps={TRACK_FPS},{SCALE}", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                               stdout=subprocess.PIPE)
        got = 0
        while got < n and (buf := dec.stdout.read(h * w * 3)) and len(buf) == h * w * 3:
            frames[got] = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            got += 1
        dec.stdout.close()
        dec.wait()
        if got == 0:
            raise RuntimeError("Couldn't decode the video for tracking")
        frames[got:] = frames[got - 1]  # pad if the stream ended a little early

        seg = get_segmenter()
        k = min(n - 1, max(0, round(prompt.get("t", 0) * TRACK_FPS)))
        order = [list(range(k, n)), list(range(k, -1, -1))]  # forward from the highlight, then backward
        done, total = 0, n + k
        for direction in order:
            session = seg.session()
            for step_i, i in enumerate(direction):
                check(cancel)
                m = seg.step(session, frames[i], step_i, prompt if step_i == 0 else None)
                masks[i] = m
                done += 1
                if progress and done % 10 == 0:
                    progress(done / total)
            del session

        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                                "-r", str(TRACK_FPS), "-i", "pipe:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                                "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
        for i in range(n):
            enc.stdin.write((frames[i] * ~masks[i][..., None]).tobytes())
        enc.stdin.close()
        if enc.wait():
            raise RuntimeError("Couldn't write the masked video")
        del frames, masks
    return out


def cache_path(folder: Path, prompt: dict) -> Path:
    """Masked videos are reused when the same highlight is isolated again."""
    key = hashlib.sha1(json.dumps([segmenter()[0], prompt], sort_keys=True).encode()).hexdigest()[:12]
    return folder / f"highlight_{key}.mp4"
