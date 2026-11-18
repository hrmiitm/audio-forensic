"""Exports: the mix (WAV / MP4) and individual tracks (WAV / ZIP), rendered block by block."""

import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import SR
from . import db
from .ingest import retry, video_source

RAMP = int(0.005 * SR)  # 5 ms fades at clip-edit boundaries (the player uses the same)
BLOCK = 30 * SR


class NotReady(Exception):
    pass


def _envelope(track: dict, a: int, n: int) -> np.ndarray:
    """Per-sample gain from the track's clip edits (track gain/mute handled by the caller)."""
    lo, hi = a - RAMP, a + n + RAMP
    env = np.ones(hi - lo, np.float32)
    for e in track["edits"]:
        i0, i1 = max(int(e["a"] * SR), lo) - lo, min(int(e["b"] * SR), hi) - lo
        if i1 > i0:
            env[i0:i1] *= 0.0 if e.get("muted") else e.get("gain", 100) / 100
    c = np.concatenate([[0.0], np.cumsum(env, dtype=np.float64)])  # centred moving average -> 5 ms ramps
    return ((c[2 * RAMP :] - c[: -2 * RAMP]) / (2 * RAMP))[:n].astype(np.float32)


def _audible(tracks: list[dict]) -> list[dict]:
    solo = any(t["solo"] for t in tracks)
    return [t for t in tracks if not t["muted"] and (t["solo"] or not solo) and t["gain"] > 0]


def _check_ready(tracks: list[dict]) -> None:
    busy = [t["name"] for t in tracks if t["status"] == "partial"]
    if busy:
        raise NotReady(f"Still processing: {', '.join(busy)}. Export when it finishes.")


def _out(source_id: int, name: str) -> Path:
    path = db.source_dir(source_id) / "exports" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_track(track: dict, total: int, path: Path) -> Path:
    with sf.SoundFile(path, "w", SR, 1, "PCM_24") as f:
        for a in range(0, total, BLOCK):
            n = min(BLOCK, total - a)
            f.write(db.read_range(track["id"], a, a + n, wait=False))
    return path


def _slug(s: str) -> str:
    return re.sub(r"[^\w.-]+", "_", s).strip("_")[:60] or "track"


def mix_wav(source_id: int) -> Path:
    src = db.one("SELECT * FROM sources WHERE id = ?", source_id)
    tracks = _audible(db.q("SELECT * FROM tracks WHERE source_id = ?", source_id))
    _check_ready(tracks)
    total = int(src["duration_s"] * SR)
    path = _out(source_id, f"{_slug(src['name'])}_mix.wav")
    with sf.SoundFile(path, "w", SR, 1, "PCM_24") as f:
        for a in range(0, total, BLOCK):
            n = min(BLOCK, total - a)
            mix = np.zeros(n, np.float32)
            for t in tracks:
                mix += db.read_range(t["id"], a, a + n, wait=False) * _envelope(t, a, n) * (t["gain"] / 100)
            f.write(np.clip(mix, -1, 1))
    return path


def mix_mp4(source_id: int) -> Path:
    src = db.one("SELECT * FROM sources WHERE id = ?", source_id)
    audio = mix_wav(source_id)
    path = _out(source_id, f"{_slug(src['name'])}_mix.mp4")
    trim = ["-ss", f"{src['offset_s']:.3f}", "-t", f"{src['duration_s']:.3f}"] if src["kind"] == "youtube" else []

    def attempt(_urls):
        video, headers = video_source(src)
        head = ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())] if headers else []
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", *head, *trim, "-i", video, "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(path)],
            capture_output=True, text=True,
        )
        if p.returncode:
            raise RuntimeError(f"ffmpeg: {p.stderr.replace(video, 'video').strip()[-300:]}")

    retry(src, attempt)
    return path


def track_wav(track_id: int) -> Path:
    t = db.one("SELECT * FROM tracks WHERE id = ?", track_id)
    _check_ready([t])
    src = db.one("SELECT * FROM sources WHERE id = ?", t["source_id"])
    return _write_track(t, int(src["duration_s"] * SR), _out(src["id"], f"{_slug(t['name'])}.wav"))


def tracks_zip(source_id: int) -> Path:
    src = db.one("SELECT * FROM sources WHERE id = ?", source_id)
    tracks = db.q("SELECT * FROM tracks WHERE source_id = ? AND status = 'complete' ORDER BY id", source_id)
    _check_ready(db.q("SELECT * FROM tracks WHERE source_id = ?", source_id))
    path = _out(source_id, f"{_slug(src['name'])}_tracks.zip")
    total = int(src["duration_s"] * SR)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for i, t in enumerate(tracks, 1):
            wav = _write_track(t, total, _out(source_id, f"{i:02d}_{_slug(t['name'])}.wav"))
            z.write(wav, wav.name)
            wav.unlink()
    return path
