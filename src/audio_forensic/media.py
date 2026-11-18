"""Audio I/O helpers (FFmpeg decode, WAV files, waveform peaks)."""

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import PEAKS_PER_SEC, SR


def _headers(headers: dict | None) -> list[str]:
    return ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())] if headers else []


def probe(src: str | Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(src)],
        capture_output=True, text=True,
    )
    if out.returncode:
        raise ValueError("Not a readable audio/video file")
    info = json.loads(out.stdout)
    streams = info.get("streams", [])
    kinds = {s["codec_type"] for s in streams if not s.get("disposition", {}).get("attached_pic")}
    return {
        "duration": float(info.get("format", {}).get("duration") or 0),
        "has_audio": "audio" in kinds,
        "has_video": "video" in kinds,
    }


def decode(src: str | Path, start: float = 0.0, dur: float | None = None, headers: dict | None = None) -> np.ndarray:
    """Decode (part of) a local file or URL to 48 kHz mono float32. Seeks without reading what's before."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", *_headers(headers)]
    if str(src).startswith("http"):
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if dur:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode:
        err = p.stderr.decode(errors="replace").replace(str(src), "input").strip()  # keep long URLs out of messages
        raise RuntimeError(err[-300:] or "ffmpeg failed")
    return np.frombuffer(p.stdout, np.float32).copy()


def fit(x: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad to exactly n samples."""
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def write_wav(path: Path, wave: np.ndarray, subtype: str = "FLOAT") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wave, SR, subtype=subtype)
    return path


def read_wav(path: Path, start: int = 0, stop: int | None = None) -> np.ndarray:
    x, sr = sf.read(path, start=start, stop=stop, dtype="float32", always_2d=True)
    return x.mean(axis=1)


def peaks(wave: np.ndarray) -> bytes:
    """Max |amplitude| per 1/PEAKS_PER_SEC s, as uint8 (0-255)."""
    hop = SR // PEAKS_PER_SEC
    n = -(-len(wave) // hop)
    buf = np.zeros(n * hop, np.float32)
    buf[: len(wave)] = np.abs(wave)
    return (np.clip(buf.reshape(n, hop).max(1), 0, 1) * 255).round().astype(np.uint8).tobytes()
