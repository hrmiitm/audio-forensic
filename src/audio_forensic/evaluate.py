"""Sanity checks on a separation result: durations, loudness, reconstruction."""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def _mono48k(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return resample_poly(x, 48000, sr).astype("float32") if sr != 48000 else x


def _db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12)


def evaluate(input_path, target_path, residual_path) -> dict:
    x, t, r = (_mono48k(Path(p)) for p in (input_path, target_path, residual_path))
    n = min(len(x), len(t), len(r))
    x, t, r = x[:n], t[:n], r[:n]
    err = x - (t + r)
    return {
        "dur_s": n / 48000,
        "input_db": _db(x),
        "target_db": _db(t),
        "residual_db": _db(r),
        # How well target + residual rebuilds the input (higher = better; watermark limits it)
        "recon_snr_db": _db(x) - _db(err),
        "finite": bool(np.isfinite(t).all() and np.isfinite(r).all()),
    }
