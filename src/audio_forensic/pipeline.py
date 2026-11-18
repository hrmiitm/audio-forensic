"""Chunked separation shared by the CLI, the benchmark and the web app.

Audio is processed in overlapping windows (chunk length + crossfade). Finished audio
is emitted in blocks (10 s first, then 30 s) so playback can start before the end.
"""

from pathlib import Path

import numpy as np

from .backends import VisualRef, check
from .config import BLOCK_S, FIRST_BLOCK_S, OVERLAP_S, SR
from .media import decode, fit, write_wav


def localize(anchors, a: float, b: float) -> list:
    """Clip absolute [sign, start, end] spans to window [a, b) and make them window-relative."""
    return [[s, max(x, a) - a, min(y, b) - a] for s, x, y in anchors or [] if y > a and x < b]


def range_mask(a: int, n: int, rng, ramp: float = 0.05) -> np.ndarray | None:
    """1 inside the selected range (with short ramps), 0 outside; None = whole clip."""
    if not rng:
        return None
    t = (a + np.arange(n)) / SR
    return np.clip(np.minimum(t - rng[0], rng[1] - t) / ramp + 0.5, 0, 1).astype(np.float32)


def make_process(backend, prompt, anchors=None, visual: VisualRef | None = None, opts=None, rng=None, cancel=None):
    """Build process(a, x) -> (target, residual) for one window starting at sample a."""

    def process(a: int, x: np.ndarray):
        mask = range_mask(a, len(x), rng)
        if mask is not None and not mask.any():  # window outside the selection: pass through
            return np.zeros_like(x), x.copy()
        t0, t1 = a / SR, (a + len(x)) / SR
        vis = (visual, t0, t1) if visual is not None else None
        target, residual = backend(x, prompt, localize(anchors, t0, t1), vis, opts, cancel)
        if mask is not None:  # outside the range, the residual keeps the input
            target, residual = target * mask, residual * mask + x * (1 - mask)
        return target, residual

    return process


def separate_stream(read, total: int, process, emit, chunk_s: float, overlap_s: float = OVERLAP_S,
                    cancel=None, progress=None) -> None:
    """read(a, b) -> samples; emit(start, target, residual) receives finished blocks in order."""
    chunk = int(max(2.0, chunk_s) * SR)
    ov = int(min(overlap_s, chunk_s / 4) * SR)
    tail, buf, out_pos, a = None, [], 0, 0
    while a < total:
        check(cancel)
        b = total if a + chunk >= total - SR else a + chunk  # fold a short remainder into the last window
        x = read(a, b)
        t, r = process(a, x)
        if tail is not None:  # crossfade with the previous window's held-back end
            k = len(tail[0])
            f = np.linspace(0, 1, k, dtype=np.float32)
            t[:k], r[:k] = tail[0] * (1 - f) + t[:k] * f, tail[1] * (1 - f) + r[:k] * f
        last = b >= total
        keep = len(t) if last else len(t) - ov
        tail = None if last else (t[keep:], r[keep:])
        buf.append((t[:keep], r[:keep]))
        size = sum(len(p[0]) for p in buf)
        if last or size >= (FIRST_BLOCK_S if out_pos == 0 else BLOCK_S) * SR:
            emit(out_pos, np.concatenate([p[0] for p in buf]), np.concatenate([p[1] for p in buf]))
            out_pos, buf = out_pos + size, []
        if progress:
            progress(b / total)
        a = b - ov if not last else total


def separate_file(src, prompt, backend, out_dir: Path, opts=None, anchors=None, visual=None, rng=None, cancel=None):
    """Offline helper: separate a whole file, write target.wav / residual.wav, return their paths."""
    opts = opts or {}
    wave = decode(src)
    target, residual = np.zeros_like(wave), np.zeros_like(wave)

    def emit(a, t, r):
        target[a : a + len(t)], residual[a : a + len(r)] = t, r

    backend.seed(int(opts.get("seed", 42)))
    process = make_process(backend, prompt, anchors, VisualRef(visual) if visual else None, opts, rng, cancel)
    separate_stream(lambda a, b: fit(wave[a:b], b - a), len(wave), process, emit,
                    opts.get("chunk", 5.0), opts.get("overlap", OVERLAP_S), cancel)
    out_dir = Path(out_dir)
    return {k: write_wav(out_dir / f"{k}.wav", v) for k, v in (("input", wave), ("target", target), ("residual", residual))}
