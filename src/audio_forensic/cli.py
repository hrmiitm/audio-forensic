"""Separate a file from the command line.

    uv run audio-forensic INPUT --text "man speaking" [--model torch-fp16|large-fp8|...] [--device cuda:0]
"""

import argparse
import json
import time
from pathlib import Path

from .backends import get_backend
from .config import CHUNK_DEFAULTS, OUTPUTS_DIR
from .devices import default_device
from .pipeline import separate_file

MODEL_IDS = list(CHUNK_DEFAULTS)


def run(src, text, model="torch-fp16", device=None, out_dir=None, anchors=None, visual=None, **opts) -> dict:
    t0 = time.perf_counter()
    backend = get_backend(model, device or default_device(model), vision=bool(visual))
    load_s = time.perf_counter() - t0
    backend.reset_peak()
    t1 = time.perf_counter()
    out = Path(out_dir or OUTPUTS_DIR / f"{Path(src).stem}_{model}")
    paths = separate_file(src, text, backend, out, {"chunk": CHUNK_DEFAULTS[model], **opts}, anchors, visual)
    return {**paths, "load_s": load_s, "infer_s": time.perf_counter() - t1, "peak_mib": backend.peak_vram_mib()}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio")
    p.add_argument("--text", required=True, help='short sound name, e.g. "dog barking"')
    p.add_argument("--model", choices=MODEL_IDS, default="torch-fp16")
    p.add_argument("--device", help="cuda:0 | vulkan:1 | cpu (default: best available)")
    p.add_argument("--out-dir")
    p.add_argument("--anchors", type=json.loads, help='time hints, e.g. \'[["+",0.5,2.0]]\'')
    p.add_argument("--visual", help="masked image/video of the sound source")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk", type=float, help="seconds per model call")
    a = p.parse_args(argv)
    opts = {"steps": a.steps, "seed": a.seed, **({"chunk": a.chunk} if a.chunk else {})}
    r = run(a.audio, a.text, a.model, a.device, a.out_dir, a.anchors, a.visual, **opts)
    print(f"load {r['load_s']:.1f}s + separate {r['infer_s']:.1f}s -> {r['target']} , {r['residual']}")


if __name__ == "__main__":
    main()
