"""Benchmark models on every sample input.

    uv run python -m audio_forensic.run_samples                 # all models
    uv run python -m audio_forensic.run_samples torch-fp16      # just one
"""

import sys
import threading
import time

from .cli import MODEL_IDS, run
from .config import INPUTS_DIR
from .devices import stats
from .evaluate import evaluate

PROMPT = "man speaking"


class GpuPeak:
    """Sample whole-GPU memory use during a block."""

    def __enter__(self):
        self.peak, self._stop = 0.0, threading.Event()
        self._t = threading.Thread(target=self._poll, daemon=True)
        self._t.start()
        return self

    def _poll(self):
        while not self._stop.wait(0.1):
            self.peak = max([self.peak] + [g["mem_used"] * 1024 for g in stats()["gpus"]])

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def main(models=MODEL_IDS, prompt: str = PROMPT) -> None:
    for model in models:
        for src in sorted(p for p in INPUTS_DIR.iterdir() if p.is_file()):
            print(f"\n== {src.name} | {model} | '{prompt}'", flush=True)
            try:
                with GpuPeak() as gpu:
                    t = time.perf_counter()
                    r = run(src, prompt, model)
            except Exception as e:
                print(f"   FAILED: {type(e).__name__}: {e}")
                continue
            m = evaluate(r["input"], r["target"], r["residual"])
            torch_peak = f", torch peak {r['peak_mib']:.0f} MiB" if r["peak_mib"] else ""
            print(
                f"   time {time.perf_counter() - t:.1f}s (load {r['load_s']:.1f}s + separate {r['infer_s']:.1f}s{torch_peak})"
                f" | GPU peak {gpu.peak:.0f} MiB | dur {m['dur_s']:.1f}s | in {m['input_db']:.1f} / target {m['target_db']:.1f}"
                f" / residual {m['residual_db']:.1f} dB | recon SNR {m['recon_snr_db']:.1f} dB | finite={m['finite']}",
                flush=True,
            )


if __name__ == "__main__":
    main(sys.argv[1:] or MODEL_IDS)
