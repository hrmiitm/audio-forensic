"""Isolation jobs: one at a time on a background worker (one model in GPU memory)."""

import queue
import threading
import time

from .. import backends
from ..backends import Cancelled, VisualRef
from ..config import CHUNK_DEFAULTS, OVERLAP_S, SR
from ..pipeline import make_process, separate_stream
from . import db

PALETTE = ["#4fb3a9", "#e07a5f", "#6b9bd1", "#c9a55c", "#b48ead", "#8fbf6a"]
_queue: "queue.Queue[int]" = queue.Queue()
_cancel = threading.Event()


def submit(source_id: int, spec: dict, visual_path: str | None) -> int:
    """spec: input ('original'|'residual'), range [a, b] | None, model, device, prompt, anchors, options."""
    tracks = db.q("SELECT id, role FROM tracks WHERE source_id = ? AND status IN ('complete', 'partial') ORDER BY id", source_id)
    residuals = [t["id"] for t in tracks if t["role"] == "residual"]
    original = next(t["id"] for t in tracks if t["role"] == "original")
    rng = spec.get("range") or [None, None]
    jid = db.insert(
        "jobs", source_id=source_id, model=spec["model"], device=spec["device"], prompt=spec["prompt"].strip().lower(),
        input_track_id=residuals[-1] if spec.get("input") == "residual" and residuals else original,
        anchors=spec.get("anchors") or [], options=spec.get("options") or {}, visual_path=visual_path,
        range_start=rng[0], range_end=rng[1], message="Waiting for the GPU",
    )
    _queue.put(jid)
    return jid


def stop() -> int:
    """Cancel queued and running jobs; the worker frees the model afterwards."""
    running = db.one("SELECT id FROM jobs WHERE status = 'running'")
    db.ex("UPDATE jobs SET status = 'cancelled', message = 'Stopped' WHERE status = 'queued'")
    if running:
        _cancel.set()
    else:
        backends.release()  # nothing running: just free the GPU
    return int(bool(running))


def _worker() -> None:
    while True:
        jid = _queue.get()
        job = db.one("SELECT * FROM jobs WHERE id = ?", jid)
        if job and job["status"] == "queued":
            _cancel.clear()
            _run(job)
            if _cancel.is_set():  # Stop also frees the GPU; otherwise keep the model loaded for the next job
                backends.release()


def _run(job: dict) -> None:
    jid, sid = job["id"], job["source_id"]
    src = db.one("SELECT * FROM sources WHERE id = ?", sid)
    n_targets = db.one("SELECT COUNT(*) AS c FROM tracks WHERE source_id = ? AND role = 'target'", sid)["c"]
    label = job["prompt"] or "visual source"
    common = dict(source_id=sid, job_id=jid, status="partial")
    target = db.insert("tracks", role="target", name=label[:1].upper() + label[1:], color=PALETTE[n_targets % len(PALETTE)], **common)
    # Residuals start muted (you hear the isolated sound); the input is muted since its parts are now separate tracks
    residual = db.insert("tracks", role="residual", name=f"Residual after “{label}”", color="--wave-residual", muted=1, **common)
    db.update("tracks", job["input_track_id"], muted=1)
    db.update("jobs", jid, status="running", message="Loading model")
    opts = {"chunk": CHUNK_DEFAULTS[job["model"]], "overlap": OVERLAP_S, "steps": 16, "seed": 42, **job["options"]}
    t0 = time.perf_counter()
    status, message = "failed", None
    try:
        backend = backends.get_backend(job["model"], job["device"], vision=bool(job["visual_path"]))
        load_s, t0 = time.perf_counter() - t0, time.perf_counter()  # separation time excludes model loading
        backend.seed(int(opts["seed"]))
        backend.reset_peak()
        total = int(src["duration_s"] * SR)
        rng = [job["range_start"], job["range_end"]] if job["range_start"] is not None else None
        visual = VisualRef(job["visual_path"]) if job["visual_path"] else None
        process = make_process(backend, job["prompt"], job["anchors"], visual, opts, rng, _cancel)

        def emit(a, t, r):
            db.add_segment(target, a, t)
            db.add_segment(residual, a, r)

        def progress(f):
            done = f * src["duration_s"]
            db.update("jobs", jid, progress=f, elapsed_s=time.perf_counter() - t0,
                      message=f"Separating · {_fmt(done)} of {_fmt(src['duration_s'])}")

        separate_stream(lambda a, b: db.read_range(job["input_track_id"], a, b, _cancel), total, process, emit,
                        float(opts["chunk"]), float(opts["overlap"]), _cancel, progress)
        status = "done"
        message = (f"{_fmt(src['duration_s'])} separated in {time.perf_counter() - t0:.1f} s"
                   + (f" (+{load_s:.0f} s model load)" if load_s > 1 else "") + (f" · {backend.note}" if backend.note else ""))
        db.update("jobs", jid, peak_vram_mib=backend.peak_vram_mib())
    except Cancelled:
        status, message = "cancelled", "Stopped"
    except Exception as e:
        message = f"{type(e).__name__}: {e}"[:300]
        if "out of memory" in message.lower():
            message = "GPU out of memory: try a shorter chunk length, close other GPU apps, or use the CPU"
            backends.release()
    track_status = {"done": "complete", "cancelled": "cancelled"}.get(status, "failed")
    for t in (target, residual):
        db.update("tracks", t, status=track_status)
    if status != "done":  # keep the input audible when nothing complete replaced it
        db.update("tracks", job["input_track_id"], muted=0)
    db.update("jobs", jid, status=status, message=message, elapsed_s=time.perf_counter() - t0,
              progress=1.0 if status == "done" else job["progress"])


def _fmt(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


def init() -> None:
    db.ex("UPDATE jobs SET status = 'failed', message = 'Interrupted by a server restart' WHERE status IN ('queued', 'running')")
    db.ex("UPDATE tracks SET status = 'failed' WHERE status = 'partial' AND role != 'original'")
    threading.Thread(target=_worker, daemon=True, name="jobs").start()
