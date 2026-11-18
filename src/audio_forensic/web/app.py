"""FastAPI app: serves the UI and a small JSON API over db / ingest / jobs / export."""

import base64
import json
from contextlib import asynccontextmanager
from functools import wraps
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import backends
from ..config import SR
from ..devices import models, stats
from . import db, export, ingest, jobs

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init()
    jobs.init()
    ingest.resume_all()
    stats()  # prime CPU % sampling
    yield


class Static(StaticFiles):
    """Static files the browser must revalidate (ETag), so UI updates show up on reload."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app = FastAPI(title="Audio Forensic", lifespan=lifespan)
app.mount("/static", Static(directory=STATIC), name="static")


def errors(fn):
    """Turn expected failures into clean HTTP errors for the UI."""
    @wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except (ValueError, export.NotReady) as e:
            raise HTTPException(400 if isinstance(e, ValueError) else 409, str(e)) from None
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from None
    return wrapper


def get(table: str, id_: int) -> dict:
    row = db.one(f"SELECT * FROM {table} WHERE id = ?", id_)
    if row is None:
        raise HTTPException(404, f"{table[:-1].title()} not found")
    return row


def download(path: Path) -> FileResponse:
    return FileResponse(path, filename=path.name)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/models")
def api_models():
    return models()


@app.get("/api/stats")
def api_stats():
    active = db.one("SELECT COUNT(*) AS c FROM jobs WHERE status IN ('queued', 'running')")["c"]
    return {**stats(), "active_jobs": active, "model_loaded": backends.loaded()}


# ───────── Sources ─────────

@app.get("/api/sources")
def list_sources():
    return db.q("""SELECT s.id, s.kind, s.name, s.yt_mode, s.duration_s, s.created_at,
                          (SELECT COUNT(*) FROM tracks t WHERE t.source_id = s.id AND t.role = 'target') AS isolated
                   FROM sources s ORDER BY s.id DESC""")


@app.post("/api/sources/upload")
@errors
def upload(file: UploadFile = File(...)):
    return {"id": ingest.add_upload(file.filename or "upload", file.file)}


@app.post("/api/sources/youtube")
@errors
def youtube(url: str = Body(...), mode: str = Body("stream"), start: float = Body(0.0), end: float | None = Body(None)):
    if not url.strip().startswith(("http://", "https://")):
        raise ValueError("Paste a full YouTube link (https://…)")
    return {"id": ingest.add_youtube(url.strip(), "save" if mode == "save" else "stream", max(0.0, start), end)}


@app.get("/api/sources/{sid}")
def source_state(sid: int):
    src = get("sources", sid)
    tracks = db.q("""SELECT id, role, name, color, muted, solo, gain, edits, status, job_id, covered FROM tracks
                     WHERE source_id = ? ORDER BY role != 'original', role = 'residual', id""", sid)
    for t in tracks:
        t["covered"] /= SR
    js = db.q("SELECT id, model, device, prompt, status, progress, message, elapsed_s, peak_vram_mib FROM jobs WHERE source_id = ? ORDER BY id", sid)
    return {"source": src, "tracks": tracks, "jobs": js}


@app.patch("/api/sources/{sid}")
def patch_source(sid: int, body: dict = Body(...)):
    get("sources", sid)
    db.update("sources", sid, **{k: v for k, v in body.items() if k in ("name", "splits")})
    return {"ok": True}


@app.delete("/api/sources/{sid}")
def delete_source(sid: int):
    if db.one("SELECT id FROM jobs WHERE source_id = ? AND status IN ('queued', 'running')", sid):
        jobs.stop()
    db.delete_source(sid)
    return {"ok": True}


@app.get("/api/sources/{sid}/segments")
def segments(sid: int, after: int = 0):
    rows = db.q("""SELECT g.id, g.track_id, g.start, g.n, g.peaks FROM segments g JOIN tracks t ON t.id = g.track_id
                   WHERE t.source_id = ? AND g.id > ? ORDER BY g.id""", sid, after)
    return [{"id": r["id"], "track_id": r["track_id"], "start": r["start"] / SR, "dur": r["n"] / SR,
             "peaks": base64.b64encode(r["peaks"]).decode()} for r in rows]


@app.get("/api/segments/{gid}")
def segment_audio(gid: int):
    return FileResponse(get("segments", gid)["path"], media_type="audio/wav")


@app.get("/api/sources/{sid}/video")
@errors
def video(sid: int, request: Request):
    src = get("sources", sid)
    path, headers = ingest.video_source(src)
    return FileResponse(path) if headers is None else ingest.proxy(src, request.headers.get("range"))


@app.get("/api/sources/{sid}/export")
@errors
def export_source(sid: int, format: str = "wav"):
    get("sources", sid)
    fn = {"wav": export.mix_wav, "mp4": export.mix_mp4, "zip": export.tracks_zip}.get(format)
    if fn is None:
        raise ValueError("Unknown export format")
    return download(fn(sid))


# ───────── Tracks ─────────

@app.patch("/api/tracks/{tid}")
def patch_track(tid: int, body: dict = Body(...)):
    get("tracks", tid)
    db.update("tracks", tid, **{k: v for k, v in body.items() if k in ("name", "muted", "solo", "gain", "edits")})
    return {"ok": True}


@app.delete("/api/tracks/{tid}")
def delete_track(tid: int):
    t = get("tracks", tid)
    if t["role"] == "original":
        raise HTTPException(400, "The original track can't be removed")
    if t["status"] == "partial":
        jobs.stop()
    db.delete_track(tid)
    return {"ok": True}


@app.get("/api/tracks/{tid}/download")
@errors
def download_track(tid: int):
    get("tracks", tid)
    return download(export.track_wav(tid))


# ───────── Jobs ─────────

@app.post("/api/jobs")
@errors
def create_job(spec: str = Form(...), visual: UploadFile | None = File(None)):
    spec = json.loads(spec)
    src = get("sources", int(spec["source_id"]))
    if not spec.get("prompt", "").strip() and visual is None:
        raise ValueError("Describe the sound, or add a visual reference")
    if visual is not None and not src["has_video"]:
        raise ValueError("A visual reference needs a video source")
    path = None
    if visual is not None:
        refs = db.source_dir(src["id"]) / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        path = refs / f"{len(list(refs.iterdir())) + 1}{Path(visual.filename or 'ref').suffix.lower()}"
        path.write_bytes(visual.file.read())
    return {"id": jobs.submit(src["id"], spec, str(path) if path else None)}


@app.post("/api/jobs/stop")
def stop_jobs():
    return {"stopped": jobs.stop()}


def main() -> None:
    import os

    import uvicorn

    # AF_HOST=0.0.0.0 inside Docker; the default only listens on this machine
    uvicorn.run("audio_forensic.web.app:app", host=os.environ.get("AF_HOST", "127.0.0.1"), port=int(os.environ.get("AF_PORT", "8000")))
