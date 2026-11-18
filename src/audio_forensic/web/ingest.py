"""Bring sources in. Local files and YouTube links are both decoded block by block into
the original track (10 s first, then 30 s), so playback can start almost immediately."""

import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from fastapi.responses import StreamingResponse

from ..config import BLOCK_S, FIRST_BLOCK_S, SR
from ..media import decode, fit, probe
from . import db

ORIGINAL_COLOR = "--wave-neutral"
YT_TTL = 2 * 3600  # re-resolve stream URLs before YouTube expires them (~6 h)
_yt: dict[int, dict] = {}


YDL_OPTS = {"quiet": True, "no_warnings": True, "noplaylist": True, "js_runtimes": {"node": {}}}
# Let yt-dlp choose: original-language audio (not dubbed / DRC copies), H.264 video up to 720p, direct HTTPS only
STREAM_FMT = ("bv*[height<=720][protocol=https][vcodec^=avc1]+ba[protocol=https]"
              "/bv*[height<=720][protocol=https]+ba[protocol=https]/ba[protocol=https]")


def yt_urls(source: dict, refresh: bool = False) -> dict:
    """Direct audio/video stream URLs (+ headers) for a YouTube source, cached (one lookup for both)."""
    cached = _yt.get(source["id"])
    if refresh or not cached or time.time() - cached["at"] > YT_TTL:
        import yt_dlp

        with yt_dlp.YoutubeDL({**YDL_OPTS, "skip_download": True, "format": STREAM_FMT}) as y:
            info = y.extract_info(source["url"], download=False)
        fmts = info.get("requested_formats") or [info]
        a = next((f for f in fmts if f.get("acodec") not in (None, "none")), None)
        v = next((f for f in fmts if f.get("vcodec") not in (None, "none")), None)
        if a is None:
            raise ValueError("No streamable audio for this video")
        cached = _yt[source["id"]] = {
            "at": time.time(), "title": info.get("title") or source["url"], "duration": float(info.get("duration") or 0),
            "audio": (a["url"], a.get("http_headers", {})), "video": (v["url"], v.get("http_headers", {})) if v else None,
        }
    return cached


def retry(source: dict, attempt, tries: int = 5):
    """attempt(urls) with retries. YouTube refuses freshly issued stream URLs for about a second
    (HTTP 403), so wait and retry; re-resolve the URLs if it keeps failing."""
    youtube = source["kind"] == "youtube"
    for i in range(tries):
        try:
            return attempt(yt_urls(source, refresh=i == 3) if youtube else None)
        except (RuntimeError, urllib.error.URLError):
            if not youtube or i == tries - 1:
                raise
            time.sleep(1)


def _read_block(source: dict, a: int, b: int):
    def attempt(urls):
        url, headers = urls["audio"] if urls else (source["path"], None)
        return decode(url, source["offset_s"] + a / SR, (b - a) / SR, headers)

    return fit(retry(source, attempt), b - a)


def _ingest(source: dict, track_id: int) -> None:
    """Fill the original track block by block in the background."""
    def run():
        try:
            total = int(source["duration_s"] * SR)
            done = db.one("SELECT covered FROM tracks WHERE id = ?", track_id)["covered"]
            for a, b in db.iter_blocks(total, FIRST_BLOCK_S, BLOCK_S):
                if b > done:  # skip blocks kept from before a restart
                    db.add_segment(track_id, a, _read_block(source, a, b))
            db.update("tracks", track_id, status="complete")
        except Exception as e:
            db.update("tracks", track_id, status="failed")
            db.update("sources", source["id"], error=f"Could not read audio: {e}"[:300])

    threading.Thread(target=run, daemon=True, name=f"ingest-{source['id']}").start()


def start(source_id: int) -> None:
    """(Re)start filling a source's original track."""
    s = db.one("SELECT * FROM sources WHERE id = ?", source_id)
    t = db.one("SELECT id FROM tracks WHERE source_id = ? AND role = 'original'", source_id)
    _ingest(s, t["id"])


def add_upload(filename: str, fileobj) -> int:
    sid = db.insert("sources", kind="audio", name=filename)  # reserve id for the folder
    path = db.source_dir(sid) / ("source" + Path(filename).suffix.lower())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        shutil.copyfileobj(fileobj, f, 1 << 20)
    try:
        info = probe(path)
        if not info["has_audio"] or info["duration"] <= 0:
            raise ValueError("The file has no audio track")
    except ValueError:
        db.delete_source(sid)
        raise
    db.update("sources", sid, kind="video" if info["has_video"] else "audio", path=str(path),
              duration_s=info["duration"], has_video=int(info["has_video"]))
    db.insert("tracks", source_id=sid, role="original", name="Original sound", color=ORIGINAL_COLOR)
    start(sid)
    return sid


def add_youtube(url: str, mode: str, start_s: float = 0.0, end_s: float | None = None) -> int:
    sid = db.insert("sources", kind="youtube", name=url, url=url, yt_mode=mode)
    try:
        info = yt_urls({"id": sid, "url": url})
    except Exception as e:
        db.delete_source(sid)
        raise ValueError(f"Could not open the YouTube link: {str(e).splitlines()[-1][:200]}") from None
    end = min(end_s or info["duration"], info["duration"])
    if end - start_s < 1:
        db.delete_source(sid)
        raise ValueError("The selected part is shorter than 1 second")
    db.update("sources", sid, name=info["title"], offset_s=start_s, duration_s=end - start_s, has_video=int(bool(info["video"])))
    db.insert("tracks", source_id=sid, role="original", name="Original sound", color=ORIGINAL_COLOR)
    start(sid)
    if mode == "save":
        threading.Thread(target=_save_copy, args=(sid, url), daemon=True).start()
    return sid


def _save_copy(source_id: int, url: str) -> None:
    """Download the full video in the background (Save a copy mode)."""
    import yt_dlp

    out = db.source_dir(source_id) / "video.%(ext)s"
    opts = {**YDL_OPTS, "format": "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]/b",
            "merge_output_format": "mp4", "outtmpl": str(out)}
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            info = y.extract_info(url, download=True)
        path = Path(y.prepare_filename(info)).with_suffix(".mp4")
        if path.exists() and db.one("SELECT id FROM sources WHERE id = ?", source_id):
            db.update("sources", source_id, path=str(path))
    except Exception as e:
        if db.one("SELECT id FROM sources WHERE id = ?", source_id):
            db.update("sources", source_id, error=f"Saving the video copy failed: {e}"[:300])


def resume_all() -> None:
    """After a restart: continue filling unfinished original tracks."""
    for t in db.q("SELECT source_id FROM tracks WHERE role = 'original' AND status = 'partial'"):
        start(t["source_id"])


def _with_video(source: dict, fn):
    """fn(video_path_or_url, headers, offset_s), retried for YouTube's fresh-URL refusals."""
    def attempt(_urls):
        video, headers = video_source(source)
        return fn(video, headers, source["offset_s"] if source["kind"] == "youtube" else 0.0)
    return retry(source, attempt)


def highlight_preview(source: dict, prompt: dict) -> bytes:
    """PNG mask of the highlighted object on one frame (fast check before a full run)."""
    from .. import segment

    return _with_video(source, lambda v, h, off: segment.preview(v, h, off + float(prompt["t"]), prompt))


def highlight_video(source: dict, prompt: dict, progress=None, cancel=None) -> Path:
    """Masked video for SAM-Audio (highlighted object blacked out); reused for the same highlight."""
    from .. import segment

    out = segment.cache_path(db.source_dir(source["id"]) / "refs", prompt)
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        _with_video(source, lambda v, h, off: segment.track(v, h, off, source["duration_s"], prompt, out, progress, cancel))
    return out


def video_source(source: dict) -> tuple[str, dict | None]:
    """Local path, or (URL, headers) to proxy, for the video preview / MP4 export."""
    if source.get("path") and Path(source["path"]).exists() and source["has_video"]:
        return source["path"], None
    if source["kind"] == "youtube":
        v = yt_urls(source)["video"]
        if v:
            return v
    raise FileNotFoundError("This source has no video")


def proxy(source: dict, range_header: str | None) -> StreamingResponse:
    """Stream a YouTube video through this server (googlevideo URLs only work from the resolving IP)."""
    def attempt(urls):
        url, headers = urls["video"]
        req = urllib.request.Request(url, headers={**headers, **({"Range": range_header} if range_header else {})})
        return urllib.request.urlopen(req, timeout=30)

    r = retry(source, attempt)
    keep = ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges")

    def body():
        with r:
            while chunk := r.read(1 << 18):
                yield chunk

    return StreamingResponse(body(), status_code=r.status, headers={k: r.headers[k] for k in keep if r.headers.get(k)})
