# AGENTS.md

Guidance for coding agents working in this repository. The user-facing documentation is in [README.md](README.md).

## What this is

A local web app + CLI around Meta's SAM-Audio: text/time/visual-prompted sound separation, with a timeline editor. Python 3.12, FastAPI, vanilla HTML/CSS/JS (no build step), SQLite, uv. The target machine has a **6 GB NVIDIA GPU**, so memory limits drive many design choices.

## Commands

```bash
uv sync                                          # install (never pip install into .venv)
uv add <pkg>                                     # add a dependency (updates pyproject.toml + uv.lock)
uv run audio-forensic-web                        # web app on 127.0.0.1:8000 (AF_HOST/AF_PORT)
uv run audio-forensic FILE --text "voice"        # CLI separation
uv run python -m audio_forensic.run_samples      # benchmark on sample-input-files/
uv run python -m audio_forensic.download [large] # weights + audio.cpp
uvx ruff check src --select F,E9                 # lint (unused/undefined names, syntax)
node --check src/audio_forensic/web/static/app.js
```

There is no unit-test suite. Verify changes by running the real thing:
- **Pipeline/backends:** run the CLI on a short clip and check `evaluate.py`'s numbers. The reconstruction SNR should stay ~12–14 dB, and the outputs must be finite.
- **Web:** run uvicorn on a spare port (e.g. 8765) and drive the API with urllib, or the UI with headless Chrome over CDP. Use `?demo` to test the GitHub Pages mode.
- **Only one process may hold a model on the GPU.** If the user's own server is running, don't load a second model; a test failing with OOM can be this, not a bug. Stop test servers by port, never with `pkill -f` patterns, which can match your own shell.

## Architecture (where to change what)

| Concern | File |
|---|---|
| Model list, labels, defaults, file locations | `config.py` → `MODEL_SPECS` (the single registry; UI, CLI and download read it) |
| Loading/running models | `backends.py` (`TorchBackend`, `GGUFBackend`, `get_backend`, `release`) |
| Chunking, crossfade, time hints, selection range | `pipeline.py` (`separate_stream`, `make_process`) |
| FP8 | `fp8.py` |
| Highlight in video (SAM 3 / SAM 2.1 tracking → masked video) | `segment.py` |
| Storage (tracks = WAV blocks + SQLite rows) | `web/db.py` |
| Uploads / YouTube / video proxy | `web/ingest.py` |
| Job queue, Stop | `web/jobs.py` |
| Exports | `web/export.py` |
| API | `web/app.py` |
| UI | `web/static/index.html`, `style.css`, `app.js`, `models.md` |

Key ideas:
- **Every track is a list of WAV blocks** (10 s first, then 30 s). Ingest, separation and playback all work block by block, which gives progressive YouTube playback and long-file support. `db.read_range` waits for blocks that are still being produced.
- **Backends share one interface:** `backend(wave, text, anchors, visual, opts, cancel) -> (target, residual)` for one window. The pipeline does windowing and crossfades.
- **Torch loading:** build the model under `accelerate.init_empty_weights`, rebuild T5 for real (it isn't in the checkpoint), rebuild the vision encoder's `Rope2D` tables (a plain object, not a submodule), then `load_state_dict(assign=True)` from a memory-mapped checkpoint. If the weights won't fit, the transformer is offloaded with `accelerate.cpu_offload`.
- **One model on the GPU at a time:** `backends.acquire()` is the single slot shared by separation backends and the segmenter. Loading one unloads the other.
- **Visual prompt convention (Meta's `mask_videos`):** frames * (mask == 0), i.e. the **target object is blacked out**. `segment.track` streams frames through the tracker; when pruning `session.processed_frames`, keep the keys (the session numbers frames by `len()`) and only free the tensors.
- **One GPU job at a time** (`jobs._worker`). The model stays loaded between jobs; `Stop` cancels and calls `backends.release()`.
- **YouTube stream URLs return 403 for about the first second;** `ingest.retry` handles this. Let yt-dlp pick formats (`STREAM_FMT`), so you get the original-language audio and not dubbed/DRC copies.

## Conventions

- Keep it small and DRY: reuse `media.py`, `db.q/one/ex/insert/update`, `pipeline.separate_stream`. Prefer extending `MODEL_SPECS` over adding model-specific branches.
- Python: type hints on public functions, short docstrings that say *why*, no dead code. Errors shown to users are short and actionable, with no URLs or tokens in them.
- UI:
  - Colours come from CSS variables; both light and dark themes must keep working.
  - New controls get a `data-help="Title|bullet|bullet"` hover tip, in plain language.
  - Demo mode (`DEMO` in `app.js`) must keep working: any new API call needs a sensible demo behaviour, or should throw `DEMO_TEXT`.
  - Bump the `?v=` on the asset links in `index.html` when you change `app.js` or `style.css`.
- **Never commit** `.env`, `data/`, `models/`, `vendor/`, sample media, or databases (all in `.gitignore`). Never print `HF_ACCESS_TOKEN`.
- Model weights are under Meta's SAM License; keep that note when you touch the docs.
