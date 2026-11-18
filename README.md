# Audio Forensic

Isolate any sound from an audio file, a video or a YouTube link. Name the sound ("dog barking", "voice", "keyboard typing") and Meta's **SAM-Audio** splits the recording into that sound and everything else. Then mute, re-level and export the parts in a timeline editor.

**[Live UI demo](https://hrmiitm.github.io/audio-forensic/)**: the real interface on GitHub Pages with illustrative data. Separating audio needs the app running locally or on a server.

- **Sources:** audio or video files, or YouTube links. YouTube streams part by part, so playback starts within seconds and nothing big has to download first.
- **Prompts:** text, plus optional time hints (where the sound is or isn't) and a visual reference (a masked video of the source).
- **Models:** small (0.6B) and large (3B) in PyTorch FP16, FP8 or GGUF (BF16 / Q8 via audio.cpp). The large model runs on a 6 GB GPU.
- **Timeline:** per-track mute, solo and level; split into clips and change each clip's level; isolate again from the residual to peel sounds off one by one.
- **Export:** the mix as WAV or MP4 (video + new audio), every track as a ZIP, or single tracks.
- **Also:** live CPU/GPU usage, a Stop button that frees GPU memory, history in SQLite, light and dark themes, a model guide, and hover help on every control.

## Contents

- [Requirements](#requirements)
- [Quick start with Docker](#quick-start-with-docker)
- [Manual setup](#manual-setup)
- [Hugging Face access and token](#hugging-face-access-and-token)
- [Models: download and cache](#models-download-and-cache)
- [Run](#run)
- [Model sizes and memory](#model-sizes-and-memory)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Linux x86-64 | Linux with an NVIDIA driver supporting CUDA 13 |
| GPU | none (CPU works, slowly) | NVIDIA, 6 GB+ (small: ~2.5 GB, large FP8: ~4.3 GB) |
| RAM | 8 GB | 16 GB (large FP16 streams ~6 GB of weights from RAM) |
| Disk | ~20 GB (small model + Python packages) | ~45 GB (all model versions) |

Software for the manual setup: [uv](https://docs.astral.sh/uv/), FFmpeg 6+, git, and Node.js 20+ (yt-dlp uses it for YouTube). The Docker image includes all of these.

## Quick start with Docker

```bash
git clone https://github.com/hrmiitm/audio-forensic.git
cd audio-forensic
cp .env.example .env            # then put your Hugging Face token in .env
docker compose up --build       # first build downloads ~6 GB of Python packages
```

Open **http://127.0.0.1:8000**. The small model downloads itself on the first isolation (~5 GB, cached in a Docker volume).

- **GPU:** install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). `docker-compose.yml` already requests all GPUs. Without the toolkit, remove the `deploy:` block and choose **CPU** in the app.
- **Optional models** (large FP16/FP8, GGUF, audio.cpp):

  ```bash
  docker compose run --rm audio-forensic uv run --no-sync python -m audio_forensic.download         # small GGUF + audio.cpp
  docker compose run --rm audio-forensic uv run --no-sync python -m audio_forensic.download large   # large FP16, Q8, builds FP8
  ```

- **Your data stays on the host:** `./data` (database, uploads, results), `./models`, `./vendor`.
- **GGUF models inside Docker:** these use Vulkan through the NVIDIA toolkit's graphics support, which is experimental. They always work on CPU.

## Manual setup

```bash
# 1. System packages (Fedora shown; on Debian/Ubuntu use apt install ffmpeg git nodejs)
sudo dnf install ffmpeg git nodejs
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Project + dependencies (Python 3.12 is installed by uv if needed)
git clone https://github.com/hrmiitm/audio-forensic.git
cd audio-forensic
uv sync

# 3. Token
cp .env.example .env            # put your Hugging Face token in it
```

**What `uv sync` installs** (pinned in `uv.lock`):

| Group | Packages |
|---|---|
| Model | [`sam-audio`](https://github.com/facebookresearch/sam-audio) from GitHub; it pulls in PyTorch (CUDA 13), torchaudio, torchcodec, transformers, perception-models, DAC-VAE, ImageBind, CLAP |
| Model support | `accelerate` (loads the large model without a second copy in RAM, and streams weights) |
| Attention kernels | `mslk`, from the PyTorch CUDA 13.2 index (configured in `pyproject.toml`) |
| Web app | `fastapi`, `uvicorn`, `python-multipart` |
| YouTube | `yt-dlp[default]` |
| Usage meters | `psutil`, `nvidia-ml-py` |
| Config and downloads | `huggingface-hub`, `python-dotenv` |

To add a dependency, use `uv add <package>`; it updates `pyproject.toml` and `uv.lock`.

## Hugging Face access and token

The official SAM-Audio weights are gated:

1. Sign in at huggingface.co and accept the license on [facebook/sam-audio-small](https://huggingface.co/facebook/sam-audio-small). Do the same on [facebook/sam-audio-large](https://huggingface.co/facebook/sam-audio-large) if you want the large model; its config is read from there.
2. Create a **read** token at <https://huggingface.co/settings/tokens>.
3. Put it in `.env` as `HF_ACCESS_TOKEN=hf_...`. Never commit `.env`; it's in `.gitignore`.

## Models: download and cache

| Model (UI name) | How it arrives | Where it lives | Size |
|---|---|---|---|
| Small · FP16 (PyTorch) | automatically, on first use | Hugging Face cache (`~/.cache/huggingface`) | 5.1 GB + T5 0.9 GB |
| Large · FP8 (PyTorch) | `download large` (built from FP16) | `models/sam-audio-large-fp8.pt` | 4.6 GB |
| Large · FP16 (PyTorch) | `download large` | `models/sam-audio-large-fp16/` | 7.4 GB |
| Small · Q8 / BF16 (GGUF) | `download` | `models/*.gguf` | 2.4 / 2.8 GB |
| Large · Q8 (GGUF) | `download large` | `models/sam-audio-large-q8_0.gguf` | 5.0 GB |
| audio.cpp runtime (for GGUF) | `download` | `vendor/audiocpp/` | 0.3 GB |

```bash
uv run python -m audio_forensic.download          # small GGUF (BF16, Q8) + audio.cpp runtime
uv run python -m audio_forensic.download large    # large Q8 GGUF + large FP16, then builds FP8
uv run python -m audio_forensic.fp8               # (re)build FP8 from the FP16 checkpoint only
```

Sources:
- **Large FP16:** the community export [`yaghmood/sam-audio-large-fp16-pytorch`](https://huggingface.co/yaghmood/sam-audio-large-fp16-pytorch). It's loaded as tensors only (no code runs), and it must match Meta's official config exactly.
- **GGUF files:** [`audio-cpp/SAM-Audio-GGUF`](https://huggingface.co/audio-cpp/SAM-Audio-GGUF), run with [audio.cpp](https://github.com/0xShug0/audio.cpp).

Models that aren't downloaded appear greyed out in the app.

## Run

### Web app

```bash
uv run audio-forensic-web        # http://127.0.0.1:8000  (AF_HOST / AF_PORT to change)
```

1. Load a source: **Audio file**, **Video file** or **YouTube**.
2. Name the sound, optionally add time hints or a visual reference, and press **Isolate sound** (Ctrl+Enter).
3. Results stream into the timeline as two new tracks: the sound, and a muted residual.
4. Play, mute, solo and re-level. Use **Split** to change the level of a part of a track.
5. **Export** the mix (WAV/MP4) or all tracks (ZIP).

Run one copy of the app at a time. Each copy keeps a model in GPU memory, and two copies don't fit on a 6 GB card.

### Command line

```bash
uv run audio-forensic input.wav --text "man speaking"                         # small FP16 on the best GPU
uv run audio-forensic clip.mp4  --text "dog barking" --model large-fp8
uv run audio-forensic input.wav --text "voice" --anchors '[["+", 2.0, 5.0]]'  # time hint: present 2–5 s
uv run audio-forensic video.mp4 --text "" --visual masked.mp4                 # visual reference
uv run audio-forensic --help                                                  # devices, steps, seed, chunk…
```

Outputs go to `sample-output-file/<name>_<model>/` as `input.wav`, `target.wav` and `residual.wav`.

### Benchmark / smoke test

Put a few files in `sample-input-files/`, then:

```bash
uv run python -m audio_forensic.run_samples              # every model
uv run python -m audio_forensic.run_samples large-fp8    # one model
```

For each file it prints load time, separation time, peak GPU memory, loudness, and how well target + residual rebuild the input.

## Model sizes and memory

Measured on an RTX 4050 Laptop GPU (6 GB) with a 35 s clip and default settings:

| Model | Params | Peak GPU memory | Separation time | Default chunk |
|---|---|---|---|---|
| Small · FP16 | 0.6B | 2.4 GB | 7.8 s (~4.5× faster than real time) | 5 s |
| Small · FP16 + visual reference | 0.6B | 4.1 GB | 42 s | 5 s |
| Large · FP8 | 3B | 4.3 GB | 78 s (~2× slower) | 5 s |
| Large · FP16 (weights streamed from RAM) | 3B | 4.0 GB | 151 s (~4× slower) | 20 s |
| Small · FP16 on CPU (FP32) | 0.6B | — (6.2 GB RAM) | ~7 s per second of audio | 5 s |

- Loading the model takes 10–15 s, once. It stays loaded until you press **Stop** or pick another model.
- **Out of GPU memory?** Lower **Advanced → Chunk length**.
- **FP8 quality:** its output matched FP16 at a correlation of 0.999 on identical input.
- More detail, including GGUF numbers and how the model works, is in the **Models** guide inside the app ([source](src/audio_forensic/web/static/models.md)).

## Project layout

```
src/audio_forensic/
  config.py      paths + MODEL_SPECS (the single model registry)
  media.py       FFmpeg decoding (files and URLs), WAV, waveform peaks
  backends.py    PyTorch (FP16/FP8, weight streaming) and GGUF (audio.cpp) behind one interface
  fp8.py         FP8 weight-only layers and the FP16 -> FP8 converter
  pipeline.py    overlapping chunks, crossfades, block output, time hints, selection ranges
  devices.py     GPU/CPU discovery and live usage
  cli.py         command line          run_samples.py / evaluate.py   benchmark
  download.py    weights + audio.cpp
  web/
    app.py       FastAPI routes        db.py      SQLite + block files
    ingest.py    uploads, YouTube      jobs.py    one-at-a-time GPU worker, Stop
    export.py    WAV / MP4 / ZIP       static/    UI (HTML/CSS/JS) + models.md
```

Runtime data lives in `data/` (SQLite database, uploads, separated blocks, exports). Delete sessions from **History** to free space.

## Troubleshooting

| Problem | Fix |
|---|---|
| "GPU out of memory" | Another app or a second copy of this app is using the GPU: press **Stop** there. Or lower the chunk length. |
| Model greyed out | Its files are missing: run the download command shown in the hover text. |
| `401` / gated-repo error | Accept the model license on Hugging Face and check `HF_ACCESS_TOKEN` in `.env`. |
| YouTube link fails | Update yt-dlp (`uv lock --upgrade-package yt-dlp && uv sync`) and make sure Node.js 20+ is installed. |
| UI looks outdated after an update | Reload the page (the app tells the browser to re-check its files). |

## License

- **Code:** [MIT](LICENSE).
- **Model weights:** Meta's [SAM License](https://github.com/facebookresearch/sam-audio/blob/main/LICENSE), including FP16/FP8/GGUF conversions. The T5 text encoder is Apache-2.0.
- **YouTube:** you are responsible for complying with YouTube's terms and with copyright when processing videos.
