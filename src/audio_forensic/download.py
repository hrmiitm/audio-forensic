"""Download model weights and the audio.cpp runtime.

    uv run python -m audio_forensic.download          # small GGUF (BF16, Q8) + audio.cpp
    uv run python -m audio_forensic.download large    # large FP16 (PyTorch) + large Q8 GGUF, then build FP8
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from .config import AUDIOCPP_ASSET, AUDIOCPP_CLI, AUDIOCPP_DIR, AUDIOCPP_URL, GGUF_REPO, LARGE_FP8, LARGE_FP16, MODELS_DIR, ROOT

LARGE_FP16_REPO = "yaghmood/sam-audio-large-fp16-pytorch"  # community FP16 export of facebook/sam-audio-large


def fetch(repo: str, filename: str, dest: Path) -> Path:
    if dest.exists():
        print(f"[skip] {dest}")
        return dest
    load_dotenv(ROOT / ".env")
    print(f"[download] {repo}/{filename}")
    return Path(hf_hub_download(repo, filename, local_dir=dest.parent, token=os.environ.get("HF_ACCESS_TOKEN")))


def download_runtime() -> None:
    if AUDIOCPP_CLI.exists():
        print(f"[skip] {AUDIOCPP_CLI}")
        return
    AUDIOCPP_DIR.mkdir(parents=True, exist_ok=True)
    archive = AUDIOCPP_DIR / AUDIOCPP_ASSET
    print(f"[download] {AUDIOCPP_URL}")
    urllib.request.urlretrieve(AUDIOCPP_URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(AUDIOCPP_DIR, filter="data")
    archive.unlink()


def download_small() -> None:
    download_runtime()
    for f in ("sam-audio-small-bf16.gguf", "sam-audio-small-q8_0.gguf"):
        fetch(GGUF_REPO, f, MODELS_DIR / f)


def download_large() -> None:
    download_runtime()
    fetch(GGUF_REPO, "sam-audio-large-q8_0.gguf", MODELS_DIR / "sam-audio-large-q8_0.gguf")
    fetch(LARGE_FP16_REPO, "checkpoint.pt", LARGE_FP16)
    if not LARGE_FP8.exists():
        from .fp8 import convert

        print(f"[build] {LARGE_FP8.name} from the FP16 checkpoint")
        print(convert(LARGE_FP16, LARGE_FP8))


if __name__ == "__main__":
    download_large() if sys.argv[1:] == ["large"] else download_small()
