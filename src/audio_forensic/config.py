"""Paths, the model registry and tuning constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
VENDOR_DIR = ROOT / "vendor"
DATA_DIR = ROOT / "data"
INPUTS_DIR = ROOT / "sample-input-files"
OUTPUTS_DIR = ROOT / "sample-output-file"

SR = 48_000  # SAM-Audio works on 48 kHz mono
PEAKS_PER_SEC = 50  # waveform resolution sent to the UI
FIRST_BLOCK_S, BLOCK_S = 10.0, 30.0  # audio is stored in blocks: short first one for a fast start
OVERLAP_S = 0.25  # crossfade between model windows

GGUF_REPO = "audio-cpp/SAM-Audio-GGUF"
LARGE_FP16 = MODELS_DIR / "sam-audio-large-fp16" / "checkpoint.pt"  # yaghmood/sam-audio-large-fp16-pytorch
LARGE_FP8 = MODELS_DIR / "sam-audio-large-fp8.pt"  # built locally from LARGE_FP16 (see fp8.py)

# kind "torch": sam_audio package. weights = (hf repo, file) downloaded on first use, or a local file.
# kind "gguf": audio.cpp CLI with a GGUF file in MODELS_DIR.
# Order = UI order: PyTorch models first, then GGUF, each from smallest to largest.
MODEL_SPECS = {
    "torch-fp16": {"kind": "torch", "label": "SAM-Audio small · FP16 (PyTorch)", "config": "facebook/sam-audio-small",
                   "weights": ("facebook/sam-audio-small", "checkpoint.pt"), "chunk": 5.0,
                   "meta": ["0.6B params", "FP16 on GPU / FP32 on CPU", "~2.5 GB GPU memory", "PyTorch"]},
    "large-fp8": {"kind": "torch", "label": "SAM-Audio large · FP8 (PyTorch)", "config": "facebook/sam-audio-large",
                  "weights": LARGE_FP8, "fp8": True, "chunk": 5.0, "cpu_dtype": "bfloat16",
                  "meta": ["3B params", "FP8 weights (built locally)", "~4.3 GB GPU memory", "~2× slower than real time on RTX 4050", "PyTorch"]},
    "large-fp16": {"kind": "torch", "label": "SAM-Audio large · FP16 (PyTorch)", "config": "facebook/sam-audio-large",
                   "weights": LARGE_FP16, "chunk": 20.0, "cpu_dtype": "bfloat16",
                   "meta": ["3B params", "FP16", "Weights streamed from RAM on a 6 GB GPU", "~4× slower than real time", "PyTorch"]},
    "q8": {"kind": "gguf", "label": "SAM-Audio small · Q8 (GGUF)", "file": "sam-audio-small-q8_0.gguf", "chunk": 30.0,
           "meta": ["0.6B params", "Q8", "2.4 GB file", "audio.cpp"]},
    "bf16": {"kind": "gguf", "label": "SAM-Audio small · BF16 (GGUF)", "file": "sam-audio-small-bf16.gguf", "chunk": 30.0,
             "meta": ["0.6B params", "BF16", "2.8 GB file", "audio.cpp"]},
    "large-q8": {"kind": "gguf", "label": "SAM-Audio large · Q8 (GGUF)", "file": "sam-audio-large-q8_0.gguf", "chunk": 30.0,
                 "meta": ["3B params", "Q8", "5.0 GB file: may not fit a 6 GB GPU, use CPU", "audio.cpp"]},
}
CHUNK_DEFAULTS = {k: v["chunk"] for k, v in MODEL_SPECS.items()}
MODELS = {k: v["file"] for k, v in MODEL_SPECS.items() if v["kind"] == "gguf"}  # GGUF files to download

AUDIOCPP_VERSION = "v0.9.5"
AUDIOCPP_ASSET = f"audio-{AUDIOCPP_VERSION}-bin-ubuntu-x64-vulkan.tar.gz"
AUDIOCPP_URL = f"https://github.com/0xShug0/audio.cpp/releases/download/{AUDIOCPP_VERSION}/{AUDIOCPP_ASSET}"
AUDIOCPP_DIR = VENDOR_DIR / "audiocpp"
AUDIOCPP_CLI = AUDIOCPP_DIR / "audiocpp_cli"
AUDIOCPP_SPEC = AUDIOCPP_DIR / "model_specs" / "sam_audio.json"


def model_path(variant: str) -> Path:
    return MODELS_DIR / MODELS[variant]


def weights_ready(spec: dict) -> bool:
    """Local weights present (HF-hosted ones download on first use)."""
    if spec["kind"] == "gguf":
        return (MODELS_DIR / spec["file"]).exists()
    return isinstance(spec["weights"], tuple) or Path(spec["weights"]).exists()
