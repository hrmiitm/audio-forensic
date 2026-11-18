"""Separation backends with one interface:

    backend(wave, text, anchors, visual, opts, cancel) -> (target, residual)

Each call handles one window of 48 kHz mono audio; the pipeline does the chunking.
`visual` is None or (VisualRef, start_s, end_s) for that window.
  - TorchBackend: official checkpoint, FP16 on CUDA (FP32 on CPU), model stays loaded.
  - GGUFBackend:  BF16 / Q8 GGUF through the audio.cpp CLI (Vulkan or CPU).
"""

import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path


from .config import AUDIOCPP_CLI, AUDIOCPP_SPEC, MODEL_SPECS, MODELS_DIR, ROOT, SR
from .media import fit, read_wav, write_wav


class Cancelled(Exception):
    pass


def check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


class VisualRef:
    """A masked image or video (everything except the sound source blacked out)."""

    VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.is_video = self.path.suffix.lower() in self.VIDEO_EXTS
        self._decoder = self._image = None

    def frames(self, a: float, b: float):
        """uint8 [T, 3, H, W] frames for seconds [a, b) (for the PyTorch backend)."""
        if not self.is_video:
            if self._image is None:
                from torchvision.io import ImageReadMode, read_image

                self._image = read_image(str(self.path), mode=ImageReadMode.RGB).unsqueeze(0)
            return self._image
        from torchcodec.decoders import VideoDecoder

        self._decoder = self._decoder or VideoDecoder(str(self.path), dimension_order="NCHW")
        end = min(b, self._decoder.metadata.duration_seconds or b)
        start = min(a, max(0.0, end - 0.5))
        return self._decoder.get_frames_played_in_range(start, end).data

    def file(self, a: float, b: float, tmp: Path) -> Path:
        """Path covering seconds [a, b) (for audio.cpp, which reads files)."""
        if not self.is_video:
            return self.path
        out = tmp / "visual.mp4"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", str(self.path),
             "-an", "-c:v", "libx264", "-preset", "ultrafast", str(out)],
            check=True,
        )
        return out


def _place(module, device, dtype) -> None:
    """Move to device and cast floating tensors to dtype (FP8 weights stay FP8)."""
    import torch

    keep = (torch.float8_e4m3fn,)
    module._apply(lambda t: t.to(device, dtype if t.is_floating_point() and t.dtype not in keep else t.dtype))


class TorchBackend:
    def __init__(self, spec: dict, device: str = "cuda:0", vision: bool = False):
        import torch
        from accelerate import cpu_offload, init_empty_weights
        from dotenv import load_dotenv
        from huggingface_hub import hf_hub_download
        from sam_audio import SAMAudio, SAMAudioProcessor
        from sam_audio.model.config import SAMAudioConfig
        from sam_audio.model.text_encoder import T5TextEncoder

        from . import fp8

        load_dotenv(ROOT / ".env")
        if token := os.environ.get("HF_ACCESS_TOKEN"):
            os.environ["HF_TOKEN"] = token
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        self.torch, self.device, self.note = torch, torch.device(device), None
        self.cuda = self.device.type == "cuda"
        self.dtype = torch.float16 if self.cuda else getattr(torch, spec.get("cpu_dtype", "float32"))
        config = json.loads(Path(hf_hub_download(spec["config"], "config.json")).read_text())
        # Rankers and span predictor are separate multi-GB models; not used here.
        config.update(text_ranker=None, visual_ranker=None, span_predictor=None)
        config["vision_encoder"]["batch_size"] = 16  # encode frames 16 at a time (default 300 overflows 6 GB)
        cfg = SAMAudioConfig(**config)

        # Build without allocating weights, then attach the checkpoint's tensors straight from the
        # memory-mapped file: a 3B model never needs a second full copy in RAM.
        with init_empty_weights(include_buffers=False):
            model = SAMAudio(cfg)
        model.text_encoder = T5TextEncoder(cfg.text_encoder)  # T5 isn't in the checkpoint: load it for real
        weights = spec["weights"]
        path = hf_hub_download(*weights) if isinstance(weights, tuple) else weights
        if not Path(path).exists():
            raise RuntimeError(f"{Path(path).name} missing: run `uv run python -m audio_forensic.download large`")
        sd = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not vision:  # ~1 GB saved when no visual reference is used
            model.vision_encoder.model = None
            sd = {k: v for k, v in sd.items() if not k.startswith("vision_encoder.model.")}
        else:  # the vision RoPE helper is a plain object (not a submodule): rebuild its computed tables
            for m in model.vision_encoder.modules():
                for v in vars(m).values():
                    if type(v).__name__ == "Rope2D":
                        v.init_tensors()
                        v.grid_size = None
        if spec.get("fp8"):
            fp8.swap_linears(model.transformer)
        missing, unexpected = torch.nn.Module.load_state_dict(model, sd, strict=False, assign=True)
        missing = [k for k in missing if not k.startswith("text_encoder.")]
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint doesn't match the model: missing {missing[:3]}, unexpected {unexpected[:3]}")

        # If the weights can't fit next to the activations, keep the transformer in RAM and stream
        # each layer to the GPU as it runs (accelerate offload): slow, but it works on 6 GB.
        size = sum(t.numel() * (1 if t.dtype == fp8.F8 else 2) for t in model.state_dict().values() if t.is_floating_point())
        offload = self.cuda and size > torch.cuda.mem_get_info(self.device)[0] - 1.5 * 2**30
        for name, child in model.named_children():
            if not (offload and name == "transformer"):
                _place(child, self.device, self.dtype)
        if offload:
            _place(model.transformer, "cpu", self.dtype)
            cpu_offload(model.transformer, execution_device=self.device)
            self.note = "weights streamed from RAM"
        self.model = model.eval()
        self.processor = SAMAudioProcessor.from_pretrained(spec["config"])
        if self.cuda:
            torch.cuda.empty_cache()

    def seed(self, seed: int) -> None:
        if seed >= 0:
            self.torch.manual_seed(seed)

    def __call__(self, wave, text, anchors=None, visual=None, opts=None, cancel=None):
        torch, n = self.torch, len(wave)
        check(cancel)
        x = fit(wave, max(n, SR))  # the model needs at least 1 s
        batch = self.processor(
            audios=[torch.from_numpy(x).unsqueeze(0)],
            descriptions=[text],
            anchors=[anchors] if anchors else None,
            masked_videos=[visual[0].frames(*visual[1:])] if visual else None,
        ).to(self.device)
        batch.audios = batch.audios.to(self.dtype)
        steps = int((opts or {}).get("steps", 16))
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=self.dtype, enabled=self.cuda):
            res = self.model.separate(batch, ode_opt={"method": "midpoint", "options": {"step_size": 1 / steps}})
        return tuple(fit(w[0].float().cpu().numpy(), n) for w in (res.target, res.residual))

    def peak_vram_mib(self) -> float | None:
        return self.torch.cuda.max_memory_allocated(self.device) / 2**20 if self.cuda else None

    def reset_peak(self) -> None:
        if self.cuda:
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def release(self) -> None:
        del self.model
        if self.cuda:
            self.torch.cuda.empty_cache()


class GGUFBackend:
    note = None

    def __init__(self, spec: dict, device: str = "vulkan:1"):
        self.model = MODELS_DIR / spec["file"]
        if not self.model.exists() or not AUDIOCPP_CLI.exists():
            raise RuntimeError("GGUF model or audio.cpp missing: run `uv run python -m audio_forensic.download`")
        kind, _, index = device.partition(":")
        self.backend = ["--backend", "cpu"] if kind == "cpu" else ["--backend", "vulkan", "--device", index or "0"]
        self._seed = 42

    def seed(self, seed: int) -> None:
        self._seed = seed

    def __call__(self, wave, text, anchors=None, visual=None, opts=None, cancel=None):
        opts, n = opts or {}, len(wave)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cmd = [
                str(AUDIOCPP_CLI), "--task", "s2s", "--family", "sam_audio", "--model", str(self.model),
                *self.backend, "--audio", str(write_wav(tmp / "in.wav", fit(wave, max(n, SR)))),
                "--text", text, "--seed", str(self._seed), "--out-dir", str(tmp),
                "--request-option", f"num_inference_steps={int(opts.get('steps', 16))}",
            ]
            if anchors:
                cmd += ["--request-option", "anchors=" + json.dumps(anchors)]
            if visual:
                ref = visual[0]
                cmd += ["--request-option", f"reference_{'video' if ref.is_video else 'image'}_file={ref.file(*visual[1:], tmp)}"]
            if opts.get("bounded", True):
                cmd += ["--model-spec-override", str(AUDIOCPP_SPEC), "--session-option", "sam_audio.memory_bounded=true"]
            with open(tmp / "run.log", "w") as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    if cancel is not None and cancel.is_set():
                        proc.kill()
                        raise Cancelled()
                    time.sleep(0.1)
            if proc.returncode:
                raise RuntimeError("audio.cpp failed: " + (tmp / "run.log").read_text()[-400:])
            return tuple(fit(read_wav(tmp / f"{k}.wav"), n) for k in ("target", "residual"))

    def peak_vram_mib(self):
        return None

    def reset_peak(self):
        pass

    def release(self):
        pass


_cache: dict = {"key": None, "backend": None}
_lock = threading.RLock()


def get_backend(model: str, device: str, vision: bool = False):
    """Return a loaded backend, reusing the previous one when nothing changed."""
    key = (model, device, vision and MODEL_SPECS[model]["kind"] == "torch")
    with _lock:
        if _cache["key"] != key:
            release()
            spec = MODEL_SPECS[model]
            _cache["backend"] = TorchBackend(spec, device, vision) if spec["kind"] == "torch" else GGUFBackend(spec, device)
            _cache["key"] = key
        return _cache["backend"]


def release() -> None:
    """Unload the current model (frees GPU memory)."""
    with _lock:
        if _cache["backend"] is not None:
            _cache["backend"].release()
        _cache.update(key=None, backend=None)


def loaded() -> str | None:
    """Model currently in memory (for the UI)."""
    return _cache["key"][0] if _cache["key"] else None
