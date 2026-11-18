"""Discover compute devices and read live CPU / GPU usage."""

import os
import re
import subprocess
from functools import cache

import psutil

from .config import AUDIOCPP_CLI, MODEL_SPECS, weights_ready


@cache
def _nvml():
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def nvidia_gpus() -> list[dict]:
    nv = _nvml()
    if nv is None:
        return []
    out = []
    for i in range(nv.nvmlDeviceGetCount()):
        h = nv.nvmlDeviceGetHandleByIndex(i)
        name = nv.nvmlDeviceGetName(h)
        out.append({"index": i, "name": name.decode() if isinstance(name, bytes) else name,
                    "mem_total": nv.nvmlDeviceGetMemoryInfo(h).total / 2**30})
    return out


@cache
def vulkan_gpus() -> list[dict]:
    """Vulkan GPUs in the order audio.cpp numbers them (software/CPU devices skipped)."""
    try:
        txt = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    found = re.findall(r"deviceType\s*=\s*PHYSICAL_DEVICE_TYPE_(\w+)\s+deviceName\s*=\s*(.+)", txt)
    return [{"index": i, "type": t, "name": n.strip()} for i, (t, n) in enumerate((t, n) for t, n in found if t != "CPU")]


def _nv_index(name: str) -> int | None:
    """Match a GPU name to its NVML index so the UI can show its usage."""
    return next((g["index"] for g in nvidia_gpus() if g["name"].lower() in name.lower() or name.lower() in g["name"].lower()), None)


def _cpu() -> dict:
    return {"id": "cpu", "name": "CPU", "detail": f"{os.cpu_count()} threads · slowest", "nv": None}


def list_devices(model: str) -> list[dict]:
    if MODEL_SPECS[model]["kind"] == "torch":
        gpus = [{"id": f"cuda:{g['index']}", "name": g["name"], "detail": f"CUDA · {g['mem_total']:.0f} GB", "nv": g["index"]}
                for g in nvidia_gpus()]
    else:
        gpus = [{"id": f"vulkan:{g['index']}", "name": g["name"],
                 "detail": "Vulkan · " + ("shared memory · slow" if g["type"] == "INTEGRATED_GPU" else "dedicated"),
                 "nv": _nv_index(g["name"])} for g in vulkan_gpus()]
        gpus.sort(key=lambda d: d["nv"] is None)  # dedicated NVIDIA first
    return gpus + [_cpu()]


def default_device(model: str) -> str:
    return list_devices(model)[0]["id"]


def _reason(model_id: str, spec: dict) -> str | None:
    if spec["kind"] == "gguf" and not AUDIOCPP_CLI.exists():
        return "audio.cpp missing: run `uv run python -m audio_forensic.download`"
    if not weights_ready(spec):
        return f"weights missing: run `uv run python -m audio_forensic.download{' large' if 'large' in model_id else ''}`"
    return None


def models() -> list[dict]:
    """Models the UI can offer, with availability and devices."""
    out = []
    for mid, spec in MODEL_SPECS.items():
        reason = _reason(mid, spec)
        note = "Downloads the checkpoint on first use" if isinstance(spec.get("weights"), tuple) and not _hf_cached(spec["weights"]) else None
        out.append({"id": mid, "label": spec["label"], "meta": spec["meta"], "note": note, "available": reason is None,
                    "reason": reason, "chunk": spec["chunk"], "devices": list_devices(mid)})
    return out


def _hf_cached(weights: tuple) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(*weights), str)
    except Exception:
        return False


def stats() -> dict:
    mem = psutil.virtual_memory()
    out = {"cpu": psutil.cpu_percent(interval=None), "ram_used": mem.used / 2**30, "ram_total": mem.total / 2**30, "gpus": []}
    nv = _nvml()
    for g in nvidia_gpus():
        h = nv.nvmlDeviceGetHandleByIndex(g["index"])
        m = nv.nvmlDeviceGetMemoryInfo(h)
        out["gpus"].append({"index": g["index"], "name": g["name"], "util": nv.nvmlDeviceGetUtilizationRates(h).gpu,
                            "mem_used": m.used / 2**30, "mem_total": m.total / 2**30})
    return out
