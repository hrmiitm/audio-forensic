"""Weight-only FP8 for the large model's transformer (no FP8 release exists, so we build one).

Each big nn.Linear in the DiT keeps its weight as float8 e4m3 plus one FP16 scale per output row,
and is dequantised to the compute dtype right before the matmul. This halves the weight memory
(~5.6 GB -> ~2.8 GB for the large DiT); activations and everything else stay FP16.

    uv run python -m audio_forensic.fp8      # models/sam-audio-large-fp16 -> models/sam-audio-large-fp8.pt
"""

import torch
import torch.nn.functional as F

F8 = torch.float8_e4m3fn
F8_MAX = torch.finfo(F8).max  # 448
MIN_NUMEL = 1 << 20  # small layers stay FP16


class FP8Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight_fp8", torch.empty(out_features, in_features, dtype=F8, device="meta"))
        self.register_buffer("scale", torch.empty(out_features, 1, dtype=torch.float16, device="meta"))
        self.bias = torch.nn.Parameter(torch.empty(out_features, device="meta")) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_fp8.to(x.dtype) * self.scale.to(x.dtype)
        return F.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row absmax scaling into the e4m3 range."""
    w = w.float()
    scale = (w.abs().amax(dim=1, keepdim=True) / F8_MAX).clamp(min=1e-12)
    return (w / scale).to(F8), scale.to(torch.float16)


def linear_names(transformer: torch.nn.Module, prefix: str = "transformer.") -> list[str]:
    return [prefix + n for n, m in transformer.named_modules()
            if isinstance(m, torch.nn.Linear) and m.weight.numel() >= MIN_NUMEL]


def swap_linears(transformer: torch.nn.Module) -> None:
    """Replace the big Linear layers with empty FP8Linear shells (weights come from the FP8 checkpoint)."""
    for name in linear_names(transformer, prefix=""):
        parent_name, _, child = name.rpartition(".")
        parent = transformer.get_submodule(parent_name) if parent_name else transformer
        lin = getattr(parent, child)
        setattr(parent, child, FP8Linear(lin.in_features, lin.out_features, lin.bias is not None))


def convert(src, dst, config_repo: str = "facebook/sam-audio-large") -> dict:
    """FP16/FP32 checkpoint -> FP8 checkpoint (transformer Linears in FP8, everything else FP16)."""
    import json
    from pathlib import Path

    from accelerate import init_empty_weights
    from huggingface_hub import hf_hub_download
    from sam_audio.model.config import SAMAudioConfig
    from sam_audio.model.transformer import DiT

    cfg = SAMAudioConfig(**json.loads(Path(hf_hub_download(config_repo, "config.json")).read_text()))
    with init_empty_weights():
        names = set(linear_names(DiT(cfg.transformer)))
    sd = torch.load(src, map_location="cpu", weights_only=True, mmap=True)
    out, stats = {}, {"fp8_layers": 0, "fp8_params": 0, "fp16_params": 0}
    for k, v in sd.items():
        base = k.removesuffix(".weight")
        if k.endswith(".weight") and base in names:
            out[base + ".weight_fp8"], out[base + ".scale"] = quantize(v)
            stats["fp8_layers"] += 1
            stats["fp8_params"] += v.numel()
        else:
            out[k] = v.to(torch.float16) if v.is_floating_point() else v
            stats["fp16_params"] += v.numel()
    if stats["fp8_layers"] != len(names):
        raise RuntimeError(f"Expected {len(names)} transformer Linear weights, found {stats['fp8_layers']}")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    return stats


if __name__ == "__main__":
    from .config import LARGE_FP8, LARGE_FP16

    print(convert(LARGE_FP16, LARGE_FP8), "->", LARGE_FP8)
