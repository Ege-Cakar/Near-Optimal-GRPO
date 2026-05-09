from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    device = device.lower()
    if device == "auto":
        return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested --device mps, but MPS is unavailable.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested --device cuda, but CUDA is unavailable.")
    return device


def save_run_config(cfg: dict, outdir: str | Path) -> None:
    with open(Path(outdir) / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def binomial_ci(p: float, n: int) -> tuple[float, float]:
    half = 1.96 * np.sqrt(max(p * (1.0 - p), 0.0) / max(n, 1))
    return max(0.0, p - half), min(1.0, p + half)


def save_dual(fig, path_no_ext: str | Path) -> None:
    path = Path(path_no_ext)
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
