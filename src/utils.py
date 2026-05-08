from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int = 42, torch_num_threads: int = 1) -> None:
    """Set common random seeds for reproducible notebook runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str | None = "auto", verbose: bool = True) -> torch.device:
    """Return a PyTorch device, preferring Apple Silicon MPS for local Mac runs."""
    requested = (preferred or "auto").lower()
    if requested not in {"auto", "mps", "cuda", "cpu"}:
        raise ValueError("preferred device must be one of: auto, mps, cuda, cpu")

    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is not available in this PyTorch environment.")
        device = torch.device("mps")
    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch environment.")
        device = torch.device("cuda")
    elif requested == "cpu":
        device = torch.device("cpu")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if verbose:
        print(f"Selected device: {device}")
    return device


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))
