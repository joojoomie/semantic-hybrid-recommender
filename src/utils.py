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


def get_device() -> torch.device:
    """Return the best available PyTorch device with CPU fallback."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))
