from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import get_device, seed_everything


def main() -> None:
    seed_everything(42)
    print(f"torch_version={torch.__version__}")
    print(f"mps_built={torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False}")
    print(f"mps_available={torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
    print(f"cuda_available={torch.cuda.is_available()}")

    device = get_device("auto")
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1)).to(device)
    x = torch.randn(16, 4, device=device)
    y = torch.randn(16, 1, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
    print(f"tiny_forward_backward=ok loss={float(loss.detach().cpu()):.6f}")


if __name__ == "__main__":
    main()
