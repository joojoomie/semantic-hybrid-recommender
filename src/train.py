from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .metrics import evaluate_leave_one_out
from .utils import get_device


EMB_DIM = 32
BATCH_SIZE = 1024
EPOCHS = 5
LR = 1e-3
TRAIN_NEGATIVES = 4
EVAL_NEGATIVES = 99
MAX_INTERACTIONS = 100_000
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5
K = 10


@dataclass
class ItemFeatureStore:
    category_by_item: np.ndarray
    dense_by_item: np.ndarray


class InteractionDataset(Dataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        item_features: ItemFeatureStore | None = None,
    ) -> None:
        self.user_idx = samples["user_idx"].to_numpy(dtype=np.int64)
        self.item_idx = samples["item_idx"].to_numpy(dtype=np.int64)
        self.labels = samples["label"].to_numpy(dtype=np.float32)
        self.item_features = item_features

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item_idx = int(self.item_idx[idx])
        row = {
            "user_idx": torch.tensor(self.user_idx[idx], dtype=torch.long),
            "item_idx": torch.tensor(item_idx, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
        if self.item_features is not None:
            row["category_idx"] = torch.tensor(
                int(self.item_features.category_by_item[item_idx]),
                dtype=torch.long,
            )
            row["dense_features"] = torch.tensor(
                self.item_features.dense_by_item[item_idx],
                dtype=torch.float32,
            )
        return row


def make_dataloader(
    samples: pd.DataFrame,
    item_features: ItemFeatureStore | None = None,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        InteractionDataset(samples, item_features),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _forward_model(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if "category_idx" in batch and "dense_features" in batch:
        return model(
            batch["user_idx"],
            batch["item_idx"],
            batch["category_idx"],
            batch["dense_features"],
        )
    return model(batch["user_idx"], batch["item_idx"])


def train_model(
    model: nn.Module,
    train_samples: pd.DataFrame,
    eval_candidates: list[dict[str, object]] | None = None,
    item_features: ItemFeatureStore | None = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    k: int = K,
    device: torch.device | None = None,
) -> nn.Module:
    device = device or get_device()
    model.to(device)
    loader = make_dataloader(train_samples, item_features, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = _forward_model(model, batch)
            loss = loss_fn(logits, batch["label"])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        message = f"epoch={epoch} train_loss={np.mean(losses):.4f}"
        if eval_candidates is not None:
            val_metrics = evaluate_leave_one_out(
                model,
                eval_candidates,
                k=k,
                scorer=make_torch_scorer(model, item_features, device),
            )
            message += f" val_Recall@{k}={val_metrics[f'Recall@{k}']:.4f}"
            message += f" val_NDCG@{k}={val_metrics[f'NDCG@{k}']:.4f}"
        print(message)
    return model


def make_torch_scorer(
    model: nn.Module,
    item_features: ItemFeatureStore | None = None,
    device: torch.device | None = None,
) -> Any:
    device = device or next(model.parameters()).device

    def scorer(user_idx: int, items: list[int]) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            user_tensor = torch.full((len(items),), int(user_idx), dtype=torch.long, device=device)
            item_tensor = torch.tensor(items, dtype=torch.long, device=device)
            if item_features is not None:
                category_tensor = torch.tensor(
                    item_features.category_by_item[items],
                    dtype=torch.long,
                    device=device,
                )
                dense_tensor = torch.tensor(
                    item_features.dense_by_item[items],
                    dtype=torch.float32,
                    device=device,
                )
                logits = model(user_tensor, item_tensor, category_tensor, dense_tensor)
            else:
                logits = model(user_tensor, item_tensor)
        return logits.detach().cpu().numpy()

    return scorer


def make_popularity_scorer(train_df: pd.DataFrame, num_items: int) -> Any:
    counts = np.zeros(num_items, dtype=np.float32)
    item_counts = train_df["item_idx"].value_counts()
    counts[item_counts.index.to_numpy(dtype=np.int64)] = item_counts.to_numpy(dtype=np.float32)

    def scorer(user_idx: int, items: list[int]) -> np.ndarray:
        del user_idx
        return counts[np.asarray(items, dtype=np.int64)]

    return scorer


def recommend_top_k(
    scorer: Any,
    user_idx: int,
    candidate_items: list[int],
    known_items: set[int] | None = None,
    k: int = 10,
) -> list[int]:
    known_items = known_items or set()
    candidates = [item for item in candidate_items if item not in known_items]
    scores = np.asarray(scorer(user_idx, candidates), dtype=np.float32)
    order = np.argsort(-scores)[:k]
    return [int(candidates[i]) for i in order]
