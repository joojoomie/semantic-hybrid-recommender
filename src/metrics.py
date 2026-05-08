from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch


def recall_at_k(ranked_items: list[int] | np.ndarray, true_item: int, k: int) -> float:
    return float(int(true_item) in list(ranked_items)[:k])


def hit_rate_at_k(ranked_items: list[int] | np.ndarray, true_item: int, k: int) -> float:
    return recall_at_k(ranked_items, true_item, k)


def ndcg_at_k(ranked_items: list[int] | np.ndarray, true_item: int, k: int) -> float:
    top_k = list(ranked_items)[:k]
    if int(true_item) not in top_k:
        return 0.0
    rank = top_k.index(int(true_item)) + 1
    return float(1.0 / np.log2(rank + 1))


def _rank_with_callable(
    scorer: Callable[[int, list[int]], np.ndarray | list[float]],
    user_idx: int,
    items: list[int],
) -> list[int]:
    scores = np.asarray(scorer(user_idx, items), dtype=np.float32)
    order = np.argsort(-scores)
    return [int(items[i]) for i in order]


def evaluate_leave_one_out(
    model: Any,
    eval_candidates: list[dict[str, object]],
    k: int = 10,
    scorer: Callable[[int, list[int]], np.ndarray | list[float]] | None = None,
) -> dict[str, float]:
    """Evaluate sampled leave-one-out candidates using a callable scorer or model object."""
    recalls = []
    hits = []
    ndcgs = []
    for row in eval_candidates:
        user_idx = int(row["user_idx"])
        true_item = int(row["true_item"])
        items = [int(item) for item in row["items"]]
        if scorer is not None:
            ranked_items = _rank_with_callable(scorer, user_idx, items)
        elif callable(model) and not isinstance(model, torch.nn.Module):
            ranked_items = _rank_with_callable(model, user_idx, items)
        else:
            raise ValueError("Provide a callable scorer for evaluate_leave_one_out.")
        recalls.append(recall_at_k(ranked_items, true_item, k))
        hits.append(hit_rate_at_k(ranked_items, true_item, k))
        ndcgs.append(ndcg_at_k(ranked_items, true_item, k))

    return {
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"HitRate@{k}": float(np.mean(hits)) if hits else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


def metrics_to_frame(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in results.items():
        rows.append({"Model": model_name, **metrics})
    return pd.DataFrame(rows)
