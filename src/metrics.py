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


def evaluate_leave_one_out_by_true_item_groups(
    eval_candidates: list[dict[str, object]],
    head_items: set[int],
    tail_items: set[int],
    k: int = 10,
    scorer: Callable[[int, list[int]], np.ndarray | list[float]] | None = None,
) -> dict[str, float | int]:
    """Evaluate leave-one-out metrics split by whether the true item is head or tail."""

    def group_metrics(prefix: str, group_items: set[int]) -> dict[str, float | int]:
        group_candidates = [
            row for row in eval_candidates if int(row["true_item"]) in group_items
        ]
        if not group_candidates:
            return {
                f"{prefix}Recall@{k}": np.nan,
                f"{prefix}HitRate@{k}": np.nan,
                f"{prefix}NDCG@{k}": np.nan,
                f"Num{prefix}EvalUsers": 0,
            }
        metrics = evaluate_leave_one_out(None, group_candidates, k=k, scorer=scorer)
        return {
            f"{prefix}Recall@{k}": metrics[f"Recall@{k}"],
            f"{prefix}HitRate@{k}": metrics[f"HitRate@{k}"],
            f"{prefix}NDCG@{k}": metrics[f"NDCG@{k}"],
            f"Num{prefix}EvalUsers": len(group_candidates),
        }

    return {
        **group_metrics("Head", head_items),
        **group_metrics("Tail", tail_items),
    }


def metrics_to_frame(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in results.items():
        rows.append({"Model": model_name, **metrics})
    return pd.DataFrame(rows)
