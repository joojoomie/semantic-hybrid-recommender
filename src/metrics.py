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


def _true_items(row: dict[str, object]) -> list[int]:
    if "true_items" in row:
        return [int(item) for item in row["true_items"]]  # type: ignore[union-attr]
    return [int(row["true_item"])]


def recall_at_k_multi(ranked_items: list[int] | np.ndarray, true_items: list[int], k: int) -> float:
    if not true_items:
        return 0.0
    top_k = set(int(item) for item in list(ranked_items)[:k])
    hits = sum(1 for item in true_items if int(item) in top_k)
    return float(hits / len(true_items))


def hit_rate_at_k_multi(ranked_items: list[int] | np.ndarray, true_items: list[int], k: int) -> float:
    top_k = set(int(item) for item in list(ranked_items)[:k])
    return float(any(int(item) in top_k for item in true_items))


def ndcg_at_k_multi(ranked_items: list[int] | np.ndarray, true_items: list[int], k: int) -> float:
    if not true_items:
        return 0.0
    true_set = set(int(item) for item in true_items)
    dcg = 0.0
    for rank, item in enumerate(list(ranked_items)[:k], start=1):
        if int(item) in true_set:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(true_set), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


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
        true_items = _true_items(row)
        items = [int(item) for item in row["items"]]
        if scorer is not None:
            ranked_items = _rank_with_callable(scorer, user_idx, items)
        elif callable(model) and not isinstance(model, torch.nn.Module):
            ranked_items = _rank_with_callable(model, user_idx, items)
        else:
            raise ValueError("Provide a callable scorer for evaluate_leave_one_out.")
        recalls.append(recall_at_k_multi(ranked_items, true_items, k))
        hits.append(hit_rate_at_k_multi(ranked_items, true_items, k))
        ndcgs.append(ndcg_at_k_multi(ranked_items, true_items, k))

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
        recalls = []
        hits = []
        ndcgs = []
        positive_count = 0
        user_count = 0
        for row in eval_candidates:
            user_idx = int(row["user_idx"])
            positives = [item for item in _true_items(row) if item in group_items]
            if not positives:
                continue
            items = [int(item) for item in row["items"]]
            if scorer is None:
                raise ValueError("Provide a callable scorer for grouped leave-one-out evaluation.")
            ranked_items = _rank_with_callable(scorer, user_idx, items)
            recalls.append(recall_at_k_multi(ranked_items, positives, k))
            hits.append(hit_rate_at_k_multi(ranked_items, positives, k))
            ndcgs.append(ndcg_at_k_multi(ranked_items, positives, k))
            positive_count += len(positives)
            user_count += 1
        if positive_count == 0:
            return {
                f"{prefix}Recall@{k}": np.nan,
                f"{prefix}HitRate@{k}": np.nan,
                f"{prefix}NDCG@{k}": np.nan,
                f"Num{prefix}EvalUsers": 0,
                f"Num{prefix}EvalPositives": 0,
            }
        return {
            f"{prefix}Recall@{k}": float(np.mean(recalls)),
            f"{prefix}HitRate@{k}": float(np.mean(hits)),
            f"{prefix}NDCG@{k}": float(np.mean(ndcgs)),
            f"Num{prefix}EvalUsers": user_count,
            f"Num{prefix}EvalPositives": positive_count,
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
