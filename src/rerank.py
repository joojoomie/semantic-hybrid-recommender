from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


ScoreFn = Callable[[int, list[int]], np.ndarray | list[float]]


def compute_item_train_counts(train_df: pd.DataFrame, num_items: int) -> np.ndarray:
    counts = np.zeros(num_items, dtype=np.float32)
    item_counts = train_df["item_idx"].value_counts()
    counts[item_counts.index.to_numpy(dtype=np.int64)] = item_counts.to_numpy(dtype=np.float32)
    return counts


def compute_cold_start_boost(
    counts: np.ndarray,
    items: list[int] | np.ndarray,
    mode: str = "none",
    threshold: int = 2,
) -> np.ndarray:
    item_counts = counts[np.asarray(items, dtype=np.int64)]
    if mode == "none":
        return np.zeros_like(item_counts, dtype=np.float32)
    if mode == "threshold":
        return (item_counts <= float(threshold)).astype(np.float32)
    if mode == "inverse_popularity":
        return (1.0 / np.sqrt(item_counts + 1.0)).astype(np.float32)
    raise ValueError(f"Unsupported cold-start boost mode: {mode}")


def make_cold_start_boost_scorer(
    base_scorer: ScoreFn,
    item_train_counts: np.ndarray,
    alpha: float,
    mode: str,
    threshold: int = 2,
) -> ScoreFn:
    def scorer(user_idx: int, items: list[int]) -> np.ndarray:
        base_scores = np.asarray(base_scorer(user_idx, items), dtype=np.float32)
        boost = compute_cold_start_boost(item_train_counts, items, mode=mode, threshold=threshold)
        return base_scores + float(alpha) * boost

    return scorer


def evaluate_ranking_exposure(
    scorer: ScoreFn,
    eval_candidates: list[dict[str, object]],
    item_train_counts: np.ndarray,
    tail_items: set[int],
    k: int = 10,
) -> dict[str, float]:
    tail_exposures = []
    avg_popularities = []
    for row in eval_candidates:
        user_idx = int(row["user_idx"])
        items = [int(item) for item in row["items"]]
        if not items:
            continue
        scores = np.asarray(scorer(user_idx, items), dtype=np.float32)
        top_indices = np.argsort(-scores)[:k]
        top_items = [items[int(idx)] for idx in top_indices]
        if not top_items:
            continue
        tail_exposures.append(np.mean([1.0 if item in tail_items else 0.0 for item in top_items]))
        avg_popularities.append(float(np.mean(item_train_counts[np.asarray(top_items, dtype=np.int64)])))

    return {
        f"TailExposure@{k}": float(np.mean(tail_exposures)) if tail_exposures else 0.0,
        f"AvgTrainPopularity@{k}": float(np.mean(avg_popularities)) if avg_popularities else 0.0,
    }
