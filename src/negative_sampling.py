from __future__ import annotations

import numpy as np
import pandas as pd


def build_user_positive_items(df: pd.DataFrame) -> dict[int, set[int]]:
    positives: dict[int, set[int]] = {}
    for user_idx, item_idx in df[["user_idx", "item_idx"]].itertuples(index=False):
        positives.setdefault(int(user_idx), set()).add(int(item_idx))
    return positives


def sample_negatives(
    user_idx: int,
    user_pos_items: dict[int, set[int]],
    num_items: int,
    num_negatives: int,
    rng: np.random.Generator,
) -> list[int]:
    positives = user_pos_items.get(int(user_idx), set())
    available = num_items - len(positives)
    if available <= 0:
        return []
    target = min(num_negatives, available)
    negatives: set[int] = set()
    while len(negatives) < target:
        candidate = int(rng.integers(0, num_items))
        if candidate not in positives:
            negatives.add(candidate)
    return list(negatives)


def build_training_samples(
    train_df: pd.DataFrame,
    user_pos_items: dict[int, set[int]],
    num_items: int,
    num_negatives: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for user_idx, item_idx in train_df[["user_idx", "item_idx"]].itertuples(index=False):
        rows.append({"user_idx": int(user_idx), "item_idx": int(item_idx), "label": 1.0})
        for neg_item_idx in sample_negatives(int(user_idx), user_pos_items, num_items, num_negatives, rng):
            rows.append({"user_idx": int(user_idx), "item_idx": neg_item_idx, "label": 0.0})
    return pd.DataFrame(rows)


def build_eval_candidates(
    eval_df: pd.DataFrame,
    user_pos_items: dict[int, set[int]],
    num_items: int,
    num_negatives: int = 99,
    seed: int = 43,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    candidates = []
    for user_idx, true_item_idx in eval_df[["user_idx", "item_idx"]].itertuples(index=False):
        user_idx = int(user_idx)
        true_item_idx = int(true_item_idx)
        negatives = sample_negatives(user_idx, user_pos_items, num_items, num_negatives, rng)
        item_candidates = [true_item_idx] + negatives
        candidates.append(
            {
                "user_idx": user_idx,
                "true_item": true_item_idx,
                "items": item_candidates,
            }
        )
    return candidates
