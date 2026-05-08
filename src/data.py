from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


USER_COLS = ("user_id", "reviewerID", "reviewer_id", "user", "customer_id")
ITEM_COLS = ("parent_asin", "asin", "item_id", "product_id")
RATING_COLS = ("rating", "overall", "stars")
TIME_COLS = ("timestamp", "unixReviewTime", "time", "review_time", "reviewTime")
CATEGORY_COLS = ("main_category", "category", "categories")
DENSE_COLS = ("price", "average_rating", "rating_number")


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffixes = [suffix.lower() for suffix in path.suffixes]
    suffix = suffixes[-1] if suffixes else ""
    if suffix == ".gz" and len(suffixes) >= 2:
        suffix = suffixes[-2]
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_reviews(path: str) -> pd.DataFrame:
    """Load Amazon review interactions from CSV, JSON, JSONL, or parquet."""
    return _read_table(path)


def load_metadata(path: str | None) -> pd.DataFrame | None:
    """Load optional Amazon product metadata."""
    if not path:
        return None
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    return _read_table(path_obj)


def _first_existing(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    lower_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def _normalize_item_column(df: pd.DataFrame) -> pd.DataFrame:
    item_col = _first_existing(df.columns, ITEM_COLS)
    if item_col is None:
        raise ValueError(f"Could not find an item id column. Expected one of {ITEM_COLS}.")
    df = df.copy()
    df["item_id"] = df[item_col].astype(str)
    return df


def normalize_columns(
    reviews: pd.DataFrame, metadata: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Normalize review and metadata column names used by the pipeline."""
    reviews = reviews.copy()

    user_col = _first_existing(reviews.columns, USER_COLS)
    item_col = _first_existing(reviews.columns, ITEM_COLS)
    rating_col = _first_existing(reviews.columns, RATING_COLS)
    time_col = _first_existing(reviews.columns, TIME_COLS)

    if user_col is None:
        raise ValueError(f"Could not find a user id column. Expected one of {USER_COLS}.")
    if item_col is None:
        raise ValueError(f"Could not find an item id column. Expected one of {ITEM_COLS}.")

    reviews["user_id"] = reviews[user_col].astype(str)
    reviews["item_id"] = reviews[item_col].astype(str)
    if rating_col is not None:
        reviews["rating"] = pd.to_numeric(reviews[rating_col], errors="coerce").fillna(0.0)
    else:
        reviews["rating"] = 1.0

    if time_col is not None:
        reviews["timestamp"] = pd.to_numeric(reviews[time_col], errors="coerce")
        if reviews["timestamp"].isna().all():
            reviews["timestamp"] = pd.to_datetime(reviews[time_col], errors="coerce").astype("int64") // 10**9
    else:
        reviews["timestamp"] = np.arange(len(reviews), dtype=np.int64)
    fallback_time = pd.Series(np.arange(len(reviews)), index=reviews.index)
    reviews["timestamp"] = reviews["timestamp"].fillna(fallback_time).astype(np.int64)

    reviews = reviews[reviews["rating"] >= 4].copy()
    reviews["label"] = 1.0

    metadata_norm = None
    if metadata is not None:
        metadata_norm = _normalize_item_column(metadata)
        metadata_norm = metadata_norm.drop_duplicates("item_id").copy()

    return reviews.reset_index(drop=True), metadata_norm


def filter_interactions(
    df: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> pd.DataFrame:
    """Iteratively keep users and items with enough positive interactions."""
    filtered = df.copy()
    while True:
        before = len(filtered)
        user_counts = filtered["user_id"].value_counts()
        item_counts = filtered["item_id"].value_counts()
        filtered = filtered[
            filtered["user_id"].isin(user_counts[user_counts >= min_user_interactions].index)
            & filtered["item_id"].isin(item_counts[item_counts >= min_item_interactions].index)
        ].copy()
        if len(filtered) == before:
            break
    return filtered.reset_index(drop=True)


def cap_interactions(df: pd.DataFrame, max_interactions: int | None, seed: int = 42) -> pd.DataFrame:
    if max_interactions is None or len(df) <= max_interactions:
        return df.reset_index(drop=True)
    sampled = df.sample(n=max_interactions, random_state=seed)
    return sampled.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leave-last-out split per user: last=test, second last=validation."""
    ordered = df.sort_values(["user_id", "timestamp", "item_id"]).copy()
    ordered["rank_from_end"] = ordered.groupby("user_id").cumcount(ascending=False)
    test_df = ordered[ordered["rank_from_end"] == 0].drop(columns="rank_from_end")
    val_df = ordered[ordered["rank_from_end"] == 1].drop(columns="rank_from_end")
    train_df = ordered[ordered["rank_from_end"] >= 2].drop(columns="rank_from_end")
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_id_mappings(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[dict[str, int], dict[str, int]]:
    """Build stable user/item mappings from all split rows."""
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    users = sorted(all_df["user_id"].astype(str).unique())
    items = sorted(all_df["item_id"].astype(str).unique())
    return {u: i for i, u in enumerate(users)}, {it: i for i, it in enumerate(items)}


def apply_id_mappings(
    df: pd.DataFrame, user_mapping: dict[str, int], item_mapping: dict[str, int]
) -> pd.DataFrame:
    mapped = df.copy()
    mapped["user_idx"] = mapped["user_id"].map(user_mapping).astype(int)
    mapped["item_idx"] = mapped["item_id"].map(item_mapping).astype(int)
    return mapped


def _stringify(value: Any) -> str:
    if isinstance(value, list):
        return " > ".join(map(str, value))
    if isinstance(value, dict):
        return " ".join(map(str, value.values()))
    if pd.isna(value):
        return ""
    return str(value)


def build_category_features(
    item_mapping: dict[str, int], metadata: pd.DataFrame | None = None
) -> tuple[np.ndarray, dict[str, int]]:
    """Return category index per item_idx and the category mapping."""
    item_categories = np.zeros(len(item_mapping), dtype=np.int64)
    category_mapping = {"unknown": 0}
    if metadata is None or metadata.empty:
        return item_categories, category_mapping

    meta = metadata.drop_duplicates("item_id").set_index("item_id")
    category_col = _first_existing(meta.columns, CATEGORY_COLS) or "main_category"
    for item_id, item_idx in item_mapping.items():
        category = "unknown"
        if item_id in meta.index and category_col in meta.columns:
            category = _stringify(meta.at[item_id, category_col]).strip() or "unknown"
        if category not in category_mapping:
            category_mapping[category] = len(category_mapping)
        item_categories[item_idx] = category_mapping[category]
    return item_categories, category_mapping


def build_dense_features(
    item_mapping: dict[str, int], metadata: pd.DataFrame | None = None
) -> tuple[np.ndarray, list[str]]:
    """Build scaled numeric item features with robust missing-column handling."""
    feature_names = list(DENSE_COLS)
    dense = np.zeros((len(item_mapping), len(feature_names)), dtype=np.float32)
    if metadata is None or metadata.empty:
        return dense, feature_names

    meta = metadata.drop_duplicates("item_id").set_index("item_id")
    for item_id, item_idx in item_mapping.items():
        if item_id not in meta.index:
            continue
        for col_idx, col in enumerate(feature_names):
            if col in meta.columns:
                raw_value = meta.at[item_id, col]
                if isinstance(raw_value, str):
                    raw_value = raw_value.replace("$", "").replace(",", "")
                dense[item_idx, col_idx] = pd.to_numeric(raw_value, errors="coerce")

    dense = np.nan_to_num(dense, nan=0.0, posinf=0.0, neginf=0.0)
    if len(dense) > 1 and np.any(dense.std(axis=0) > 0):
        dense = StandardScaler().fit_transform(dense).astype(np.float32)
    return dense.astype(np.float32), feature_names


def generate_synthetic_amazon_data(
    num_users: int = 50,
    num_items: int = 100,
    min_interactions_per_user: int = 10,
    max_interactions_per_user: int = 18,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a tiny Amazon-like dataset for end-to-end notebook smoke tests."""
    rng = np.random.default_rng(seed)
    categories = ["Electronics", "Books", "Home", "Beauty", "Sports"]
    adjectives = ["Wireless", "Compact", "Premium", "Classic", "Smart", "Portable"]
    nouns = ["Headphones", "Cookbook", "Lamp", "Serum", "Backpack", "Speaker", "Planner"]

    item_rows = []
    for item_idx in range(num_items):
        category = categories[item_idx % len(categories)]
        title = f"{adjectives[item_idx % len(adjectives)]} {nouns[item_idx % len(nouns)]} {item_idx}"
        item_rows.append(
            {
                "item_id": f"item_{item_idx:04d}",
                "title": title,
                "main_category": category,
                "categories": [category, f"{category} Accessories"],
                "description": f"{title} for everyday use with reliable quality and practical features.",
                "price": round(float(rng.uniform(8, 250)), 2),
                "average_rating": round(float(rng.uniform(3.7, 4.9)), 2),
                "rating_number": int(rng.integers(10, 5000)),
            }
        )
    metadata = pd.DataFrame(item_rows)

    interactions = []
    timestamp = 1_700_000_000
    popularity = np.linspace(1.0, 0.2, num_items)
    popularity = popularity / popularity.sum()
    for user_idx in range(num_users):
        n_interactions = int(rng.integers(min_interactions_per_user, max_interactions_per_user + 1))
        preferred_category = categories[user_idx % len(categories)]
        candidate_items = metadata.index[metadata["main_category"] == preferred_category].to_numpy()
        sampled = set(rng.choice(num_items, size=max(2, n_interactions // 2), replace=False, p=popularity))
        sampled.update(rng.choice(candidate_items, size=min(len(candidate_items), n_interactions), replace=False))
        for item_idx in list(sampled)[:n_interactions]:
            interactions.append(
                {
                    "user_id": f"user_{user_idx:03d}",
                    "item_id": f"item_{item_idx:04d}",
                    "rating": int(rng.choice([4, 5], p=[0.35, 0.65])),
                    "timestamp": timestamp,
                }
            )
            timestamp += int(rng.integers(60, 86_400))
    reviews = pd.DataFrame(interactions)
    return reviews, metadata


def merge_item_features(
    df: pd.DataFrame, item_category: np.ndarray, item_dense: np.ndarray
) -> pd.DataFrame:
    enriched = df.copy()
    enriched["category_idx"] = item_category[enriched["item_idx"].to_numpy()]
    return enriched
