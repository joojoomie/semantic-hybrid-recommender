from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import filter_interactions, normalize_columns


MOVIES_REVIEWS_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "review_categories/Movies_and_TV.jsonl.gz"
)
MOVIES_METADATA_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "meta_categories/meta_Movies_and_TV.jsonl.gz"
)


def iter_gzip_jsonl(url: str, max_rows: int | None = None) -> Iterable[dict]:
    with urllib.request.urlopen(url, timeout=60) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            for idx, raw_line in enumerate(gz):
                if max_rows is not None and idx >= max_rows:
                    break
                if raw_line.strip():
                    yield json.loads(raw_line)


def canonical_item_id(row: dict) -> str:
    return str(row.get("parent_asin") or row.get("asin") or row.get("item_id") or "")


def order_items_by_selection_mode(item_counts, selection_mode: str, target_items: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    if selection_mode == "popular":
        return list(item_counts.sort_values(ascending=False).index)
    if selection_mode == "random":
        items = item_counts.index.to_numpy(copy=True)
        rng.shuffle(items)
        return list(items)
    if selection_mode != "long_tail":
        raise ValueError(f"Unknown selection mode: {selection_mode}")

    sorted_items = item_counts.sort_values(ascending=True).index.to_numpy()
    num_buckets = min(10, max(1, len(sorted_items)))
    buckets = [bucket.copy() for bucket in np.array_split(sorted_items, num_buckets) if len(bucket)]
    for bucket in buckets:
        rng.shuffle(bucket)

    # Tail-heavy but still mixed: cycle tail-to-head so sparse items enter early,
    # while popular items are still available to preserve enough 5-core users.
    ordered: list[str] = []
    max_bucket_len = max(len(bucket) for bucket in buckets)
    for position in range(max_bucket_len):
        for bucket in buckets:
            if position < len(bucket):
                ordered.append(str(bucket[position]))

    if target_items > 0 and len(ordered) > target_items * 3:
        return ordered[: target_items * 3]
    return ordered


def select_subset(filtered, args):
    if args.selection_mode == "full":
        return filtered.sort_values(["user_id", "timestamp", "item_id"]).copy()

    item_counts = filtered["item_id"].value_counts()
    target_items = min(args.target_items, len(item_counts)) if args.target_items else len(item_counts)
    ordered_items = order_items_by_selection_mode(
        item_counts,
        selection_mode=args.selection_mode,
        target_items=target_items,
        seed=args.seed,
    )

    selected_items: set[str] = set()
    best_subset = filtered.iloc[0:0].copy()
    batch_size = max(100, target_items // 5)
    for item_id in ordered_items:
        selected_items.add(str(item_id))
        if len(selected_items) % batch_size != 0 and len(selected_items) < len(ordered_items):
            continue
        candidate = filtered[filtered["item_id"].isin(selected_items)].copy()
        candidate = filter_interactions(
            candidate,
            min_user_interactions=args.min_user_interactions,
            min_item_interactions=args.min_item_interactions,
        )
        if candidate["item_id"].nunique() > best_subset["item_id"].nunique():
            best_subset = candidate
        if candidate["item_id"].nunique() >= target_items:
            best_subset = candidate
            break

    if best_subset.empty:
        raise RuntimeError(
            "Subset selection became empty after preserving core constraints. "
            "Increase --max-review-rows, --target-items, or lower interaction thresholds."
        )

    if args.target_interactions and len(best_subset) > args.target_interactions:
        # Keep a diverse item universe by sampling per item, then re-apply core filters.
        per_item_quota = max(args.min_item_interactions, int(np.ceil(args.target_interactions / best_subset["item_id"].nunique())))
        capped = (
            best_subset.sort_values(["item_id", "timestamp"])
            .groupby("item_id", group_keys=False)
            .head(per_item_quota)
            .copy()
        )
        capped = filter_interactions(
            capped,
            min_user_interactions=args.min_user_interactions,
            min_item_interactions=args.min_item_interactions,
        )
        if not capped.empty and capped["item_id"].nunique() >= min(target_items, best_subset["item_id"].nunique()):
            best_subset = capped

    return best_subset.sort_values(["user_id", "timestamp", "item_id"]).copy()


def build_subset(args: argparse.Namespace) -> None:
    reviews_out = Path(args.reviews_out)
    metadata_out = Path(args.metadata_out)
    reviews_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)

    positive_rows = []
    print(f"Streaming reviews from {args.reviews_url}")
    for row in iter_gzip_jsonl(args.reviews_url, max_rows=args.max_review_rows):
        if float(row.get("rating", row.get("overall", 0)) or 0) >= 4:
            positive_rows.append(row)

    if not positive_rows:
        raise RuntimeError("No positive review rows found in the streamed sample.")

    import pandas as pd

    reviews_norm, _ = normalize_columns(pd.DataFrame(positive_rows), None)
    filtered = filter_interactions(
        reviews_norm,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
    )
    if filtered.empty:
        raise RuntimeError(
            "The streamed sample did not contain enough repeated users/items. "
            "Increase --max-review-rows or lower interaction thresholds."
        )

    selected_df = select_subset(filtered, args)
    selected_rows = selected_df.to_dict(orient="records")
    selected_items = {str(item_id) for item_id in selected_df["item_id"].unique()}
    selected_items.discard("")
    if not selected_items:
        raise RuntimeError("No item IDs selected for metadata matching.")

    print(f"Writing {len(selected_rows)} filtered reviews to {reviews_out}")
    with reviews_out.open("w", encoding="utf-8") as fp:
        for row in selected_rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Streaming metadata from {args.metadata_url}")
    written_meta = 0
    with metadata_out.open("w", encoding="utf-8") as fp:
        for row in iter_gzip_jsonl(args.metadata_url, max_rows=args.max_metadata_rows):
            item_id = canonical_item_id(row)
            if item_id in selected_items:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                written_meta += 1
                if written_meta >= len(selected_items):
                    break

    user_counts = Counter(str(row["user_id"]) for row in selected_rows)
    item_counts = Counter(str(row["item_id"]) for row in selected_rows)
    print(
        "Subset stats: "
        f"users={len(user_counts)} items={len(item_counts)} reviews={len(selected_rows)} metadata={written_meta}"
    )
    print(
        "Selection config: "
        f"mode={args.selection_mode} target_items={args.target_items} "
        f"target_interactions={args.target_interactions} "
        f"min_user={args.min_user_interactions} min_item={args.min_item_interactions}"
    )
    print("These files are under data/raw/ and are ignored by git.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small ignored Movies_and_TV subset from Amazon Reviews 2023."
    )
    parser.add_argument("--reviews-url", default=MOVIES_REVIEWS_URL)
    parser.add_argument("--metadata-url", default=MOVIES_METADATA_URL)
    parser.add_argument("--reviews-out", default="data/raw/movies_tv_reviews_subset.jsonl")
    parser.add_argument("--metadata-out", default="data/raw/movies_tv_metadata_subset.jsonl")
    parser.add_argument("--max-review-rows", type=int, default=250_000)
    parser.add_argument("--max-metadata-rows", type=int, default=1_000_000)
    parser.add_argument("--target-interactions", type=int, default=50_000)
    parser.add_argument("--target-items", type=int, default=2_000)
    parser.add_argument("--selection-mode", choices=["long_tail", "popular", "random", "full"], default="long_tail")
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--min-item-interactions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    build_subset(parse_args())
