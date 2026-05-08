from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable

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

    if len(filtered) > args.max_output_reviews:
        item_counts = filtered["item_id"].value_counts()
        selected_items_for_subset = []
        selected_total = 0
        for item_id, count in item_counts.items():
            selected_items_for_subset.append(item_id)
            selected_total += int(count)
            if selected_total >= args.max_output_reviews * 2:
                break
        selected_df = filtered[filtered["item_id"].isin(selected_items_for_subset)].copy()
        selected_df = filter_interactions(
            selected_df,
            min_user_interactions=args.min_user_interactions,
            min_item_interactions=args.min_item_interactions,
        )
    else:
        selected_df = filtered.copy()

    if selected_df.empty:
        raise RuntimeError(
            "Subset selection became empty after preserving 5-core constraints. "
            "Increase --max-output-reviews."
        )
    selected_df = selected_df.sort_values(["user_id", "timestamp", "item_id"])
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
    parser.add_argument("--max-output-reviews", type=int, default=2_000)
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--min-item-interactions", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    build_subset(parse_args())
