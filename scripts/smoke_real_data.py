from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import normalize_columns


def read_sample(path: str | Path, rows: int) -> pd.DataFrame:
    path = Path(path)
    suffixes = [suffix.lower() for suffix in path.suffixes]
    suffix = suffixes[-1] if suffixes else ""
    if suffix == ".gz" and len(suffixes) >= 2:
        suffix = suffixes[-2]
    if suffix == ".csv":
        return pd.read_csv(path, nrows=rows)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True, nrows=rows)
    if suffix == ".json":
        try:
            return pd.read_json(path, lines=True, nrows=rows)
        except ValueError:
            return pd.read_json(path).head(rows)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path).head(rows)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print normalized columns from a small local Amazon Reviews sample."
    )
    parser.add_argument("--reviews", required=True, help="Path to a local reviews file.")
    parser.add_argument("--metadata", default=None, help="Optional path to a local metadata file.")
    parser.add_argument("--rows", type=int, default=5, help="Number of rows to inspect.")
    args = parser.parse_args()

    reviews_raw = read_sample(args.reviews, args.rows)
    metadata_raw = read_sample(args.metadata, args.rows) if args.metadata else None
    reviews, metadata = normalize_columns(reviews_raw, metadata_raw)

    print("Raw review columns:")
    print(list(reviews_raw.columns))
    print("\nNormalized review columns:")
    print(list(reviews.columns))
    print("\nNormalized review sample:")
    print(reviews.head(args.rows).to_string(index=False))

    if metadata is not None:
        print("\nNormalized metadata columns:")
        print(list(metadata.columns))
        print("\nNormalized metadata sample:")
        print(metadata.head(args.rows).to_string(index=False))


if __name__ == "__main__":
    main()
