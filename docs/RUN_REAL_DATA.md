# Run With Local Amazon Reviews Data

The notebook defaults to a synthetic demo dataset. To run on local Amazon Reviews files, keep the files under `data/raw/` or another local path and point the notebook variables at them.

Do not commit real Amazon Reviews data, processed datasets, generated embeddings, checkpoints, or result dumps. The repo `.gitignore` excludes those paths and artifact types by default.

## Expected Inputs

Review files can be CSV, JSON, JSONL, NDJSON, or parquet.

Common review column aliases are supported:

```text
user: reviewerID, user_id, user
item: parent_asin, asin, item_id
time: unixReviewTime, timestamp, time
rating: overall, rating
```

Metadata files are optional. Common item columns are:

```text
parent_asin or asin
title
main_category
categories
description
price
average_rating
rating_number
```

`parent_asin` is preferred when present; otherwise the code falls back to `asin` or `item_id`.

## Notebook Configuration

In `notebooks/01_amazon_semantic_hybrid_recommender.ipynb`, set:

```python
REVIEWS_PATH = "data/raw/your_reviews.jsonl"
METADATA_PATH = "data/raw/your_metadata.jsonl"
```

Leave both as `None` to use the synthetic fallback:

```python
REVIEWS_PATH = None
METADATA_PATH = None
```

For offline-safe notebook execution, keep:

```python
SEMANTIC_MODEL_NAME = "tfidf-svd"
```

For pretrained semantic embeddings, change it to one of:

```text
sentence-transformers/all-MiniLM-L6-v2
BAAI/bge-small-en-v1.5
intfloat/e5-small-v2
```

Generated embeddings are cached under `data/embeddings/` and ignored by git.

## Smoke Check A Real File Sample

Before running a full notebook job, inspect normalized columns on a small sample:

```bash
python scripts/smoke_real_data.py \
  --reviews data/raw/your_reviews.jsonl \
  --metadata data/raw/your_metadata.jsonl \
  --rows 5
```

Metadata is optional:

```bash
python scripts/smoke_real_data.py --reviews data/raw/your_reviews.jsonl --rows 5
```

The script prints raw review columns, normalized review columns, normalized metadata columns when provided, and a few normalized rows after applying the implicit-positive filter.

## Full Run Checklist

1. Put local data files somewhere ignored by git, usually `data/raw/`.
2. Run the smoke script and confirm `user_id`, `item_id`, `rating`, and `timestamp` are normalized correctly.
3. Set `REVIEWS_PATH` and optional `METADATA_PATH` in the notebook.
4. Start with `NOTEBOOK_EPOCHS = 1` and a small `NOTEBOOK_MAX_INTERACTIONS`.
5. Increase `EPOCHS`, `MAX_INTERACTIONS`, and semantic model quality after the small run succeeds.
