# Semantic Hybrid Recommender on Amazon Reviews

This project builds a semantic hybrid recommendation system on Amazon Reviews. It starts from popularity and ID-based collaborative filtering, extends to a compact DLRM-style feature interaction model, and then incorporates frozen pretrained semantic item embeddings from product text. The hybrid model combines behavioral similarity from user-item interactions with semantic similarity from product content, which is especially useful for sparse, long-tail, and cold-start items.

This project implements a compact DLRM-inspired ranking model for educational and experimental purposes, not a full-scale production DLRM system.

## Project Motivation

Traditional collaborative and DLRM-style recommenders learn item representations mainly from user-item interaction signals. This can work well for popular items, but sparse, long-tail, and cold-start items often lack enough interactions to learn stable embeddings.

Pretrained semantic encoders such as MiniLM, BGE, and e5 can represent products from title, category, and description text before many interactions exist. The hybrid model combines:

- collaborative signal from user-item behavior
- feature interactions from a Mini-DLRM-style ranking model
- semantic signal from frozen product text embeddings

## Project Evolution

The notebook follows a staged experiment:

```text
Popularity baseline
→ ID-based Two-Tower
→ Mini-DLRM-style ranking model
→ Hybrid Mini-DLRM + frozen semantic item embeddings
```

Each stage adds one modeling capability so the final comparison is easy to interpret.

## Dataset Format

The notebook expects Amazon Reviews-style interactions and optional metadata.

Review fields:

```text
user_id
parent_asin or asin
rating
timestamp
title/text optional
```

Metadata fields:

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

`parent_asin` is used as the canonical item id when available. Otherwise the code falls back to `asin` or `item_id`.

If no local files are provided, the notebook uses a tiny synthetic Amazon-like dataset for smoke testing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run:

```bash
jupyter notebook notebooks/01_amazon_semantic_hybrid_recommender.ipynb
```

## How To Use Real Data

In the notebook setup cell, set:

```python
REVIEWS_PATH = "data/raw/your_reviews.jsonl"
METADATA_PATH = "data/raw/your_metadata.jsonl"
```

Leave both as `None` to use the synthetic fallback.

See [docs/RUN_REAL_DATA.md](docs/RUN_REAL_DATA.md) for a real-data checklist and a small sample-normalization smoke script.

Large data, processed files, embeddings, model checkpoints, and generated results are ignored by git.

## Models

### Popularity

Ranks candidate items by training-set interaction count.

### Two-Tower

Learns user and item ID embeddings, then scores a pair with a dot product.

### Mini-DLRM

A compact DLRM-inspired ranker using user, item, category, and dense item features. It computes pairwise dot-product feature interactions and feeds them to a top MLP.

### Hybrid Mini-DLRM + Semantic

Adds a frozen semantic item embedding table created from product text. A trainable projection maps semantic vectors into the recommender latent dimension so they can participate in feature interactions.

## Retrieval vs Ranking

This project focuses on ranking-style recommendation experiments on sampled candidate sets. It does not implement full-scale industrial approximate nearest neighbor retrieval over millions of items. The goal is to compare collaborative, feature-interaction, and semantic-hybrid ranking signals.

## Evaluation Metrics

The notebook evaluates sampled leave-one-out ranking with:

- `Recall@10`
- `HitRate@10`
- `NDCG@10`

For leave-one-out evaluation, Recall@K and HitRate@K are equivalent, but both are reported for clarity.

## Cold-start And Long-tail

Collaborative embeddings require user-item interactions to become meaningful. Long-tail or new items often have too few interactions to learn stable embeddings. A semantic encoder can place items into a meaningful content space based on title, category, and description before sufficient user feedback exists.

The notebook includes a long-tail analysis that compares Recall@10 on head and tail test items.

## Future Work

- hard negative mining
- popularity-aware negative sampling
- ANN retrieval with Faiss
- SASRec or BERT4Rec sequence modeling
- LLM reranking
- multimodal product embeddings
- fine-tuning semantic encoders on recommendation pairs
