# Semantic Hybrid Recommender on Amazon Reviews

This project builds a semantic hybrid recommender for sparse, long-tail Amazon Reviews ranking. It starts with popularity and ID-based collaborative baselines, adds a compact Mini-DLRM-style interaction model, then injects frozen MiniLM product-text embeddings into the ranker. The goal is to test when semantic item priors help recommendation quality beyond user-item interaction signals.

This is a research-engineering portfolio project: compact, reproducible, and designed for comparison, not a full-scale production Meta DLRM or industrial retrieval stack.

## Key Results

Real-data experiment: Amazon Reviews 2023 `Movies_and_TV` subset.

| Setting | Value |
|---|---:|
| Users | 832 |
| Items | 2,215 |
| Positive interactions | 7,356 |
| Sparsity | 0.9960 |

| Model | Recall@10 | NDCG@10 |
|---|---:|---:|
| Mini-DLRM | 0.2188 | 0.1135 |
| Hybrid Mini-DLRM + Semantic | 0.2404 | 0.1347 |

Controlled multi-seed ablation at `epochs=5`, `lr=1e-3`, `emb_dim=32`, `train_negatives=4`:

| Model | Recall@10 mean | Recall@10 std |
|---|---:|---:|
| Mini-DLRM | 0.2264 | 0.0117 |
| Hybrid Mini-DLRM + Semantic | 0.2472 | 0.0049 |

The hybrid model improved overall ranking quality and showed lower seed variance in the sparse long-tail setting. In the earlier tiny dense setting with roughly 47 items, semantic embeddings did not help, which is the useful contrast: dataset regime matters.

## Project Highlights

- Sparse recommendation setup with explicit long-tail evaluation
- Staged baselines: popularity, Two-Tower, Mini-DLRM, Hybrid Mini-DLRM + Semantic
- Frozen semantic item embeddings from product title/category/description
- Leave-last-out sampled ranking with Recall@10, HitRate@10, and NDCG@10
- Multi-seed ablation runner for controlled comparisons
- Synthetic fallback so the notebook runs without local Amazon data
- Generated raw data, embeddings, checkpoints, and metrics are ignored by git

## Architecture

```text
Amazon Reviews interactions          Product metadata text
        |                                     |
        v                                     v
 user_id, item_id                      title/category/description
        |                                     |
        v                                     v
 collaborative embeddings        frozen MiniLM/e5/BGE embeddings
        |                                     |
        |                          semantic projection layer
        |                                     |
        +----------- Mini-DLRM-style feature interaction --------+
                                      |
                                      v
                              ranking logit / score
```

The semantic embedding table is frozen. Only the recommender embeddings, dense feature layers, semantic projection, interaction layer, and top MLP are trained.

## Quick Start

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the notebook with synthetic fallback data:

```bash
jupyter notebook notebooks/01_amazon_semantic_hybrid_recommender.ipynb
```

Build a local ignored Movies_and_TV subset:

```bash
python scripts/build_real_subset.py \
  --max-review-rows 500000 \
  --max-metadata-rows 1000000 \
  --target-items 2000 \
  --target-interactions 50000 \
  --selection-mode long_tail \
  --min-user-interactions 5 \
  --min-item-interactions 2
```

Run a small controlled ablation:

```bash
python scripts/run_ablation.py --quick
```

For real-data setup details, see [docs/RUN_REAL_DATA.md](docs/RUN_REAL_DATA.md).

## Repository Structure

```text
notebooks/   Main runnable experiment and reporting notebook
src/         Reusable data, sampling, metrics, model, training, and semantic modules
scripts/     Real-data subset builder, smoke checks, and ablation runner
data/        Local raw/processed/embedding files; ignored except .gitkeep files
results/     Local generated metrics and figures; ignored except sample outputs
```

## Models

| Model | Signal Used | Purpose |
|---|---|---|
| Popularity | train interaction counts | simple non-personalized baseline |
| Two-Tower | user/item ID embeddings | collaborative baseline |
| Mini-DLRM | user/item/category/dense feature interactions | compact DLRM-inspired ranker |
| Hybrid Mini-DLRM + Semantic | Mini-DLRM + frozen item text embeddings | semantic hybrid ranker for sparse items |

The project evaluates sampled candidate ranking, not full-catalog ANN retrieval over millions of items.

## Dataset Format

Review files may be CSV, JSON, JSONL, NDJSON, or parquet. Common Amazon Reviews aliases are normalized:

```text
user: reviewerID, user_id, user
item: parent_asin, asin, item_id
time: unixReviewTime, timestamp, time
rating: overall, rating
```

Optional metadata fields:

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

`parent_asin` is preferred as the canonical item id when present; otherwise the code falls back to `asin` or `item_id`.

## Experiment Findings

The dense 47-item regime was too small for semantic item representations to add much value. Collaborative repetition dominated the sampled candidate ranking problem.

The larger sparse long-tail regime was more informative. Mini-DLRM ranked head items strongly but was sharply head-biased; adding frozen MiniLM product-text embeddings improved Mini-DLRM overall and improved tail Recall@10/NDCG@10 relative to Mini-DLRM.

The multi-seed ablation showed that semantic priors improved both mean Recall@10 and stability at the best observed setting. This suggests semantic embeddings can act as a useful regularizing prior when item interactions are sparse and metadata is informative.

## Insights

- Semantic embeddings help more in sparse long-tail settings than in tiny dense item universes.
- DLRM-style interaction models can become strongly head-biased.
- Frozen semantic priors can improve seed stability and generalization.
- Recommendation quality depends heavily on dataset regime, metadata coverage, negative sampling, and feature richness.

## Evaluation Metrics

The experiments use leave-last-out sampled ranking and report:

- `Recall@10`
- `HitRate@10`
- `NDCG@10`

For leave-one-out ranking, Recall@K and HitRate@K are equivalent, but both are included for readability.

## Future Work

- Hard negative mining
- Popularity-aware negative sampling
- ANN retrieval with Faiss
- SASRec or BERT4Rec sequence modeling
- LLM reranking
- Multimodal product embeddings
- Fine-tuning semantic encoders on recommendation pairs

## References And Resources

- [Amazon Reviews 2023 dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023), McAuley Lab
- DLRM-style recommendation models with sparse embeddings, dense features, and feature interactions
- Semantic embedding models: MiniLM, e5, and BGE
- Long-tail recommendation research on sparse feedback and popularity skew
