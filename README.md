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

Final tuned sampled-ranking benchmark:

| Model | Setting | Recall@10 | HitRate@10 | NDCG@10 |
|---|---|---:|---:|---:|
| Popularity | sparse long-tail baseline | 0.2139 | 0.2139 | 0.1136 |
| Two-Tower | sparse long-tail baseline | 0.1094 | 0.1094 | 0.0479 |
| Mini-DLRM | tuned negatives=16 | 0.2592 | 0.2592 | 0.1445 |
| Hybrid MiniLM | epochs=5, emb_dim=32, negatives=12 | 0.2596 | 0.2596 | 0.1440 |
| Hybrid BGE | epochs=8, emb_dim=64, negatives=12 | ~0.275 | ~0.275 | ~0.137 |

Hybrid MiniLM is the best balanced/stable configuration. Hybrid BGE achieves the strongest observed Recall@10 after encoder-specific tuning. Mini-DLRM also improves substantially with stronger negative sampling, showing that optimization quality matters. Because leave-last-out evaluation has one positive test item per user, HitRate@10 and Recall@10 are effectively identical in this setup.

These results are from sampled candidate ranking, not full-catalog retrieval.

Current recommended configuration from tuning:

```text
Hybrid Mini-DLRM + Semantic
epochs = 5
lr = 1e-3
emb_dim = 32
train_negatives = 12
```

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

## Hyperparameter Tuning Findings

Epoch tuning compared 5 versus 10 epochs with `lr=1e-3`, `emb_dim=32`, and `train_negatives=4` across seeds 42, 123, and 2026.

| Model | Epochs | Recall@10 mean | Recall@10 std | NDCG@10 mean | NDCG@10 std |
|---|---:|---:|---:|---:|---:|
| Mini-DLRM | 5 | 0.2264 | 0.0117 | 0.1255 | 0.0055 |
| Mini-DLRM | 10 | 0.2336 | 0.0049 | 0.1263 | 0.0068 |
| Hybrid Mini-DLRM + Semantic | 5 | 0.2472 | 0.0049 | 0.1318 | 0.0034 |
| Hybrid Mini-DLRM + Semantic | 10 | 0.2392 | 0.0167 | 0.1254 | 0.0063 |

Hybrid performs best at 5 epochs. Extending to 10 epochs reduces mean Recall/NDCG and increases variance, suggesting the semantic prior helps early but longer training may overfit sparse collaborative signals or increase head/popularity bias. Mini-DLRM improves slightly with more epochs, which is consistent with a purely collaborative model needing more optimization to fit sparse interaction patterns.

Negative sampling tuning compared `train_negatives=8,12,16` with `epochs=5`, `lr=1e-3`, and `emb_dim=32`.

| Model | Train negatives | Recall@10 mean | Recall@10 std | NDCG@10 mean | NDCG@10 std |
|---|---:|---:|---:|---:|---:|
| Mini-DLRM | 8 | 0.2444 | 0.0218 | 0.1366 | 0.0099 |
| Mini-DLRM | 12 | 0.2488 | 0.0156 | 0.1390 | 0.0099 |
| Mini-DLRM | 16 | 0.2592 | 0.0163 | 0.1445 | 0.0105 |
| Hybrid Mini-DLRM + Semantic | 8 | 0.2576 | 0.0066 | 0.1411 | 0.0037 |
| Hybrid Mini-DLRM + Semantic | 12 | 0.2596 | 0.0073 | 0.1440 | 0.0007 |
| Hybrid Mini-DLRM + Semantic | 16 | 0.2604 | 0.0050 | 0.1429 | 0.0035 |

Increasing negatives from 4 to 8/12/16 improves ranking quality. Hybrid remains more stable across seeds than Mini-DLRM, and `train_negatives=12` is the best balanced hybrid setting: strong Recall@10, best NDCG@10, and very low NDCG variance. At 16 negatives, Mini-DLRM nearly catches up in mean performance but still has much higher variance.

Final tuning takeaway: semantic embeddings improve sparse recommendation not only by raising mean ranking quality, but also by reducing seed sensitivity and stabilizing optimization under harder negative sampling.

## Final Findings

The final experimental setting is a sparse Amazon Reviews 2023 `Movies_and_TV` subset with 832 users, 2,215 items, 7,356 interactions, and 0.9960 sparsity. Evaluation uses leave-last-out ranking with 1 held-out positive and 99 sampled negatives per user. Because each test case has a single positive item, `HitRate@10` and `Recall@10` are effectively identical in this setup.

The project progressed through five stages:

1. Collaborative baselines: popularity, Two-Tower, and Mini-DLRM established the ranking baseline. On the initial tiny 47-item dense subset, Two-Tower and Mini-DLRM were similar, and semantic priors had little room to help.
2. Sparse long-tail regime: expanding to 2,215 items made tail-item behavior central. Hybrid semantic recommendation began outperforming purely collaborative models.
3. Negative sampling: increasing train negatives from 4 to 8/12/16 consistently improved ranking quality, especially NDCG. Semantic hybrid models remained more stable across seeds under harder negative sampling.
4. Embedding capacity: tuning embedding dimensions from 16 to 96 exposed a capacity/stability tradeoff. Larger dimensions increased capacity but could increase variance or over-parameterize sparse interactions.
5. Semantic encoder comparison: MiniLM, BGE-small, and e5-small behaved differently. MiniLM was the strongest balanced/stable baseline, BGE-small achieved the strongest tuned Recall, and e5-small was weaker in this recommendation setting.

Compact final benchmark:

| Model | Best observed setting | Recall@10 | HitRate@10 | NDCG@10 |
|---|---|---:|---:|---:|
| Popularity | sparse long-tail baseline | 0.2139 | 0.2139 | 0.1136 |
| Two-Tower | sparse long-tail baseline | 0.1094 | 0.1094 | 0.0479 |
| Mini-DLRM | tuned negatives=16 | 0.2592 | 0.2592 | 0.1445 |
| Hybrid MiniLM | epochs=5, emb_dim=32, negatives=12 | 0.2596 | 0.2596 | 0.1440 |
| Hybrid BGE | epochs=8, emb_dim=64, negatives=12 | ~0.275 | ~0.275 | ~0.137 |

Best balanced/stable configuration:

```text
Model: Hybrid Mini-DLRM + Semantic
Semantic encoder: sentence-transformers/all-MiniLM-L6-v2
epochs = 5
lr = 1e-3
emb_dim = 32
train_negatives = 12
```

Best retrieval-oriented configuration:

```text
Model: Hybrid Mini-DLRM + Semantic
Semantic encoder: BAAI/bge-small-en-v1.5
epochs = 8
lr = 1e-3
emb_dim = 64
train_negatives = 12
Recall@10 mean ~= 0.275
NDCG@10 mean ~= 0.137
std ~= 0.012
```

The main conclusion is that semantic item embeddings become increasingly valuable under sparse long-tail recommendation settings. They improve tail-item discrimination, reduce seed sensitivity, and provide a stabilizing inductive bias when negative sampling becomes harder. The strongest general retrieval encoder is not automatically the best recommender semantic prior; alignment between semantic embedding geometry and recommender optimization matters.

## Lessons Learned / Key Insights

- Semantic embeddings help more in sparse long-tail settings than in tiny dense item universes.
- DLRM-style interaction models can become strongly head-biased.
- Frozen semantic priors can improve seed stability and generalization.
- Harder negative sampling improves ranking quality, with `train_negatives=12` the best balanced hybrid setting observed.
- Semantic encoder choice materially affects hybrid recommender performance.
- Sparse recommendation has a real capacity/stability tradeoff; larger embeddings are not always better.
- Recommendation quality depends heavily on dataset regime, metadata coverage, negative sampling, and feature richness.

Most important insight: the project evolved from a simple "DLRM + semantic embeddings" implementation into a systematic analysis of semantic priors under sparse long-tail recommendation optimization.

## Evaluation Metrics

The experiments use leave-last-out sampled ranking and report:

- `Recall@10`
- `HitRate@10`
- `NDCG@10`

For leave-one-out ranking, Recall@K and HitRate@K are equivalent, but both are included for readability.

## Limitations And Future Work

Limitations:

- Evaluation uses sampled candidate ranking, not full-catalog retrieval.
- The project does not include sequential user modeling.
- The project does not include graph-based collaborative modeling.
- The project does not separate retrieval and reranking stages.
- Encoder-specific tuning was informative but not exhaustive.

Future work:

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
- [Negative Sampling in Recommendation: A Survey and Future Directions](https://dl.acm.org/doi/10.1145/3793855), ACM Digital Library
