# Semantic Hybrid Recommender on Amazon Reviews

## Overview

This project studies sparse, long-tail recommendation on Amazon Reviews 2023 using sampled ranking experiments. It compares simple and neural baselines: Popularity, ID-based Two-Tower, and a compact Mini-DLRM-style ranking model.

The main model is a semantic Hybrid Mini-DLRM: a Mini-DLRM ranker augmented with frozen `sentence-transformers/all-MiniLM-L6-v2` item-text embeddings built from product title, category, and description. This is a research-engineering portfolio project for analyzing semantic priors under sparse recommendation, not a production recommender, full Meta DLRM implementation, or full-catalog retrieval system.

## Key Results

Main benchmark: Amazon Reviews 2023 `Movies_and_TV`, leave-last-out sampled ranking with 1 held-out positive item and 99 sampled non-interacted negatives per user.

| Setting | Value |
|---|---:|
| Users | 13,509 |
| Items | 20,896 |
| Positive interactions | 138,919 |
| Sparsity | 0.99951 |
| Minimum user interactions | 5 |
| Minimum item interactions | 2 |
| Eval protocol | 1 positive + 99 negatives |

| Model | Recall@10 | NDCG@10 | HeadRecall@10 | TailRecall@10 | TailNDCG@10 | TailExposure@10 | AvgTrainPopularity@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Two-Tower | 0.363424 | 0.216993 | 0.700602 | 0.003596 | 0.001116 | 0.019172 | 22.715645 |
| Popularity | 0.383152 | 0.230059 | 0.742042 | 0.000153 | 0.000046 | 0.000596 | 23.318332 |
| Mini-DLRM | 0.413835 | 0.242458 | 0.665974 | 0.144759 | 0.061567 | 0.394333 | 14.313709 |
| Hybrid Mini-DLRM + Semantic | 0.465727 | 0.280997 | 0.745698 | 0.166947 | 0.070298 | 0.348845 | 14.923451 |
| Hybrid + inverse-popularity boost alpha=0.15 | 0.465282 | 0.280582 | 0.741898 | 0.170084 | 0.071741 | 0.355037 | 14.814953 |

The model hierarchy is clean: `Two-Tower < Popularity < Mini-DLRM < Hybrid Mini-DLRM + Semantic`. Popularity captures head demand and looks competitive overall, but nearly fails on tail items. Mini-DLRM improves over simple baselines through feature interactions. Hybrid improves both overall ranking and tail relevance by adding semantic item priors. The inverse-popularity boost gives a modest tail exposure and tail relevance shift with a tiny head/overall cost.

## Multi-positive Evaluation

Additional robustness check: `num_test_positives=3`, where each eligible user has 3 held-out positive items ranked against 99 sampled non-interacted negatives. Users without enough history for train, validation, and 3 test positives are excluded. This does not replace the main benchmark; it gives a more realistic view of users having multiple relevant items.

| Setting | Value |
|---|---:|
| Evaluated users | 4,899 |
| Items | 19,707 |
| Interactions | 138,919 |
| Sparsity | 0.99856 |
| NumHeadEvalPositives | 6,843 |
| NumTailEvalPositives | 7,854 |

| Model | Recall@10 | HitRate@10 | NDCG@10 | HeadRecall@10 | TailRecall@10 | TailHitRate@10 | TailNDCG@10 | TailExposure@10 | AvgTrainPopularity@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mini-DLRM | 0.404368 | 0.731986 | 0.316013 | 0.765454 | 0.084463 | 0.148112 | 0.036546 | 0.193856 | 11.100204 |
| Hybrid Mini-DLRM + Semantic | 0.435701 | 0.768932 | 0.342207 | 0.724939 | 0.172435 | 0.282914 | 0.084377 | 0.347295 | 9.444050 |
| Hybrid + inverse-popularity boost alpha=0.15 | 0.435667 | 0.769034 | 0.341914 | 0.721874 | 0.175137 | 0.286544 | 0.086249 | 0.354266 | 9.362564 |

Multi-positive evaluation reduces single-item noise. The semantic Hybrid advantage is clearer here: Hybrid improves overall Recall/NDCG and roughly doubles TailRecall and TailNDCG versus Mini-DLRM. Mini-DLRM has stronger HeadRecall but much weaker tail relevance, suggesting stronger head-item memorization and weaker tail generalization. The inverse-popularity boost remains modest: it slightly improves TailRecall, TailNDCG, and TailExposure while leaving overall Recall/NDCG nearly unchanged.

## Current Recommended Configuration

```text
Model: Hybrid Mini-DLRM + Semantic
Semantic encoder: sentence-transformers/all-MiniLM-L6-v2
epochs: 4
lr: 1e-3
emb_dim: 32
train_negatives: 8
eval_negatives: 99
optional exploration boost: inverse_popularity alpha=0.15
```

The boost is a post-ranking exploration heuristic, not a new model or production cold-start solution.

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

The semantic embedding table is frozen. Only recommender embeddings, dense feature layers, semantic projection, interaction layers, and the top MLP are trained.

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

Build a small demo subset:

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

Build the current 138k-scale subset:

```bash
python scripts/build_real_subset.py \
  --max-review-rows 2000000 \
  --max-metadata-rows 2500000 \
  --target-items 20000 \
  --target-interactions 200000 \
  --selection-mode long_tail \
  --min-user-interactions 5 \
  --min-item-interactions 2
```

Run the current main benchmark:

```bash
python scripts/run_ablation.py \
  --run-name movies_tv_138k_main_baselines \
  --models popularity,two_tower,mini_dlrm,hybrid \
  --seeds 42,43,44 \
  --epochs 4 \
  --learning-rates 0.001 \
  --embedding-dims 32 \
  --train-negatives 8 \
  --eval-negatives 99 \
  --num-test-positives 1 \
  --batch-size 4096 \
  --semantic-model sentence-transformers/all-MiniLM-L6-v2 \
  --cold-start-boost-mode inverse_popularity \
  --cold-start-boost-alpha 0.15
```

For real-data setup details, see [docs/RUN_REAL_DATA.md](docs/RUN_REAL_DATA.md).

## Repository Structure

```text
notebooks/   Main runnable experiment and reporting notebook
src/         Reusable data, sampling, metrics, model, training, reranking, and semantic modules
scripts/     Real-data subset builder, smoke checks, and ablation runner
data/        Local raw/processed/embedding files; ignored except .gitkeep files
results/     Local generated metrics and figures; ignored except sample outputs
```

## Models

| Model | Signal Used | Purpose |
|---|---|---|
| Popularity | train interaction counts | non-personalized head-demand baseline |
| Two-Tower | user/item ID embeddings | collaborative baseline |
| Mini-DLRM | user/item/category/dense feature interactions | compact DLRM-inspired ranker |
| Hybrid Mini-DLRM + Semantic | Mini-DLRM + frozen item text embeddings | semantic hybrid ranker for sparse items |
| Hybrid + inverse-popularity boost | post-ranking score adjustment | exploration analysis for tail exposure |

This project evaluates sampled candidate ranking, not full-catalog ANN retrieval over millions of items.

## Evaluation Metrics

Single-positive evaluation:

- Each user has 1 held-out positive plus sampled negatives.
- `Recall@K` equals `HitRate@K` because there is only one positive item.
- This is the main benchmark protocol for comparability.

Multi-positive evaluation:

- Each eligible user has N held-out positives plus `eval_negatives` sampled negatives.
- `Recall@K` is the fraction of held-out positives retrieved in top K.
- `HitRate@K` is whether at least one held-out positive appears in top K.
- `NDCG@K` accounts for the ranks of all held-out positives.

Long-tail diagnostics:

- `TailExposure@10`: fraction of recommended top-10 slots assigned to tail items.
- `AvgTrainPopularity@10`: average train interaction count of recommended top-10 items.
- Head/tail relevance metrics are split by whether held-out positive items are head or tail.

## Main Findings

1. Semantic priors help sparse long-tail recommendation.
2. Popularity is highly head-biased and almost fails on tail positives.
3. Mini-DLRM improves over simple baselines but remains weaker on tail relevance.
4. Hybrid Mini-DLRM + Semantic improves overall ranking and tail relevance.
5. Multi-positive evaluation shows stronger semantic Hybrid gains, especially on tail positives.
6. Larger embedding dimensions can overfit; `emb_dim=32` is best overall in the 138k run.
7. Negative sampling optimum is regime-dependent; `train_negatives=8` is best in the 138k run.
8. Cold-start boosting is a modest exploration heuristic, not a replacement for relevance learning.

## Earlier Exploratory Results

Earlier 49k-scale and encoder-comparison experiments were useful for shaping the final setup, but they are not the current main result.

| Observation | Result |
|---|---|
| Tiny dense regime | Semantic Hybrid did not help much because collaborative repetition dominated. |
| Earlier sparse subset | Hybrid MiniLM improved over Mini-DLRM and showed better stability. |
| Earlier BGE-small tuning | BGE was promising on a smaller subset after encoder-specific tuning. |
| Earlier negative sampling | `train_negatives=12` was strong before scaling; `train_negatives=8` is better in the 138k run. |

MiniLM is the current main encoder for the 138k benchmark. BGE is retained as historical context, not claimed as the final best model.

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

## Limitations and Future Work

Limitations:

- Evaluation uses sampled candidate ranking, not full-catalog retrieval.
- The project does not include sequential user modeling.
- The project does not include graph-based collaborative modeling.
- The project does not separate retrieval and reranking stages.
- Encoder-specific tuning was informative but not exhaustive.
- The cold-start boost is a heuristic exposure analysis, not a production exploration system.

Future work:

- Hard negative mining
- Popularity-aware negative sampling
- ANN retrieval with Faiss
- SASRec or BERT4Rec sequence modeling
- LLM reranking
- Multimodal product embeddings
- Fine-tuning semantic encoders on recommendation pairs

## References and Resources

- [Amazon Reviews 2023 dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023), McAuley Lab
- DLRM-style recommendation models with sparse embeddings, dense features, and feature interactions
- Semantic embedding models: MiniLM, e5, and BGE
- Long-tail recommendation research on sparse feedback and popularity skew
- [Negative Sampling in Recommendation: A Survey and Future Directions](https://dl.acm.org/doi/10.1145/3793855), ACM Digital Library
