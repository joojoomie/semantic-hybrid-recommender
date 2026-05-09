# Semantic Hybrid Recommender on Amazon Reviews

## Overview

This project studies sparse, long-tail recommendation on Amazon Reviews 2023 using sampled ranking experiments. It compares simple and neural baselines: Popularity, ID-based Two-Tower, and a compact Mini-DLRM-style ranking model.

The main model is a semantic Hybrid Mini-DLRM: a Mini-DLRM ranker augmented with frozen `sentence-transformers/all-MiniLM-L6-v2` item-text embeddings built from product title, category, and description. This is a research-engineering portfolio project for analyzing semantic priors under sparse recommendation, not a production recommender, full Meta DLRM implementation, or full-catalog retrieval system.

## Key Results

Main benchmark: Amazon Reviews 2023 `Movies_and_TV`, leave-last-N-out sampled ranking with `num_test_positives=3`. Each eligible user has 3 held-out positive items ranked against 99 sampled non-interacted negatives.

### Dataset and Evaluation Population

The original filtered subset contains 13,509 users, 20,896 items, and 138,919 positive interactions. Under the fixed multi-positive protocol, users without enough history for train, validation, and 3 test positives are excluded from evaluation, leaving the eligible population below.

| Setting | Value |
|---|---:|
| Eligible users | 4,899 |
| Eligible items | 19,707 |
| Positive interactions | 138,919 |
| Sparsity | 0.99856 |
| Minimum user interactions | 5 |
| Minimum item interactions | 2 |
| Eval protocol | 3 positives + 99 negatives |
| NumHeadEvalUsers mean | 3,834 |
| NumTailEvalUsers mean | 4,132 |
| NumHeadEvalPositives mean | 6,843 |
| NumTailEvalPositives mean | 7,854 |

## Multi-positive Evaluation: 3 Positives + 99 Negatives

| Model | Recall@10 | HitRate@10 | NDCG@10 | HeadRecall@10 | HeadNDCG@10 | TailRecall@10 | TailHitRate@10 | TailNDCG@10 | TailExposure@10 | AvgTrainPopularity@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Two-Tower | 0.301456 | 0.615126 | 0.230036 | 0.634455 | 0.404431 | 0.005183 | 0.009681 | 0.001984 | 0.030966 | 14.141478 |
| Popularity | 0.320576 | 0.636967 | 0.245931 | 0.681512 | 0.435126 | 0.000202 | 0.000363 | 0.000071 | 0.000388 | 14.691437 |
| Mini-DLRM | 0.370926 | 0.705042 | 0.277062 | 0.604134 | 0.397169 | 0.152085 | 0.259439 | 0.080227 | 0.426628 | 8.582466 |
| Hybrid Mini-DLRM + Semantic | 0.434306 | 0.767810 | 0.341041 | 0.718288 | 0.505856 | 0.175722 | 0.284729 | 0.089091 | 0.366850 | 9.187212 |
| Hybrid + inverse-popularity boost alpha=0.15 | 0.434340 | 0.767197 | 0.340624 | 0.715289 | 0.503142 | 0.178243 | 0.288238 | 0.090643 | 0.372658 | 9.122362 |

Multi-positive evaluation is now the main protocol because it better reflects users having multiple relevant held-out items. Popularity remains strongly head-biased: it has high HeadRecall but almost zero TailRecall and TailExposure. Two-Tower performs worst overall and has weak tail generalization. Mini-DLRM improves over simple baselines, but Hybrid Mini-DLRM + Semantic is the strongest non-boosted model, improving Recall@10 from 0.370926 to 0.434306 and NDCG@10 from 0.277062 to 0.341041 versus Mini-DLRM. It also improves TailRecall@10 from 0.152085 to 0.175722 and TailNDCG@10 from 0.080227 to 0.089091, showing that frozen MiniLM item-text embeddings provide useful semantic priors in sparse long-tail recommendation.

The inverse-popularity boost remains optional and modest. Compared with the base Hybrid, Recall@10 is essentially unchanged, NDCG@10 drops slightly, TailRecall@10 improves from 0.175722 to 0.178243, TailNDCG@10 improves from 0.089091 to 0.090643, TailExposure@10 improves from 0.366850 to 0.372658, and AvgTrainPopularity@10 decreases from 9.187212 to 9.122362. This is best interpreted as a small exploration-exploitation tradeoff, not as a new model or production cold-start solution.

## Hard Retrieval Stress Test: 3 Positives + 499 Negatives

This is not the main benchmark. It is a robustness analysis with a much larger sampled candidate set: each eligible user ranks 3 held-out positive items against 499 sampled non-interacted negatives. Absolute scores are lower because the top-10 ranking task is substantially harder.

| Setting | Value |
|---|---:|
| Eligible users | 4,899 |
| Eligible items | 19,707 |
| Positive interactions | 138,919 |
| Sparsity | 0.998561 |
| Eval protocol | 3 positives + 499 negatives |
| NumHeadEvalUsers mean | 3,834 |
| NumTailEvalUsers mean | 4,132 |
| NumHeadEvalPositives mean | 6,843 |
| NumTailEvalPositives mean | 7,854 |

| Model | Recall@10 | HitRate@10 | NDCG@10 | HeadRecall@10 | HeadNDCG@10 | TailRecall@10 | TailHitRate@10 | TailNDCG@10 | TailExposure@10 | AvgTrainPopularity@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mini-DLRM | 0.179084 | 0.414983 | 0.132253 | 0.369718 | 0.229910 | 0.007160 | 0.014037 | 0.003308 | 0.047714 | 18.452878 |
| Hybrid Mini-DLRM + Semantic | 0.202626 | 0.461523 | 0.150502 | 0.389976 | 0.247414 | 0.031885 | 0.059656 | 0.016594 | 0.157726 | 14.273791 |
| Hybrid + inverse-popularity boost alpha=0.15 | 0.202082 | 0.459992 | 0.149852 | 0.387585 | 0.245234 | 0.033196 | 0.061834 | 0.017449 | 0.165411 | 14.128322 |

Hybrid remains stronger than Mini-DLRM under harder retrieval: Recall@10 improves from 0.179084 to 0.202626 and NDCG@10 improves from 0.132253 to 0.150502. The tail gap becomes especially clear: TailRecall@10 improves from 0.007160 to 0.031885 and TailNDCG@10 improves from 0.003308 to 0.016594. This suggests semantic item priors improve robustness as sampled retrieval difficulty increases. Mini-DLRM collapses more strongly on tail positives when the candidate set becomes larger. Hybrid + inverse-popularity boost slightly increases TailRecall, TailNDCG, and TailExposure, but the effect remains modest.

## Single-positive Compatibility Benchmark

The earlier single-positive protocol is retained for standard benchmark compatibility, not as the main project result. It uses the original 13,509-user, 20,896-item filtered population with 1 held-out positive item plus 99 negatives per user. In this setting, `Recall@K` and `HitRate@K` are equivalent because there is only one positive item.

| Model | Recall@10 | NDCG@10 | HeadRecall@10 | TailRecall@10 | TailNDCG@10 | TailExposure@10 | AvgTrainPopularity@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Two-Tower | 0.363424 | 0.216993 | 0.700602 | 0.003596 | 0.001116 | 0.019172 | 22.715645 |
| Popularity | 0.383152 | 0.230059 | 0.742042 | 0.000153 | 0.000046 | 0.000596 | 23.318332 |
| Mini-DLRM | 0.413835 | 0.242458 | 0.665974 | 0.144759 | 0.061567 | 0.394333 | 14.313709 |
| Hybrid Mini-DLRM + Semantic | 0.465727 | 0.280997 | 0.745698 | 0.166947 | 0.070298 | 0.348845 | 14.923451 |
| Hybrid + inverse-popularity boost alpha=0.15 | 0.465282 | 0.280582 | 0.741898 | 0.170084 | 0.071741 | 0.355037 | 14.814953 |

The single-positive result shows the same broad pattern: popularity is strongly head-biased, Mini-DLRM improves over simpler collaborative baselines, and the semantic Hybrid gives the strongest overall ranking and tail relevance.

## Current Recommended Configuration

```text
Model: Hybrid Mini-DLRM + Semantic
Semantic encoder: sentence-transformers/all-MiniLM-L6-v2
epochs: 5
lr: 1e-3
emb_dim: 32
train_negatives: 8
eval_negatives: 99
num_test_positives: 3 for main evaluation
optional exploration boost: inverse_popularity alpha=0.15
```

The boost is a post-ranking exploration heuristic, not a new model or production cold-start solution.

## Final Benchmark Visualizations

### Tail Relevance Metrics

![Tail Metrics](results/figures/final_tail_metrics.png)

Popularity and Two-Tower nearly fail on tail relevance. Hybrid substantially improves TailRecall and TailNDCG, showing that semantic item-text priors provide stronger tail generalization under sparse recommendation.

### Head vs Tail Recall

![Head Tail Recall](results/figures/final_head_tail_recall.png)

This view makes the head-tail imbalance explicit. Mini-DLRM retains high tail exposure but weaker tail relevance, while Hybrid improves tail recall without collapsing head-item performance.

### Overall Ranking Metrics

![Overall Metrics](results/figures/final_overall_metrics.png)

Hybrid achieves the best overall Recall@10 and NDCG@10. Mini-DLRM improves over simpler collaborative baselines, while Two-Tower performs worst in the sparse long-tail regime.

### Exposure vs Popularity Tradeoff

![Exposure Tradeoff](results/figures/final_exposure_tradeoff.png)

Popularity is extremely head-biased. Hybrid + Boost shifts slightly toward better tail exposure, while Mini-DLRM produces high tail exposure but lower tail relevance. Hybrid balances relevance and exposure more effectively.

The figures are generated from the fixed final benchmark metrics and saved as both PNG and PDF with:

```bash
python scripts/plot_final_results.py
```

## Architecture

```text
Amazon Reviews interactions                           Product metadata text
        |                                                        |
        v                                                        v
 user_id, item_id                                   title/category/description
        |                                                        |
        v                                                        v
 collaborative embeddings                         frozen MiniLM/e5/BGE embeddings
        |                                                        |
        |                                            semantic projection layer
        |                                                        |
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
  --run-name movies_tv_138k_multipos3_main \
  --models mini_dlrm,hybrid \
  --seeds 42,43,44 \
  --epochs 5 \
  --learning-rates 0.001 \
  --embedding-dims 32 \
  --train-negatives 8 \
  --eval-negatives 99 \
  --num-test-positives 3 \
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

Multi-positive evaluation:

- Each eligible user has N held-out positives plus `eval_negatives` sampled negatives.
- `Recall@K` is the fraction of held-out positives retrieved in top K.
- `HitRate@K` is whether at least one held-out positive appears in top K.
- `NDCG@K` accounts for the ranks of all held-out positives.

Single-positive compatibility evaluation:

- Each user has 1 held-out positive plus sampled negatives.
- `Recall@K` equals `HitRate@K` because there is only one positive item.
- This is retained for comparison with common leave-one-out ranking setups, but the main project benchmark is multi-positive.

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
6. Larger embedding dimensions can overfit; `emb_dim=32` is best overall in the final 138k multi-positive run.
7. Negative sampling optimum is regime-dependent; `train_negatives=8` is best in the 138k run.
8. Cold-start boosting is a modest exploration heuristic, not a replacement for relevance learning.
9. Under the harder 499-negative stress test, the semantic Hybrid retains much stronger tail relevance than Mini-DLRM, suggesting better robustness as retrieval difficulty increases.

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
- The 499-negative stress test is still sampled evaluation, not full-catalog retrieval.

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
