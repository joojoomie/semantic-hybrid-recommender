# Semantic Hybrid Recommender on Amazon Reviews

This project builds a semantic hybrid recommender for sparse, long-tail Amazon Reviews ranking. It starts with popularity and ID-based collaborative baselines, adds a compact Mini-DLRM-style interaction model, then injects frozen MiniLM product-text embeddings into the ranker. The goal is to test when semantic item priors help recommendation quality beyond user-item interaction signals.

This is a research-engineering portfolio project: compact, reproducible, and designed for comparison, not a full-scale production Meta DLRM or industrial retrieval stack.

## Key Results

Latest real-data experiment: Amazon Reviews 2023 `Movies_and_TV` subset, evaluated with leave-last-out sampled ranking using 1 positive item plus 99 sampled negatives per user.

| Setting | Value |
|---|---:|
| Users | 13,509 |
| Items | 20,896 |
| Positive interactions | 138,919 |
| Sparsity | 0.99951 |
| Minimum user interactions | 5 |
| Minimum item interactions | 2 |

Latest 138k-scale baseline comparison:

| Model | Recall@10 | NDCG@10 | HeadRecall@10 | TailRecall@10 | TailNDCG@10 | TailExposure@10 | AvgTrainPopularity@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Two-Tower | 0.3634 | 0.2170 | 0.7006 | 0.0036 | 0.0011 | 0.0192 | 22.7156 |
| Popularity | 0.3832 | 0.2301 | 0.7420 | 0.0002 | 0.0000 | 0.0006 | 23.3183 |
| Mini-DLRM | 0.4138 | 0.2425 | 0.6660 | 0.1448 | 0.0616 | 0.3943 | 14.3137 |
| Hybrid Mini-DLRM + Semantic | 0.4657 | 0.2810 | 0.7457 | 0.1669 | 0.0703 | 0.3488 | 14.9235 |
| Hybrid + inverse-popularity boost alpha=0.15 | 0.4653 | 0.2806 | 0.7419 | 0.1701 | 0.0717 | 0.3550 | 14.8150 |

The model hierarchy is now clean: `Two-Tower < Popularity < Mini-DLRM < Hybrid Mini-DLRM + Semantic`. The semantic Hybrid is the strongest overall model on the 138k-scale benchmark. Popularity is competitive on overall metrics because it captures head demand, but it almost completely fails on tail items: TailRecall@10 is near zero, TailExposure@10 is near zero, and AvgTrainPopularity@10 is highest.

Current recommended configuration:

```text
Hybrid Mini-DLRM + Semantic
semantic encoder = sentence-transformers/all-MiniLM-L6-v2
epochs = 4
lr = 1e-3
emb_dim = 32
train_negatives = 8
eval_negatives = 99
optional exploration boost = inverse_popularity, alpha = 0.15
```

The inverse-popularity boost is optional and should be interpreted as an exploration analysis, not a new model. It slightly improves tail relevance and exposure while slightly reducing head Recall and overall NDCG.

Earlier exploratory benchmark on the smaller sparse subset:

| Model | Setting | Recall@10 | HitRate@10 | NDCG@10 |
|---|---|---:|---:|---:|
| Popularity | sparse long-tail baseline | 0.2139 | 0.2139 | 0.1136 |
| Two-Tower | sparse long-tail baseline | 0.1094 | 0.1094 | 0.0479 |
| Mini-DLRM | tuned negatives=16 | 0.2592 | 0.2592 | 0.1445 |
| Hybrid MiniLM | epochs=5, emb_dim=32, negatives=12 | 0.2596 | 0.2596 | 0.1440 |
| Hybrid BGE | epochs=8, emb_dim=64, negatives=12 | ~0.275 | ~0.275 | ~0.137 |

These earlier results are useful historical context, not the current main benchmark. The BGE result came from an earlier subset and should not be read as the final best model. Because leave-last-out evaluation has one positive test item per user, HitRate@10 and Recall@10 are effectively identical in this setup.

These results are from sampled candidate ranking, not full-catalog retrieval.

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

## Latest 138k-scale Results

The project was later scaled to a larger `Movies_and_TV` subset with 13,509 users, 20,896 items, 138,919 positive interactions, and 0.99951 sparsity. This setting is more representative than the earlier ~49k-interaction subset because the item universe is larger, the matrix is sparser, and tail-item ranking is a more prominent part of the sampled-candidate task.

The main 3-seed setting used `epochs=4`, `lr=1e-3`, `emb_dim=32`, and `train_negatives=8`. The Hybrid model used `sentence-transformers/all-MiniLM-L6-v2`; the boosted row applies an inverse-popularity post-ranking adjustment with `alpha=0.15`. Evaluation remains sampled ranking with one held-out positive and 99 sampled negatives per user.

| Model | Recall@10 | NDCG@10 | HeadRecall@10 | TailRecall@10 | TailNDCG@10 | TailExposure@10 | AvgTrainPopularity@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Two-Tower | 0.363424 | 0.216993 | 0.700602 | 0.003596 | 0.001116 | 0.019172 | 22.715645 |
| Popularity | 0.383152 | 0.230059 | 0.742042 | 0.000153 | 0.000046 | 0.000596 | 23.318332 |
| Mini-DLRM | 0.413835 | 0.242458 | 0.665974 | 0.144759 | 0.061567 | 0.394333 | 14.313709 |
| Hybrid Mini-DLRM + Semantic | 0.465727 | 0.280997 | 0.745698 | 0.166947 | 0.070298 | 0.348845 | 14.923451 |
| Hybrid + inverse-popularity boost alpha=0.15 | 0.465282 | 0.280582 | 0.741898 | 0.170084 | 0.071741 | 0.355037 | 14.814953 |

The hierarchy is now clear: Two-Tower < Popularity < Mini-DLRM < Hybrid Mini-DLRM + Semantic. Pure ID-based Two-Tower embeddings struggle in this sparse setting. Popularity is strong on head demand but almost completely misses tail positives. Mini-DLRM improves long-tail relevance through richer feature interactions. The semantic Hybrid improves both overall ranking and tail relevance relative to Mini-DLRM, showing that frozen MiniLM item-text embeddings provide useful priors when collaborative item signals are sparse.

### Tuning Observations

- The best overall configuration is still relatively compact: `emb_dim=32`, `lr=1e-3`, `train_negatives=8`, and `epochs=4`.
- Increasing embedding dimension to 64 improves tail relevance somewhat but lowers overall Recall/NDCG; `emb_dim=96` clearly overfits and hurts overall performance.
- Smaller learning rates underfit within the fixed 4-epoch budget, while `lr=1e-3` remains strongest.
- Increasing train negatives to 12 or 16 does not improve overall ranking in this sparse regime and appears to hurt optimization.

### Interpretation

The 138k-scale run strengthens the sparse long-tail conclusion: semantic item priors are especially useful when many items have weak collaborative signals. Popularity-based ranking can look competitive overall because it captures head demand, but it remains extremely popularity-biased and does not solve long-tail recommendation. The inverse-popularity cold-start boost consistently shifts exposure toward tail items: `TailExposure@10` increases, `AvgTrainPopularity@10` decreases, and TailRecall/TailNDCG improve slightly. The tradeoff is small but real: HeadRecall and overall NDCG decrease slightly. This supports using smooth post-ranking exploration priors for long-tail visibility, while avoiding overclaiming relevance gains or production readiness.

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

On the earlier subset, increasing negatives from 4 to 8/12/16 improved ranking quality. Hybrid remained more stable across seeds than Mini-DLRM, and `train_negatives=12` was the best balanced hybrid setting in that regime. In the larger 138k-scale run, `train_negatives=8` worked better; increasing negatives to 12 or 16 appeared to hurt optimization.

Final tuning takeaway: semantic embeddings improve sparse recommendation not only by raising mean ranking quality, but also by reducing seed sensitivity and stabilizing optimization under harder negative sampling.

## Cold-start Boost Reranking

The project also includes an optional post-ranking exploration heuristic for cold-start and long-tail analysis. This is not a new model and does not retrain the recommender. It adjusts already-computed Hybrid scores with a small popularity-based boost:

```text
final_score = model_score + alpha * cold_start_boost
```

Two modes are supported: `threshold`, which boosts items with train interaction counts at or below a threshold, and `inverse_popularity`, which gives larger boosts to lower-popularity items. The goal is to simulate exploration traffic allocation for sparse or new items using train-set popularity as a proxy.

This analysis should be interpreted as an accuracy versus exposure tradeoff. A boost can increase tail exposure without proving better relevance, so the runner reports both ranking metrics and exposure diagnostics:

```text
Recall@10, NDCG@10, HeadRecall@10, TailRecall@10, TailExposure@10, AvgTrainPopularity@10
```

`TailExposure@10` measures how many recommended top-10 slots go to tail items. `TailRecall@10` and `TailNDCG@10` instead measure whether users whose true held-out item is a tail item are better served. Exposure improvement alone does not prove relevance improvement; the split relevance metrics make that tradeoff explicit.

Example smoke commands:

```bash
python scripts/run_ablation.py \
  --quick \
  --models hybrid \
  --epochs 1 \
  --train-negatives 2 \
  --cold-start-boost-mode threshold \
  --cold-start-boost-alpha 0.02,0.05 \
  --cold-start-threshold 2
```

```bash
python scripts/run_ablation.py \
  --quick \
  --models hybrid \
  --epochs 1 \
  --train-negatives 2 \
  --cold-start-boost-mode inverse_popularity \
  --cold-start-boost-alpha 0.02,0.05
```

## Final Findings

The latest experimental setting is a sparse Amazon Reviews 2023 `Movies_and_TV` subset with 13,509 users, 20,896 items, 138,919 interactions, and 0.99951 sparsity. Evaluation uses leave-last-out ranking with 1 held-out positive and 99 sampled negatives per user. Because each test case has a single positive item, `HitRate@10` and `Recall@10` are effectively identical in this setup.

The project progressed through five stages:

1. Collaborative baselines: popularity, Two-Tower, and Mini-DLRM established the ranking baseline. On the initial tiny 47-item dense subset, Two-Tower and Mini-DLRM were similar, and semantic priors had little room to help.
2. Sparse long-tail regime: expanding through the 2,215-item and 20,896-item subsets made tail-item behavior central. Hybrid semantic recommendation began outperforming purely collaborative models, and the larger 138k-interaction run produced more representative sparse-ranking behavior than the earlier ~49k subset.
3. Negative sampling: harder negative sampling improved ranking quality in the earlier subset, but the larger 138k-scale run showed a lower optimum around `train_negatives=8`; 12 or 16 negatives did not improve overall ranking in that sparse regime.
4. Embedding capacity: tuning embedding dimensions from 16 to 96 exposed a capacity/stability tradeoff. Larger dimensions increased capacity but could increase variance or over-parameterize sparse interactions.
5. Semantic encoder comparison: MiniLM, BGE-small, and e5-small behaved differently in earlier experiments. BGE-small was promising on a smaller subset after tuning, but MiniLM is the current main encoder for the 138k-scale benchmark.

Compact final benchmark:

| Model | Setting | Recall@10 | NDCG@10 | HeadRecall@10 | TailRecall@10 |
|---|---|---:|---:|---:|---:|
| Two-Tower | 138k sparse baseline | 0.363424 | 0.216993 | 0.700602 | 0.003596 |
| Popularity | 138k sparse baseline | 0.383152 | 0.230059 | 0.742042 | 0.000153 |
| Mini-DLRM | epochs=4, emb_dim=32, negatives=8 | 0.413835 | 0.242458 | 0.665974 | 0.144759 |
| Hybrid MiniLM | epochs=4, emb_dim=32, negatives=8 | 0.465727 | 0.280997 | 0.745698 | 0.166947 |
| Hybrid MiniLM + inverse-popularity boost | alpha=0.15 | 0.465282 | 0.280582 | 0.741898 | 0.170084 |

Current recommended configuration:

```text
Model: Hybrid Mini-DLRM + Semantic
Semantic encoder: sentence-transformers/all-MiniLM-L6-v2
epochs = 4
lr = 1e-3
emb_dim = 32
train_negatives = 8
eval_negatives = 99
optional exploration boost = inverse_popularity, alpha = 0.15
```

Earlier subset observation:

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

BGE-small was an earlier retrieval-oriented observation, not the final main result. The main conclusion is that semantic item embeddings become increasingly valuable under sparse long-tail recommendation settings, but the benefit depends on capacity, negative sampling, and optimization budget. Smooth inverse-popularity boosting can slightly improve tail relevance and exposure, while slightly reducing head and overall metrics. The strongest general retrieval encoder is not automatically the best recommender semantic prior; alignment between semantic embedding geometry and recommender optimization matters.

## Lessons Learned / Key Insights

- Semantic embeddings help more in sparse long-tail settings than in tiny dense item universes.
- DLRM-style interaction models can become strongly head-biased.
- Frozen semantic priors can improve seed stability and generalization.
- Negative sampling has a regime-dependent optimum: `train_negatives=12` was strongest in the earlier subset, while `train_negatives=8` worked better in the 138k-scale sparse run.
- Semantic encoder choice materially affects hybrid recommender performance.
- Sparse recommendation has a real capacity/stability tradeoff; larger embeddings are not always better, and `emb_dim=96` overfit in the larger sparse setting.
- Cold-start boosting is best treated as a modest post-ranking exploration heuristic: it can shift exposure and slightly improve tail relevance, but it does not replace model-side relevance learning.
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
