from __future__ import annotations

import argparse
import itertools
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import (
    apply_id_mappings,
    build_category_features,
    build_dense_features,
    build_id_mappings,
    cap_interactions,
    filter_interactions,
    load_metadata,
    load_reviews,
    normalize_columns,
    temporal_split,
)
from src.metrics import evaluate_leave_one_out, evaluate_leave_one_out_by_true_item_groups
from src.models import HybridDLRM, MiniDLRM, TwoTower
from src.negative_sampling import (
    build_eval_candidates,
    build_eval_candidates_multi_positive,
    build_training_samples,
    build_user_positive_items,
)
from src.rerank import compute_item_train_counts, evaluate_ranking_exposure, make_cold_start_boost_scorer
from src.semantic_embeddings import build_item_text, load_or_create_semantic_embeddings
from src.train import ItemFeatureStore, make_popularity_scorer, make_torch_scorer, train_model
from src.utils import get_device, seed_everything


MODEL_NAMES = ("popularity", "two_tower", "mini_dlrm", "hybrid")


@dataclass(frozen=True)
class AblationConfig:
    seed: int
    epochs: int
    lr: float
    emb_dim: int
    train_negatives: int


@dataclass(frozen=True)
class PreparedData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    metadata: pd.DataFrame | None
    item_features: ItemFeatureStore
    category_mapping: dict[str, int]
    item_mapping: dict[str, int]
    known_positive_items: dict[int, set[int]]
    num_users: int
    num_items: int
    dense_dim: int
    stats: dict[str, float | int | str]


def parse_int_list(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def parse_models(raw: str) -> list[str]:
    models = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = sorted(set(models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown models {unknown}. Supported models: {MODEL_NAMES}")
    return models


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def resolve_path(path: str | None) -> Path | None:
    if not path:
        return None
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = REPO_ROOT / path_obj
    return path_obj


def build_configs(args: argparse.Namespace) -> list[AblationConfig]:
    configs = [
        AblationConfig(seed=seed, epochs=epochs, lr=lr, emb_dim=emb_dim, train_negatives=train_negatives)
        for seed, epochs, lr, emb_dim, train_negatives in itertools.product(
            parse_int_list(args.seeds),
            parse_int_list(args.epochs),
            parse_float_list(args.learning_rates),
            parse_int_list(args.embedding_dims),
            parse_int_list(args.train_negatives),
        )
    ]
    if args.quick:
        return configs[: args.quick_configs]
    return configs


def parse_boost_alphas(args: argparse.Namespace) -> list[float]:
    if args.cold_start_boost_mode == "none":
        return []
    return parse_float_list(args.cold_start_boost_alpha)


def limit_eval_candidates(
    candidates: list[dict[str, object]],
    limit: int | None,
    seed: int,
) -> list[dict[str, object]]:
    if limit is None or limit <= 0 or len(candidates) <= limit:
        return candidates
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(candidates), size=limit, replace=False))
    return [candidates[int(idx)] for idx in indices]


def candidate_true_items(row: dict[str, object]) -> list[int]:
    if "true_items" in row:
        return [int(item) for item in row["true_items"]]  # type: ignore[union-attr]
    return [int(row["true_item"])]


def build_head_tail_items(train_df: pd.DataFrame, num_items: int) -> tuple[set[int], set[int]]:
    popularity = train_df["item_idx"].value_counts()
    head_cutoff = max(1, int(np.ceil(num_items * 0.2)))
    head_items = set(popularity.sort_values(ascending=False).head(head_cutoff).index.astype(int))
    tail_items = set(range(num_items)) - head_items
    return head_items, tail_items


def temporal_split_with_test_positives(
    df: pd.DataFrame,
    num_test_positives: int,
    min_train_interactions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if num_test_positives == 1:
        return temporal_split(df)
    if num_test_positives < 1:
        raise ValueError("--num-test-positives must be at least 1.")

    ordered = df.sort_values(["user_id", "timestamp", "item_id"]).copy()
    sizes = ordered.groupby("user_id")["item_id"].transform("size")
    required = min_train_interactions + num_test_positives + 1
    eligible = ordered[sizes >= required].copy()
    if eligible.empty:
        raise RuntimeError(
            "No users have enough interactions for the requested multi-positive split. "
            "Use a smaller --num-test-positives or lower --min-user-interactions."
        )
    eligible["rank_from_end"] = eligible.groupby("user_id").cumcount(ascending=False)
    test_df = eligible[eligible["rank_from_end"] < num_test_positives].drop(columns="rank_from_end")
    val_df = eligible[eligible["rank_from_end"] == num_test_positives].drop(columns="rank_from_end")
    train_df = eligible[eligible["rank_from_end"] > num_test_positives].drop(columns="rank_from_end")
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def prepare_data(args: argparse.Namespace) -> PreparedData:
    reviews_path = resolve_path(args.reviews)
    metadata_path = resolve_path(args.metadata)
    if reviews_path is None or not reviews_path.exists():
        raise FileNotFoundError(
            f"Reviews file not found: {reviews_path}. Build the ignored real subset with scripts/build_real_subset.py first."
        )

    reviews_raw = load_reviews(str(reviews_path))
    metadata_raw = load_metadata(str(metadata_path)) if metadata_path else None
    reviews, metadata = normalize_columns(reviews_raw, metadata_raw)
    reviews = cap_interactions(reviews, args.max_interactions, seed=args.data_seed)
    interactions = filter_interactions(
        reviews,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
    )
    if interactions.empty:
        raise RuntimeError("No interactions remain after filtering. Relax the min interaction thresholds.")

    train_df, val_df, test_df = temporal_split_with_test_positives(
        interactions,
        num_test_positives=args.num_test_positives,
        min_train_interactions=args.min_user_interactions,
    )
    user_mapping, item_mapping = build_id_mappings(train_df, val_df, test_df)
    train_df = apply_id_mappings(train_df, user_mapping, item_mapping)
    val_df = apply_id_mappings(val_df, user_mapping, item_mapping)
    test_df = apply_id_mappings(test_df, user_mapping, item_mapping)

    category_by_item, category_mapping = build_category_features(item_mapping, metadata)
    dense_by_item, _ = build_dense_features(item_mapping, metadata)
    all_positive_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    known_positive_items = build_user_positive_items(all_positive_df)

    num_users = len(user_mapping)
    num_items = len(item_mapping)
    interactions_count = len(interactions)
    stats = {
        "dataset": args.dataset_name,
        "users": num_users,
        "items": num_items,
        "interactions": interactions_count,
        "sparsity": 1.0 - (interactions_count / max(1, num_users * num_items)),
        "min_user_interactions": args.min_user_interactions,
        "min_item_interactions": args.min_item_interactions,
        "max_interactions": args.max_interactions if args.max_interactions is not None else "all",
        "eval_negatives": args.eval_negatives,
        "num_test_positives": args.num_test_positives,
        "k": args.k,
    }

    return PreparedData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        metadata=metadata,
        item_features=ItemFeatureStore(category_by_item=category_by_item, dense_by_item=dense_by_item),
        category_mapping=category_mapping,
        item_mapping=item_mapping,
        known_positive_items=known_positive_items,
        num_users=num_users,
        num_items=num_items,
        dense_dim=dense_by_item.shape[1],
        stats=stats,
    )


def evaluate_long_tail(
    scorer_by_model: dict[str, object],
    test_candidates: list[dict[str, object]],
    head_items: set[int],
    tail_items: set[int],
    item_train_counts: np.ndarray,
    k: int,
    scorer_metadata: dict[str, dict[str, object]] | None = None,
) -> pd.DataFrame:
    def filter_candidates_by_items(candidates: list[dict[str, object]], allowed_items: set[int]) -> list[dict[str, object]]:
        return [row for row in candidates if any(item in allowed_items for item in candidate_true_items(row))]

    scorer_metadata = scorer_metadata or {}
    rows = []
    for model_name, scorer in scorer_by_model.items():
        for group_name, item_group in (("head", head_items), ("tail", tail_items)):
            group_candidates = filter_candidates_by_items(test_candidates, item_group)
            metrics = evaluate_leave_one_out(None, group_candidates, k=k, scorer=scorer)
            exposure = evaluate_ranking_exposure(scorer, group_candidates, item_train_counts, tail_items, k=k)
            rows.append(
                {
                    "Model": model_name,
                    "Group": group_name,
                    "Test Users": len(group_candidates),
                    "Test Items": len(
                        {
                            item
                            for row in group_candidates
                            for item in candidate_true_items(row)
                            if item in item_group
                        }
                    ),
                    **metrics,
                    **exposure,
                    **scorer_metadata.get(model_name, {}),
                }
            )
    return pd.DataFrame(rows)


def add_metadata_columns(frame: pd.DataFrame, args: argparse.Namespace, config: AblationConfig, data: PreparedData) -> pd.DataFrame:
    enriched = frame.copy()
    for key, value in {
        "run_name": args.run_name,
        "run_timestamp": args.run_timestamp,
        "quick": args.quick,
        "semantic_model": args.semantic_model,
        **data.stats,
        **asdict(config),
        "device": getattr(args, "selected_device", args.device),
        "device_request": args.device,
        "eval_user_limit": args.eval_user_limit if args.eval_user_limit else "full",
        "cold_start_threshold": args.cold_start_threshold,
    }.items():
        if key not in enriched.columns:
            enriched[key] = value
    return enriched


def summarize_metrics(df: pd.DataFrame, metric_cols: list[str], extra_group_cols: list[str] | None = None) -> pd.DataFrame:
    extra_group_cols = extra_group_cols or []
    group_cols = [
        "Model",
        *extra_group_cols,
        "epochs",
        "lr",
        "emb_dim",
        "train_negatives",
        "cold_start_boost_mode",
        "cold_start_boost_alpha",
        "cold_start_threshold",
    ]
    present_group_cols = [col for col in group_cols if col in df.columns]
    present_metric_cols = [col for col in metric_cols if col in df.columns]
    summary = df.groupby(present_group_cols, dropna=False)[present_metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part) if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    return summary.fillna(0.0)


def evaluate_scorer_row(
    model_name: str,
    scorer: object,
    test_candidates: list[dict[str, object]],
    item_train_counts: np.ndarray,
    head_items: set[int],
    tail_items: set[int],
    k: int,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata = metadata or {}
    metrics = evaluate_leave_one_out(None, test_candidates, k=k, scorer=scorer)
    split_metrics = evaluate_leave_one_out_by_true_item_groups(
        test_candidates,
        head_items,
        tail_items,
        k=k,
        scorer=scorer,
    )
    exposure = evaluate_ranking_exposure(scorer, test_candidates, item_train_counts, tail_items, k=k)
    return {"Model": model_name, **metrics, **split_metrics, **exposure, **metadata}


def base_scorer_metadata() -> dict[str, object]:
    return {"cold_start_boost_mode": "none", "cold_start_boost_alpha": 0.0}


def boosted_scorer_name(mode: str, alpha: float) -> str:
    return f"Hybrid + Cold-start Boost {mode} alpha={alpha:g}"


def run_one_config(
    args: argparse.Namespace,
    config: AblationConfig,
    data: PreparedData,
    models: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Running config: {config}")
    seed_everything(config.seed)
    device = get_device(args.device)
    args.selected_device = str(device)

    train_samples = build_training_samples(
        data.train_df,
        data.known_positive_items,
        data.num_items,
        num_negatives=config.train_negatives,
        seed=config.seed,
    )
    val_candidates = build_eval_candidates(
        data.val_df,
        data.known_positive_items,
        data.num_items,
        num_negatives=args.eval_negatives,
        seed=config.seed + 1,
    )
    if args.num_test_positives == 1:
        test_candidates = build_eval_candidates(
            data.test_df,
            data.known_positive_items,
            data.num_items,
            num_negatives=args.eval_negatives,
            seed=config.seed + 2,
        )
    else:
        test_candidates = build_eval_candidates_multi_positive(
            data.test_df,
            data.known_positive_items,
            data.num_items,
            num_negatives=args.eval_negatives,
            seed=config.seed + 2,
        )
    val_candidates_for_training = limit_eval_candidates(
        val_candidates,
        args.eval_user_limit,
        seed=config.seed + 3,
    )
    item_train_counts = compute_item_train_counts(data.train_df, data.num_items)
    head_items, tail_items = build_head_tail_items(data.train_df, data.num_items)

    result_rows: list[dict[str, object]] = []
    scorers: dict[str, object] = {}
    scorer_metadata: dict[str, dict[str, object]] = {}

    if "popularity" in models:
        scorer = make_popularity_scorer(data.train_df, data.num_items)
        scorers["Popularity"] = scorer
        scorer_metadata["Popularity"] = base_scorer_metadata()
        result_rows.append(
            evaluate_scorer_row(
                "Popularity",
                scorer,
                test_candidates,
                item_train_counts,
                head_items,
                tail_items,
                args.k,
                scorer_metadata["Popularity"],
            )
        )

    if "two_tower" in models:
        model = TwoTower(data.num_users, data.num_items, emb_dim=config.emb_dim)
        model = train_model(
            model,
            train_samples,
            eval_candidates=val_candidates_for_training,
            epochs=config.epochs,
            batch_size=args.batch_size,
            lr=config.lr,
            k=args.k,
            device=device,
        )
        scorer = make_torch_scorer(model, device=device)
        scorers["Two-Tower"] = scorer
        scorer_metadata["Two-Tower"] = base_scorer_metadata()
        result_rows.append(
            evaluate_scorer_row(
                "Two-Tower",
                scorer,
                test_candidates,
                item_train_counts,
                head_items,
                tail_items,
                args.k,
                scorer_metadata["Two-Tower"],
            )
        )

    if "mini_dlrm" in models:
        model = MiniDLRM(
            data.num_users,
            data.num_items,
            len(data.category_mapping),
            data.dense_dim,
            emb_dim=config.emb_dim,
        )
        model = train_model(
            model,
            train_samples,
            eval_candidates=val_candidates_for_training,
            item_features=data.item_features,
            epochs=config.epochs,
            batch_size=args.batch_size,
            lr=config.lr,
            k=args.k,
            device=device,
        )
        scorer = make_torch_scorer(model, item_features=data.item_features, device=device)
        scorers["Mini-DLRM"] = scorer
        scorer_metadata["Mini-DLRM"] = base_scorer_metadata()
        result_rows.append(
            evaluate_scorer_row(
                "Mini-DLRM",
                scorer,
                test_candidates,
                item_train_counts,
                head_items,
                tail_items,
                args.k,
                scorer_metadata["Mini-DLRM"],
            )
        )

    if "hybrid" in models:
        texts = build_item_text(data.metadata, data.item_mapping)
        semantic_slug = slugify(args.semantic_model)
        cache_path = REPO_ROOT / "data" / "embeddings" / f"ablation_{semantic_slug}_{data.num_items}.npy"
        semantic_embeddings = load_or_create_semantic_embeddings(
            texts,
            cache_path=cache_path,
            model_name=args.semantic_model,
        )
        model = HybridDLRM(
            data.num_users,
            data.num_items,
            len(data.category_mapping),
            data.dense_dim,
            semantic_embeddings=semantic_embeddings,
            semantic_dim=semantic_embeddings.shape[1],
            emb_dim=config.emb_dim,
        )
        model = train_model(
            model,
            train_samples,
            eval_candidates=val_candidates_for_training,
            item_features=data.item_features,
            epochs=config.epochs,
            batch_size=args.batch_size,
            lr=config.lr,
            k=args.k,
            device=device,
        )
        scorer = make_torch_scorer(model, item_features=data.item_features, device=device)
        scorers["Hybrid Mini-DLRM + Semantic"] = scorer
        scorer_metadata["Hybrid Mini-DLRM + Semantic"] = base_scorer_metadata()
        result_rows.append(
            evaluate_scorer_row(
                "Hybrid Mini-DLRM + Semantic",
                scorer,
                test_candidates,
                item_train_counts,
                head_items,
                tail_items,
                args.k,
                scorer_metadata["Hybrid Mini-DLRM + Semantic"],
            )
        )

        for alpha in parse_boost_alphas(args):
            boosted_name = boosted_scorer_name(args.cold_start_boost_mode, alpha)
            boosted_scorer = make_cold_start_boost_scorer(
                scorer,
                item_train_counts,
                alpha=alpha,
                mode=args.cold_start_boost_mode,
                threshold=args.cold_start_threshold,
            )
            scorers[boosted_name] = boosted_scorer
            scorer_metadata[boosted_name] = {
                "cold_start_boost_mode": args.cold_start_boost_mode,
                "cold_start_boost_alpha": alpha,
            }
            result_rows.append(
                evaluate_scorer_row(
                    boosted_name,
                    boosted_scorer,
                    test_candidates,
                    item_train_counts,
                    head_items,
                    tail_items,
                    args.k,
                    scorer_metadata[boosted_name],
                )
            )

    overall = add_metadata_columns(pd.DataFrame(result_rows), args, config, data)
    long_tail = add_metadata_columns(
        evaluate_long_tail(
            scorers,
            test_candidates,
            head_items,
            tail_items,
            item_train_counts,
            args.k,
            scorer_metadata,
        ),
        args,
        config,
        data,
    )
    return overall, long_tail


def write_outputs(args: argparse.Namespace, overall: pd.DataFrame, long_tail: pd.DataFrame) -> None:
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{slugify(args.run_name)}_{args.run_timestamp}"
    overall_path = results_dir / f"ablation_overall_{stem}.csv"
    long_tail_path = results_dir / f"ablation_long_tail_{stem}.csv"
    overall_summary_path = results_dir / f"ablation_overall_summary_{stem}.csv"
    long_tail_summary_path = results_dir / f"ablation_long_tail_summary_{stem}.csv"

    metric_cols = [
        f"Recall@{args.k}",
        f"HitRate@{args.k}",
        f"NDCG@{args.k}",
        f"HeadRecall@{args.k}",
        f"HeadHitRate@{args.k}",
        f"HeadNDCG@{args.k}",
        f"TailRecall@{args.k}",
        f"TailHitRate@{args.k}",
        f"TailNDCG@{args.k}",
        "NumHeadEvalUsers",
        "NumTailEvalUsers",
        "NumHeadEvalPositives",
        "NumTailEvalPositives",
        f"TailExposure@{args.k}",
        f"AvgTrainPopularity@{args.k}",
    ]
    overall_summary = summarize_metrics(overall, metric_cols)
    long_tail_summary = summarize_metrics(long_tail, metric_cols, extra_group_cols=["Group"])

    overall.to_csv(overall_path, index=False)
    long_tail.to_csv(long_tail_path, index=False)
    overall_summary.to_csv(overall_summary_path, index=False)
    long_tail_summary.to_csv(long_tail_summary_path, index=False)

    print(f"Saved overall metrics to {overall_path}")
    print(f"Saved long-tail metrics to {long_tail_path}")
    print(f"Saved overall summary to {overall_summary_path}")
    print(f"Saved long-tail summary to {long_tail_summary_path}")
    print("\nOverall summary:")
    print(overall_summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled ablations for the semantic hybrid recommender.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", dest="quick", action="store_true", default=True, help="Run a capped smoke-size sweep.")
    mode.add_argument("--full", dest="quick", action="store_false", help="Run the full requested grid.")
    parser.add_argument("--reviews", default="data/raw/movies_tv_reviews_subset.jsonl")
    parser.add_argument("--metadata", default="data/raw/movies_tv_metadata_subset.jsonl")
    parser.add_argument("--dataset-name", default="Movies_and_TV")
    parser.add_argument("--run-name", default="ablation_quick")
    parser.add_argument("--run-timestamp", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--models", default="popularity,two_tower,mini_dlrm,hybrid")
    parser.add_argument("--seeds", default="42,7")
    parser.add_argument("--epochs", default="1")
    parser.add_argument("--learning-rates", default="0.001")
    parser.add_argument("--embedding-dims", default="32")
    parser.add_argument("--train-negatives", default="2")
    parser.add_argument("--quick-configs", type=int, default=2)
    parser.add_argument("--eval-negatives", type=int, default=99)
    parser.add_argument(
        "--num-test-positives",
        type=int,
        default=1,
        choices=[1, 3, 5],
        help="Number of last interactions per user to hold out as test positives. Default 1 preserves prior behavior.",
    )
    parser.add_argument("--max-interactions", type=int, default=None)
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--min-item-interactions", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Training device. auto prefers Apple Silicon MPS, then CUDA, then CPU.",
    )
    parser.add_argument(
        "--eval-user-limit",
        type=int,
        default=None,
        help="Optional validation candidate limit during training. Final test evaluation remains full.",
    )
    parser.add_argument("--semantic-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--cold-start-boost-mode",
        choices=["none", "threshold", "inverse_popularity"],
        default="none",
        help="Optional post-ranking cold-start boost. none keeps existing scoring unchanged.",
    )
    parser.add_argument(
        "--cold-start-boost-alpha",
        default="0.0",
        help="Comma-separated boost strengths, used only when cold-start boost mode is not none.",
    )
    parser.add_argument(
        "--cold-start-threshold",
        type=int,
        default=2,
        help="Train interaction threshold for threshold-mode cold-start boost.",
    )
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = parse_models(args.models)
    configs = build_configs(args)
    if not configs:
        raise ValueError("No ablation configs were generated.")
    print(f"Mode: {'quick' if args.quick else 'full'}")
    print(f"Models: {models}")
    print(f"Configs: {len(configs)}")

    data = prepare_data(args)
    print(f"Prepared data: {data.stats}")

    overall_frames = []
    long_tail_frames = []
    for config in configs:
        overall, long_tail = run_one_config(args, config, data, models)
        overall_frames.append(overall)
        long_tail_frames.append(long_tail)

    write_outputs(args, pd.concat(overall_frames, ignore_index=True), pd.concat(long_tail_frames, ignore_index=True))


if __name__ == "__main__":
    main()
