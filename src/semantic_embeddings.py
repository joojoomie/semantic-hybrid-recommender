from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_MODEL_NAMES = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2",
)


def _stringify(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(map(str, value))
    if isinstance(value, dict):
        return " ".join(map(str, value.values()))
    if pd.isna(value):
        return ""
    return str(value)


def build_item_text(metadata_df: pd.DataFrame | None, item_mapping: dict[str, int]) -> list[str]:
    """Build text per item_idx from title, category, categories, and description."""
    texts = [""] * len(item_mapping)
    if metadata_df is None or metadata_df.empty:
        for item_id, item_idx in item_mapping.items():
            texts[item_idx] = f"item {item_id}"
        return texts

    metadata = metadata_df.drop_duplicates("item_id").set_index("item_id")
    for item_id, item_idx in item_mapping.items():
        if item_id not in metadata.index:
            texts[item_idx] = f"item {item_id}"
            continue
        row = metadata.loc[item_id]
        title = _stringify(row.get("title", ""))
        main_category = _stringify(row.get("main_category", ""))
        categories = _stringify(row.get("categories", ""))
        description = _stringify(row.get("description", ""))
        text = f"{title} [CATEGORY] {main_category} {categories} [DESC] {description}".strip()
        texts[item_idx] = text or f"item {item_id}"
    return texts


def generate_sentence_transformer_embeddings(
    texts: list[str],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 64,
) -> np.ndarray:
    """Generate frozen semantic item embeddings with a SentenceTransformer model."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError("sentence_transformers is unavailable") from exc

    encoded_texts = texts
    if "e5" in model_name.lower():
        encoded_texts = [f"passage: {text}" for text in texts]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        encoded_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def generate_tfidf_svd_embeddings(texts: list[str], n_components: int = 384) -> np.ndarray:
    """Offline-safe semantic fallback using TF-IDF followed by TruncatedSVD."""
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    max_components = max(1, min(n_components, matrix.shape[0] - 1, matrix.shape[1] - 1))
    if max_components < 1:
        return np.zeros((len(texts), n_components), dtype=np.float32)
    svd = TruncatedSVD(n_components=max_components, random_state=42)
    reduced = svd.fit_transform(matrix)
    if reduced.shape[1] < n_components:
        padding = np.zeros((len(texts), n_components - reduced.shape[1]), dtype=np.float32)
        reduced = np.hstack([reduced, padding])
    reduced = normalize(reduced)
    return reduced.astype(np.float32)


def load_or_create_semantic_embeddings(
    texts: list[str],
    cache_path: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    fallback_components: int = 384,
) -> np.ndarray:
    """Load cached semantic embeddings or create and save them."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return np.load(cache_path).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if model_name.lower() in {"tfidf-svd", "tfidf_svd", "offline"}:
        print("Using TF-IDF + SVD semantic embeddings for offline-safe execution.")
        embeddings = generate_tfidf_svd_embeddings(texts, n_components=fallback_components)
    else:
        try:
            embeddings = generate_sentence_transformer_embeddings(texts, model_name=model_name)
        except Exception as exc:
            print(f"SentenceTransformer unavailable or failed ({exc}). Falling back to TF-IDF + SVD.")
            embeddings = generate_tfidf_svd_embeddings(texts, n_components=fallback_components)
    np.save(cache_path, embeddings)
    return embeddings.astype(np.float32)
