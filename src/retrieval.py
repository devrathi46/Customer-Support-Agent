"""Embedding retrieval over historical (customer, brand-reply) pairs.

Splits threads into an index set (used for retrieval grounding) and a
held-out set (source for the golden eval set) so retrieval can't trivially
find the exact gold answer for an eval example. See config.HELDOUT_FRACTION.
"""
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

_model = None


def embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def split_index_holdout(threads: pd.DataFrame, holdout_frac: float = config.HELDOUT_FRACTION, seed: int = config.RANDOM_SEED):
    holdout = threads.sample(frac=holdout_frac, random_state=seed)
    index_set = threads.drop(holdout.index)
    return index_set.reset_index(drop=True), holdout.reset_index(drop=True)


def embed_texts(texts: list) -> np.ndarray:
    vecs = embedder().encode(list(texts), show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")


def build_index(index_set: pd.DataFrame):
    """Returns (faiss_index, metadata_df) — metadata rows align with index order."""
    vecs = embed_texts(index_set["customer_text"].tolist())
    idx = faiss.IndexFlatIP(vecs.shape[1])  # cosine sim via normalized vectors + inner product
    idx.add(vecs)
    return idx, index_set.reset_index(drop=True)


def save_index(idx, metadata: pd.DataFrame, brand: str):
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(config.PROCESSED_DIR / f"{brand}_index.faiss"))
    metadata.to_parquet(config.PROCESSED_DIR / f"{brand}_index_meta.parquet", index=False)


def load_index(brand: str):
    idx = faiss.read_index(str(config.PROCESSED_DIR / f"{brand}_index.faiss"))
    metadata = pd.read_parquet(config.PROCESSED_DIR / f"{brand}_index_meta.parquet")
    return idx, metadata


def retrieve(query_text: str, idx, metadata: pd.DataFrame, k: int = config.RETRIEVAL_TOP_K) -> pd.DataFrame:
    q = embed_texts([query_text])
    scores, indices = idx.search(q, k)
    rows = metadata.iloc[indices[0]].copy()
    rows["similarity"] = scores[0]
    return rows.reset_index(drop=True)
