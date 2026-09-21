"""Intent taxonomy definition (data-driven) + Groq-based classifier.

Taxonomy is brand-specific and NOT hardcoded here: `suggest_clusters()` does
a quick embedding + KMeans pass over a sample of customer messages to assist
manual open-coding (Phase 2). The human-reviewed result is saved to
data/eda/intent_taxonomy.json as a list of {"name", "description"} objects,
which `classify_intent` then loads and uses as the fixed label set.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from groq_client import chat_json

TAXONOMY_PATH = config.EDA_DIR / "intent_taxonomy.json"


def fit_tfidf_kmeans(texts: pd.Series, n_clusters: int = 10, seed: int = config.RANDOM_SEED):
    """Fit TF-IDF + KMeans and return (vectorizer, kmeans, labels) for `texts`
    (already deduped/sampled by the caller). Local, no API cost."""
    vec = TfidfVectorizer(max_features=5000, stop_words="english", min_df=3)
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    return vec, km, labels


def suggest_clusters(texts: pd.Series, n_clusters: int = 10, sample_size: int = 3000, seed: int = config.RANDOM_SEED):
    """Cheap, local (no API) clustering pass to bootstrap taxonomy design.

    Returns a dict cluster_id -> {top_terms, example_texts} for manual review.
    Uses TF-IDF, not the sentence-transformer, so this can run before the
    retrieval index / embedding model is even loaded.
    """
    texts = texts.dropna()
    if len(texts) > sample_size:
        texts = texts.sample(sample_size, random_state=seed)
    texts = texts.reset_index(drop=True)

    vec, km, labels = fit_tfidf_kmeans(texts, n_clusters=n_clusters, seed=seed)

    terms = np.array(vec.get_feature_names_out())
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    clusters = {}
    for i in range(n_clusters):
        top_terms = list(terms[order_centroids[i, :12]])
        examples = list(texts[labels == i].head(5))
        clusters[i] = {"size": int((labels == i).sum()), "top_terms": top_terms, "examples": examples}
    return clusters


def load_taxonomy() -> list:
    if not TAXONOMY_PATH.exists():
        raise FileNotFoundError(
            f"{TAXONOMY_PATH} not found. Run the clustering pass (suggest_clusters), "
            "hand-review the output, and save it as a JSON list of {name, description}."
        )
    return json.loads(TAXONOMY_PATH.read_text())


def _system_prompt(taxonomy: list) -> str:
    options = "\n".join(f"- {t['name']}: {t['description']}" for t in taxonomy)
    return (
        "You are an intent classifier for customer support tweets sent to a brand. "
        "Classify the customer's message into exactly one of these intents:\n"
        f"{options}\n\n"
        'Respond with JSON only: {"intent": "<one of the intent names above>", "confidence": <0-1 float>}'
    )


def classify_intent(text: str, taxonomy: list = None) -> dict:
    taxonomy = taxonomy or load_taxonomy()
    system = _system_prompt(taxonomy)
    result = chat_json(system=system, user=text, model=config.GROQ_MODEL_FAST)
    valid_names = {t["name"] for t in taxonomy}
    if result.get("intent") not in valid_names:
        result["intent"] = "Other"
        result["confidence"] = 0.0
    return result


def classify_batch(texts: list, taxonomy: list = None) -> list:
    taxonomy = taxonomy or load_taxonomy()
    return [classify_intent(t, taxonomy) for t in tqdm(texts, desc="classifying intent")]
