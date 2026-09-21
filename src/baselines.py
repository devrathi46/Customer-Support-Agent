"""Two baselines to compare the agent against (assignment requirement):

Trivial baseline:
  - intent: always predict the single most common intent (from silver labels)
  - reply: a single canned reply (the modal historical brand reply)
  - escalation: a fixed policy — whichever of {auto_handle, escalate} the
    brand's own historical behavior (DM-deflection rate) favors overall

Simple baseline:
  - intent: TF-IDF + Logistic Regression, trained on LLM-generated "silver"
    labels over the index set (see NOTE below) — NOT the golden set, to
    avoid training on eval data
  - reply: top-1 retrieved historical brand reply, used verbatim (no LLM
    rewriting)
  - escalation: rule-only (safety-pattern regex + deflection-rate threshold,
    no LLM call)

NOTE on silver labels: we don't have brand-specific ground-truth intent
labels outside the golden set, so the simple baseline's training labels come
from the same LLM classifier used by the agent, applied to the (larger,
non-golden) index set. This is a real limitation — the simple baseline
partially inherits the LLM's own labeling biases — and should be called out
explicitly in the report's "what's misleading about my headline number"
section, not hidden.
"""
import sys
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from escalation import DEFLECTION_RATE_ESCALATE_THRESHOLD, check_safety_triggers, historical_deflection_rate
from intents import classify_intent, load_taxonomy
from retrieval import retrieve


# ---------------- trivial baseline ----------------

class TrivialBaseline:
    def __init__(self, index_set: pd.DataFrame, silver_labels: list, index_meta: pd.DataFrame):
        self.majority_intent = Counter(silver_labels).most_common(1)[0][0]
        self.canned_reply = index_set["brand_reply_text"].value_counts().idxmax()
        overall_deflection = historical_deflection_rate(index_meta)
        self.fixed_decision = "escalate" if overall_deflection >= 0.5 else "auto_handle"
        self.overall_deflection_rate = overall_deflection

    def process(self, customer_text: str) -> dict:
        return {
            "customer_text": customer_text,
            "intent": self.majority_intent,
            "drafted_reply": self.canned_reply,
            "decision": self.fixed_decision,
            "decision_reason": (
                f"Trivial policy: brand's overall historical deflection-to-DM rate is "
                f"{self.overall_deflection_rate:.2f}, so always predict "
                f"'{self.fixed_decision}'."
            ),
        }


# ---------------- simple baseline ----------------

def build_silver_labels(index_set: pd.DataFrame, taxonomy: list, sample_size: int = 1500, seed: int = config.RANDOM_SEED):
    sample = index_set.sample(min(sample_size, len(index_set)), random_state=seed)
    labels = []
    for text in tqdm(sample["customer_text"], desc="silver-labeling for baseline training"):
        labels.append(classify_intent(text, taxonomy)["intent"])
    return sample["customer_text"].tolist(), labels


class SimpleBaseline:
    def __init__(self, index_set: pd.DataFrame, faiss_index, index_meta: pd.DataFrame, taxonomy: list = None):
        self.taxonomy = taxonomy or load_taxonomy()
        texts, labels = build_silver_labels(index_set, self.taxonomy)
        self.silver_texts = texts
        self.silver_labels = labels
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", min_df=2)
        X = self.vectorizer.fit_transform(texts)
        self.clf = LogisticRegression(max_iter=1000)
        self.clf.fit(X, labels)
        self.faiss_index = faiss_index
        self.index_meta = index_meta

    def predict_intent(self, text: str) -> str:
        X = self.vectorizer.transform([text])
        return self.clf.predict(X)[0]

    def process(self, customer_text: str) -> dict:
        intent = self.predict_intent(customer_text)

        retrieved = retrieve(customer_text, self.faiss_index, self.index_meta, k=1)
        reply = retrieved.iloc[0]["brand_reply_text"] if not retrieved.empty else ""

        safety_hit = check_safety_triggers(customer_text)
        if safety_hit:
            decision, reason = "escalate", f"Rule trigger: {safety_hit}."
        else:
            top5 = retrieve(customer_text, self.faiss_index, self.index_meta, k=config.RETRIEVAL_TOP_K)
            rate = historical_deflection_rate(top5)
            decision = "escalate" if rate >= DEFLECTION_RATE_ESCALATE_THRESHOLD else "auto_handle"
            reason = f"Rule-only policy: historical deflection rate {rate:.2f} vs threshold {DEFLECTION_RATE_ESCALATE_THRESHOLD}."

        return {
            "customer_text": customer_text,
            "intent": intent,
            "drafted_reply": reply,
            "decision": decision,
            "decision_reason": reason,
        }

    def save(self, brand: str):
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump((self.vectorizer, self.clf), config.PROCESSED_DIR / f"{brand}_simple_baseline.joblib")
