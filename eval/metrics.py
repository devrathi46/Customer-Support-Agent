"""Automated metrics: intent accuracy/F1, escalation P/R/F1, retrieval hit-rate@k."""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def intent_metrics(y_true: list, y_pred: list) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {k: v for k, v in report.items() if k not in ("accuracy", "macro avg", "weighted avg")},
    }


def escalation_metrics(y_true: list, y_pred: list, positive_label: str = "escalate") -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=positive_label, zero_division=0
    )
    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def retrieval_hit_rate_at_k(gold_relevant_ids: list, retrieved_ids_lists: list) -> float:
    """gold_relevant_ids[i]: set of thread_ids considered relevant for example i.
    retrieved_ids_lists[i]: list of retrieved thread_ids (top-k) for example i.
    Hit if any overlap. Use only where a real "same underlying issue" label exists
    (e.g. via intent match) — see run_eval.py for how this is derived."""
    hits = 0
    for gold, retrieved in zip(gold_relevant_ids, retrieved_ids_lists):
        if set(gold) & set(retrieved):
            hits += 1
    return hits / len(gold_relevant_ids) if gold_relevant_ids else float("nan")
