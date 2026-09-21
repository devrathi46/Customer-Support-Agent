"""Auto-handle vs. escalate decision: rule triggers + historical deflection
signal + an LLM judgment call. Every decision carries a `reason` string —
that's a stated deliverable requirement, not debug output.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from groq_client import chat_json

# hard rule triggers: never auto-handle these regardless of model opinion
SAFETY_PATTERNS = [
    (r"\blawyer\b|\blawsuit\b|\bsue\b|legal action", "legal threat language"),
    (r"\bfraud\b|unauthorized charge|hacked|stolen", "possible fraud/security issue"),
    (r"suicide|kill myself|self.?harm", "safety/self-harm language"),
    (r"\bpolice\b|\bcops\b", "law enforcement mention"),
]
DM_DEFLECTION_RE = re.compile("|".join(config.DM_DEFLECTION_PATTERNS), re.IGNORECASE)

DEFLECTION_RATE_ESCALATE_THRESHOLD = 0.6

SYSTEM_PROMPT = """You are deciding whether an incoming customer support message \
can be auto-handled with a drafted reply, or must be escalated to a human agent.

Escalate when: the issue needs account-specific action the brand can't do over \
public/scripted reply (refunds, account access, order changes), the customer is \
clearly frustrated/repeating themselves, or the retrieved historical examples show \
this brand itself usually hands this off to a human channel.

Auto-handle when: it's a general question, status check, or something the brand's \
past replies resolved directly and completely in-thread.

Respond with JSON only: {"decision": "auto_handle" or "escalate", "reason": "<one \
sentence, concrete, referencing what drove the decision>"}"""


def check_safety_triggers(text: str):
    for pattern, label in SAFETY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return label
    return None


def historical_deflection_rate(retrieved: pd.DataFrame) -> float:
    if retrieved.empty:
        return 0.0
    hits = retrieved["brand_reply_text"].apply(lambda t: bool(DM_DEFLECTION_RE.search(t)))
    return float(hits.mean())


def decide(customer_text: str, intent: str, retrieved: pd.DataFrame) -> dict:
    safety_hit = check_safety_triggers(customer_text)
    if safety_hit:
        return {
            "decision": "escalate",
            "reason": f"Rule trigger: {safety_hit}.",
            "signals": {"safety_trigger": safety_hit, "deflection_rate": None},
        }

    deflection_rate = historical_deflection_rate(retrieved)

    user = (
        f"Customer message: {customer_text}\n"
        f"Predicted intent: {intent}\n"
        f"Historical deflection rate for similar past issues (fraction of retrieved "
        f"examples where this brand redirected to DM/phone instead of resolving "
        f"publicly): {deflection_rate:.2f}\n\n"
        "Decide auto_handle or escalate."
    )
    result = chat_json(system=SYSTEM_PROMPT, user=user, model=config.GROQ_MODEL_FAST)
    result.setdefault("decision", "escalate")
    result.setdefault("reason", "Model did not provide a reason; defaulting to escalate.")
    result["signals"] = {"safety_trigger": None, "deflection_rate": deflection_rate}
    return result
