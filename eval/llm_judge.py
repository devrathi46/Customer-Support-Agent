"""LLM-as-judge: rubric-based reply-quality scoring.

Scores each drafted reply 1-5 on four axes (groundedness, tone match,
completeness, actionability) plus an overall score. `judge_agreement.py`
validates this against human scores on a subset — never trust this module's
numbers without that check.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config
from groq_client import chat_json

RUBRIC = """Score the drafted customer-support reply on these axes, 1 (poor) to 5 (excellent):

- groundedness: does it avoid inventing facts/policies/order details not present \
in the customer message or the reference notes? (5 = fully grounded, 1 = fabricates \
specifics)
- tone: does it match a professional, brand-appropriate support tone?
- completeness: does it address what the customer actually asked, per the \
reference notes (key facts a good reply must contain)?
- actionability: does it give the customer a clear next step or resolution, \
rather than being vague?

Respond with JSON only:
{"groundedness": <1-5>, "tone": <1-5>, "completeness": <1-5>, "actionability": <1-5>, \
"overall": <1-5>, "rationale": "<one sentence>"}"""


def judge_reply(customer_text: str, drafted_reply: str, reference_reply_notes: str = "") -> dict:
    user = (
        f"Customer message: {customer_text}\n\n"
        f"Key facts a good reply must contain (reference notes from human labeller): "
        f"{reference_reply_notes or '(none provided)'}\n\n"
        f"Drafted reply to score:\n{drafted_reply}"
    )
    return chat_json(system=RUBRIC, user=user, model=config.GROQ_MODEL_STRONG, temperature=0.0)
