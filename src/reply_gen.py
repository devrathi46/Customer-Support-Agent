"""Draft a reply grounded in retrieved historical brand resolutions."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from groq_client import chat_text

SYSTEM_PROMPT = """You are drafting a customer-support reply on behalf of a brand's \
Twitter support account. You are given the customer's message, its intent, and \
several examples of how this brand has actually replied to similar past issues.

Rules:
- Match the brand's real tone and conventions from the examples (length, \
signature style, whether it asks the customer to DM, etc.).
- The examples' brand replies start with "@<customer handle>" because that's how \
Twitter replies are addressed — that is handled separately by the platform, not \
part of the message content. Do NOT start your reply with an @mention or any \
placeholder like "@yourhandle"; write only the reply body.
- Only state facts/policies that appear in the examples or the customer's own \
message. Do not invent order numbers, refund amounts, timelines, or policies \
not evidenced in the examples.
- If the examples don't clearly cover this case, write a brief, honest reply \
that acknowledges the issue and asks for the specific info needed to help \
(don't fabricate a resolution).
- Output only the reply text, no preamble, no quotes."""


def format_examples(retrieved: pd.DataFrame) -> str:
    blocks = []
    for i, row in retrieved.iterrows():
        blocks.append(
            f"Example {i + 1} (similarity={row['similarity']:.2f}):\n"
            f"Customer: {row['customer_text']}\n"
            f"Brand reply: {row['brand_reply_text']}"
        )
    return "\n\n".join(blocks)


def draft_reply(customer_text: str, intent: str, retrieved: pd.DataFrame) -> str:
    user = (
        f"Customer message: {customer_text}\n"
        f"Predicted intent: {intent}\n\n"
        f"Similar past resolutions from this brand:\n{format_examples(retrieved)}\n\n"
        "Draft the brand's reply to the customer message above."
    )
    return chat_text(system=SYSTEM_PROMPT, user=user, model=config.GROQ_MODEL_STRONG)
