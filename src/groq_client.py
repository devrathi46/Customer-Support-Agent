"""Thin Groq wrapper: retries + JSON-mode helper, single place model choice lives."""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

load_dotenv(config.ROOT / ".env")

_client = None


def client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set. Copy .env.example to .env and fill it in.")
        _client = Groq(api_key=api_key)
    return _client


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20))
def chat_json(system: str, user: str, model: str = config.GROQ_MODEL_FAST, temperature: float = 0.0) -> dict:
    """Call Groq chat completion, forcing JSON-object output, and parse it."""
    resp = client().chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content
    return json.loads(content)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20))
def chat_text(system: str, user: str, model: str = config.GROQ_MODEL_STRONG, temperature: float = 0.3) -> str:
    resp = client().chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()
