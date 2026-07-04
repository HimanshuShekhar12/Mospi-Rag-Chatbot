"""
rag/llm.py
-----------
LLaMA 3 client for the RAG chatbot, served via the Groq API.

Groq hosts LLaMA 3 Instruct models and exposes them over a fast,
free-tier HTTP API — so the chatbot runs without a local Ollama
server or Docker. Set GROQ_API_KEY in .env (get a free key at
https://console.groq.com/keys).

Assignment requirement:
  "Generator: LLaMA 3 Instruct (via Ollama or llama.cpp)"
  "A toggle for k and temperature"
"""

from typing import Generator, List

from groq import Groq, GroqError

from scraper.config import settings
from scraper.utils import get_logger

logger = get_logger(__name__)


def _client() -> Groq:
    """Build a Groq client, failing clearly if no API key is set."""
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file. "
            "Get a free key at https://console.groq.com/keys"
        )
    return Groq(api_key=settings.groq_api_key)


# ── Health check ──────────────────────────────────────────────────────────────
def is_llm_available() -> bool:
    """True if the LLM backend is reachable and the model is available."""
    if not settings.groq_api_key:
        logger.error("GROQ_API_KEY not configured")
        return False
    try:
        models = _client().models.list()
        model_ids = [m.id for m in models.data]
        available = settings.groq_model in model_ids
        if not available:
            logger.warning(
                "Configured model not found on Groq",
                extra={"expected": settings.groq_model, "available": model_ids},
            )
        return available
    except Exception as exc:
        logger.error("Groq API not reachable", extra={"error": str(exc)})
        return False


# Backwards-compatible alias (health endpoint / older callers).
is_ollama_running = is_llm_available


# ── Generate ──────────────────────────────────────────────────────────────────
def generate(
    messages:    List[dict],
    temperature: float = settings.llm_temperature,
    max_tokens:  int   = settings.llm_max_tokens,
) -> str:
    """
    Send a prompt to LLaMA 3 (via Groq) and return the full response.

    Args:
        messages    : list of {"role": ..., "content": ...} dicts
        temperature : controls randomness (0.0 = focused, 1.0 = creative)
        max_tokens  : maximum tokens in the response

    Returns:
        The model's response as a plain string.

    Raises:
        RuntimeError if the Groq API is unreachable or misconfigured.
    """
    try:
        logger.info(
            "Calling LLaMA via Groq",
            extra={
                "model":       settings.groq_model,
                "temperature": temperature,
                "max_tokens":  max_tokens,
            },
        )

        response = _client().chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        answer = (response.choices[0].message.content or "").strip()
        logger.info("LLaMA response received", extra={"response_length": len(answer)})
        return answer

    except GroqError as exc:
        logger.error("Groq API error", extra={"error": str(exc)})
        raise RuntimeError(
            f"Groq error: {exc}. Check your GROQ_API_KEY and model name."
        ) from exc

    except Exception as exc:
        logger.error("Unexpected LLM error", extra={"error": str(exc)})
        raise RuntimeError(
            "Could not reach the Groq API. "
            "Set GROQ_API_KEY in .env (https://console.groq.com/keys)."
        ) from exc


def generate_stream(
    messages:    List[dict],
    temperature: float = settings.llm_temperature,
    max_tokens:  int   = settings.llm_max_tokens,
) -> Generator[str, None, None]:
    """
    Stream LLaMA 3 tokens back one at a time as a generator.

    Yields:
        Individual token strings as they arrive from Groq.

    Usage:
        for token in generate_stream(messages):
            print(token, end="", flush=True)
    """
    try:
        stream = _client().chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    except Exception as exc:
        logger.error("Streaming error", extra={"error": str(exc)})
        yield "\n\n[Error: Could not reach the Groq API. Is GROQ_API_KEY set?]"
