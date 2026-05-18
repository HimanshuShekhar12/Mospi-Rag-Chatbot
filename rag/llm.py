"""
rag/llm.py
-----------
Ollama LLaMA client for the RAG chatbot.

Calls the local Ollama server with a chat prompt and returns
the response — either as a complete string or as a stream of tokens.

Assignment requirement:
  "Generator: LLaMA 3 Instruct (via Ollama or llama.cpp)"
  "A toggle for k and temperature"
"""

import os
from typing import Generator, List, Optional

import ollama
from ollama import Client, ResponseError

from scraper.config import settings
from scraper.utils import get_logger

logger = get_logger(__name__)

_client = Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))


# ── Health check ──────────────────────────────────────────────────────────────
def is_ollama_running() -> bool:
    try:
        result = _client.list()
        model_names = [m.model for m in result.models]
        available = any(settings.ollama_model in name for name in model_names)
        if not available:
            logger.warning(
                "LLaMA model not found in Ollama",
                extra={"expected": settings.ollama_model, "available": model_names},
            )
        return available
    except Exception as exc:
        logger.error("Ollama server not reachable", extra={"error": str(exc)})
        return False


# ── Generate ──────────────────────────────────────────────────────────────────

def generate(
    messages:    List[dict],
    temperature: float = settings.llm_temperature,
    max_tokens:  int   = settings.llm_max_tokens,
) -> str:
    """
    Send a prompt to LLaMA and return the complete response as a string.

    Args:
        messages    : list of {"role": ..., "content": ...} dicts
        temperature : controls randomness (0.0 = focused, 1.0 = creative)
        max_tokens  : maximum tokens in the response

    Returns:
        The model's response as a plain string.

    Raises:
        RuntimeError if Ollama is not running or the model is unavailable.
    """
    try:
        logger.info(
            "Calling LLaMA",
            extra={
                "model":       settings.ollama_model,
                "temperature": temperature,
                "max_tokens":  max_tokens,
            },
        )

        response = _client.chat(
            model=settings.ollama_model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )

        answer = response["message"]["content"].strip()
        logger.info(
            "LLaMA response received",
            extra={"response_length": len(answer)},
        )
        return answer

    except ResponseError as exc:
        logger.error(
            "Ollama API error",
            extra={"error": str(exc)},
        )
        raise RuntimeError(
            f"Ollama error: {exc}. "
            f"Make sure Ollama is running: ollama serve"
        ) from exc

    except Exception as exc:
        logger.error(
            "Unexpected LLM error",
            extra={"error": str(exc)},
        )
        raise RuntimeError(
            "Could not connect to Ollama. "
            "Run: ollama serve  and  ollama pull llama3:8b-instruct-q4_0"
        ) from exc


def generate_stream(
    messages:    List[dict],
    temperature: float = settings.llm_temperature,
    max_tokens:  int   = settings.llm_max_tokens,
) -> Generator[str, None, None]:
    """
    Stream LLaMA tokens back one at a time as a generator.

    Used by the Streamlit UI for a real-time typewriter effect.

    Yields:
        Individual token strings as they arrive from Ollama.

    Usage:
        for token in generate_stream(messages):
            print(token, end="", flush=True)
    """
    try:
        stream = _client.chat(
            model=settings.ollama_model,
            messages=messages,
            stream=True,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token

    except Exception as exc:
        logger.error(
            "Streaming error",
            extra={"error": str(exc)},
        )
        yield "\n\n[Error: Could not reach Ollama. Is it running?]"
