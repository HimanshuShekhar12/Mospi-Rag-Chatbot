"""
rag/prompt.py
--------------
Builds the prompt sent to LLaMA at query time.

Assignment requirement:
  System prompt: "Answer strictly from the provided context.
  If not found, say 'I don't have that in my data.'
  Include bullet citations with the source title and URL."

Two functions:
  build_context_window() : formats retrieved chunks into a readable context block
  build_prompt()         : assembles the full system + user prompt
"""

from typing import List

from rag.retriever import RetrievedChunk

# ── Constants ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful research assistant for Indian government \
statistical data from the Ministry of Statistics and Programme Implementation (MoSPI).

Answer STRICTLY from the provided context below.
- If the answer is not found in the context, respond with exactly:
  "I don't have that in my data."
- Do NOT use any prior knowledge outside the provided context.
- Keep answers concise and factual.
- At the end of your answer, always include a "Sources:" section with
  bullet points listing the title and URL of every source you used.

Format your Sources like this:
Sources:
• [Title of Document](URL)
• [Another Document](URL)
"""

# maximum characters allowed in context window
# keeps the prompt within LLaMA's context length
MAX_CONTEXT_CHARS = 6000


# ── Builders ──────────────────────────────────────────────────────────────────

def build_context_window(chunks: List[RetrievedChunk]) -> str:
    """
    Format a list of RetrievedChunks into a numbered context block.

    Each chunk shows its rank, source title, URL and text content.
    Total length is capped at MAX_CONTEXT_CHARS to stay within
    LLaMA's context window.

    Example output:
        [1] GDP Advance Estimate Q3 2024
            Source: https://mospi.gov.in/press-releases/gdp-q3-2024
            The National Statistical Office releases GDP estimates...

        [2] Consumer Price Index December 2023
            ...
    """
    if not chunks:
        return "No relevant context found."

    parts = []
    total_chars = 0

    for chunk in chunks:
        block = (
            f"[{chunk.rank}] {chunk.title}\n"
            f"    Source: {chunk.url}\n"
            f"    {chunk.text}\n"
        )
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break   # stop adding chunks once we hit the limit
        parts.append(block)
        total_chars += len(block)

    return "\n".join(parts)


def build_prompt(
    question: str,
    chunks:   List[RetrievedChunk],
) -> List[dict]:
    """
    Build the full message list for the Ollama chat API.

    Returns a list of message dicts in OpenAI/Ollama chat format:
        [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
        ]

    Args:
        question : the user's question
        chunks   : retrieved chunks from the vector search
    """
    context = build_context_window(chunks)

    user_message = (
        f"Context:\n"
        f"{'─' * 60}\n"
        f"{context}\n"
        f"{'─' * 60}\n\n"
        f"Question: {question.strip()}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]


def format_citations(chunks: List[RetrievedChunk]) -> List[dict]:
    """
    Format retrieved chunks as a clean list of citation objects.
    Used by the API to return structured citations in the response.

    Returns list of dicts:
        [{"title": ..., "url": ..., "snippet": ..., "score": ...}]
    """
    return [chunk.to_dict() for chunk in chunks]
