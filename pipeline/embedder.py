"""
pipeline/embedder.py
---------------------
Converts text chunks into dense vector embeddings using SentenceTransformers.

Assignment requirement:
  "Embeddings: any open model (e.g. all-MiniLM-L6-v2) via SentenceTransformers"

The embedder is a singleton — model is loaded once and reused across calls
to avoid reloading the 80MB model file on every batch.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from scraper.config import settings
from scraper.utils import get_logger
from pipeline.chunker import Chunk

logger = get_logger(__name__)

# ── Singleton model ───────────────────────────────────────────────────────────

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """
    Load the SentenceTransformer model once and cache it globally.
    Subsequent calls return the cached model immediately.
    """
    global _model
    if _model is None:
        logger.info(
            "Loading embedding model",
            extra={"model": settings.embedding_model},
        )
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model loaded")
    return _model


# ── Public API ────────────────────────────────────────────────────────────────

def embed_chunks(
    chunks: List[Chunk],
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed a list of Chunks into a 2D numpy array of float32 vectors.

    Args:
        chunks        : list of Chunk objects to embed
        batch_size    : how many chunks to embed at once (memory vs speed)
        show_progress : show tqdm progress bar during embedding

    Returns:
        np.ndarray of shape (len(chunks), embedding_dim)
        e.g. for all-MiniLM-L6-v2: (N, 384)

    Raises:
        ValueError if chunks list is empty
    """
    if not chunks:
        raise ValueError("embed_chunks called with empty chunk list")

    model  = _get_model()
    texts  = [chunk.text for chunk in chunks]

    logger.info(
        "Embedding chunks",
        extra={"n_chunks": len(texts), "batch_size": batch_size},
    )

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalise — better cosine similarity
    )

    logger.info(
        "Embedding complete",
        extra={
            "n_chunks": len(chunks),
            "shape":    list(embeddings.shape),
        },
    )

    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string into a 1D float32 vector.

    Used by the retriever at query time.

    Returns:
        np.ndarray of shape (embedding_dim,)  e.g. (384,)
    """
    if not query or not query.strip():
        raise ValueError("Query string cannot be empty")

    model = _get_model()

    vector = model.encode(
        [query.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return vector[0].astype(np.float32)


def get_embedding_dim() -> int:
    """Return the dimensionality of the current embedding model."""
    model = _get_model()
    return model.get_sentence_embedding_dimension()
