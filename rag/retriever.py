"""
rag/retriever.py
-----------------
Vector search retriever for the RAG chatbot.

At query time:
  1. Embed the user's question into a vector
  2. Search the FAISS index for the top-k most similar chunks
  3. Return chunks with scores and full source metadata

Assignment requirement:
  "At query time: retrieve k chunks, build a context window"
  "Use MMR/top-k with a clear, configurable k"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from scraper.config import settings
from scraper.utils import get_logger
from pipeline.chunker import Chunk
from pipeline.embedder import embed_query
from pipeline.indexer import load_index

logger = get_logger(__name__)


# ── Retrieved chunk model ─────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    A Chunk returned by the retriever, with its similarity score.

    Used by prompt.py to build the context window and by the
    API/UI to display source citations.
    """
    chunk:   Chunk
    score:   float    # cosine similarity score (0–1, higher = more relevant)
    rank:    int      # position in results (1 = most relevant)

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def title(self) -> str:
        return self.chunk.title

    @property
    def url(self) -> str:
        return self.chunk.url

    @property
    def category(self) -> str:
        return self.chunk.category

    def to_dict(self) -> dict:
        """Serialise for API response / JSON output."""
        return {
            "rank":     self.rank,
            "score":    round(float(self.score), 4),
            "title":    self.title,
            "url":      self.url,
            "category": self.category,
            "snippet":  self.text[:300] + "..." if len(self.text) > 300 else self.text,
        }


# ── Retriever ─────────────────────────────────────────────────────────────────

class Retriever:
    """
    Loads the FAISS index once and answers repeated queries efficiently.

    Usage:
        retriever = Retriever()
        results = retriever.search("What is India's GDP growth rate?", k=5)
        for r in results:
            print(r.score, r.title, r.url)
    """

    def __init__(self) -> None:
        self._index  = None
        self._chunks: List[Chunk] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load the index on first query."""
        if not self._loaded:
            self._index, self._chunks = load_index()
            self._loaded = True
            logger.info(
                "Retriever ready",
                extra={
                    "n_vectors": self._index.ntotal,
                    "n_chunks":  len(self._chunks),
                },
            )

    def search(
        self,
        query: str,
        k:     int = settings.retrieval_top_k,
    ) -> List[RetrievedChunk]:
        """
        Search for the top-k chunks most relevant to the query.

        Args:
            query : user question string
            k     : number of results to return (configurable via .env)

        Returns:
            List of RetrievedChunk sorted by relevance (best first).
        """
        self._ensure_loaded()

        if not query or not query.strip():
            logger.warning("Empty query received by retriever")
            return []

        # embed the query
        query_vector = embed_query(query)
        query_vector = query_vector.reshape(1, -1)  # FAISS needs 2D array

        # clamp k to available vectors
        k = min(k, self._index.ntotal)
        if k == 0:
            return []

        # search FAISS index
        scores, indices = self._index.search(query_vector, k)

        results: List[RetrievedChunk] = []
        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]), start=1
        ):
            if idx < 0 or idx >= len(self._chunks):
                continue  # FAISS returns -1 for empty slots

            results.append(
                RetrievedChunk(
                    chunk=self._chunks[idx],
                    score=float(score),
                    rank=rank,
                )
            )

        logger.info(
            "Search complete",
            extra={
                "query":   query[:60],
                "k":       k,
                "results": len(results),
            },
        )
        return results

    def reload(self) -> None:
        """Force reload the index from disk (called after /ingest)."""
        self._loaded = False
        self._ensure_loaded()


# ── Singleton retriever ───────────────────────────────────────────────────────
# Shared across API requests — index loaded only once per server start
retriever = Retriever()
