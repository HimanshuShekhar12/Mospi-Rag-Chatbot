"""
pipeline/indexer.py
--------------------
Builds a FAISS vector index from chunk embeddings and saves it to disk.

Assignment requirement:
  "Vector store: FAISS or Chroma. Use MMR/top-k with a clear, configurable k."

Two files are saved to data/processed/:
  faiss.index  — the FAISS index (vector search structure)
  chunks.pkl   — list of Chunk objects (text + metadata for citations)

Both files must exist together — the index gives us vector IDs,
chunks.pkl gives us the actual text and source info for those IDs.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

from scraper.config import settings
from scraper.utils import get_logger
from pipeline.chunker import Chunk

logger = get_logger(__name__)


# ── Build index ───────────────────────────────────────────────────────────────

def build_index(
    embeddings: np.ndarray,
    chunks: List[Chunk],
) -> faiss.Index:
    """
    Build a FAISS flat L2 index from a numpy embedding matrix.

    Args:
        embeddings : float32 array of shape (N, embedding_dim)
        chunks     : list of N Chunk objects matching the embeddings

    Returns:
        A trained and populated faiss.IndexFlatIP (inner product / cosine)

    We use IndexFlatIP (inner product) because embeddings are
    L2-normalised in embedder.py — inner product == cosine similarity.
    """
    if len(embeddings) == 0:
        raise ValueError("Cannot build index from empty embeddings")

    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Embeddings count ({len(embeddings)}) != "
            f"chunks count ({len(chunks)})"
        )

    embedding_dim = embeddings.shape[1]

    logger.info(
        "Building FAISS index",
        extra={
            "n_vectors":     len(embeddings),
            "embedding_dim": embedding_dim,
        },
    )

    # IndexFlatIP = exact nearest neighbour search using inner product
    # Good for small-medium corpora (< 100k vectors)
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)

    logger.info(
        "FAISS index built",
        extra={"total_vectors": index.ntotal},
    )
    return index


# ── Save & Load ───────────────────────────────────────────────────────────────

def save_index(
    index: faiss.Index,
    chunks: List[Chunk],
    index_path: Optional[str] = None,
    chunks_path: Optional[str] = None,
) -> None:
    """
    Save FAISS index and chunk metadata to disk.

    Args:
        index       : built FAISS index
        chunks      : list of Chunk objects
        index_path  : path for faiss.index file (default from settings)
        chunks_path : path for chunks.pkl file (default from settings)
    """
    index_path  = index_path  or settings.faiss_index_path
    chunks_path = chunks_path or settings.chunks_pkl_path

    # ensure output directory exists
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)

    # save FAISS index
    faiss.write_index(index, index_path)
    logger.info("FAISS index saved", extra={"path": index_path})

    # save chunks metadata
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(
        "Chunks metadata saved",
        extra={"path": chunks_path, "n_chunks": len(chunks)},
    )


def load_index(
    index_path:  Optional[str] = None,
    chunks_path: Optional[str] = None,
) -> Tuple[faiss.Index, List[Chunk]]:
    """
    Load FAISS index and chunk metadata from disk.

    Returns:
        (faiss.Index, List[Chunk])

    Raises:
        FileNotFoundError if either file is missing.
    """
    index_path  = index_path  or settings.faiss_index_path
    chunks_path = chunks_path or settings.chunks_pkl_path

    if not Path(index_path).exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{index_path}'. "
            "Run 'make index' or 'python -m pipeline.run' first."
        )
    if not Path(chunks_path).exists():
        raise FileNotFoundError(
            f"Chunks file not found at '{chunks_path}'. "
            "Run 'make index' or 'python -m pipeline.run' first."
        )

    index = faiss.read_index(index_path)
    logger.info(
        "FAISS index loaded",
        extra={"path": index_path, "total_vectors": index.ntotal},
    )

    with open(chunks_path, "rb") as f:
        chunks: List[Chunk] = pickle.load(f)
    logger.info(
        "Chunks metadata loaded",
        extra={"path": chunks_path, "n_chunks": len(chunks)},
    )

    return index, chunks


def index_exists() -> bool:
    """Return True if both index files exist on disk."""
    return (
        Path(settings.faiss_index_path).exists()
        and Path(settings.chunks_pkl_path).exists()
    )
