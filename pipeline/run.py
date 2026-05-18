"""
pipeline/run.py
----------------
Master ETL pipeline — runs all stages in order:

  1. Load documents from SQLite
  2. Validate documents (quality checks)
  3. Chunk valid documents into overlapping text segments
  4. Embed chunks into float32 vectors
  5. Build and save FAISS index
  6. Export documents to Parquet
  7. Generate datasets/catalog.json

Assignment requirement:
  "Artifacts should land under /data/processed and be reproducible
   by a single command: make etl  # or: python -m pipeline.run"

Entry point:
    python -m pipeline.run
    make etl
"""

import sys
import time
from pathlib import Path

import pandas as pd

from scraper.config import settings
from scraper.db import get_all_documents
from scraper.utils import get_logger
from pipeline.validate import validate_documents
from pipeline.chunker import chunk_documents
from pipeline.embedder import embed_chunks
from pipeline.indexer import build_index, save_index
from pipeline.catalog import build_catalog

logger = get_logger(__name__)


# ── Stage runners ─────────────────────────────────────────────────────────────

def stage_load() -> list:
    """Stage 1 — Load all documents from SQLite."""
    logger.info("Stage 1/7 — Loading documents from database")
    docs = get_all_documents()
    if not docs:
        logger.error(
            "No documents found. Run scraper first: python -m scraper.crawl"
        )
        sys.exit(1)
    logger.info("Documents loaded", extra={"count": len(docs)})
    return docs


def stage_validate(docs: list) -> tuple:
    """Stage 2 — Validate document quality."""
    logger.info("Stage 2/7 — Validating documents")
    valid_docs, results = validate_documents(docs)
    logger.info(
        "Validation done",
        extra={"valid": len(valid_docs), "total": len(docs)},
    )
    return valid_docs, results


def stage_chunk(valid_docs: list) -> list:
    """Stage 3 — Split documents into overlapping chunks."""
    logger.info("Stage 3/7 — Chunking documents")
    chunks = chunk_documents(valid_docs)
    if not chunks:
        logger.error("No chunks produced — check document content")
        sys.exit(1)
    logger.info("Chunking done", extra={"n_chunks": len(chunks)})
    return chunks


def stage_embed(chunks: list):
    """Stage 4 — Embed chunks into vectors."""
    logger.info("Stage 4/7 — Embedding chunks (this may take a minute)")
    embeddings = embed_chunks(chunks, show_progress=True)
    logger.info(
        "Embedding done",
        extra={"shape": list(embeddings.shape)},
    )
    return embeddings


def stage_index(embeddings, chunks: list) -> None:
    """Stage 5 — Build and save FAISS index."""
    logger.info("Stage 5/7 — Building FAISS index")
    index = build_index(embeddings, chunks)
    save_index(index, chunks)
    logger.info("Index saved")


def stage_parquet(docs: list) -> None:
    """Stage 6 — Export clean documents to Parquet."""
    logger.info("Stage 6/7 — Exporting documents to Parquet")
    processed_dir = Path(settings.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "id":             doc.id,
            "title":          doc.title,
            "url":            doc.url,
            "category":       doc.category,
            "date_published": (
                doc.date_published.isoformat() if doc.date_published else None
            ),
            "summary":        doc.summary,
            "content_hash":   doc.content_hash,
        }
        for doc in docs
    ]

    df = pd.DataFrame(records)
    parquet_path = processed_dir / "documents.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info(
        "Parquet exported",
        extra={"path": str(parquet_path), "rows": len(df)},
    )


def stage_catalog(docs: list, chunks: list, validation_results: list) -> dict:
    """Stage 7 — Generate datasets/catalog.json."""
    logger.info("Stage 7/7 — Building catalog")
    catalog = build_catalog(
        n_chunks=len(chunks),
        validation_results=validation_results,
    )
    logger.info("Catalog written")
    return catalog


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """Run the full ETL pipeline end to end."""
    start = time.monotonic()
    logger.info("Pipeline starting")

    docs                      = stage_load()
    valid_docs, val_results   = stage_validate(docs)
    chunks                    = stage_chunk(valid_docs)
    embeddings                = stage_embed(chunks)
    stage_index(embeddings, chunks)
    stage_parquet(valid_docs)
    catalog                   = stage_catalog(valid_docs, chunks, val_results)

    elapsed = time.monotonic() - start

    print("\n── Pipeline Complete ────────────────────────────")
    print(f"  Documents loaded    : {len(docs)}")
    print(f"  Documents valid     : {len(valid_docs)}")
    print(f"  Chunks created      : {len(chunks)}")
    print(f"  Vectors indexed     : {len(chunks)}")
    print(f"  Catalog written     : datasets/catalog.json")
    print(f"  Time elapsed        : {elapsed:.1f}s")
    print("─────────────────────────────────────────────────\n")


if __name__ == "__main__":
    run_pipeline()
