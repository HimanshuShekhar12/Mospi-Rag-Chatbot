"""
rag/api.py
-----------
FastAPI backend for the MoSPI RAG chatbot.

Endpoints (assignment requirement):
  POST /ask     {question: str, k?: int, temperature?: float}
                → {answer: str, citations: list}
  POST /ingest  → rebuilds the FAISS index from current DB
  GET  /health  → {status, ollama, index, n_vectors}

Run locally:
    uvicorn rag.api:app --host 0.0.0.0 --port 8000 --reload
"""

import subprocess
import sys
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from scraper.config import settings
from scraper.utils import get_logger
from pipeline.indexer import index_exists
from rag.retriever import retriever
from rag.prompt import build_prompt, format_citations
from rag.llm import generate, is_ollama_running

logger = get_logger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MoSPI RAG Chatbot API",
    description=(
        "Q&A API that answers questions strictly from scraped "
        "MoSPI statistical publications using LLaMA 3."
    ),
    version="1.0.0",
)

# allow Streamlit UI (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question:    str   = Field(..., min_length=3, description="User question")
    k:           int   = Field(default=settings.retrieval_top_k, ge=1, le=20)
    temperature: float = Field(default=settings.llm_temperature, ge=0.0, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "question":    "What was India's GDP growth rate in 2023-24?",
                "k":           5,
                "temperature": 0.1,
            }
        }


class Citation(BaseModel):
    rank:     int
    score:    float
    title:    str
    url:      str
    category: str
    snippet:  str


class AskResponse(BaseModel):
    answer:    str
    citations: List[Citation]
    question:  str
    k_used:    int


class IngestResponse(BaseModel):
    status:   str
    message:  str


class HealthResponse(BaseModel):
    status:    str
    ollama:    bool
    index:     bool
    n_vectors: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health_check() -> HealthResponse:
    """
    Check if all system components are ready.

    Returns:
      - status   : "ok" or "degraded"
      - ollama   : True if Ollama + LLaMA model are available
      - index    : True if FAISS index exists on disk
      - n_vectors: number of vectors in the loaded index
    """
    ollama_ok  = is_ollama_running()
    index_ok   = index_exists()
    n_vectors  = 0

    if index_ok:
        try:
            retriever._ensure_loaded()
            n_vectors = retriever._index.ntotal
        except Exception:
            index_ok = False

    status = "ok" if (ollama_ok and index_ok) else "degraded"

    logger.info(
        "Health check",
        extra={"status": status, "ollama": ollama_ok, "index": index_ok},
    )

    return HealthResponse(
        status=status,
        ollama=ollama_ok,
        index=index_ok,
        n_vectors=n_vectors,
    )


@app.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(request: AskRequest) -> AskResponse:
    """
    Answer a question using the RAG pipeline.

    Steps:
      1. Retrieve top-k relevant chunks from FAISS
      2. Build a prompt with context window
      3. Call LLaMA via Ollama
      4. Return answer + structured citations

    Raises:
      400 if question is empty
      503 if Ollama is not running
      500 for unexpected errors
    """
    logger.info(
        "Question received",
        extra={"question": request.question[:80], "k": request.k},
    )

    # ── Guard: index must exist ────────────────────────────────────
    if not index_exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Vector index not found. "
                "Run 'python -m pipeline.run' or 'make etl' first."
            ),
        )

    # ── Guard: Ollama must be running ──────────────────────────────
    if not is_ollama_running():
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama is not running. "
                "Start it with: ollama serve"
            ),
        )

    try:
        # 1. Retrieve relevant chunks
        chunks = retriever.search(request.question, k=request.k)

        # 2. Build prompt
        messages = build_prompt(request.question, chunks)

        # 3. Call LLaMA
        answer = generate(
            messages,
            temperature=request.temperature,
            max_tokens=settings.llm_max_tokens,
        )

        # 4. Format citations
        citations = [Citation(**c) for c in format_citations(chunks)]

        logger.info(
            "Answer generated",
            extra={
                "question":  request.question[:60],
                "citations": len(citations),
            },
        )

        return AskResponse(
            answer=answer,
            citations=citations,
            question=request.question,
            k_used=len(chunks),
        )

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected error in /ask", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/ingest", response_model=IngestResponse, tags=["ops"])
def ingest() -> IngestResponse:
    """
    Rebuild the FAISS vector index from the current database.

    Runs the full pipeline (validate → chunk → embed → index).
    Use this after scraping new documents.

    Note: This is a blocking call — may take several minutes
    depending on corpus size.
    """
    logger.info("Ingest triggered via API")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pipeline.run"],
            capture_output=True,
            text=True,
            timeout=600,   # 10 minute timeout
        )

        if result.returncode != 0:
            logger.error(
                "Pipeline failed during ingest",
                extra={"stderr": result.stderr[:500]},
            )
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline failed: {result.stderr[:300]}",
            )

        # reload the retriever with the new index
        retriever.reload()

        logger.info("Ingest complete — index reloaded")
        return IngestResponse(
            status="ok",
            message="Index rebuilt successfully. Retriever reloaded.",
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Pipeline timed out after 10 minutes",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Ingest error", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail=str(exc))


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "rag.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
