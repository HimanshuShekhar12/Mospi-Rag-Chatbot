"""
rag/tests/test_retriever.py
-----------------------------
Unit tests for rag/retriever.py

Tests:
  - Retriever.search()     returns correct number of results
  - RetrievedChunk.to_dict() serialises correctly
  - retriever handles empty query gracefully
  - retriever handles k > available vectors gracefully
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from pipeline.chunker import Chunk
from rag.retriever import Retriever, RetrievedChunk


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_chunks():
    """Create 10 sample chunks for testing."""
    return [
        Chunk(
            text=f"This is sample chunk number {i} about Indian GDP statistics and economic data.",
            document_id=i,
            chunk_index=0,
            title=f"Document {i}",
            url=f"https://mospi.gov.in/doc-{i}",
            category="gdp",
            date_published="2024-01-01",
        )
        for i in range(10)
    ]


@pytest.fixture
def mock_retriever(sample_chunks):
    """
    Create a Retriever with a mocked FAISS index and chunks.
    Avoids loading real index files from disk.
    """
    import faiss

    # create real FAISS index with random vectors
    dim = 384
    embeddings = np.random.rand(len(sample_chunks), dim).astype(np.float32)
    # L2 normalise
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    retriever = Retriever()
    retriever._index  = index
    retriever._chunks = sample_chunks
    retriever._loaded = True

    return retriever


# ── RetrievedChunk tests ──────────────────────────────────────────────────────

class TestRetrievedChunk:
    def test_to_dict_has_required_keys(self, sample_chunks):
        chunk = RetrievedChunk(chunk=sample_chunks[0], score=0.95, rank=1)
        d = chunk.to_dict()
        assert "rank"     in d
        assert "score"    in d
        assert "title"    in d
        assert "url"      in d
        assert "category" in d
        assert "snippet"  in d

    def test_score_is_rounded(self, sample_chunks):
        chunk = RetrievedChunk(chunk=sample_chunks[0], score=0.123456789, rank=1)
        d = chunk.to_dict()
        assert len(str(d["score"]).split(".")[-1]) <= 4

    def test_snippet_truncated_at_300_chars(self, sample_chunks):
        long_chunk = Chunk(
            text="A" * 500,
            document_id=1,
            chunk_index=0,
            title="Test",
            url="https://mospi.gov.in/test",
            category="test",
        )
        rc = RetrievedChunk(chunk=long_chunk, score=0.9, rank=1)
        d  = rc.to_dict()
        assert len(d["snippet"]) <= 304  # 300 + "..."

    def test_properties_delegate_to_chunk(self, sample_chunks):
        chunk = sample_chunks[0]
        rc    = RetrievedChunk(chunk=chunk, score=0.8, rank=1)
        assert rc.text     == chunk.text
        assert rc.title    == chunk.title
        assert rc.url      == chunk.url
        assert rc.category == chunk.category


# ── Retriever.search tests ────────────────────────────────────────────────────

class TestRetrieverSearch:
    def test_returns_list(self, mock_retriever):
        results = mock_retriever.search("What is India GDP growth rate?", k=3)
        assert isinstance(results, list)

    def test_returns_correct_k_results(self, mock_retriever):
        """Should return exactly k results when k <= available vectors."""
        results = mock_retriever.search("GDP growth India 2024", k=5)
        assert len(results) == 5

    def test_returns_retrieved_chunk_objects(self, mock_retriever):
        """Every result should be a RetrievedChunk instance."""
        results = mock_retriever.search("consumer price index", k=3)
        for r in results:
            assert isinstance(r, RetrievedChunk)

    def test_results_sorted_by_rank(self, mock_retriever):
        """Results should be ranked 1, 2, 3 ... in order."""
        results = mock_retriever.search("industrial production", k=4)
        for i, r in enumerate(results, start=1):
            assert r.rank == i

    def test_empty_query_returns_empty_list(self, mock_retriever):
        """Empty query should return empty list not raise."""
        results = mock_retriever.search("", k=5)
        assert results == []

    def test_k_clamped_to_available_vectors(self, mock_retriever):
        """k larger than index size should return all available vectors."""
        results = mock_retriever.search("GDP", k=999)
        assert len(results) <= 10   # only 10 chunks in fixture

    def test_scores_are_floats(self, mock_retriever):
        """Every score should be a float."""
        results = mock_retriever.search("GDP statistics India", k=3)
        for r in results:
            assert isinstance(r.score, float)

    def test_whitespace_only_query_returns_empty(self, mock_retriever):
        """Whitespace-only query should return empty list."""
        results = mock_retriever.search("   ", k=5)
        assert results == []


# ── Retriever.reload tests ────────────────────────────────────────────────────

class TestRetrieverReload:
    def test_reload_resets_loaded_flag(self, mock_retriever):
        """reload() should set _loaded to False triggering re-load."""
        with patch.object(mock_retriever, "_ensure_loaded") as mock_load:
            mock_retriever.reload()
            assert mock_retriever._loaded is False
            mock_load.assert_called_once()
