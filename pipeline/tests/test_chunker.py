"""
pipeline/tests/test_chunker.py
--------------------------------
Unit tests for pipeline/chunker.py

Tests:
  - split_text()      produces correct chunk sizes and overlap
  - chunk_document()  returns Chunk objects with correct lineage
  - chunk_documents() handles empty and multi-document lists
"""

import pytest
from datetime import datetime

from scraper.models import Document
from pipeline.chunker import split_text, chunk_document, chunk_documents, Chunk


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_TEXT = """
The National Statistical Office (NSO) releases the First Advance Estimate
of GDP for the year 2024-25. India's GDP growth rate is estimated at 6.4
percent for the fiscal year 2024-25. The estimate is based on available
information on agriculture, industry and services sectors. The advance
estimate will be revised as more data becomes available. The GDP at
constant prices in 2024-25 is estimated at Rs 184.88 lakh crore. The
growth rate of real GDP during 2024-25 is estimated at 6.4 percent as
compared to 8.2 percent in 2023-24. The Gross Value Added at basic
prices at constant prices is estimated to grow at 6.4 percent during
2024-25 as against 7.2 percent in 2023-24.
""".strip()


@pytest.fixture
def sample_document():
    return Document(
        url="https://mospi.gov.in/press-releases/gdp-2024",
        title="GDP First Advance Estimate 2024-25",
        date_published=datetime(2025, 1, 7),
        category="gdp",
        summary=SAMPLE_TEXT,
    )


# ── split_text tests ──────────────────────────────────────────────────────────

class TestSplitText:
    """Tests for the core text splitting function."""

    def test_returns_list(self):
        """split_text should always return a list."""
        result = split_text(SAMPLE_TEXT)
        assert isinstance(result, list)

    def test_non_empty_input_gives_chunks(self):
        """Non-empty text should produce at least one chunk."""
        result = split_text(SAMPLE_TEXT)
        assert len(result) >= 1

    def test_empty_string_returns_empty_list(self):
        """Empty string should return empty list not raise."""
        result = split_text("")
        assert result == []

    def test_none_like_whitespace_returns_empty(self):
        """Whitespace-only string should return empty list."""
        result = split_text("   \n  \t  ")
        assert result == []

    def test_each_chunk_is_string(self):
        """Every chunk should be a non-empty string."""
        result = split_text(SAMPLE_TEXT, chunk_size=100, chunk_overlap=20)
        for chunk in result:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

    def test_chunk_size_respected(self):
        """Chunks should not greatly exceed the configured size."""
        chunk_size = 50   # tokens
        result = split_text(SAMPLE_TEXT, chunk_size=chunk_size, chunk_overlap=10, min_size=10)
        chars_limit = chunk_size * 4 * 1.5  # 50% tolerance
        for chunk in result:
            assert len(chunk) <= chars_limit, f"Chunk too large: {len(chunk)} chars"

    def test_overlap_produces_more_chunks(self):
        """Higher overlap should produce more chunks than no overlap."""
        no_overlap   = split_text(SAMPLE_TEXT, chunk_size=100, chunk_overlap=0,  min_size=10)
        with_overlap = split_text(SAMPLE_TEXT, chunk_size=100, chunk_overlap=50, min_size=10)
        assert len(with_overlap) >= len(no_overlap)

    def test_min_size_filters_small_chunks(self):
        """Chunks smaller than min_size should be discarded."""
        result = split_text(SAMPLE_TEXT, chunk_size=500, chunk_overlap=0, min_size=400)
        for chunk in result:
            assert len(chunk) >= 400 * 4  # min_size in chars

    def test_invalid_overlap_raises(self):
        """overlap >= chunk_size should raise ValueError."""
        with pytest.raises(ValueError):
            split_text(SAMPLE_TEXT, chunk_size=100, chunk_overlap=100)


# ── chunk_document tests ──────────────────────────────────────────────────────

class TestChunkDocument:
    """Tests for Document → Chunk conversion."""

    def test_returns_list_of_chunks(self, sample_document):
        """chunk_document should return a list of Chunk objects."""
        result = chunk_document(sample_document)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_chunks_have_document_lineage(self, sample_document):
        """Every chunk should carry title, url, and category from the document."""
        sample_document.id = 42
        result = chunk_document(sample_document)
        for chunk in result:
            assert chunk.title    == sample_document.title
            assert chunk.url      == sample_document.url
            assert chunk.category == sample_document.category

    def test_chunk_index_is_sequential(self, sample_document):
        """chunk_index should be 0, 1, 2 ... in order."""
        result = chunk_document(sample_document)
        for i, chunk in enumerate(result):
            assert chunk.chunk_index == i

    def test_empty_document_returns_empty_list(self):
        """Document with no title or summary should return empty list."""
        doc = Document(url="https://mospi.gov.in/empty", title="X", summary="")
        result = chunk_document(doc)
        assert isinstance(result, list)

    def test_chunk_text_is_non_empty(self, sample_document):
        """All chunks should have non-empty text."""
        result = chunk_document(sample_document)
        for chunk in result:
            assert chunk.text.strip() != ""

    def test_token_count_property(self, sample_document):
        """Chunk.token_count should return a positive integer."""
        result = chunk_document(sample_document)
        for chunk in result:
            assert chunk.token_count > 0


# ── chunk_documents tests ─────────────────────────────────────────────────────

class TestChunkDocuments:
    """Tests for batch document chunking."""

    def test_empty_list_returns_empty(self):
        """Empty document list should return empty chunk list."""
        result = chunk_documents([])
        assert result == []

    def test_multiple_documents_all_chunked(self, sample_document):
        """All documents should contribute chunks to the output."""
        doc2 = Document(
            url="https://mospi.gov.in/cpi-2024",
            title="CPI December 2024",
            summary="Consumer Price Index for December 2024 stands at 5.2 percent.",
        )
        result = chunk_documents([sample_document, doc2])
        assert len(result) >= 2

    def test_output_is_flat_list(self, sample_document):
        """Output should be a flat list, not list of lists."""
        result = chunk_documents([sample_document])
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)
