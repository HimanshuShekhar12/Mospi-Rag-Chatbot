"""
scraper/db.py
-------------
SQLite database layer for the MoSPI scraper.

Responsibilities:
  - init_db()        : create tables if they don't exist
  - save_document()  : insert a Document (skip if hash already exists)
  - save_file()      : insert a PDFFile record
  - save_table()     : insert an ExtractedTable record
  - get_all_documents() : fetch all documents (used by pipeline)
  - get_run_summary()   : stats for scraper.report

Schema (matches assignment spec exactly):
  documents(id, title, url, date_published, summary, category, hash, created_at)
  files(id, document_id, file_url, file_path, file_hash, file_type, pages, created_at)
  tables(id, document_id, source_file_id, table_json, n_rows, n_cols)
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional

from scraper.config import settings
from scraper.models import Document, ExtractedTable, PDFFile

logger = logging.getLogger(__name__)


# ── Connection helper ─────────────────────────────────────────────────────────

@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that yields a SQLite connection and always closes it.
    Uses Row factory so columns are accessible by name (row["title"]).
    """
    db_path = Path(settings.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)  # create data/ if needed

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row          # access columns by name
    conn.execute("PRAGMA journal_mode=WAL") # better concurrent write performance
    conn.execute("PRAGMA foreign_keys=ON")  # enforce FK constraints
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────────────

_CREATE_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    title          TEXT    NOT NULL,
    url            TEXT    NOT NULL UNIQUE,
    date_published TEXT,                        -- ISO-8601 string or NULL
    summary        TEXT    DEFAULT '',
    category       TEXT    DEFAULT 'uncategorized',
    hash           TEXT    NOT NULL UNIQUE,     -- sha256 fingerprint for dedup
    created_at     TEXT    NOT NULL
);
"""

_CREATE_FILES = """
CREATE TABLE IF NOT EXISTS files (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  INTEGER NOT NULL REFERENCES documents(id),
    file_url     TEXT    NOT NULL,
    file_path    TEXT    NOT NULL,
    file_hash    TEXT    NOT NULL,
    file_type    TEXT    DEFAULT 'pdf',
    pages        INTEGER DEFAULT 0,
    created_at   TEXT    NOT NULL
);
"""

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS tables (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id    INTEGER NOT NULL REFERENCES documents(id),
    source_file_id INTEGER NOT NULL REFERENCES files(id),
    table_json     TEXT    NOT NULL,   -- JSON-serialised list-of-lists
    n_rows         INTEGER DEFAULT 0,
    n_cols         INTEGER DEFAULT 0
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);",
    "CREATE INDEX IF NOT EXISTS idx_files_document ON files(document_id);",
    "CREATE INDEX IF NOT EXISTS idx_tables_document ON tables(document_id);",
]


# ── Public API ────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create all tables and indexes if they don't already exist.
    Safe to call multiple times (idempotent).
    """
    with get_connection() as conn:
        conn.execute(_CREATE_DOCUMENTS)
        conn.execute(_CREATE_FILES)
        conn.execute(_CREATE_TABLES)
        for idx_sql in _CREATE_INDEXES:
            conn.execute(idx_sql)
    logger.info("Database initialised at %s", settings.db_path)


def save_document(doc: Document) -> Optional[int]:
    """
    Insert a Document into the database.

    Returns the new row id on success.
    Returns None if the document already exists (dedup via hash).
    """
    sql = """
        INSERT OR IGNORE INTO documents
            (title, url, date_published, summary, category, hash, created_at)
        VALUES
            (?, ?, ?, ?, ?, ?, ?)
    """
    date_str = (
        doc.date_published.isoformat() if doc.date_published else None
    )
    with get_connection() as conn:
        cursor = conn.execute(
            sql,
            (
                doc.title,
                doc.url,
                date_str,
                doc.summary,
                doc.category,
                doc.content_hash,
                doc.created_at.isoformat(),
            ),
        )
        if cursor.lastrowid and cursor.rowcount > 0:
            logger.debug("Saved document id=%d  %s", cursor.lastrowid, doc.title[:60])
            return cursor.lastrowid

    logger.debug("Duplicate skipped: %s", doc.url)
    return None  # already exists


def save_file(pdf_file: PDFFile) -> Optional[int]:
    """
    Insert a PDFFile record linked to a Document.
    Returns the new row id, or None if already stored (dedup via file_hash).
    """
    sql = """
        INSERT OR IGNORE INTO files
            (document_id, file_url, file_path, file_hash, file_type, pages, created_at)
        VALUES
            (?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        cursor = conn.execute(
            sql,
            (
                pdf_file.document_id,
                pdf_file.file_url,
                pdf_file.file_path,
                pdf_file.file_hash,
                pdf_file.file_type,
                pdf_file.pages,
                pdf_file.created_at.isoformat(),
            ),
        )
        if cursor.lastrowid and cursor.rowcount > 0:
            logger.debug("Saved file id=%d  %s", cursor.lastrowid, pdf_file.file_path)
            return cursor.lastrowid

    return None


def save_table(table: ExtractedTable) -> Optional[int]:
    """
    Insert an ExtractedTable record into the database.
    table_data is serialised to JSON for storage.
    """
    sql = """
        INSERT INTO tables
            (document_id, source_file_id, table_json, n_rows, n_cols)
        VALUES
            (?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        cursor = conn.execute(
            sql,
            (
                table.document_id,
                table.source_file_id,
                json.dumps(table.table_data),
                table.n_rows,
                table.n_cols,
            ),
        )
        logger.debug("Saved table id=%d  %dx%d", cursor.lastrowid, table.n_rows, table.n_cols)
        return cursor.lastrowid


def get_all_documents() -> List[Document]:
    """
    Fetch every document from the database.
    Used by the pipeline to load the corpus for chunking + embedding.
    """
    sql = "SELECT * FROM documents ORDER BY date_published DESC"
    docs: List[Document] = []

    with get_connection() as conn:
        rows = conn.execute(sql).fetchall()

    for row in rows:
        date_published = None
        if row["date_published"]:
            try:
                date_published = datetime.fromisoformat(row["date_published"])
            except ValueError:
                pass  # keep None if date is malformed

        doc = Document(
            url=row["url"],
            title=row["title"],
            date_published=date_published,
            category=row["category"],
            summary=row["summary"],
        )
        doc.id = row["id"]
        doc.content_hash = row["hash"]
        docs.append(doc)

    logger.info("Loaded %d documents from database", len(docs))
    return docs


def get_run_summary() -> dict:
    """
    Return counts for scraper.report — documents, files, tables stored so far.
    """
    with get_connection() as conn:
        n_docs   = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        n_files  = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        n_tables = conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
        categories = conn.execute(
            "SELECT category, COUNT(*) as cnt FROM documents GROUP BY category"
        ).fetchall()

    return {
        "total_documents": n_docs,
        "total_files":     n_files,
        "total_tables":    n_tables,
        "by_category":     {row["category"]: row["cnt"] for row in categories},
    }
