"""
scraper/report.py
-----------------
Prints a human-readable summary of everything the scraper has collected.

Shows:
  - Total documents, files, tables in the database
  - Breakdown by category
  - Breakdown by year
  - Most recently scraped documents
  - Any documents missing a date or summary

Entry point:
    python -m scraper.report
"""

import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List

from rich.console import Console
from rich.table import Table
from rich import box

from scraper.config import settings
from scraper.db import get_all_documents, get_connection, get_run_summary
from scraper.utils import get_logger

logger  = get_logger(__name__)
console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_recent_documents(limit: int = 5) -> List[dict]:
    """Fetch the most recently added documents from SQLite."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT title, url, category, date_published, created_at
              FROM documents
             ORDER BY created_at DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def _get_missing_data_counts() -> dict:
    """Count documents with missing date or empty summary."""
    with get_connection() as conn:
        missing_date = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE date_published IS NULL"
        ).fetchone()[0]

        missing_summary = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE summary = '' OR summary IS NULL"
        ).fetchone()[0]

        missing_files = conn.execute(
            """
            SELECT COUNT(*) FROM documents d
             WHERE NOT EXISTS (
                 SELECT 1 FROM files f WHERE f.document_id = d.id
             )
            """
        ).fetchone()[0]

    return {
        "missing_date":    missing_date,
        "missing_summary": missing_summary,
        "missing_files":   missing_files,
    }


def _get_docs_by_year() -> dict:
    """Count documents grouped by publication year."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT SUBSTR(date_published, 1, 4) as year, COUNT(*) as cnt
              FROM documents
             WHERE date_published IS NOT NULL
             GROUP BY year
             ORDER BY year DESC
            """
        ).fetchall()
    return {row["year"]: row["cnt"] for row in rows}


def _db_size() -> str:
    """Return human-readable size of the SQLite database file."""
    db_path = Path(settings.db_path)
    if not db_path.exists():
        return "not found"
    size = db_path.stat().st_size
    if size < 1024:
        return f"{size} B"
    elif size < 1024 ** 2:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / 1024 ** 2:.1f} MB"


# ── Report sections ───────────────────────────────────────────────────────────

def _print_header() -> None:
    console.rule("[bold cyan]MoSPI Scraper — Run Report[/bold cyan]")
    console.print(
        f"  Generated : [green]{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC[/green]"
    )
    console.print(f"  Database  : [green]{settings.db_path}[/green]  ({_db_size()})")
    console.print()


def _print_totals(summary: dict) -> None:
    console.print("[bold]── Totals ──────────────────────────────────────[/bold]")

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column("Metric", style="cyan")
    t.add_column("Count",  style="white", justify="right")

    t.add_row("Documents collected", str(summary["total_documents"]))
    t.add_row("PDF files downloaded", str(summary["total_files"]))
    t.add_row("Tables extracted",    str(summary["total_tables"]))

    console.print(t)


def _print_by_category(summary: dict) -> None:
    by_cat = summary.get("by_category", {})
    if not by_cat:
        return

    console.print("[bold]── By Category ─────────────────────────────────[/bold]")

    t = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
    t.add_column("Category", style="cyan")
    t.add_column("Documents", style="white", justify="right")

    for category, count in sorted(by_cat.items(), key=lambda x: -x[1]):
        t.add_row(category, str(count))

    console.print(t)


def _print_by_year() -> None:
    by_year = _get_docs_by_year()
    if not by_year:
        return

    console.print("[bold]── By Year ──────────────────────────────────────[/bold]")

    t = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
    t.add_column("Year",      style="cyan")
    t.add_column("Documents", style="white", justify="right")

    for year, count in by_year.items():
        t.add_row(year or "unknown", str(count))

    console.print(t)


def _print_recent(limit: int = 5) -> None:
    recent = _get_recent_documents(limit)
    if not recent:
        return

    console.print(f"[bold]── Last {limit} Documents Added ─────────────────────[/bold]")

    t = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    t.add_column("Title",    style="white",  max_width=55)
    t.add_column("Category", style="cyan",   max_width=18)
    t.add_column("Date",     style="yellow", max_width=12)

    for doc in recent:
        date_str = (doc["date_published"] or "")[:10]  # just YYYY-MM-DD
        t.add_row(
            doc["title"][:55],
            doc["category"],
            date_str,
        )

    console.print(t)


def _print_data_quality() -> None:
    missing = _get_missing_data_counts()

    console.print("[bold]── Data Quality ────────────────────────────────[/bold]")

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column("Check",  style="cyan")
    t.add_column("Count",  style="white", justify="right")

    def _status(n: int) -> str:
        return f"[red]{n}[/red]" if n > 0 else "[green]0 ✓[/green]"

    t.add_row("Missing publication date", _status(missing["missing_date"]))
    t.add_row("Missing summary",          _status(missing["missing_summary"]))
    t.add_row("Documents with no PDFs",   _status(missing["missing_files"]))

    console.print(t)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Print the full scraper run report."""
    db_path = Path(settings.db_path)
    if not db_path.exists():
        console.print(
            "[red]Database not found.[/red] "
            "Run [cyan]python -m scraper.crawl[/cyan] first."
        )
        sys.exit(1)

    summary = get_run_summary()

    _print_header()
    _print_totals(summary)
    _print_by_category(summary)
    _print_by_year()
    _print_recent(limit=5)
    _print_data_quality()

    console.rule()


if __name__ == "__main__":
    main()
