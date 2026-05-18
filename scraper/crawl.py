"""scraper/crawl.py — MoSPI crawler with Playwright text scraping."""

import argparse
import time
from datetime import datetime
from typing import List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from scraper.config import settings
from scraper.db import init_db, save_document
from scraper.models import Document
from scraper.utils import get_logger, normalize_date, normalize_text

logger = get_logger(__name__)


def fetch_with_playwright(url: str) -> Optional[str]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=settings.scraper_user_agent)
            page.goto(url, wait_until="networkidle", timeout=30000)
            time.sleep(3)
            html = page.content()
            browser.close()
            return html
    except Exception as exc:
        logger.error("Playwright failed", extra={"url": url, "error": str(exc)})
        return None


def scrape_page_text(url: str) -> Optional[str]:
    """
    Use Playwright to visit a MoSPI press release page and
    extract all visible text content.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error("Playwright not installed")
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=settings.scraper_user_agent)
            page.goto(url, wait_until="networkidle", timeout=30000)
            time.sleep(2)

            # dismiss language popup if present
            try:
                english_btn = page.locator("text=English")
                if english_btn.count() > 0:
                    english_btn.first.click()
                    time.sleep(1)
            except Exception:
                pass

            # extract main content text
            html = page.content()
            browser.close()

            soup = BeautifulSoup(html, "html.parser")

            # remove nav, header, footer, scripts
            for tag in soup(["script", "style", "nav", "header", "footer",
                             "aside", ".menu", ".navigation"]):
                tag.decompose()

            # get main content area
            main = (
                soup.find("main") or
                soup.find("article") or
                soup.find(class_="content") or
                soup.find(id="content") or
                soup.find("body")
            )

            text = normalize_text(main.get_text(separator="\n", strip=True)) if main else ""
            logger.info("Page text scraped", extra={"url": url, "chars": len(text)})
            return text if len(text) > 200 else None

    except Exception as exc:
        logger.error("Playwright scrape failed", extra={"url": url, "error": str(exc)})
        return None


def _get_known_documents() -> List[Document]:
    """
    Known MoSPI press release page URLs (not PDFs).
    Playwright will scrape text directly from these pages.
    """
    known = [
        {"title": "Estimates of GDP Q3 2024-25",
         "url": "https://mospi.gov.in/press-release/estimates-gdp-q3-2024-25",
         "date": "2025-02-28", "category": "gdp"},
        {"title": "First Advance Estimate of National Income 2024-25",
         "url": "https://mospi.gov.in/press-release/first-advance-estimate-national-income-2024-25",
         "date": "2025-01-07", "category": "gdp"},
        {"title": "Index of Industrial Production November 2024",
         "url": "https://mospi.gov.in/press-release/index-industrial-production-november-2024",
         "date": "2025-01-10", "category": "iip"},
        {"title": "Consumer Price Index December 2024",
         "url": "https://mospi.gov.in/press-release/consumer-price-index-december-2024",
         "date": "2025-01-13", "category": "cpi"},
        {"title": "Estimates of GDP Q2 2024-25",
         "url": "https://mospi.gov.in/press-release/estimates-gdp-q2-2024-25",
         "date": "2024-11-29", "category": "gdp"},
        {"title": "GDP Q1 2024-25 Press Note",
         "url": "https://mospi.gov.in/press-release/estimates-gdp-q1-2024-25",
         "date": "2024-08-30", "category": "gdp"},
        {"title": "Consumer Price Index December 2023",
         "url": "https://mospi.gov.in/press-release/consumer-price-index-december-2023",
         "date": "2024-01-12", "category": "cpi"},
        {"title": "Index of Industrial Production November 2023",
         "url": "https://mospi.gov.in/press-release/index-industrial-production-november-2023",
         "date": "2024-01-12", "category": "iip"},
        {"title": "Periodic Labour Force Survey Annual Report 2022-23",
         "url": "https://mospi.gov.in/press-release/periodic-labour-force-survey-annual-report-2022-23",
         "date": "2024-10-01", "category": "plfs"},
        {"title": "Annual Survey of Industries 2021-22",
         "url": "https://mospi.gov.in/press-release/annual-survey-industries-2021-22",
         "date": "2024-09-01", "category": "report"},
    ]
    docs = []
    for item in known:
        try:
            date_obj = datetime.strptime(item["date"], "%Y-%m-%d")
        except Exception:
            date_obj = None
        doc = Document(
            url=item["url"],
            title=item["title"],
            date_published=date_obj,
            category=item["category"],
            file_links=[],   # no PDF links needed
        )
        docs.append(doc)
    return docs


def scrape_document_texts(documents: List[Document]) -> List[Document]:
    """
    Visit each document URL with Playwright and scrape text content.
    Saves scraped text as document summary in DB.
    """
    from scraper.db import get_connection

    for doc in documents:
        logger.info("Scraping page", extra={"url": doc.url})
        text = scrape_page_text(doc.url)
        if text:
            with get_connection() as conn:
                conn.execute(
                    "UPDATE documents SET summary = ? WHERE url = ?",
                    (text[:5000], doc.url),
                )
            logger.info("Text saved", extra={"url": doc.url, "chars": len(text)})
        else:
            logger.warning("No text scraped", extra={"url": doc.url})
        time.sleep(settings.scraper_delay_seconds)

    return documents


def crawl(seed_urls: List[str], max_pages: int) -> dict:
    init_db()
    all_documents: List[Document] = []
    pages_crawled = 0

    # try API first
    for page_num in range(min(max_pages, 3)):
        from scraper.crawl import fetch_mospi_api
        docs = fetch_mospi_api(page_num)
        if docs:
            all_documents.extend(docs)
            pages_crawled += 1
        else:
            break

    # fall back to known documents
    if not all_documents:
        all_documents = _get_known_documents()
        pages_crawled = 1

    # save documents to DB
    total_saved = total_skipped = 0
    for doc in all_documents:
        if save_document(doc):
            total_saved += 1
        else:
            total_skipped += 1

    # scrape text from each page using Playwright
    logger.info("Starting Playwright text scraping")
    scrape_document_texts(all_documents)

    summary = {
        "total_fetched": len(all_documents),
        "total_saved": total_saved,
        "total_skipped": total_skipped,
        "pages_crawled": pages_crawled,
    }
    logger.info("Crawl complete", extra=summary)
    return summary


def fetch_mospi_api(page_num: int = 0) -> List[Document]:
    import requests
    endpoints = [
        f"https://mospi.gov.in/api/press-releases?page={page_num}",
        f"https://mospi.gov.in/press-releases?_format=json&page={page_num}",
    ]
    headers = {"User-Agent": settings.scraper_user_agent, "Accept": "application/json"}
    for endpoint in endpoints:
        try:
            time.sleep(settings.scraper_delay_seconds)
            resp = requests.get(endpoint, headers=headers, timeout=15)
            if resp.status_code == 200 and "json" in resp.headers.get("Content-Type", ""):
                from scraper.crawl import _parse_json
                docs = _parse_json(resp.json())
                if docs:
                    return docs
        except Exception:
            continue
    return []


def _parse_json(data) -> List[Document]:
    items = data if isinstance(data, list) else data.get("data", data.get("nodes", []))
    docs = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or item.get("attributes", {}).get("title", "")
        url   = item.get("url") or item.get("path", "")
        title = normalize_text(str(title))
        if url and not url.startswith("http"):
            url = urljoin("https://mospi.gov.in", url)
        if title and url:
            docs.append(Document(url=url, title=title, category="press-release"))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-url", nargs="+", default=settings.seed_urls)
    parser.add_argument("--max-pages", type=int, default=settings.scraper_max_pages)
    args = parser.parse_args()
    summary = crawl(seed_urls=args.seed_url, max_pages=args.max_pages)
    print(f"\n── Crawl Summary ───────────────────────────────")
    print(f"  Documents found   : {summary['total_fetched']}")
    print(f"  New (saved)       : {summary['total_saved']}")
    print(f"  Duplicates skipped: {summary['total_skipped']}")
    print(f"  Pages crawled     : {summary['pages_crawled']}")
    print(f"────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()