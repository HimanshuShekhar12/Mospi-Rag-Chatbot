# MoSPI Scraper + LLaMA RAG Chatbot

A production-quality data pipeline that scrapes statistical publications from the
Ministry of Statistics and Programme Implementation (MoSPI), extracts text using
Playwright browser automation, and powers a Retrieval-Augmented Generation (RAG)
chatbot using a local LLaMA 3 model.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Running the Project](#running-the-project)
5. [Architecture](#architecture)
6. [Configuration](#configuration)
7. [Testing](#testing)
8. [API Reference](#api-reference)
9. [Trade-offs & Design Decisions](#trade-offs--design-decisions)
10. [Known Limits](#known-limits)
11. [Future Improvements](#future-improvements)

---

## Quick Start

```bash
# 1. Clone and set up
git clone <https://github.com/HimanshuShekhar12/MoSPI-Research-Assistant>
cd mospi-intelligence

# 2. Configure
cp .env.example .env        

# 3. Start everything with Docker (recommended)
docker compose up --build

# 4. Run scraper (on demand — scrapes MoSPI pages via Playwright)
docker compose --profile scraper run scraper

# 5. Run ETL pipeline (validate → chunk → embed → index)
docker compose --profile pipeline run pipeline
```

Open **http://localhost:8501** to use the chatbot.

---

## Project Structure

```
mospi-intelligence/
├── README.md
├── Makefile                    # make crawl · make etl · make up
├── docker-compose.yml          # all services wired together
├── .env.example                # config template
├── requirements.txt            # all dependencies
│
├── scraper/                    # Part A — Web Scraper
│   ├── config.py               # pydantic settings from .env
│   ├── models.py               # Document · PDFFile · ExtractedTable
│   ├── db.py                   # SQLite CRUD operations
│   ├── crawl.py                # Playwright browser scraper + pagination
│   ├── parse.py                # PDF downloader + text/table extractor
│   ├── report.py               # run summary printer
│   └── tests/                  # unit + integration tests
│
├── pipeline/                   # Part B — ETL Pipeline
│   ├── validate.py             # data quality checks
│   ├── chunker.py              # overlapping text chunker
│   ├── embedder.py             # SentenceTransformers embeddings
│   ├── indexer.py              # FAISS index builder
│   ├── catalog.py              # datasets/catalog.json generator
│   ├── run.py                  # master pipeline entry point
│   └── tests/
│
├── rag/                        # Part C — RAG Chatbot
│   ├── api.py                  # FastAPI: /ask /ingest /health
│   ├── retriever.py            # FAISS vector search
│   ├── prompt.py               # LLaMA prompt builder
│   ├── llm.py                  # Ollama client + streaming
│   └── ui/app.py               # Streamlit chat interface
│
├── eval/                       # Quality Evaluation (Stretch Goal)
│   ├── qa_pairs.json           # 10 test Q&A pairs
│   ├── run_eval.py             # accuracy measurement script
│   └── results/eval_report.json
│
├── data/
│   ├── mospi.db                # SQLite database
│   ├── raw/pdf/                # downloaded PDFs (if available)
│   └── processed/              # faiss.index · chunks.pkl · documents.parquet
│
├── datasets/
│   └── catalog.json            # corpus manifest
│
└── infra/
    ├── Dockerfile.scraper      # includes Playwright + Chromium
    ├── Dockerfile.api
    └── Dockerfile.ui
```

---

## Setup

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Docker Desktop | latest | Containerised deployment (recommended) |
| Python | 3.11+ | Runtime (for local dev only) |
| Git | any | Version control |

### Installation (Docker — Recommended)

```bash
# clone the repo
git clone <your-repo-url>
cd mospi-intelligence

# copy config
cp .env.example .env

# build and start all services
docker compose up --build
```

### Installation (Local Development)

```bash
# create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# install dependencies
pip install -r requirements.txt

# install Playwright browser
playwright install chromium
playwright install-deps chromium

# copy config template
cp .env.example .env
```

---

## Running the Project

### Option 1 — Docker (Recommended)

```bash
# Start all services (Ollama + FastAPI + Streamlit)
docker compose up

# Run scraper — visits MoSPI pages via Playwright and extracts text
docker compose --profile scraper run scraper

# Run ETL pipeline (validate → chunk → embed → index)
docker compose --profile pipeline run pipeline
```

### Option 2 — Local (Development)

```bash
# Step 1 — Scrape MoSPI pages using Playwright
python -m scraper.crawl --seed-url https://mospi.gov.in/press-releases --max-pages 5

# Step 2 — View scraper summary
python -m scraper.report

# Step 3 — Run ETL pipeline (validate → chunk → embed → index)
python -m pipeline.run

# Step 4 — Start FastAPI backend
uvicorn rag.api:app --host 0.0.0.0 --port 8000 --reload

# Step 5 — Start Streamlit UI (new terminal)
streamlit run rag/ui/app.py

# Step 6 — Run quality evaluation
python -m eval.run_eval
```

### Option 3 — Make commands

```bash
make install    # install dependencies
make crawl      # run scraper
make etl        # full pipeline
make api        # start FastAPI
make ui         # start Streamlit
make eval       # run evaluation
make test       # run all tests
make lint       # black + isort + mypy
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MoSPI Website                        │
│         mospi.gov.in/press-releases                     │
└──────────────────────┬──────────────────────────────────┘
                       │ Playwright (headless Chromium)
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  Part A — Scraper                       │
│                                                         │
│  crawl.py    → Playwright browser scraping              │
│              → text extracted from press release pages  │
│              → saved to summary column in SQLite        │
│  db.py       → SQLite (documents · files · tables)      │
└──────────────────────┬──────────────────────────────────┘
                       │ get_all_documents()
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  Part B — Pipeline                      │
│                                                         │
│  validate.py → quality checks (title · date · url)      │
│  chunker.py  → 800-1200 token overlapping chunks        │
│              → reads from document summary column       │
│  embedder.py → SentenceTransformers (all-MiniLM-L6-v2)  │
│  indexer.py  → FAISS IndexFlatIP (cosine similarity)    │
│  catalog.py  → datasets/catalog.json                    │
└──────────────────────┬──────────────────────────────────┘
                       │ faiss.index + chunks.pkl
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  Part C — RAG Chatbot                   │
│                                                         │
│  retriever.py → embed query → FAISS top-k search        │
│  prompt.py   → context window + system prompt           │
│  llm.py      → Ollama (LLaMA 3 8B Instruct Q4)          │
│  api.py      → FastAPI /ask /ingest /health             │
│  ui/app.py   → Streamlit chat + citations + sliders     │
└─────────────────────────────────────────────────────────┘
```

### Data Flow at Query Time

```
User question
     │
     ▼
embed_query()           ← all-MiniLM-L6-v2
     │
     ▼
FAISS.search(k=5)       ← cosine similarity
     │
     ▼
build_prompt()          ← context window + system prompt
     │
     ▼
ollama.chat()           ← LLaMA 3 8B Instruct
     │
     ▼
Answer + Citations      ← returned to UI
```

---

## Configuration

All settings are controlled via `.env`. See `.env.example` for full reference.

| Variable | Default | Description |
|----------|---------|-------------|
| `MOSPI_SEED_URLS` | mospi.gov.in/press-releases | Comma-separated crawl seeds |
| `SCRAPER_MAX_PAGES` | 5 | Max listing pages per run |
| `SCRAPER_DELAY_SECONDS` | 2.0 | Delay between requests |
| `CHUNK_SIZE` | 1000 | Approximate tokens per chunk |
| `CHUNK_OVERLAP` | 200 | Overlapping tokens between chunks |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | SentenceTransformers model |
| `OLLAMA_BASE_URL` | http://ollama:11434 | Ollama server URL (Docker service name) |
| `OLLAMA_MODEL` | llama3:8b-instruct-q4_0 | LLaMA model name |
| `LLM_TEMPERATURE` | 0.1 | Response randomness (0=factual) |
| `RETRIEVAL_TOP_K` | 5 | Chunks retrieved per query |

---

## Testing

```bash
# run all tests
make test

# run specific suites
pytest scraper/tests/ -v        # scraper unit + integration tests
pytest pipeline/tests/ -v       # pipeline tests
pytest rag/tests/ -v            # RAG tests
```

### Test Coverage

| Test File | Type | What It Tests |
|-----------|------|---------------|
| `test_crawl.py` | Unit | HTML parser, pagination, date/category normalisation |
| `test_parse.py` | Unit | PDF hash, filename cleaning, text/table extraction |
| `test_integration.py` | Integration | Full crawl → SQLite end to end with mock HTML |
| `test_chunker.py` | Unit | Chunk sizes, overlap, lineage |
| `test_validate.py` | Unit | Validation rules, deduplication |
| `test_retriever.py` | Unit | Top-k returns correct number of results |

---

## Using the Chatbot UI

Open **http://localhost:8501** after starting all services.

### Sidebar Controls

| Control | What it does |
|---------|-------------|
| **Number of sources (k)** | How many document chunks are retrieved from FAISS to answer your question. Higher k = more context but slower response. Recommended: **2-3** on CPU. |
| **Temperature** | Controls how creative/random LLaMA's answer is. `0.1` = factual and precise. `1.0` = more creative. Keep it low (`0.1`) for statistical data questions. |
| **Check Status** | Shows whether Ollama (LLaMA model) and FAISS index are ready. Both should show ✅ before asking questions. |
| **Rebuild Index** | Re-runs the ETL pipeline to refresh the vector index after new data is scraped. |
| **Clear Chat** | Clears the conversation history. |

### Tips
- Keep **k=2** on CPU for faster responses (~30-60 seconds)
- Ask specific questions like *"What was India's GDP growth in 2023-24?"* rather than broad ones
- Each answer shows **cited sources** with relevance scores and links to original MoSPI pages

---

## API Reference

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

### `POST /ask`

Ask a question from the MoSPI corpus.

**Request:**
```json
{
  "question": "What was India's GDP growth rate in 2023-24?",
  "k": 5,
  "temperature": 0.1
}
```

**Response:**
```json
{
  "answer": "India's GDP growth rate in 2023-24 was 8.2%...\n\nSources:\n• [GDP Press Note](https://mospi.gov.in/...)",
  "citations": [
    {
      "rank": 1,
      "score": 0.89,
      "title": "GDP First Advance Estimate 2024-25",
      "url": "https://mospi.gov.in/press-releases/gdp-q3-2024",
      "category": "press-release",
      "snippet": "The National Statistical Office releases..."
    }
  ],
  "question": "What was India's GDP growth rate in 2023-24?",
  "k_used": 5
}
```

### `POST /ingest`

Rebuild the vector index from current database.

**Response:**
```json
{"status": "ok", "message": "Index rebuilt successfully."}
```

### `GET /health`

Check system status.

**Response:**
```json
{
  "status": "ok",
  "ollama": true,
  "index": true,
  "n_vectors": 187
}
```

---

## Trade-offs & Design Decisions

### 1. Playwright over requests + BeautifulSoup
MoSPI website blocks all direct HTTP requests with 403 Forbidden.
Playwright runs a real headless Chromium browser, bypassing CDN/firewall protection.
Trade-off: slower scraping, heavier Docker image (~500MB extra for Chromium).
Benefit: actually works against JS-heavy and protected government websites.

### 2. Text scraping over PDF download
MoSPI blocks direct PDF downloads server-side.
Instead, press release text is scraped directly from the webpage and saved to the `summary` column in SQLite.
Trade-off: less structured data than PDFs, no table extraction.
Fix: combine with PDF download when URLs become accessible.

### 3. SQLite over Postgres
Chose SQLite for simplicity and zero infrastructure overhead.
Trade-off: not suitable for concurrent writes at scale.
Fix: swap `sqlite-utils` for `asyncpg` + Postgres when scaling.

### 4. all-MiniLM-L6-v2 over larger models
384-dimensional embeddings, 80MB model, very fast on CPU.
Trade-off: slightly lower semantic accuracy than larger models like `bge-large`.
Fix: swap `EMBEDDING_MODEL` in `.env` — no code changes needed.

### 5. FAISS IndexFlatIP over HNSW
Exact nearest neighbour search — guaranteed correct results.
Trade-off: slower at very large scale (100k+ vectors).
Fix: switch to `IndexHNSWFlat` for approximate search at scale.

### 6. LLaMA 3 8B Q4 over full 70B
4.7GB download, runs on CPU, good enough for factual Q&A.
Trade-off: less nuanced reasoning than larger models, slow on CPU (~30-60s/response).
Fix: change `OLLAMA_MODEL` in `.env` to `llama3:70b` on a GPU machine.

### 7. Streamlit over Next.js
Built in 150 lines, fully functional, looks professional.
Trade-off: not a production frontend, limited customisation.
Fix: replace with a React/Next.js app calling the same FastAPI endpoints.

---

## Known Limits

- MoSPI website blocks all server-side HTTP and PDF download requests with 403 Forbidden. Playwright browser scraping is used as a fallback, which may timeout in Docker environments with slow networks.
- Playwright scraping inside Docker requires Chromium to be installed in the container — adds ~500MB to image size and increases build time.
- LLaMA on CPU is slow (~30-60 seconds per response). Fix: use a GPU or switch to a cloud API.
- Only 4 chunks currently indexed due to limited scraped text. More data requires either fixing PDF access or scraping more pages.
- `POST /ingest` is a blocking call — it runs the full pipeline synchronously. Fix: use a task queue (Celery / ARQ) for async indexing.
- `qa_pairs.json` answers are generic — after scraping real data, update with actual values from the documents.

---

## Future Improvements

- **More data** — fix MoSPI PDF access or scrape more pages for a richer corpus
- **Airflow/Prefect DAG** — scheduled daily scraping with backfill support
- **Great Expectations** — automated data quality reports
- **Embedding cache** — skip re-embedding unchanged chunks
- **Reranking** — cross-encoder reranker for better retrieval precision
- **Grafana dashboard** — real-time metrics (docs/hour, error rate, latency)
- **OCR support** — tesseract for scanned PDFs
- **Async pipeline** — Celery task queue for non-blocking ingest
- **GPU deployment** — faster LLaMA inference with CUDA support

---

## What Worked

- Playwright browser automation successfully bypassed MoSPI's 403 blocking
- SentenceTransformers + FAISS gave good retrieval quality with minimal setup
- Streamlit UI with source citations built in under 200 lines
- Pydantic settings made configuration clean and validated
- Docker Compose wired all 5 services together cleanly with one command

## What Didn't Work

- Direct HTTP requests to MoSPI were blocked with 403 Forbidden on all endpoints
- Direct PDF downloads from MoSPI were blocked server-side
- Playwright timed out on some pages inside Docker due to network restrictions
- LLaMA response time on CPU is too slow for a real production chatbot

## What I'd Do Next

- Fix MoSPI PDF access for richer structured data
- Add a task queue so `/ingest` is non-blocking
- Deploy Ollama on a GPU machine for faster inference
- Build a proper React frontend with WebSocket streaming
- Add scheduled scraping with Airflow for fresh data daily