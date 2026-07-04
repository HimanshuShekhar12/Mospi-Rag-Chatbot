# MoSPI Research Assistant — RAG Chatbot over Indian Government Statistics

A question-answering system built on the official publications of India's
**Ministry of Statistics and Programme Implementation (MoSPI)**. It scrapes
real statistical reports from `mospi.gov.in`, extracts their text, builds a
searchable vector index, and answers questions using a **LLaMA 3** model —
every answer backed by citations to the source document.

Ask *"What does the report say about labour market dynamics in million-plus
cities?"* and get an answer grounded in the actual MoSPI PDF, with a link to it.

---

## Contents

1. [How it works](#how-it-works)
2. [Tech stack](#tech-stack)
3. [Prerequisites](#prerequisites)
4. [Setup](#setup)
5. [Run it](#run-it)
6. [Using the chatbot](#using-the-chatbot)
7. [API reference](#api-reference)
8. [Project structure](#project-structure)
9. [Configuration](#configuration)
10. [Testing](#testing)
11. [Notes](#notes)

---

## How it works

The project is three stages: **scrape → index → ask.**

```
┌──────────────────────────────────────────────────────────────┐
│  1. SCRAPE                                          scraper/  │
│  ──────────                                                   │
│  MoSPI is a JavaScript site backed by a JSON API.            │
│  crawl.py calls that API, lists publications, downloads      │
│  each PDF, and extracts its text with pdfplumber.            │
│  → stored in SQLite (data/mospi.db)                          │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│  2. INDEX (ETL pipeline)                          pipeline/   │
│  ──────────────────────                                       │
│  validate → chunk (800–1200 tokens, overlapping)             │
│           → embed (SentenceTransformers, all-MiniLM-L6-v2)   │
│           → FAISS vector index (cosine similarity)           │
│  → data/processed/faiss.index + chunks.pkl                  │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│  3. ASK (RAG chatbot)                                  rag/   │
│  ─────────────────                                            │
│  question → embed → FAISS top-k search → build prompt        │
│          → LLaMA 3 (via Groq API) → answer + citations       │
│  Served by FastAPI, with a Streamlit chat UI.               │
└──────────────────────────────────────────────────────────────┘
```

**Query-time flow:**

```
User question
   → embed_query()        (all-MiniLM-L6-v2)
   → FAISS.search(k)      (cosine similarity)
   → build_prompt()       (retrieved chunks + system prompt)
   → LLaMA 3 via Groq
   → Answer + Citations
```

---

## Tech stack

| Layer          | Technology                                             |
|----------------|--------------------------------------------------------|
| Scraping       | `requests` (MoSPI JSON API), `pdfplumber` (PDF text)   |
| Storage        | SQLite                                                  |
| Embeddings     | SentenceTransformers — `all-MiniLM-L6-v2` (384-dim)    |
| Vector search  | FAISS (`IndexFlatIP`, exact cosine similarity)         |
| LLM            | LLaMA 3 via the **Groq** API (`llama-3.1-8b-instant`)  |
| Backend        | FastAPI                                                 |
| Frontend       | Streamlit                                               |
| Config         | Pydantic Settings (`.env`)                              |
| Tests          | pytest (125 tests)                                      |

---

## Prerequisites

- **Python 3.11+**
- **A free Groq API key** — sign up at
  [console.groq.com/keys](https://console.groq.com/keys) (no credit card).
  This is what runs the LLaMA 3 model, so the project needs no GPU and no
  local model download.

That's it — no Docker, no Ollama, no browser drivers required.

---

## Setup

```powershell
# 1. Clone
git clone https://github.com/HimanshuShekhar12/Mospi-rag-chatbot
cd Mospi-rag-chatbot

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows (PowerShell)
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your config file
copy .env.example .env         # Windows
# cp .env.example .env         # macOS / Linux
```

Then open `.env` and paste in your Groq API key:

```
GROQ_API_KEY=gsk_your_key_here
```

---

## Run it

You need the data indexed once, then two servers running (API + UI).

### Step 1 — Build the corpus (scrape + index)

```powershell
# Scrape MoSPI publications (downloads PDFs, extracts text into SQLite)
python -m scraper.crawl --max-pages 6

# Build the vector index (validate → chunk → embed → FAISS)
python -m pipeline.run
```

`--max-pages 6` fetches ~60 publications; raise it for a larger corpus.

### Step 2 — Start the two servers

Open **two terminals** (activate the venv in each):

```powershell
# Terminal 1 — FastAPI backend
uvicorn rag.api:app --host 127.0.0.1 --port 8000

# Terminal 2 — Streamlit UI
streamlit run rag/ui/app.py
```

### Step 3 — Open the chatbot

Go to **http://localhost:8501** and start asking questions.

> The backend API and its interactive docs are at http://localhost:8000/docs.

---

## Using the chatbot

Open **http://localhost:8501**.

**Sidebar controls:**

| Control              | What it does                                                        |
|----------------------|--------------------------------------------------------------------|
| **Number of sources (k)** | How many document chunks are retrieved to answer a question.  |
| **Temperature**      | Answer randomness. `0.1` = factual and precise (recommended).      |
| **Check Status**     | Shows whether the LLM backend and FAISS index are ready.          |
| **Rebuild Index**    | Re-runs the ETL pipeline after new data is scraped.               |
| **Clear Chat**       | Clears the conversation.                                           |

**Tips**
- Ask specific questions, e.g. *"What was the GDP growth rate in 2024-25?"*
- Each answer lists its **cited sources** with relevance scores and links to
  the original MoSPI PDF.

---

## API reference

Base URL: `http://localhost:8000` · Interactive docs: `/docs`

### `POST /ask`

```json
{ "question": "What is India's GDP growth rate?", "k": 5, "temperature": 0.1 }
```

Response:

```json
{
  "answer": "India's GDP growth rate ...\n\nSources:\n• [Title](https://mospi.gov.in/...)",
  "citations": [
    { "rank": 1, "score": 0.89, "title": "...", "url": "https://mospi.gov.in/...",
      "category": "gdp", "snippet": "..." }
  ],
  "question": "What is India's GDP growth rate?",
  "k_used": 5
}
```

### `POST /ingest`

Rebuilds the FAISS index from the current database. Returns `{"status": "ok"}`.

### `GET /health`

Reports readiness: `{ "status": "ok", "ollama": true, "index": true, "n_vectors": 210 }`
(the `ollama` field indicates the LLM backend is reachable.)

---

## Project structure

```
mospi-intelligence/
├── scraper/                 # Part 1 — scrape MoSPI + extract PDF text
│   ├── crawl.py             #   calls MoSPI's JSON API, downloads PDFs
│   ├── parse.py             #   standalone PDF downloader + table extractor
│   ├── db.py                #   SQLite layer (documents · files · tables)
│   ├── models.py            #   Document / PDFFile / ExtractedTable dataclasses
│   ├── config.py            #   Pydantic settings (reads .env)
│   ├── utils.py             #   logging, rate limiting, text/date normalisation
│   ├── report.py            #   prints a scrape run summary
│   └── tests/               #   unit + integration tests
│
├── pipeline/                # Part 2 — ETL: validate → chunk → embed → index
│   ├── validate.py          #   quality checks + deduplication
│   ├── chunker.py           #   overlapping token chunks with doc lineage
│   ├── embedder.py          #   SentenceTransformers embeddings
│   ├── indexer.py           #   FAISS index build/load
│   ├── catalog.py           #   writes datasets/catalog.json
│   ├── run.py               #   pipeline entry point
│   └── tests/
│
├── rag/                     # Part 3 — retrieval + chatbot
│   ├── retriever.py         #   embed query → FAISS top-k search
│   ├── prompt.py            #   builds the context window + citations
│   ├── llm.py               #   LLaMA 3 client (Groq API)
│   ├── api.py               #   FastAPI backend (/ask /ingest /health)
│   ├── ui/app.py            #   Streamlit chat UI
│   └── tests/
│
├── eval/                    # chatbot quality evaluation
│   ├── qa_pairs.json
│   └── run_eval.py
│
├── data/                    # SQLite DB + downloaded PDFs + FAISS index
├── datasets/                # catalog.json
├── infra/                   # Dockerfiles
├── requirements.txt
├── Makefile
└── .env.example
```

---

## Configuration

All settings live in `.env` (see `.env.example`). The common ones:

| Variable              | Default                    | Description                              |
|-----------------------|----------------------------|------------------------------------------|
| `GROQ_API_KEY`        | *(required)*               | Your Groq API key                        |
| `GROQ_MODEL`          | `llama-3.1-8b-instant`     | LLaMA 3 model served by Groq             |
| `SCRAPER_MAX_PAGES`   | `5`                        | Listing pages to fetch per crawl         |
| `SCRAPER_DELAY_SECONDS` | `2.0`                    | Delay between requests                   |
| `CHUNK_SIZE`          | `1000`                     | Approx. tokens per chunk                 |
| `CHUNK_OVERLAP`       | `200`                      | Overlap between consecutive chunks       |
| `EMBEDDING_MODEL`     | `all-MiniLM-L6-v2`         | SentenceTransformers model               |
| `LLM_TEMPERATURE`     | `0.1`                      | Answer randomness (0 = factual)          |
| `RETRIEVAL_TOP_K`     | `5`                        | Chunks retrieved per query               |

---

## Testing

```powershell
# Run the full suite (125 tests)
pytest scraper/tests/ pipeline/tests/ rag/tests/ -v

# Or per component
pytest scraper/tests/     # scraper: API parsing, PDF handling, integration
pytest pipeline/tests/    # chunking + validation
pytest rag/tests/         # retrieval
```

| Test file            | Covers                                                        |
|----------------------|---------------------------------------------------------------|
| `test_crawl.py`      | API item parsing, title cleaning, URL building, categories    |
| `test_parse.py`      | PDF download, hashing, text + table extraction                |
| `test_integration.py`| Full crawl → SQLite, end to end (HTTP mocked)                 |
| `test_chunker.py`    | Chunk sizes, overlap, short-document handling                 |
| `test_validate.py`   | Validation rules, deduplication                               |
| `test_retriever.py`  | Top-k retrieval                                               |

---

## Notes

- **The LLM runs on Groq's free API**, so responses are fast (~1–2s) and need
  no GPU or local model. A `GROQ_API_KEY` in `.env` is required.
- **Scraping uses MoSPI's public JSON API**, not browser automation — it's
  fast and needs no browser drivers. Very large PDFs (>30 MB) and files with
  no extractable text layer are skipped automatically.
- **Embeddings download once.** The first run fetches the `all-MiniLM-L6-v2`
  model (~80 MB) from Hugging Face; after that everything is local.
- A Docker Compose file is included under `infra/`, but it targets the older
  local-Ollama setup — the local run steps above are the maintained path.
