# ================================================================
# MoSPI Scraper + LLaMA RAG Chatbot — Makefile
# ================================================================
# Usage:
#   make install    install all dependencies
#   make crawl      run the web scraper
#   make parse      download PDFs and extract text/tables
#   make etl        run full pipeline (validate+chunk+embed+index)
#   make index      alias for etl
#   make up         start api + ui + ollama via Docker
#   make eval       run chatbot quality evaluation
#   make test       run all pytest tests
#   make report     print scraper run summary
#   make clean      remove generated data files
#   make lint       run black + isort + mypy
# ================================================================

.PHONY: install crawl parse etl index up eval test report clean lint help

# ── Default ───────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  MoSPI RAG — Available Commands"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make install   Install Python dependencies"
	@echo "  make crawl     Run web scraper (MoSPI listing pages)"
	@echo "  make parse     Download PDFs + extract text/tables"
	@echo "  make etl       Full pipeline: validate+chunk+embed+index"
	@echo "  make index     Alias for make etl"
	@echo "  make up        Start all services via Docker Compose"
	@echo "  make eval      Run chatbot quality evaluation"
	@echo "  make test      Run all pytest tests"
	@echo "  make report    Print scraper run summary"
	@echo "  make lint      Run black + isort + mypy"
	@echo "  make clean     Remove generated data files"
	@echo "  ─────────────────────────────────────────────"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# ── Scraper (Part A) ──────────────────────────────────────────────
crawl:
	python -m scraper.crawl --max-pages 5
	@echo "✅ Crawl complete"

parse:
	python -m scraper.parse --max-pdfs 10
	@echo "✅ Parse complete"

report:
	python -m scraper.report

# ── Pipeline (Part B) ─────────────────────────────────────────────
etl:
	python -m pipeline.run
	@echo "✅ ETL pipeline complete"

index: etl

# ── Docker (Ops) ──────────────────────────────────────────────────
up:
	docker compose up --build
	@echo "✅ All services started"

down:
	docker compose down

# ── RAG API (Part C) ──────────────────────────────────────────────
api:
	uvicorn rag.api:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run rag/ui/app.py

# ── Evaluation (Stretch Goal) ─────────────────────────────────────
eval:
	python -m eval.run_eval
	@echo "✅ Evaluation complete — see eval/results/eval_report.json"

# ── Tests ─────────────────────────────────────────────────────────
test:
	pytest scraper/tests/ pipeline/tests/ rag/tests/ -v --tb=short
	@echo "✅ All tests complete"

test-scraper:
	pytest scraper/tests/ -v

test-pipeline:
	pytest pipeline/tests/ -v

test-rag:
	pytest rag/tests/ -v

# ── Code quality ──────────────────────────────────────────────────
lint:
	black scraper/ pipeline/ rag/ eval/
	isort scraper/ pipeline/ rag/ eval/
	mypy scraper/ pipeline/ rag/ --ignore-missing-imports
	@echo "✅ Lint complete"

# ── Cleanup ───────────────────────────────────────────────────────
clean:
	rm -rf data/raw/pdf/*.pdf
	rm -rf data/processed/faiss.index
	rm -rf data/processed/chunks.pkl
	rm -rf data/processed/documents.parquet
	rm -rf datasets/catalog.json
	rm -rf eval/results/eval_report.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	@echo "✅ Clean complete"

# ── Full run (scrape → etl → start) ──────────────────────────────
all: crawl parse etl
	@echo "✅ Full pipeline complete — run 'make api' and 'make ui' to start"
