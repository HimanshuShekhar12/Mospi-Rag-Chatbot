"""
scraper/config.py
-----------------
Centralised configuration for the scraper package.
All settings are read from environment variables (via .env).
Import `settings` anywhere in the scraper package to access config.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List


class ScraperSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Scraper
    mospi_seed_urls: str = Field(default="https://mospi.gov.in/press-releases")
    scraper_max_pages: int = Field(default=5, ge=1)
    scraper_delay_seconds: float = Field(default=2.0, ge=0.5)
    scraper_concurrency: int = Field(default=2, ge=1, le=5)
    scraper_user_agent: str = Field(default="MoSPI-Research-Bot/1.0 (educational project)")
    scraper_max_retries: int = Field(default=3, ge=1)

    # Storage
    database_url: str = Field(default="data/mospi.db")
    pdf_download_dir: str = Field(default="data/raw/pdf")
    processed_dir: str = Field(default="data/processed")

    # Pipeline
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)
    chunk_min_size: int = Field(default=100, ge=10)

    # Embeddings
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    faiss_index_path: str = Field(default="data/processed/faiss.index")
    chunks_pkl_path: str = Field(default="data/processed/chunks.pkl")

    # LLaMA 3 via Groq (free hosted API — no local Ollama/Docker needed)
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.1-8b-instant")
    # Legacy Ollama settings (kept for optional local/offline use)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3:8b-instruct-q4_0")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=512, ge=64)

    # RAG
    retrieval_top_k: int = Field(default=5, ge=1, le=20)

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # UI
    ui_port: int = Field(default=8501)

    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    @property
    def seed_urls(self) -> List[str]:
        return [url.strip() for url in self.mospi_seed_urls.split(",") if url.strip()]

    @property
    def db_path(self) -> str:
        return self.database_url


settings = ScraperSettings()
