"""Настройки из .env. Импорт модуля не обращается к сети и не требует ключа."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
ANSWER_MODE = os.getenv("ANSWER_MODE", "openai").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini").strip()
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.35"))
TOP_K = int(os.getenv("TOP_K", "3"))
HOST = os.getenv("HOST", "127.0.0.1").strip() or "127.0.0.1"
PORT = int(os.getenv("PORT", "8000"))
INDEX_TXT_DOCS = _flag("INDEX_TXT_DOCS", "false")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

DATA_DIR = ROOT / "data"
FAQ_PATH = DATA_DIR / "faqs.json"
INDEX_PATH = DATA_DIR / "faiss_index.bin"
META_PATH = DATA_DIR / "faqs_metadata.json"
METRICS_PATH = ROOT / "docs" / "metrics.json"

CORS_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

REFUSAL_TEXT = "В базе знаний нет ответа на этот вопрос. Напишите нам: hello@mnwhl.ru"
NO_PRICE_TEXT = "В базе знаний нет сведений о ценах и обороте. Напишите нам: hello@mnwhl.ru"

MAX_HISTORY_MESSAGES = 6
MAX_SESSIONS = 500
TEMPERATURE = 0.0
