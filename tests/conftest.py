"""Окружение тестов: лексический индекс во временной папке, без сети и без ключа."""

import os
import tempfile
from pathlib import Path

os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["EMBEDDING_MODEL"] = "hash-lexical"
os.environ["ANSWER_MODE"] = "extractive"
os.environ["SCORE_THRESHOLD"] = "0.2"
os.environ["INDEX_TXT_DOCS"] = "false"
os.environ["OPENAI_API_KEY"] = ""

from backend import config
from backend.embeddings import reset_embedder

_TMP = Path(tempfile.mkdtemp(prefix="faq-test-"))
config.EMBEDDING_PROVIDER = "hash"
config.EMBEDDING_MODEL = "hash-lexical"
config.ANSWER_MODE = "extractive"
config.SCORE_THRESHOLD = 0.2
config.INDEX_TXT_DOCS = False
config.OPENAI_API_KEY = ""
config.INDEX_PATH = _TMP / "faiss_index.bin"
config.META_PATH = _TMP / "faqs_metadata.json"
reset_embedder()

from backend.build_index import main

main()


import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    from backend.app import app

    return TestClient(app)
