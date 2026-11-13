"""Провайдеры эмбеддингов: openai, local и hash (только для тестов без сети)."""

import hashlib
import re
import time

import numpy as np

from . import config

_EMBEDDER = None
_STOPWORDS = {
    "и", "в", "во", "на", "с", "со", "к", "ко", "о", "об", "от", "до", "по",
    "за", "из", "у", "а", "но", "или", "как", "что", "это", "для", "при",
    "не", "ни", "же", "ли", "бы", "то", "все", "уже", "ещё", "еще",
}


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (matrix / norms).astype("float32")


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[0-9a-zа-яё]+", text.casefold())
    return [word for word in words if word not in _STOPWORDS and len(word) > 1]


def _hash_embed(texts: list[str]) -> np.ndarray:
    """Лексические векторы для pytest: без скачивания моделей и без API."""
    dim = 384
    vectors = np.zeros((len(texts), dim), dtype="float32")
    for row, text in enumerate(texts):
        tokens = _tokens(text)
        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vectors[row, bucket] += sign
    return l2_normalize(vectors)


class HashEmbedder:
    def embed(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        del is_query
        return _hash_embed(texts)


class OpenAIEmbedder:
    def __init__(self) -> None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to the .env file. "
                "It is used for embeddings and, in openai answer mode, for generation."
            )
        from openai import OpenAI

        self.client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=60.0)
        self.model = config.EMBEDDING_MODEL

    def embed(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        del is_query
        from openai import APIError, RateLimitError

        ordered: list[list[float]] = []
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            response = None
            for attempt in range(4):
                try:
                    response = self.client.embeddings.create(model=self.model, input=chunk)
                    break
                except (RateLimitError, APIError):
                    if attempt == 3:
                        raise
                    time.sleep(1.5 * (attempt + 1))
            ranked = sorted(response.data, key=lambda item: item.index)
            ordered.extend(item.embedding for item in ranked)
        return l2_normalize(np.asarray(ordered, dtype="float32"))


class LocalEmbedder:
    """Локальная модель sentence-transformers. Импорт тяжёлой библиотеки только здесь."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = config.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        # У multilingual-e5 префиксы обязательны, иначе качество на коротких запросах падает.
        self.uses_e5_prefixes = "e5" in self.model_name.lower()

    def embed(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        prepared = texts
        if self.uses_e5_prefixes:
            prefix = "query: " if is_query else "passage: "
            prepared = [prefix + text for text in texts]
        vectors = self.model.encode(
            prepared,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    provider = config.EMBEDDING_PROVIDER
    if provider == "openai":
        _EMBEDDER = OpenAIEmbedder()
    elif provider == "local":
        _EMBEDDER = LocalEmbedder()
    elif provider == "hash":
        _EMBEDDER = HashEmbedder()
    else:
        raise RuntimeError(
            f"Unknown EMBEDDING_PROVIDER: {provider}. Use openai, local, or hash."
        )
    return _EMBEDDER


def reset_embedder() -> None:
    global _EMBEDDER
    _EMBEDDER = None


def embed_passages(texts: list[str]) -> np.ndarray:
    if not texts:
        raise ValueError("No texts to embed")
    return get_embedder().embed(texts, is_query=False)


def embed_query(text: str) -> np.ndarray:
    return get_embedder().embed([text], is_query=True)
