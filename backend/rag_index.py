"""Загрузка FAISS-индекса и поиск по косинусному сходству."""

import json
import os
import re
from typing import Any

import faiss
import numpy as np

_TOKEN = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
_STOP = {
    "и", "в", "во", "на", "с", "со", "к", "ко", "о", "об", "от", "до", "по",
    "за", "из", "у", "а", "но", "или", "как", "что", "это", "для", "при",
    "не", "ни", "же", "ли", "бы", "то", "все", "уже", "ещё", "еще",
    "ваши", "ваших", "есть", "какие", "какой", "какая",
}


def load_faq_data(path: str) -> list[dict[str, Any]]:
    """Читает FAQ и проверяет обязательные поля. Тексты не изменяет."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("faqs.json must contain a list of objects")

    seen: set[str] = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"FAQ item {index} is not an object")
        question = item.get("question")
        answer = item.get("answer")
        faq_id = item.get("id")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"FAQ item {index} has an empty question")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"FAQ item {index} has an empty answer")
        if not isinstance(faq_id, str) or not faq_id.strip():
            raise ValueError(f"FAQ item {index} has an empty id")
        if faq_id in seen:
            raise ValueError(f"Duplicate FAQ id: {faq_id}")
        seen.add(faq_id)
    return data


def load_index(index_path: str, meta_path: str) -> tuple[faiss.Index, dict[str, Any]]:
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise RuntimeError(
            "FAISS index or metadata not found. "
            "Run `python -m backend.build_index` first to build the RAG index."
        )
    index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict) or not isinstance(metadata.get("items"), list):
        raise RuntimeError(
            "Metadata file is invalid. "
            "Run `python -m backend.build_index` to rebuild the index."
        )
    return index, metadata


def _canon(token: str) -> str:
    folded = token.casefold()
    if folded in {"mnwhl", "mnwhlru"}:
        return "минвайл"
    for prefix, label in (
        ("заработ", "зарплат"),
        ("зарпла", "зарплат"),
        ("продукт", "продукт"),
        ("сопровожд", "сопровожд"),
        ("портал", "портал"),
        ("сайт", "сайт"),
        ("технолог", "технолог"),
    ):
        if folded.startswith(prefix):
            return label
    return folded


def _token_list(text: str, expand: bool = False) -> list[str]:
    words = [
        _canon(word)
        for word in _TOKEN.findall(text.casefold())
        if len(word) > 1 and word not in _STOP
    ]
    if not expand:
        return words
    extra: list[str] = []
    joined = " ".join(words)
    if "занимается" in joined or "занимаетесь" in joined:
        extra.extend(["разработчик", "прикладного"])
    if "продукт" in words:
        extra.extend(["линии", "портфеле"])
    if "час" in words:
        extra.append("производительность")
    return words + extra


def _tokens(text: str, expand: bool = False) -> set[str]:
    return set(_token_list(text, expand=expand))


def _contains(token: str, corpus: set[str]) -> bool:
    if token in corpus:
        return True
    if len(token) < 5:
        return False
    stem = token[:5]
    return any(word.startswith(stem) or token.startswith(word[:5]) for word in corpus if len(word) >= 5)


def _sequence_bonus(query: str, item: dict[str, Any]) -> float:
    query_tokens = _token_list(query, expand=False)
    blob = _token_list(f"{item.get('question', '')} {item.get('answer', '')}", expand=False)
    if len(query_tokens) < 2 or len(blob) < 2:
        return 0.0
    blob_pairs = set(zip(blob, blob[1:]))
    for left, right in zip(query_tokens, query_tokens[1:]):
        if (left, right) in blob_pairs:
            return 0.28
    return 0.0


def lexical_overlap(query: str, item: dict[str, Any]) -> float:
    query_tokens = _tokens(query, expand=True)
    if not query_tokens:
        return 0.0
    question = _tokens(item.get("question", ""), expand=False)
    answer = _tokens(item.get("answer", ""), expand=False)
    hits = 0.0
    for token in query_tokens:
        if _contains(token, question):
            hits += 2.0
        elif _contains(token, answer):
            hits += 1.0
    return hits / (2.0 * len(query_tokens)) + _sequence_bonus(query, item)


def search_similar(
    index: faiss.Index,
    items: list[dict[str, Any]],
    query_vec: np.ndarray,
    k: int = 3,
    query_text: str = "",
    exclude_ids: list[str] | None = None,
    vector_map: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Ранжирует по косинусу и лексическому пересечению. В ответе score — косинус."""
    if k < 1:
        k = 1
    k = min(k, max(1, len(items)))
    vector = np.asarray(query_vec, dtype="float32")
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    pool = max(index.ntotal, k)
    scores, indices = index.search(vector, pool)
    excluded = set(exclude_ids or [])
    best: dict[str, dict[str, Any]] = {}
    for score, idx in zip(scores[0], indices[0]):
        position = int(idx)
        if position < 0:
            continue
        if vector_map is not None:
            if position >= len(vector_map):
                continue
            item_index = int(vector_map[position])
        else:
            item_index = position
        if item_index < 0 or item_index >= len(items):
            continue
        item = items[item_index]
        cosine = float(score)
        current = best.get(item["id"])
        if current is None or cosine > current["score"]:
            row = dict(item)
            row["score"] = cosine
            best[item["id"]] = row
    ranked = []
    for item in best.values():
        lex = lexical_overlap(query_text, item) if query_text else 0.0
        row = dict(item)
        cosine = float(row["score"])
        row["score"] = round(cosine, 4)
        row["_rank"] = cosine + 0.32 * lex
        row["_excluded"] = row.get("id") in excluded
        row["_nolex"] = lex <= 0
        ranked.append(row)
    ranked.sort(key=lambda row: (row["_nolex"], row["_excluded"], -row["_rank"]))
    for item in ranked:
        item.pop("_rank", None)
        item.pop("_excluded", None)
        item.pop("_nolex", None)
    return ranked[:k]
