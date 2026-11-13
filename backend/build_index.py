"""Построение FAISS-индекса по data/faqs.json."""

import json
import os
import sys

import faiss

from . import config
from .embeddings import embed_passages
from .rag_index import load_faq_data


def load_txt_documents(directory: str) -> list[dict]:
    """Необязательный формат урока. По умолчанию выключен через INDEX_TXT_DOCS."""
    docs = []
    if not os.path.isdir(directory):
        return docs

    for name in sorted(os.listdir(directory)):
        if not name.lower().endswith(".txt"):
            continue
        path = os.path.join(directory, name)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                content = handle.read().strip()
        except OSError:
            continue
        if not content:
            continue
        lines = content.splitlines()
        title = next((line.strip() for line in lines if line.strip()), name)
        body_lines = lines[1:] if len(lines) > 1 else []
        body = "\n".join(body_lines).strip() or content
        docs.append(
            {
                "id": f"txt-{os.path.splitext(name)[0]}",
                "category": "Документы",
                "question": title,
                "answer": body,
                "source": name,
            }
        )
    return docs


def passage_texts(item: dict) -> list[str]:
    question = item["question"]
    category = item.get("category") or ""
    answer = item["answer"]
    return [
        f"{question}\n{question}\n{category}",
        f"{question}\n{answer}",
    ]


def main() -> None:
    items: list[dict] = []
    faq_path = str(config.FAQ_PATH)
    if os.path.exists(faq_path):
        faqs = load_faq_data(faq_path)
        items.extend(faqs)
        print(f"Loaded {len(faqs)} FAQ items from faqs.json")

    if config.INDEX_TXT_DOCS:
        txt_docs = load_txt_documents(str(config.DATA_DIR))
        items.extend(txt_docs)
        print(f"Loaded {len(txt_docs)} TXT documents from {config.DATA_DIR}")

    if not items:
        raise RuntimeError("No data found to build index (no faqs.json and no txt files).")

    texts: list[str] = []
    vector_map: list[int] = []
    for item_index, item in enumerate(items):
        for text in passage_texts(item):
            texts.append(text)
            vector_map.append(item_index)
    print(f"Embedding {len(texts)} items with {config.EMBEDDING_PROVIDER}:{config.EMBEDDING_MODEL}...")
    embeddings = embed_passages(texts)
    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(str(config.INDEX_PATH)), exist_ok=True)
    faiss.write_index(index, str(config.INDEX_PATH))
    metadata = {
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_provider": config.EMBEDDING_PROVIDER,
        "embedding_dim": dim,
        "vector_map": vector_map,
        "items": [
            {
                "id": item["id"],
                "category": item.get("category", ""),
                "question": item["question"],
                "answer": item["answer"],
            }
            for item in items
        ],
    }
    with open(config.META_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"Index built and saved to {config.INDEX_PATH}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
