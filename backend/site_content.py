"""Тексты страниц берутся из faqs.json, чтобы цифры не расходились с базой."""

import json

from . import config
from .rag_index import load_faq_data

HOME_SPECS = (
    ("faq-001", "Компания"),
    ("faq-003", "Что продаёт Минвайл"),
    ("faq-006", "Продуктовые линии"),
    ("faq-009", "Подход к искусственному интеллекту"),
)

CASE_SPECS = (
    ("ГОРИЗОНТ / 01", ("faq-013", "faq-015")),
    ("Корпоративный поиск и машиночитаемые документы", ("faq-020", "faq-021")),
    ("«Кубрик» и ИИ-секретарь", ("faq-027", "faq-028")),
    ("Промышленный ИИ", ("faq-036", "faq-037")),
    ("Smart Data", ("faq-039", "faq-040")),
)


def _by_id() -> dict:
    items = load_faq_data(str(config.FAQ_PATH))
    return {item["id"]: item for item in items}


def _fmt_number(value) -> str:
    if isinstance(value, str):
        return value.replace(".", ",")
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return text.replace(".", ",")


def home_blocks() -> list[dict]:
    items = _by_id()
    return [
        {"id": faq_id, "title": title, "answer": items[faq_id]["answer"]}
        for faq_id, title in HOME_SPECS
    ]


def assistant_card() -> dict:
    parts = [
        "Модуль поиска по базе из 40 вопросов и ответов на этом сайте. "
        "Отвечает только по найденному контексту и сообщает, если в базе нет ответа."
    ]
    if config.METRICS_PATH.exists():
        metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
        parts.append(
            "Стек: FastAPI, FAISS с косинусным сходством, "
            f"эмбеддинги {metrics['embedding_model']}, генерация {metrics['llm_model']}, "
            f"режим {metrics['answer_mode']}."
        )
        parts.append(
            "Измерения на проверочном наборе: "
            f"Recall@1 {_fmt_number(metrics['recall_at_1'])}, "
            f"Recall@3 {_fmt_number(metrics['recall_at_3'])}, "
            f"верных отказов вне базы {metrics['out_of_scope_refusals']}, "
            f"среднее время запроса {metrics['mean_query_ms']} мс, "
            f"порог сходства {_fmt_number(metrics['threshold'])}."
        )
    else:
        parts.append(
            "Стек: FastAPI, FAISS, эмбеддинги и генерация через OpenAI. "
            "Измеренные метрики публикуются в docs/MODEL_CHOICE.md после прогона оценки."
        )
    return {"title": "FAQ-ассистент этого сайта", "parts": parts}


def case_cards() -> list[dict]:
    items = _by_id()
    cards = []
    for title, ids in CASE_SPECS:
        cards.append(
            {
                "title": title,
                "parts": [items[faq_id]["answer"] for faq_id in ids],
            }
        )
    cards.append(assistant_card())
    return cards


def faq_groups() -> list[dict]:
    groups: list[dict] = []
    index: dict[str, list] = {}
    for item in load_faq_data(str(config.FAQ_PATH)):
        category = item.get("category") or "Прочее"
        if category not in index:
            bucket: list = []
            index[category] = bucket
            groups.append({"category": category, "items": bucket})
        index[category].append(item)
    return groups
