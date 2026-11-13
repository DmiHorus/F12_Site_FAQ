"""Ответ по найденным записям: extractive, OpenAI или локальная LLM."""

import json
import re
import urllib.error
import urllib.request

from . import config

PRICE_QUERY = re.compile(
    r"(сколько стоит|какая цена|какой оборот|стоимость|бюджет|тариф)",
    re.IGNORECASE,
)
MONEY = re.compile(
    r"(руб\.?|рубл\w*|₽|\$|€|\busd\b|\beur\b|доллар\w*|\bевро\b|"
    r"миллион\w*|миллиард\w*|\bмлн\b|\bмлрд\b)",
    re.IGNORECASE,
)
SYSTEM_PROMPT = (
    "Ты FAQ-ассистент компании Минвайл. Отвечай кратко и по-русски, только по приведённому контексту. "
    "Сохраняй числа, проценты и названия из контекста без изменений, включая десятичную запятую. "
    "Если в контексте нет ответа, скажи об этом и предложи написать на hello@mnwhl.ru. "
    "Не называй цены, сроки и факты, которых нет в контексте."
)
REWRITE_PROMPT = (
    "Составь один поисковый запрос к базе FAQ на русском языке. "
    "Последняя реплика уточняет предыдущий вопрос пользователя, а не начинает новую тему. "
    "Если пользователь спрашивает про результаты, в запросе должны быть тема прошлого вопроса "
    "и слова про результаты, эффект, пилот и цифры, а не повтор определения. "
    "Верни только текст запроса, одной строкой, без кавычек и пояснений."
)


def sanitize_answer(query: str, answer: str) -> str:
    if PRICE_QUERY.search(query) and MONEY.search(answer or ""):
        return config.NO_PRICE_TEXT
    return answer


def _context_block(items: list[dict]) -> str:
    parts = []
    for item in items:
        parts.append(
            f"[{item.get('id', '')}]\nВопрос: {item['question']}\nОтвет: {item['answer']}"
        )
    return "\n\n".join(parts)


def _content_words(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[0-9a-zа-яё]{4,}", text.casefold())
        if word not in {"какие", "какой", "какая", "какое", "это", "этом", "него", "неё"}
    ]


def reformulate_query(previous: str, message: str, mode: str) -> str:
    prompt = (
        f"Предыдущий вопрос пользователя: {previous.strip()}\n"
        f"Уточнение: {message.strip()}"
    )
    messages = [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if mode == "openai":
        rewritten = _openai_chat(messages, max_tokens=80)
    elif mode == "local_llm":
        rewritten = _ollama_chat(messages)
    else:
        return ""
    rewritten = (rewritten or "").strip()
    if not rewritten:
        return ""
    rewritten = rewritten.splitlines()[0].strip(" «»\"'")
    words = _content_words(message)
    folded = rewritten.casefold()
    if words and not any(word in folded for word in words):
        rewritten = f"{rewritten} {message.strip()}"
    return rewritten


def resolve_search_query(message: str, history: list[dict], mode: str) -> str:
    from .memory import build_search_query, is_followup, last_user_message

    base = build_search_query(message, history)
    previous = last_user_message(history)
    if not (previous and is_followup(message)):
        return base
    extra = []
    if re.search(r"результат", message, re.IGNORECASE):
        extra.append("результаты эффект цифры")
    del mode
    return " ".join([base, *extra]).strip()


def generate_answer(query: str, items: list[dict], history: list[dict], mode: str) -> str:
    if mode == "extractive":
        answer = items[0]["answer"]
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history:
            if turn.get("role") in {"user", "assistant"} and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Вопрос пользователя: {query}\n\nКонтекст FAQ:\n{_context_block(items)}"
                ),
            }
        )
        if mode == "openai":
            answer = _openai_chat(messages, max_tokens=500)
        elif mode == "local_llm":
            answer = _ollama_chat(messages)
        else:
            raise RuntimeError(
                f"Unknown ANSWER_MODE: {mode}. Use extractive, local_llm, or openai."
            )
    return sanitize_answer(query, (answer or "").strip())


def _openai_chat(messages: list[dict], max_tokens: int) -> str:
    if not config.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to the .env file. "
            "It is used for embeddings and, in openai answer mode, for generation."
        )
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=60.0)
    completion = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=config.TEMPERATURE,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content or ""


def _ollama_chat(messages: list[dict]) -> str:
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": config.TEMPERATURE},
    }
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            "Локальная LLM недоступна. Запустите Ollama и загрузите модель из LLM_MODEL."
        ) from exc
    return data.get("message", {}).get("content", "")
