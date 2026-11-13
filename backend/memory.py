"""Память диалога в процессе: последние 6 реплик на сессию."""

import re
import uuid
from collections import OrderedDict

from . import config

_sessions: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
_last_hits: dict[str, list[str]] = {}
_FOLLOWUP = re.compile(r"^(а|и)\s+", re.IGNORECASE)


def last_hit_ids(session_id: str) -> list[str]:
    return list(_last_hits.get(session_id, []))


def set_last_hit_ids(session_id: str, ids: list[str]) -> None:
    _last_hits[session_id] = list(ids)


def get_or_create_session(session_id: str | None) -> str:
    if session_id and session_id.strip():
        sid = session_id.strip()[:128]
    else:
        sid = uuid.uuid4().hex
    if sid not in _sessions:
        if len(_sessions) >= config.MAX_SESSIONS:
            _sessions.popitem(last=False)
        _sessions[sid] = []
    else:
        _sessions.move_to_end(sid)
    return sid


def get_history(session_id: str) -> list[dict[str, str]]:
    return list(_sessions.get(session_id, []))


def append_turn(session_id: str, user_text: str, assistant_text: str) -> None:
    history = _sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": assistant_text})
    overflow = len(history) - config.MAX_HISTORY_MESSAGES
    if overflow > 0:
        del history[:overflow]


def last_user_message(history: list[dict[str, str]]) -> str | None:
    for item in reversed(history):
        if item.get("role") == "user" and item.get("content"):
            return item["content"]
    return None


def is_followup(message: str) -> bool:
    return bool(_FOLLOWUP.match(message.strip()))


def build_search_query(message: str, history: list[dict[str, str]]) -> str:
    """Короткая реплика-продолжение ищется вместе с предыдущим вопросом."""
    text = message.strip()
    previous = last_user_message(history)
    if previous and is_followup(text):
        return f"{previous.strip()} {text}"
    return text


def clear_memory() -> None:
    _sessions.clear()
    _last_hits.clear()
