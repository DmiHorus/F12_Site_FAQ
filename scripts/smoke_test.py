"""Проверка запущенного сервера. Печатает PASS или FAIL по каждой проверке."""

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MONEY = re.compile(
    r"(руб\.?|рубл\w*|₽|\$|€|\busd\b|\beur\b|доллар\w*|\bевро\b|"
    r"миллион\w*|миллиард\w*|\bмлн\b|\bмлрд\b)",
    re.IGNORECASE,
)
REFUSAL = "В базе знаний нет ответа на этот вопрос. Напишите нам: hello@mnwhl.ru"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    line = f"{status} {name}"
    if detail and not ok:
        line += f" — {detail}"
    print(line)


def request(base: str, method: str, path: str, payload: dict | None = None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=90) as response:
        body = response.read().decode("utf-8")
        return response.status, body


def chat(base: str, message: str, session_id: str | None = None) -> dict:
    payload = {"message": message, "top_k": 3}
    if session_id:
        payload["session_id"] = session_id
    status, body = request(base, "POST", "/chat", payload)
    if status != 200:
        raise RuntimeError(f"HTTP {status}")
    return json.loads(body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка FAQ-ассистента Минвайл")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    base = f"http://{args.host}:{args.port}"

    try:
        status, body = request(base, "GET", "/health")
        health = json.loads(body)
        check(
            "health",
            status == 200 and health.get("status") == "ok" and health.get("faq_count") == 40,
            json.dumps(health, ensure_ascii=False),
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        check("health", False, str(exc))
        print("Итог: сервер недоступен")
        return 1

    for path in ("/", "/cases", "/faq", "/contacts"):
        try:
            status, html = request(base, "GET", path)
            check(f"page {path}", status == 200 and 'id="chat-launcher"' in html)
            if path == "/cases":
                check("cases cards", html.count("case-card") >= 6, f"count={html.count('case-card')}")
                check("cases facts", "76%" in html and "0,8%" in html)
            if path == "/faq":
                check("faq list", html.count("faq-item") == 40)
            if path == "/contacts":
                check("contacts", "hello@mnwhl.ru" in html and "mnwhl.ru" in html)
        except urllib.error.URLError as exc:
            check(f"page {path}", False, str(exc))

    try:
        status, body = request(base, "GET", "/api/faq/search?q=" + urllib.parse.quote("заменить Zabbix") + "&top_k=5")
        data = json.loads(body)
        top = data["results"][0]["id"] if data.get("results") else None
        check("search Zabbix", status == 200 and top == "faq-034", f"top={top}")
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
        check("search Zabbix", False, str(exc))

    expected_chats = [
        ("Как связаться с компанией?", "faq-002", ["hello@mnwhl.ru"]),
        ("Что такое ГОРИЗОНТ / 01?", "faq-013", ["электронн"]),
        ("Какая точность маршрутизации получилась на пилоте?", "faq-015", ["76"]),
        ("Кто запишет совещание и разнесёт поручения?", "faq-028", ["поручен"]),
        ("Чем заменить Zabbix и RabbitMQ?", "faq-034", ["мониторинг"]),
        ("Насколько снизился брак на производстве?", "faq-037", ["0,8"]),
        ("Как скрыть персональные данные перед отправкой в нейросеть?", "faq-023", ["маскир"]),
        ("Ответ гражданину уходит без проверки человеком?", "faq-012", ["подпись"]),
    ]
    for index, (message, faq_id, pieces) in enumerate(expected_chats, start=1):
        try:
            data = chat(base, message, session_id=f"smoke-q{index}")
            top = data["context"][0]["id"] if data.get("context") else None
            missing = [piece for piece in pieces if piece.casefold() not in data["answer"].casefold()]
            check(
                f"chat {faq_id}",
                data.get("found") is True and top == faq_id and not missing,
                f"top={top}, missing={missing}, answer={data.get('answer', '')[:180]}",
            )
        except Exception as exc:
            check(f"chat {faq_id}", False, str(exc))

    for message in ("Какая сегодня погода в Москве?", "Как оформить возврат товара?"):
        try:
            data = chat(base, message, session_id=f"smoke-out-{message[:12]}")
            check(
                f"refusal {message}",
                data.get("found") is False and data.get("answer") == REFUSAL,
                data.get("answer", ""),
            )
        except Exception as exc:
            check(f"refusal {message}", False, str(exc))

    spec = json.loads((ROOT / "data" / "faq_eval.json").read_text(encoding="utf-8"))
    for row in spec["no_price"]:
        try:
            data = chat(base, row["query"], session_id="smoke-price-" + row["query"][:12])
            check(
                f"no price {row['query']}",
                not MONEY.search(data.get("answer", "")),
                data.get("answer", "")[:180],
            )
        except Exception as exc:
            check(f"no price {row['query']}", False, str(exc))

    for dialog in spec["memory_dialogs"]:
        session_id = "smoke-" + dialog["name"]
        try:
            first, second = dialog["turns"]
            chat(base, first["message"], session_id=session_id)
            data = chat(base, second["message"], session_id=session_id)
            top = data["context"][0]["id"] if data.get("context") else None
            missing = [piece for piece in second.get("must_contain", []) if piece not in data.get("answer", "")]
            check(
                f"memory {dialog['name']}",
                data.get("found") is True and top in second["expected_ids"] and not missing,
                f"top={top}, missing={missing}, answer={data.get('answer', '')[:180]}",
            )
        except Exception as exc:
            check(f"memory {dialog['name']}", False, str(exc))

    failed = [name for name, ok, _detail in RESULTS if not ok]
    print(f"Итог: {len(RESULTS) - len(failed)} PASS, {len(failed)} FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
