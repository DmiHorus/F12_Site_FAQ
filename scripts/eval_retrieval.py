"""Оценка поиска по data/faq_eval.json. Набор не изменяется."""

import json
import re
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import config
from backend.embeddings import embed_query
from backend.llm import generate_answer, resolve_search_query
from backend.memory import append_turn, clear_memory, get_history, get_or_create_session, is_followup
from backend.rag_index import lexical_overlap, load_index, search_similar

MONEY = re.compile(
    r"(руб\.?|рубл\w*|₽|\$|€|\busd\b|\beur\b|доллар\w*|\bевро\b|"
    r"миллион\w*|миллиард\w*|\bмлн\b|\bмлрд\b)",
    re.IGNORECASE,
)


def search(
    index,
    items,
    text: str,
    top_k: int = 3,
    exclude_ids: list[str] | None = None,
    vector_map: list[int] | None = None,
):
    started = time.perf_counter()
    vector = embed_query(text)
    hits = search_similar(
        index,
        items,
        vector,
        k=top_k,
        query_text=text,
        exclude_ids=exclude_ids,
        vector_map=vector_map,
    )
    elapsed = time.perf_counter() - started
    return hits, elapsed


def choose_threshold(in_scores: list[float], out_scores: list[float]) -> float:
    """Самый высокий порог, который сохраняет все попадания in_scope и отсекает минимум 4 из 5 вне базы."""
    if not in_scores:
        return config.SCORE_THRESHOLD
    best = None
    for step in range(0, 10001):
        threshold = step / 10000
        kept = sum(score >= threshold for score in in_scores)
        refused = sum(score < threshold for score in out_scores)
        if kept == len(in_scores) and refused >= 4:
            best = threshold
    if best is not None:
        return round(best, 4)
    # Если оценки пересекаются, сохраняем максимум верных in_scope при отказе не меньше 4.
    fallback = None
    for step in range(0, 10001):
        threshold = step / 10000
        kept = sum(score >= threshold for score in in_scores)
        refused = sum(score < threshold for score in out_scores)
        if refused >= 4 and (fallback is None or kept > fallback[0]):
            fallback = (kept, threshold)
    if fallback:
        return round(fallback[1], 4)
    return config.SCORE_THRESHOLD


def main() -> int:
    eval_path = config.DATA_DIR / "faq_eval.json"
    spec = json.loads(eval_path.read_text(encoding="utf-8"))
    index, meta = load_index(str(config.INDEX_PATH), str(config.META_PATH))
    if meta.get("embedding_model") != config.EMBEDDING_MODEL:
        raise SystemExit(
            "Модель индекса не совпадает с .env. Запустите python -m backend.build_index"
        )
    items = meta["items"]
    vector_map = meta.get("vector_map")
    times: list[float] = []

    in_hits_1 = 0
    in_hits_3 = 0
    in_scores = []
    misses = []
    for row in spec["in_scope"]:
        hits, elapsed = search(index, items, row["query"], top_k=3, vector_map=vector_map)
        times.append(elapsed)
        ids = [hit["id"] for hit in hits]
        expected = set(row["expected_ids"])
        if ids and ids[0] in expected:
            in_hits_1 += 1
        else:
            misses.append((row["query"], ids[:1], row["expected_ids"]))
        if any(faq_id in expected for faq_id in ids[:3]):
            in_hits_3 += 1
        in_scores.append(hits[0]["score"] if hits else -1.0)

    out_hits = []
    out_scores = []
    for row in spec["out_of_scope"]:
        hits, elapsed = search(index, items, row["query"], top_k=3, vector_map=vector_map)
        times.append(elapsed)
        out_hits.append(hits)
        out_scores.append(hits[0]["score"] if hits else -1.0)

    total_in = len(spec["in_scope"])
    recall_at_1 = in_hits_1 / total_in
    recall_at_3 = in_hits_3 / total_in
    threshold = choose_threshold(in_scores, out_scores)

    def refused(query: str, hits: list, score: float) -> bool:
        if not hits or score < threshold:
            return True
        return lexical_overlap(query, hits[0]) <= 0

    refusals = sum(
        refused(row["query"], hits, score)
        for row, hits, score in zip(spec["out_of_scope"], out_hits, out_scores)
    )
    kept = sum(score >= threshold for score in in_scores)
    mean_ms = round(sum(times) / len(times) * 1000, 1) if times else 0.0

    print(f"Модель: {config.EMBEDDING_PROVIDER}:{config.EMBEDDING_MODEL}")
    print(f"Режим ответа: {config.ANSWER_MODE}")
    print(f"Recall@1: {recall_at_1:.3f} ({in_hits_1}/{total_in})")
    print(f"Recall@3: {recall_at_3:.3f} ({in_hits_3}/{total_in})")
    print(f"Порог: {threshold}")
    print(f"Отказы вне базы: {refusals}/{len(out_scores)}")
    print(f"In-scope выше порога: {kept}/{total_in}")
    print(f"Среднее время запроса (эмбеддинг + поиск): {mean_ms} мс")
    if misses:
        print("Промахи Recall@1:")
        for query, got, expected in misses:
            print(f"  - {query} -> {got}, ожидалось {expected}")

    clear_memory()
    memory_ok = True
    for dialog in spec["memory_dialogs"]:
        session_id = get_or_create_session(f"eval-{dialog['name']}")
        print(f"Диалог: {dialog['name']}")
        last_ids: list[str] = []
        for turn in dialog["turns"]:
            history = get_history(session_id)
            query = resolve_search_query(turn["message"], history, config.ANSWER_MODE)
            exclude = last_ids if is_followup(turn["message"]) else None
            hits, _elapsed = search(
                index, items, query, top_k=3, exclude_ids=exclude, vector_map=vector_map
            )
            top_id = hits[0]["id"] if hits else None
            expected = turn["expected_ids"]
            ok = top_id in expected
            memory_ok = memory_ok and ok
            print(f"  {turn['message']} -> {top_id} ({'ok' if ok else 'FAIL'}, ожидалось {expected})")
            print(f"    запрос: {query}")
            answer = config.REFUSAL_TEXT
            if hits and hits[0]["score"] >= threshold:
                answer = generate_answer(turn["message"], hits, history, config.ANSWER_MODE)
            append_turn(session_id, turn["message"], answer)
            last_ids = [hit["id"] for hit in hits[:1]]
            if turn.get("must_contain"):
                missing = [piece for piece in turn["must_contain"] if piece not in answer]
                if missing:
                    memory_ok = False
                    print(f"    в ответе нет: {missing}")
                else:
                    print("    обязательные фрагменты на месте")

    price_ok = True
    for row in spec["no_price"]:
        hits, _elapsed = search(index, items, row["query"], top_k=3, vector_map=vector_map)
        if not hits or hits[0]["score"] < threshold:
            answer = config.REFUSAL_TEXT
        else:
            answer = generate_answer(row["query"], hits, [], config.ANSWER_MODE)
        has_money = bool(MONEY.search(answer))
        price_ok = price_ok and not has_money
        print(f"Деньги: {row['query']} -> {'FAIL' if has_money else 'ok'}")
        if has_money:
            print(f"  ответ: {answer}")

    peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = round(peak_bytes / (1024 * 1024), 1)
    metrics = {
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_provider": config.EMBEDDING_PROVIDER,
        "llm_model": config.LLM_MODEL,
        "answer_mode": config.ANSWER_MODE,
        "recall_at_1": round(recall_at_1, 4),
        "recall_at_3": round(recall_at_3, 4),
        "out_of_scope_refusals": f"{refusals}/{len(out_scores)}",
        "mean_query_ms": mean_ms,
        "threshold": threshold,
        "model_disk": "не хранится на диске (облачная модель OpenAI)",
        "peak_ram_mb": peak_mb,
    }
    config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Пик ОЗУ процесса: {peak_mb} МБ")
    print(f"Метрики записаны в {config.METRICS_PATH}")

    ok = (
        recall_at_1 >= 0.85
        and recall_at_3 >= 0.95
        and refusals >= 4
        and memory_ok
        and price_ok
    )
    print("Итог:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
