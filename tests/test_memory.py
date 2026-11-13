from backend.memory import build_search_query, is_followup


def test_followup_keeps_previous_question():
    history = [
        {"role": "user", "content": "Что такое ГОРИЗОНТ / 01?"},
        {"role": "assistant", "content": "Электронная экспедиция."},
    ]
    query = build_search_query("А какие результаты?", history)
    assert "ГОРИЗОНТ / 01" in query
    assert "А какие результаты?" in query


def test_first_message_is_not_rewritten():
    assert build_search_query("А какие результаты?", []) == "А какие результаты?"


def test_standalone_question_ignores_history():
    history = [{"role": "user", "content": "Что такое ГОРИЗОНТ / 01?"}]
    query = build_search_query("Как связаться с компанией?", history)
    assert query == "Как связаться с компанией?"
    assert is_followup("А что делает в нём ИИ-секретарь?")
    assert not is_followup("Как связаться с компанией?")
