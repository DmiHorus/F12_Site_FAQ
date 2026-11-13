from backend import config


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["faq_count"] == 40
    assert body["answer_mode"] == "extractive"
    assert body["embedding_model"] == "hash-lexical"


def test_chat_finds_contact(client):
    response = client.post("/chat", json={"message": "Как связаться с компанией Минвайл?"})
    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert "hello@mnwhl.ru" in body["answer"]
    assert body["context"][0]["id"] == "faq-002"
    assert body["session_id"]
    assert body["answer_mode"] == "extractive"


def test_chat_refuses_out_of_scope(client):
    response = client.post("/chat", json={"message": "Какая сегодня погода в Москве?"})
    assert response.status_code == 200
    body = response.json()
    assert body["found"] is False
    assert body["answer"] == config.REFUSAL_TEXT
    assert body["context"] == []


def test_empty_message_is_400(client):
    response = client.post("/chat", json={"message": "   "})
    assert response.status_code == 400


def test_search_returns_answer_and_score(client):
    response = client.get("/api/faq/search", params={"q": "Как связаться с компанией", "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["results"]
    hit = body["results"][0]
    assert hit["id"] == "faq-002"
    assert "hello@mnwhl.ru" in hit["answer"]
    assert "score" in hit
    assert hit["category"]


def test_faq_list_has_forty(client):
    response = client.get("/api/faq")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 40
    assert len(body["items"]) == 40


def test_pages_and_widget(client):
    for path in ("/", "/cases", "/faq", "/contacts"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert 'id="chat-launcher"' in response.text
        assert "support@example.com" not in response.text
        assert "возврат товара" not in response.text
        assert "+7 (999)" not in response.text

    cases = client.get("/cases")
    assert cases.text.count("case-card") >= 6
    assert "76%" in cases.text
    assert "0,8%" in cases.text

    faq = client.get("/faq")
    assert faq.text.count("faq-item") == 40

    contacts = client.get("/contacts")
    assert "hello@mnwhl.ru" in contacts.text
    assert "mnwhl.ru" in contacts.text

    logo = client.get("/static/img/mnwhl-logo-white.png")
    assert logo.status_code == 200
    script = client.get("/static/js/chat-widget.js")
    assert "Привет! Я FAQ-ассистент Минвайл. Спросите о компании, продуктах и подходе" in script.text
    assert 'fetch("/chat"' in script.text
