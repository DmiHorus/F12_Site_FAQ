"""Сайт-портфолио Минвайл и API FAQ-ассистента."""

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from . import config, embeddings, memory
from .llm import generate_answer, resolve_search_query
from .rag_index import lexical_overlap, load_faq_data, load_index, search_similar
from .site_content import case_cards, faq_groups, home_blocks

app = FastAPI(title="FAQ-ассистент Минвайл")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=str(config.ROOT / "frontend" / "templates"))
app.mount(
    "/static",
    StaticFiles(directory=str(config.ROOT / "frontend" / "static")),
    name="static",
)

_state: dict | None = None


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    top_k: int = Field(default=3, ge=1, le=10)


def reset_state() -> None:
    global _state
    _state = None


def get_state() -> dict:
    global _state
    if _state is not None:
        return _state
    index, meta = load_index(str(config.INDEX_PATH), str(config.META_PATH))
    indexed_model = meta.get("embedding_model")
    if indexed_model != config.EMBEDDING_MODEL:
        raise RuntimeError(
            f"Индекс собран моделью {indexed_model}, а в .env указана {config.EMBEDDING_MODEL}. "
            "Пересоберите индекс командой python -m backend.build_index"
        )
    if int(meta.get("embedding_dim", index.d)) != int(index.d):
        raise RuntimeError(
            "Размерность индекса не совпадает с метаданными. "
            "Пересоберите индекс командой python -m backend.build_index"
        )
    _state = {"index": index, "meta": meta}
    return _state


def _public_context(items: list[dict]) -> list[dict]:
    return [
        {
            "id": item["id"],
            "question": item["question"],
            "category": item.get("category", ""),
            "score": item["score"],
        }
        for item in items
    ]


def _render(request: Request, template: str, page: str, **extra):
    return templates.TemplateResponse(
        request,
        template,
        {"page": page, **extra},
    )


@app.get("/health")
def health():
    state = get_state()
    return {
        "status": "ok",
        "faq_count": len(state["meta"]["items"]),
        "embedding_model": config.EMBEDDING_MODEL,
        "answer_mode": config.ANSWER_MODE,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty")

    try:
        session_id = memory.get_or_create_session(req.session_id)
        history = memory.get_history(session_id)
        search_text = resolve_search_query(message, history, config.ANSWER_MODE)
        state = get_state()
        query_vec = embeddings.embed_query(search_text)
        exclude = memory.last_hit_ids(session_id) if memory.is_followup(message) else None
        hits = search_similar(
            state["index"],
            state["meta"]["items"],
            query_vec,
            k=req.top_k,
            query_text=search_text,
            exclude_ids=exclude,
            vector_map=state["meta"].get("vector_map"),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    best = hits[0]["score"] if hits else -1.0
    covered = bool(hits) and lexical_overlap(search_text, hits[0]) > 0
    if not hits or best < config.SCORE_THRESHOLD or not covered:
        answer = config.REFUSAL_TEXT
        found = False
        context: list[dict] = []
    else:
        try:
            answer = generate_answer(message, hits, history, config.ANSWER_MODE)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        found = True
        context = _public_context(hits)

    memory.append_turn(session_id, message, answer)
    memory.set_last_hit_ids(session_id, [context[0]["id"]] if found and context else [])
    return {
        "answer": answer,
        "found": found,
        "session_id": session_id,
        "answer_mode": config.ANSWER_MODE,
        "context": context,
    }


@app.get("/api/faq/search")
def faq_search(q: str = "", top_k: int = Query(default=5, ge=1, le=10)):
    query = q.strip()
    if not query:
        return {"query": q, "results": []}
    try:
        state = get_state()
        query_vec = embeddings.embed_query(query)
        hits = search_similar(
            state["index"],
            state["meta"]["items"],
            query_vec,
            k=top_k,
            query_text=query,
            vector_map=state["meta"].get("vector_map"),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    results = [
        {
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "category": item.get("category", ""),
            "score": item["score"],
        }
        for item in hits
    ]
    return {"query": query, "results": results}


@app.get("/api/faq")
def faq_all():
    items = load_faq_data(str(config.FAQ_PATH))
    return {"count": len(items), "items": items}


@app.get("/")
def home(request: Request):
    return _render(request, "index.html", "home", blocks=home_blocks())


@app.get("/cases")
def cases(request: Request):
    return _render(request, "cases.html", "cases", cards=case_cards())


@app.get("/faq")
def faq_page(request: Request):
    return _render(request, "faq.html", "faq", groups=faq_groups())


@app.get("/contacts")
def contacts(request: Request):
    return _render(request, "contacts.html", "contacts")
