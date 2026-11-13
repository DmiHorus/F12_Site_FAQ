# FAQ-ассистент Минвайл

Сайт-портфолио компании Минвайл со встроенным поиском по базе из 40 вопросов и ответов. Данные → FAISS → FastAPI → страницы и чат-виджет.

Ответы только по найденному фрагменту базы. Если сходство ниже порога, ассистент отказывается и предлагает написать на hello@mnwhl.ru.

## Запуск за 5 команд

Нужны Python 3.11 и ключ OpenAI в `.env` (имя переменной `OPENAI_API_KEY`).

```bash
python3.11 -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # затем впишите OPENAI_API_KEY
python -m backend.build_index
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Откройте http://127.0.0.1:8000. Во втором терминале:

```bash
python scripts/smoke_test.py
pytest -q
python scripts/eval_retrieval.py
```

Порт для `smoke_test.py`: `python scripts/smoke_test.py --port 8001`.

## Режим и модель

Сценарий ответа: OpenAI (`ANSWER_MODE=openai`).

- Эмбеддинги: `text-embedding-3-small`
- Генерация: `gpt-4.1-mini`
- Другие режимы в `.env`: `extractive` (текст FAQ без генерации) и `local` / `local_llm`

## Измеренные метрики

На `data/faq_eval.json` (набор не подгоняли):

- Recall@1 = 0,933
- Recall@3 = 1,000
- верных отказов вне базы: 5 из 5
- среднее время запроса: 337 мс
- порог сходства: 0,31

Подробности выбора и характеристики ПК: [docs/MODEL_CHOICE.md](docs/MODEL_CHOICE.md).

## Структура

```text
backend/     FastAPI, индекс, поиск, память диалога
frontend/    шаблоны страниц, стили, чат-виджет
data/        faqs.json, faq_eval.json; индекс собирается скриптом
scripts/     eval_retrieval.py, smoke_test.py
tests/       pytest без сети
docs/        промпт, выбор модели, сценарий проверки
```

## Страницы

- `/` — компания и подход
- `/cases` — кейсы-продукты
- `/faq` — поиск и 40 вопросов
- `/contacts` — hello@mnwhl.ru, mnwhl.ru

Пункт 4 домашнего задания (деплой на удалённый сервер) в этой версии не выполнялся.
