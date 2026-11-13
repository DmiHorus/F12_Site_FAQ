import json

import pytest

from backend import config
from backend.rag_index import load_faq_data


def test_dataset_has_forty_unique_records():
    items = load_faq_data(str(config.FAQ_PATH))
    assert len(items) == 40
    assert [item["id"] for item in items] == [f"faq-{i:03d}" for i in range(1, 41)]
    for item in items:
        assert item["question"].strip()
        assert item["answer"].strip()
        assert item["category"].strip()


def test_duplicate_id_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            [
                {"id": "faq-001", "question": "a", "answer": "b"},
                {"id": "faq-001", "question": "c", "answer": "d"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_faq_data(str(path))


def test_empty_answer_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps([{"id": "faq-001", "question": "a", "answer": "  "}], ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="empty answer"):
        load_faq_data(str(path))
