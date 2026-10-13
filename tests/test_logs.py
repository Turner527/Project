import sqlite3

import pytest

from agent.llm import LLMError
from agent.loop import Agent
from agent.stub_llm import ScriptedLLM
from agent.tool_adapter import make_adapter
from logs.pilot_summary import format_summary, summarize
from logs.store import LogStore
from tests.helpers import FakeLLM

Q = "Почему мы отстаём по цели роста розничного кредитования и что делать?"


@pytest.fixture(scope="module")
def good_result():
    return Agent(ScriptedLLM(), make_adapter("direct")).run(Q)


@pytest.fixture
def store(tmp_path):
    return LogStore(tmp_path / "pilot.db")


def test_log_request_stores_everything_needed(store, good_result):
    rid = store.log_request(good_result, client_id="u1")
    row = store.get(rid)
    assert row["question"] == Q and row["client_id"] == "u1"
    assert row["verification_status"] == "проверен" and row["error"] is None
    assert row["latency_ms"] == good_result.latency_ms and row["tokens_total"] == good_result.usage.total_tokens
    assert row["prompt_version"] == good_result.prompt_version and row["llm"] == "stub"
    assert row["rating"] is None and row["comment"] is None and row["ts"].endswith("+00:00")

    import json

    calls = json.loads(row["tool_calls"])
    assert [c["tool"] for c in calls] == ["find_kpis_by_goal", "diagnose_deviation"]
    assert calls[1]["arguments"] and calls[1]["result"]["call_id"] == calls[1]["call_id"]
    assert json.loads(row["answer"])["diagnosis"]["key_segments"] == ["Молодёжь"]


def test_failed_request_is_logged_with_error_status(store):
    failed = Agent(FakeLLM([LLMError("Нет связи с сервисом GigaChat")]), make_adapter("direct")).run("вопрос")
    row = store.get(store.log_request(failed, client_id="u1"))
    assert row["verification_status"] == "ошибка" and "Нет связи" in row["error"] and row["answer"] is None


def test_feedback_rating_and_comment_are_independent(store, good_result):
    rid = store.log_request(good_result)
    assert store.set_feedback(rid, rating=-1) is True
    assert store.get(rid)["rating"] == -1 and store.get(rid)["comment"] is None
    assert store.set_feedback(rid, comment="  Нужны цифры по ИП  ") is True
    row = store.get(rid)
    assert row["rating"] == -1 and row["comment"] == "Нужны цифры по ИП" and row["rated_at"]
    assert store.set_feedback(rid, rating=1) and store.get(rid)["rating"] == 1
    assert store.set_feedback("нет-такого", rating=1) is False
    assert store.set_feedback(rid) is False
    with pytest.raises(ValueError):
        store.set_feedback(rid, rating=5)


def test_comment_is_stored_as_data_not_executed(store, good_result):
    rid = store.log_request(good_result)
    nasty = "'); DROP TABLE requests; --"
    store.set_feedback(rid, comment=nasty)
    assert store.get(rid)["comment"] == nasty and store.get(rid) is not None


def test_pilot_summary(store, good_result):
    failed = Agent(FakeLLM([LLMError("сбой")]), make_adapter("direct")).run("вопрос")
    ids = [
        store.log_request(good_result, "u1"),
        store.log_request(good_result, "u1"),
        store.log_request(good_result, "u2"),
        store.log_request(failed, "u3"),
    ]
    other_q = Agent(ScriptedLLM(), make_adapter("direct")).run("Что с оттоком клиентов?")
    ids.append(store.log_request(other_q, "u2"))
    store.set_feedback(ids[0], rating=1)
    store.set_feedback(ids[1], rating=1)
    store.set_feedback(ids[2], rating=-1)
    # u1 вернулся на следующий день
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE requests SET ts = '2026-10-01T10:00:00+00:00' WHERE id = ?", (ids[0],))
        conn.execute("UPDATE requests SET ts = '2026-10-02T10:00:00+00:00' WHERE id = ?", (ids[1],))

    s = summarize(store.path)
    assert (s["total"], s["errors"], s["answered"], s["verified"]) == (5, 1, 4, 4)
    assert s["verified_share"] == 1.0
    assert (s["rated"], s["useful"]) == (3, 2) and s["useful_share"] == pytest.approx(2 / 3)
    assert s["rating_coverage"] == pytest.approx(3 / 5)
    assert s["top_questions"][0] == (Q, 3)
    assert (s["users"], s["repeat_users"], s["returned_other_day"]) == (3, 2, 1)
    assert s["avg_latency_ms"] is not None and s["avg_tokens"] > 0

    text = format_summary(s)
    for fragment in ("Запросов всего", "5", "«проверен»", "100%", "67%", "Самые частые вопросы", "3× " + Q):
        assert fragment in text


def test_questions_are_grouped_ignoring_case_and_punctuation(store, good_result):
    for q in ("Что с оттоком?", "что с оттоком", "  Что  с   оттоком!!  "):
        r = Agent(ScriptedLLM(), make_adapter("direct")).run(q)
        store.log_request(r, "u1")
    assert summarize(store.path)["top_questions"][0][1] == 3


def test_empty_log_summary_does_not_crash(store):
    s = summarize(store.path)
    assert s["total"] == 0 and s["verified_share"] is None and s["useful_share"] is None
    assert "пока нет данных" in format_summary(s)
