import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.llm import LLMError
from agent.stub_llm import ScriptedLLM
from agent.tool_adapter import make_adapter
from app.main import MAX_UPLOAD_BYTES, create_app
from data import generate
from logs.store import LogStore
from tests.conftest import DEMO_XLSX
from tests.helpers import FakeLLM
from tools import datastore

Q = "Почему мы отстаём по цели роста розничного кредитования и что делать?"


@pytest.fixture
def store(tmp_path):
    return LogStore(tmp_path / "pilot.db")


@pytest.fixture
def client(store):
    app = create_app(llm=ScriptedLLM(), adapter=make_adapter("direct"), store=store, load_env_file=False)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    """Загрузки уходят во временный каталог, а активной витриной становится «загруженная, если есть»."""
    monkeypatch.delenv("KPI_DATA_PATH", raising=False)
    monkeypatch.setattr(datastore, "UPLOAD_PATH", tmp_path / "uploads" / "active.xlsx")
    monkeypatch.setattr(datastore, "DEFAULT_PATH", DEMO_XLSX)
    return datastore.UPLOAD_PATH


def xlsx_bytes(**mutations) -> bytes:
    goals, facts, links = generate.build_frames()
    frames = {"goals": goals, "facts": facts, "links": links}
    for sheet, fn in mutations.items():
        frames[sheet] = fn(frames[sheet]) if fn else None
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in frames.items():
            if df is not None:
                df.to_excel(xw, sheet_name=name, index=False)
    return buf.getvalue()


def upload(client, content, name="kpi.xlsx"):
    return client.post("/api/upload", files={"file": (name, content, "application/vnd.ms-excel")})


# ---------- страница и статус ----------


def test_index_serves_russian_interface(client):
    html = client.get("/").text
    for fragment in ("AI-аналитик", "Загрузить xlsx", "Полезно", "Не полезно", "Как получен ответ", "lang=\"ru\"",
                     "Подключить GigaChat", "Новый разговор"):
        assert fragment in html


def test_index_has_dark_green_default_and_light_theme_without_external_resources(client):
    html = client.get("/").text
    # тёмно-зелёная тема по умолчанию, светлая — по переключателю; системную настройку браузера не читаем
    assert '<html lang="ru" data-theme="dark">' in html and ':root[data-theme="light"]' in html
    assert 'id="theme-btn"' in html and "prefers-color-scheme" not in html
    # интерфейс должен работать офлайн: ни CDN, ни внешних шрифтов
    assert "https://" not in html and "http://" not in html.replace("http://www.w3.org/2000/svg", "")


def test_status(client):
    s = client.get("/api/status").json()
    assert s["offline"] is True and s["llm"] == "stub" and s["llm_error"] is None
    assert s["dataset"]["kpis"] == 18 and s["dataset"]["is_upload"] is False
    assert s["dataset"]["first_month"] == "2025-04" and s["prompt_version"].startswith("v1-")


# ---------- вопрос-ответ ----------


def test_ask_returns_memo_trace_and_logs_request(client, store):
    r = client.post("/api/ask", json={"question": Q, "client_id": "tester"})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "проверен" and data["error"] is None
    assert data["memo"]["diagnosis"]["key_segments"] == ["Молодёжь"]
    assert [c["tool"] for c in data["tool_calls"]] == ["find_kpis_by_goal", "diagnose_deviation"]
    assert data["verification"]["unverified"] == [] and data["tokens"]["total"] > 0
    row = store.get(data["request_id"])
    assert row["question"] == Q and row["client_id"] == "tester" and row["verification_status"] == "проверен"


def test_ask_validates_input(client):
    assert client.post("/api/ask", json={"question": "а"}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 1001}).status_code == 422
    assert client.post("/api/ask", json={}).status_code == 422


def test_model_failure_is_a_normal_response_with_error_status(store):
    app = create_app(llm=FakeLLM([LLMError("Превышен лимит запросов (429).")]), adapter=make_adapter("direct"),
                     store=store, load_env_file=False)
    data = TestClient(app).post("/api/ask", json={"question": Q}).json()
    assert data["status"] == "ошибка" and "429" in data["error"] and data["memo"] is None
    assert store.get(data["request_id"])["verification_status"] == "ошибка"


def test_unconfigured_gigachat_is_explained(store, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "gigachat")
    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    app = create_app(adapter=make_adapter("direct"), store=store, load_env_file=False)
    with TestClient(app) as c:
        assert "GIGACHAT_CREDENTIALS" in c.get("/api/status").json()["llm_error"]
        r = c.post("/api/ask", json={"question": Q})
        assert r.status_code == 503 and "GIGACHAT_CREDENTIALS" in r.json()["detail"]


# ---------- оценки ----------


def test_feedback_flow(client, store):
    rid = client.post("/api/ask", json={"question": Q}).json()["request_id"]
    assert client.post("/api/feedback", json={"request_id": rid, "rating": "not_useful"}).json() == {"ok": True}
    assert store.get(rid)["rating"] == -1
    assert client.post("/api/feedback", json={"request_id": rid, "comment": "Мало деталей по ИП"}).status_code == 200
    row = store.get(rid)
    assert row["rating"] == -1 and row["comment"] == "Мало деталей по ИП"
    client.post("/api/feedback", json={"request_id": rid, "rating": "useful"})
    assert store.get(rid)["rating"] == 1


def test_feedback_errors(client):
    assert client.post("/api/feedback", json={"request_id": "нет", "rating": "useful"}).status_code == 404
    assert client.post("/api/feedback", json={"request_id": "x"}).status_code == 422
    assert client.post("/api/feedback", json={"request_id": "x", "rating": "отлично"}).status_code == 422


# ---------- загрузка xlsx ----------


def test_valid_upload_becomes_active_dataset(client, uploads):
    r = upload(client, xlsx_bytes())
    assert r.status_code == 200 and r.json()["ok"] and r.json()["dataset"]["is_upload"] is True
    assert uploads.exists()
    assert client.get("/api/status").json()["dataset"]["is_upload"] is True
    assert client.post("/api/ask", json={"question": Q}).json()["status"] == "проверен"


def test_upload_with_problems_returns_russian_errors_and_keeps_old_data(client, uploads):
    bad = xlsx_bytes(links=None, facts=lambda df: df.drop(columns=["план"]))
    r = upload(client, bad)
    assert r.status_code == 422
    body = r.json()
    assert body["ok"] is False and body["detail"] == "Файл не прошёл проверку."
    assert any("Нет листа «links»" in e for e in body["errors"])
    assert any("нет колонок «план»" in e for e in body["errors"])
    assert not uploads.exists() and client.get("/api/status").json()["dataset"]["is_upload"] is False


def test_upload_rejects_wrong_files(client, uploads):
    assert "не в формате .xlsx" in upload(client, b"a,b\n1,2\n", "data.csv").json()["detail"]
    assert "не похож на книгу Excel" in upload(client, "привет".encode(), "kpi.xlsx").json()["detail"]
    broken = b"PK\x03\x04" + b"\x00" * 50  # похоже на zip, но не xlsx
    r = upload(client, broken)
    assert r.status_code == 422 and "Не удалось открыть файл как Excel" in r.json()["errors"][0]
    huge = upload(client, b"PK\x03\x04" + b"0" * (MAX_UPLOAD_BYTES + 10))
    assert huge.status_code == 413 and "больше 10 МБ" in huge.json()["detail"]
    assert not uploads.exists()
