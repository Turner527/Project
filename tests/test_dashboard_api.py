"""Дашборд через API: построение, хранение, привязка к отчёту, ошибки модели, чат после дашборда, журнал."""
import json
import os
import re
import shutil
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from agent.llm import LLMError
from agent.stub_llm import ScriptedLLM
from agent.tool_adapter import make_adapter
from app.main import create_app
from data import generate_samples
from logs.pilot_summary import format_summary, summarize
from logs.store import LogStore
from tests.conftest import DEMO_XLSX
from tests.helpers import FakeLLM, activate_flat, memo_json, plan_json, text
from tools import dashboard_store, datastore

STRATEGIC_QUESTION = "Что делать дальше, в каком порядке действовать?"


@pytest.fixture
def store(tmp_path):
    return LogStore(tmp_path / "pilot.db")


def make_client(store, llm=None) -> TestClient:
    app = create_app(llm=llm or ScriptedLLM(), adapter=make_adapter("direct"), store=store, load_env_file=False)
    return TestClient(app)


@pytest.fixture
def client(store):
    with make_client(store) as c:
        yield c


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    """Загрузки и дашборды уходят во временную папку; активным отчётом становится «загруженный, если есть»."""
    monkeypatch.delenv("KPI_DATA_PATH", raising=False)
    monkeypatch.setattr(datastore, "DEFAULT_PATH", DEMO_XLSX)
    datastore._cache.clear()
    return datastore.UPLOAD_PATH


def sample_bytes(tmp_path, name) -> bytes:
    path = tmp_path / f"{name}.xlsx"
    generate_samples.write_sample(name, path)
    return path.read_bytes()


def load_report(client, content: bytes, name: str, edit=None) -> dict:
    """Загружает плоский отчёт и подтверждает роли столбцов (edit правит спецификацию перед подтверждением)."""
    body = client.post("/api/upload", files={"file": (name, content, "application/vnd.ms-excel")}).json()
    spec = body["profile"]["spec"]
    if edit:
        edit(spec)
    r = client.post("/api/datasets/confirm", json={"upload_id": body["upload_id"], "spec": spec})
    assert r.status_code == 200, r.text
    return r.json()


def build(client, **body) -> dict:
    r = client.post("/api/dashboard", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def current(client):
    return client.get("/api/dashboard").json()["dashboard"]


# ======================================================================================================================
# Построение и показ
# ======================================================================================================================


def test_there_is_no_dashboard_until_one_is_built(client):
    assert client.get("/api/dashboard").json() == {"dashboard": None}
    assert client.get("/api/status").json()["dashboard"] is None


def test_building_a_dashboard_on_the_strategic_demo(client, store):
    d = build(client, client_id="u1")
    assert d["plan_source"] == "model" and d["error"] is None and d["llm"] == "stub"
    assert d["verification"]["status"] == "проверен" and d["verification"]["unverified"] == []
    assert len(d["widgets"]) >= 3 and {w["type"] for w in d["widgets"]} == {"kpi", "trend", "breakdown"}
    assert re.fullmatch(r"[0-9a-f]{16}", d["fingerprint"]) and d["request_id"] and d["prompt_version"].startswith("d1-")
    sent = d["sent_to_model"]
    assert sent["chars"] > 500 and sent["metrics"] == len(sent["digest"]["metrics"]) and sent["dimensions"] >= 1
    assert d["tokens"]["total"] > 0 and d["latency_ms"] >= 0
    # дашборд сохранён: GET возвращает тот же, а в статусе есть краткая карточка
    assert current(client) == d
    assert client.get("/api/status").json()["dashboard"] == {
        "title": d["plan"]["title"], "created_at": d["created_at"], "plan_source": "model", "widgets": len(d["widgets"])}
    row = store.get(d["request_id"])
    assert row["question"].startswith("[дашборд] ") and row["verification_status"] == "проверен" and row["client_id"] == "u1"
    assert json.loads(row["answer"])["title"] == d["plan"]["title"] and row["error"] is None


def test_a_dashboard_can_be_requested_without_a_body(client):
    r = client.post("/api/dashboard")
    assert r.status_code == 200 and r.json()["widgets"]
    assert client.post("/api/dashboard", json={"client_id": "x" * 65}).status_code == 422


def test_dashboard_of_an_uploaded_report_names_the_problem_segment(client, uploads, tmp_path):
    load_report(client, sample_bytes(tmp_path, "marketing"), "marketing.xlsx")
    d = build(client)
    assert d["plan_source"] == "model" and d["verification"]["status"] == "проверен"
    assert d["sent_to_model"]["digest"]["report"]["source"] == "marketing.xlsx"
    assert d["plan"]["insights"][0]["severity"] == "problem" and any("Поиск" in i["text"] for i in d["plan"]["insights"])
    # в модель ушла сводка, а не строки отчёта: ни одной даты построчных данных и никаких заметок о файле
    sent = json.dumps(d["sent_to_model"]["digest"], ensure_ascii=False)
    assert "notes" not in d["sent_to_model"]["digest"] and sent.count("2026-08-0") < 40
    assert "Лист «Данные»" not in sent


def test_the_digest_in_the_response_is_exactly_what_the_model_received(store, uploads, tmp_path):
    llm = FakeLLM([text(plan_json())])
    with make_client(store, llm) as c:
        load_report(c, sample_bytes(tmp_path, "marketing"), "marketing.xlsx")
        d = build(c)
    prompt = llm.calls[0]["messages"][1]["content"]
    digest = json.loads(prompt.split("<<СВОДКА>>", 1)[1].rsplit("<<КОНЕЦ СВОДКИ>>", 1)[0])
    assert digest == d["sent_to_model"]["digest"] and d["sent_to_model"]["chars"] > 0
    assert llm.calls[0]["functions"] is None  # модель не вызывает инструментов: ей нужен только текст плана


# ======================================================================================================================
# Дашборд привязан к своему отчёту
# ======================================================================================================================


def test_dashboard_is_bound_to_its_report(client, uploads, tmp_path):
    demo = build(client)
    data = sample_bytes(tmp_path, "marketing")
    load_report(client, data, "marketing.xlsx")
    assert current(client) is None and client.get("/api/status").json()["dashboard"] is None  # другой отчёт — другого дашборда нет
    marketing = build(client)
    assert marketing["fingerprint"] != demo["fingerprint"]
    assert current(client)["fingerprint"] == marketing["fingerprint"]
    client.delete("/api/upload")  # вернулись к демо-данным: их дашборд на месте
    assert current(client)["fingerprint"] == demo["fingerprint"]
    load_report(client, data, "marketing.xlsx")  # тот же файл загружен снова: найден его дашборд, модель повторно не нужна
    assert current(client)["fingerprint"] == marketing["fingerprint"]
    assert current(client)["plan"]["title"] == marketing["plan"]["title"]


def test_other_column_roles_mean_another_dashboard(client, uploads, tmp_path):
    data = sample_bytes(tmp_path, "marketing")
    load_report(client, data, "marketing.xlsx")
    first = build(client)
    load_report(client, data, "marketing.xlsx", edit=lambda spec: spec["metrics"].pop())  # пользователь убрал показатель
    assert current(client) is None
    second = build(client)
    assert second["fingerprint"] != first["fingerprint"]
    assert len(second["sent_to_model"]["digest"]["metrics"]) == len(first["sent_to_model"]["digest"]["metrics"]) - 1


def test_rebuilding_replaces_the_stored_dashboard_of_the_same_report(client):
    first = build(client)
    second = build(client)
    assert second["fingerprint"] == first["fingerprint"] and second["request_id"] != first["request_id"]
    assert current(client)["request_id"] == second["request_id"]
    assert len(list(dashboard_store._dir().glob("*.json"))) == 1


# ======================================================================================================================
# Ошибки и недоверие к модели
# ======================================================================================================================


def test_a_failed_model_gives_the_default_dashboard_with_the_reason(store):
    with make_client(store, FakeLLM([LLMError("Превышен лимит запросов (429).")])) as c:
        d = build(c)
        assert d["plan_source"] == "default" and "429" in d["error"] and "по умолчанию" in d["error"]
        assert len(d["widgets"]) >= 3 and d["plan"]["insights"] and d["verification"]["status"] == "проверен"
        assert d["tokens"]["total"] == 0
        assert current(c)["plan_source"] == "default"  # показан и сохранён, чтобы пользователь не остался ни с чем
        assert c.get("/api/status").json()["dashboard"]["plan_source"] == "default"
    row = store.get(d["request_id"])
    assert "429" in row["error"] and row["verification_status"] == "проверен"


def test_an_unusable_answer_is_retried_once_with_the_schema_error(store):
    llm = FakeLLM([text("Вот ваш дашборд: он получился отличным."), text('{"title": "Только заголовок"}')])
    with make_client(store, llm) as c:
        d = build(c)
    assert d["plan_source"] == "default" and "не вернул план" in d["error"] and d["raw_text"]
    assert len(llm.calls) == 2 and "не прошёл проверку схемы" in llm.calls[1]["messages"][-1]["content"]
    assert d["tokens"]["total"] == 60 and d["widgets"]


def test_a_second_attempt_can_repair_the_plan(store):
    llm = FakeLLM([text("{не json"), text(plan_json())])
    with make_client(store, llm) as c:
        d = build(c)
    assert d["plan_source"] == "model" and d["error"] is None and d["raw_text"] == ""


def test_invented_numbers_are_flagged_but_the_dashboard_is_still_shown(store, uploads, tmp_path):
    lie = plan_json(insights=[{"title": "Расходы", "text": "Расходы выросли на 77,7%.", "severity": "problem", "widgets": []},
                              {"title": "Конверсия", "text": "Конверсия в лиды упала в канале «Поиск».", "severity": "risk",
                               "widgets": []}])
    llm = FakeLLM([text(lie), text("ок"), text(memo_json("Главное — просадка конверсии в «Поиске».", used=[]))])
    with make_client(store, llm) as c:
        load_report(c, sample_bytes(tmp_path, "marketing"), "marketing.xlsx")
        d = build(c)
        assert d["plan_source"] == "model" and d["verification"]["status"] == "есть непроверенные числа"
        assert [u["value"] for u in d["verification"]["unverified"]] == ["77,7"]
        assert d["verification"]["unverified"][0]["field"] == "insights[1].text"
        assert store.get(d["request_id"])["verification_status"] == "есть непроверенные числа"
        # ложное число не попадает в контекст чата, а проверенный инсайт попадает
        c.post("/api/ask", json={"question": "Что делать дальше?"})
    system = llm.calls[1]["messages"][0]["content"]
    assert "77,7" not in system and "Конверсия в лиды упала в канале" in system


def test_the_dashboard_needs_a_configured_model(store, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "gigachat")
    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    app = create_app(adapter=make_adapter("direct"), store=store, load_env_file=False)
    with TestClient(app) as c:
        r = c.post("/api/dashboard", json={})
        assert r.status_code == 503 and "GIGACHAT_CREDENTIALS" in r.json()["detail"]
        assert c.get("/api/dashboard").json() == {"dashboard": None}  # просмотр работает и без модели


def test_a_dashboard_that_could_not_be_built_is_not_saved(store, monkeypatch):
    import agent.dashboard as agent_dashboard

    def broken(*args, **kwargs):
        raise RuntimeError("сломалось")

    monkeypatch.setattr(agent_dashboard.D, "assemble", broken)
    with make_client(store) as c:
        d = build(c)
        assert d["widgets"] == [] and d["plan"] is None and "сломалось" in d["error"]
        assert current(c) is None and not list(dashboard_store._dir().glob("*.json"))
    assert store.get(d["request_id"])["verification_status"] == "ошибка"


# ======================================================================================================================
# После дашборда: вопросы по данным и стратегические вопросы
# ======================================================================================================================


def test_a_strategic_question_after_the_dashboard_is_answered_from_its_insights(client):
    dash = build(client)
    r = client.post("/api/ask", json={"question": STRATEGIC_QUESTION}).json()
    assert r["status"] == "проверен" and r["error"] is None and "дашборд" in r["memo"]["used_tool_calls"]
    assert r["tool_calls"] == [] and r["memo"]["manual_checks"]  # инструменты не нужны: ответ строится по выводам дашборда
    assert dash["plan"]["insights"][0]["title"] in r["memo"]["diagnosis"]["summary"]


def test_without_a_dashboard_the_chat_works_as_before(client):
    r = client.post("/api/ask", json={"question": STRATEGIC_QUESTION}).json()
    assert "дашборд" not in (r["memo"] or {}).get("used_tool_calls", [])
    assert r["status"] in ("проверен", "есть непроверенные числа", "ошибка") and r["memo"] is not None


def test_a_factual_question_after_the_dashboard_still_goes_through_the_tools(client, uploads, tmp_path):
    load_report(client, sample_bytes(tmp_path, "marketing"), "marketing.xlsx")
    build(client)
    r = client.post("/api/ask", json={"question": "Почему выросли расходы и в каком канале?"}).json()
    assert r["status"] == "проверен" and [c["tool"] for c in r["tool_calls"]] == ["find_metrics", "diagnose_change"]
    assert r["memo"]["diagnosis"]["key_segments"] == ["Поиск"]


def test_the_chat_does_not_get_the_dashboard_of_another_report(store, uploads, tmp_path):
    llm = FakeLLM([text(plan_json()), text("ок"), text(memo_json())])
    with make_client(store, llm) as c:
        load_report(c, sample_bytes(tmp_path, "marketing"), "marketing.xlsx")
        build(c)
        c.delete("/api/upload")  # вернулись к демо-данным, где дашборда нет
        c.post("/api/ask", json={"question": "Что делать дальше?"})
    assert "Контекст дашборда (построен" not in llm.calls[1]["messages"][0]["content"]


# ======================================================================================================================
# Оценки и журнал пилота
# ======================================================================================================================


def test_feedback_on_a_dashboard_and_the_pilot_summary(client, store):
    first = build(client, client_id="u1")
    client.post("/api/ask", json={"question": "Почему мы отстаём по цели роста розничного кредитования и что делать?",
                                  "client_id": "u1"})
    assert client.post("/api/feedback", json={"request_id": first["request_id"], "rating": "useful"}).json() == {"ok": True}
    assert store.get(first["request_id"])["rating"] == 1
    s = summarize(store.path)
    assert s["total"] == 1 and s["rated"] == 0 and s["top_questions"][0][1] == 1  # дашборд не искажает статистику вопросов
    assert (s["dashboards"], s["dashboards_verified"], s["dashboards_rated"], s["dashboards_useful"]) == (1, 1, 1, 1)
    assert s["dashboards_with_model_error"] == 0 and s["users"] == 1
    report = format_summary(s)
    assert "Дашборды:" in report and "построено: 1, из них со статусом «проверен»: 1" in report


def test_pilot_summary_counts_dashboards_built_without_the_model(store):
    with make_client(store, FakeLLM([LLMError("Нет связи с сервисом GigaChat.")])) as c:
        build(c)
    s = summarize(store.path)
    assert s["dashboards"] == 1 and s["dashboards_with_model_error"] == 1 and s["total"] == 0
    assert "с ошибкой или планом по умолчанию: 1" in format_summary(s)


# ======================================================================================================================
# Страница
# ======================================================================================================================


def test_index_has_the_dashboard_interface_and_stays_offline(client):
    html = client.get("/").text
    for fragment in ("Построить дашборд", "Открыть дашборд", 'id="dash"', "Как построен дашборд", "Рекомендаций GigaChat нет",
                     "Спросить у аналитика", "Что сделать дальше", "/api/dashboard", "п.п.", 'id="dash-btn"', 'id="dc-build"'):
        assert fragment in html, fragment
    # графики рисует сама страница: ни библиотек, ни внешних файлов, ни зависимости от системной темы
    assert "<script src" not in html and '<link rel="stylesheet"' not in html and "@import" not in html
    assert "https://" not in html and "http://" not in html.replace("http://www.w3.org/2000/svg", "")
    assert "prefers-color-scheme" not in html
    # цвета рядов заданы для обеих тем: тёмной по умолчанию и светлой
    assert html.count("--s1:") == 2 and html.count("--s8:") == 2


@pytest.mark.skipif(shutil.which("node") is None, reason="для проверки синтаксиса скрипта страницы нужен node")
def test_the_page_script_parses(client, tmp_path):
    html = client.get("/").text
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(scripts) == 1
    path = tmp_path / "page.js"
    path.write_text(scripts[0], "utf-8")
    done = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr


def test_the_privacy_note_names_what_goes_to_the_model(client):
    html = client.get("/").text
    assert "сводка отчёта (названия показателей и разрезов, итоги, ряды по периодам, вклады значений)" in html
    assert "а не строки таблицы" in html


# ======================================================================================================================
# Хранилище дашбордов и отпечаток отчёта
# ======================================================================================================================


def stored(fp, **extra) -> dict:
    return {"fingerprint": fp, "plan": {"title": "Т"}, "widgets": [], **extra}


def test_store_roundtrip_and_input_checks():
    fp = "a" * 16
    dashboard_store.save(stored(fp, note="Кириллица и «кавычки»"))
    assert dashboard_store.load(fp)["note"] == "Кириллица и «кавычки»"
    for bad in ("", "../../etc/passwd", "A" * 16, "a" * 15, "g" * 16):
        assert dashboard_store.load(bad) is None
        with pytest.raises(ValueError):
            dashboard_store.save(stored(bad))
    with pytest.raises(ValueError):
        dashboard_store.save({"plan": {}})
    assert dashboard_store.load(None) is None and dashboard_store.load("b" * 16) is None
    (dashboard_store._dir() / f"{'c' * 16}.json").write_text("{не json", "utf-8")
    assert dashboard_store.load("c" * 16) is None  # повреждённый файл — как будто дашборда нет
    dashboard_store.discard(fp)
    dashboard_store.discard("../../x")  # не падает и ничего не трогает
    assert dashboard_store.load(fp) is None


def test_store_keeps_only_the_latest_dashboards(monkeypatch):
    monkeypatch.setattr(dashboard_store, "KEEP", 3)
    base = time.time() - 1000
    for i in range(5):
        dashboard_store.save(stored(f"{i:016x}"))
        os.utime(dashboard_store._dir() / f"{i:016x}.json", (base + i, base + i))
    dashboard_store.save(stored(f"{9:016x}"))
    assert {p.stem for p in dashboard_store._dir().glob("*.json")} == {f"{3:016x}", f"{4:016x}", f"{9:016x}"}


def test_fingerprint_follows_the_content_not_the_path_or_the_time(tmp_path, monkeypatch):
    first = tmp_path / "first"
    first.mkdir()
    activate_flat(first, monkeypatch, "marketing")
    fp = datastore.fingerprint()
    assert re.fullmatch(r"[0-9a-f]{16}", fp) and datastore.fingerprint() == fp
    second = tmp_path / "second"
    shutil.copytree(first, second)
    os.utime(second / "data.xlsx", (1_000_000, 1_000_000))
    monkeypatch.setenv("KPI_DATA_PATH", str(second / "spec.json"))
    assert datastore.fingerprint() == fp  # та же таблица и те же роли: тот же отпечаток
    meta = json.loads((second / "spec.json").read_text("utf-8"))
    meta["spec"]["metrics"].pop()
    (second / "spec.json").write_text(json.dumps(meta, ensure_ascii=False), "utf-8")
    assert datastore.fingerprint() != fp  # другие роли столбцов
    third = tmp_path / "third"
    third.mkdir()
    activate_flat(third, monkeypatch, "support")
    assert datastore.fingerprint() not in (fp, "")  # другие данные


def test_fingerprint_of_the_strategic_demo_is_stable_and_differs_from_flat_reports(tmp_path, monkeypatch):
    demo = datastore.fingerprint()
    assert demo == datastore.fingerprint()
    flat = tmp_path / "flat"
    flat.mkdir()
    activate_flat(flat, monkeypatch, "branches")
    assert datastore.fingerprint() != demo
    monkeypatch.setenv("KPI_DATA_PATH", str(DEMO_XLSX))
    assert datastore.fingerprint() == demo


def test_snapshot_returns_the_table_together_with_its_fingerprint():
    table, fp = datastore.snapshot()
    assert table is datastore.get_table() and fp == datastore.fingerprint() and table.source == "kpi.xlsx"
