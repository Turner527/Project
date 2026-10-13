import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.llm import LLMError
from agent.stub_llm import ScriptedLLM
from agent.tool_adapter import make_adapter
from app.main import MAX_UPLOAD_BYTES, create_app
from data import generate, generate_samples
from logs.store import LogStore
from tests.conftest import DEMO_XLSX
from tests.helpers import FakeLLM, xlsx_book
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
    for fragment in ("AI-аналитик", "Загрузить отчёт", "Полезно", "Не полезно", "Как получен ответ", "lang=\"ru\"",
                     "Подключить GigaChat", "Новый разговор", "Проверьте роли и типы столбцов", "Вернуться к демо-данным",
                     'accept=".xlsx,.csv"'):
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
    assert s["dataset"]["first_month"] == "2025-04" and s["prompt_version"].startswith("v2-")


# ---------- вопрос-ответ ----------


def test_ask_returns_memo_trace_and_logs_request(client, store):
    r = client.post("/api/ask", json={"question": Q, "client_id": "tester"})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "проверен" and data["error"] is None
    assert data["memo"]["diagnosis"]["key_segments"] == ["Молодёжь"]
    assert [c["tool"] for c in data["tool_calls"]] == ["find_metrics", "diagnose_change"]
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
    assert r.json()["mode"] == "ready" and len(r.json()["questions"]) == 4  # формат из трёх листов принимается сразу
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
    assert "не в формате .xlsx или .csv" in upload(client, b"a,b\n1,2\n", "data.txt").json()["detail"]
    assert "слишком мало данных" in upload(client, b"a,b\n1,2\n", "data.csv").json()["errors"][0]
    assert "устарел" in upload(client, b"x", "old.xls").json()["detail"]
    assert "не похож на книгу Excel" in upload(client, "привет".encode(), "kpi.xlsx").json()["detail"]
    broken = b"PK\x03\x04" + b"\x00" * 50  # похоже на zip, но не xlsx
    r = upload(client, broken)
    assert r.status_code == 422 and "Не удалось открыть файл как Excel" in r.json()["errors"][0]
    huge = upload(client, b"PK\x03\x04" + b"0" * (MAX_UPLOAD_BYTES + 10))
    assert huge.status_code == 413 and "больше 15 МБ" in huge.json()["detail"]
    assert not uploads.exists()


# ---------- любые отчёты: загрузка плоской таблицы, проверка ролей столбцов, подтверждение ----------


def sample_bytes(tmp_path, name):
    path = tmp_path / f"{name}.xlsx"
    generate_samples.write_sample(name, path)
    return path.read_bytes()


def confirm(client, upload_id, spec):
    return client.post("/api/datasets/confirm", json={"upload_id": upload_id, "spec": spec})


def test_flat_xlsx_goes_to_review_and_nothing_changes_until_confirmed(client, uploads, tmp_path):
    r = upload(client, sample_bytes(tmp_path, "marketing"), "marketing.xlsx")
    body = r.json()
    assert r.status_code == 200 and body["ok"] and body["mode"] == "review" and len(body["upload_id"]) == 32
    report = body["profile"]
    assert report["filename"] == "marketing.xlsx" and report["grain"] == "day" and report["rows"] == 488
    assert {c["name"]: c["role"] for c in report["columns"]}["Канал"] == "dimension"
    assert [m["name"] for m in report["spec"]["metrics"]][:2] == ["Расходы", "Показы"]
    assert report["ratios"][0]["name"] == "CTR" and report["notes"]
    status = client.get("/api/status").json()["dataset"]
    assert status["is_upload"] is False and status["kind"] == "strategic"  # демо-данные на месте
    assert not uploads.exists() and not (uploads.parent / "active.spec.json").exists()


def test_confirmed_flat_report_becomes_active_and_the_agent_answers_on_it(client, uploads, tmp_path):
    body = upload(client, sample_bytes(tmp_path, "marketing"), "marketing.xlsx").json()
    r = confirm(client, body["upload_id"], body["profile"]["spec"])
    assert r.status_code == 200 and r.json()["ok"]
    dataset = r.json()["dataset"]
    assert dataset["is_upload"] is True and dataset["kind"] == "flat" and dataset["grain"] == "day"
    assert dataset["first_period"] == "2026-08-01" and dataset["dimensions"] == ["Канал", "Кампания"]
    questions = r.json()["questions"]
    assert 1 <= len(questions) <= 4 and any("«" in q for q in questions)  # подсказки строятся из названий в данных
    status = client.get("/api/status").json()
    assert status["dataset"]["kind"] == "flat" and status["questions"] == questions
    ask = client.post("/api/ask", json={"question": "Почему выросли расходы и в каком канале?"}).json()
    assert ask["status"] == "проверен" and ask["error"] is None
    assert ask["memo"]["diagnosis"]["key_segments"] == ["Поиск"]
    assert [c["tool"] for c in ask["tool_calls"]] == ["find_metrics", "diagnose_change"]
    assert "предыдущий период" in ask["memo"]["diagnosis"]["summary"]  # база сравнения названа
    # плоский отчёт сохранён на диск целиком: спецификация и сам файл
    assert (uploads.parent / "active.spec.json").exists() and (uploads.parent / "active.data.xlsx").exists()


def test_user_edits_of_the_spec_are_respected(client, uploads, tmp_path):
    body = upload(client, sample_bytes(tmp_path, "marketing"), "marketing.xlsx").json()
    spec = body["profile"]["spec"]
    spec["dimensions"] = ["Канал"]  # пользователь убрал «Кампанию» из измерений
    spec["metrics"] = [m for m in spec["metrics"] if m["name"] in ("Расходы", "Лиды", "Стоимость лида")]
    spec["metrics"][0]["name"] = "Затраты на рекламу"
    spec["metrics"][2]["num"] = "Расходы, руб"
    r = confirm(client, body["upload_id"], spec)
    assert r.status_code == 200 and r.json()["dataset"]["dimensions"] == ["Канал"] and r.json()["dataset"]["metrics"] == 3
    ask = client.post("/api/ask", json={"question": "Почему выросли затраты на рекламу?"}).json()
    assert ask["status"] == "проверен" and ask["memo"]["diagnosis"]["key_segments"] == ["Поиск"]
    assert ask["tool_calls"][1]["arguments"]["metric"] == "Затраты на рекламу"


def test_confirm_rejects_a_broken_spec_and_keeps_the_previous_report(client, uploads, tmp_path):
    body = upload(client, sample_bytes(tmp_path, "support"), "support.xlsx").json()
    spec = body["profile"]["spec"]
    bad = {**spec, "date": "Нет такого столбца", "metrics": spec["metrics"] + [{"name": "Расходы пропали", "column": "Расходы"}]}
    r = confirm(client, body["upload_id"], bad)
    assert r.status_code == 422 and r.json()["detail"] == "Настройки столбцов не прошли проверку."
    assert any("Столбца «Нет такого столбца» нет в таблице" in e for e in r.json()["errors"])
    assert any("столбца «Расходы» нет в таблице" in e for e in r.json()["errors"])
    assert client.get("/api/status").json()["dataset"]["is_upload"] is False
    wrong_date = {**spec, "date": "Очередь", "dimensions": ["Канал"]}  # текст вместо даты: сообщение с номерами строк
    errors = confirm(client, body["upload_id"], wrong_date).json()["errors"]
    assert "Столбец даты «Очередь»" in errors[0] and "в строках 2" in errors[0]
    assert confirm(client, body["upload_id"], {**spec, "metrics": []}).status_code == 422
    assert confirm(client, body["upload_id"], "не словарь").status_code == 422  # pydantic: spec должен быть объектом


def test_confirm_checks_the_upload_id(client, uploads):
    r = confirm(client, "../../etc/passwd", {})
    assert r.status_code == 422 and "Неверный идентификатор" in r.json()["errors"][0]
    r = confirm(client, "0" * 32, {})
    assert r.status_code == 422 and "не найден или устарел" in r.json()["errors"][0]
    assert client.post("/api/datasets/profile", json={"upload_id": "x" * 32}).status_code == 422


def test_other_sheet_of_an_uploaded_file_can_be_profiled(client, uploads, tmp_path):
    first = [["Дата", "Канал", "Клики"]] + [[f"2026-03-0{d}", "А", d] for d in range(1, 6)]
    second = [["Месяц", "Отдел", "Выручка", "Расходы"]] + [[f"2026-0{m}", o, 10 * m, 5 * m] for m in range(1, 5) for o in ("Х", "Ю")]
    body = upload(client, xlsx_book({"Первый": first, "Второй": second}), "two.xlsx").json()
    assert body["profile"]["sheet"] == "Второй" and sorted(body["profile"]["sheets"]) == ["Второй", "Первый"]
    r = client.post("/api/datasets/profile", json={"upload_id": body["upload_id"], "sheet": "Первый"})
    assert r.status_code == 200 and r.json()["profile"]["sheet"] == "Первый"
    assert r.json()["profile"]["spec"]["date"] == "Дата"
    spec = r.json()["profile"]["spec"]
    assert confirm(client, body["upload_id"], spec).json()["dataset"]["metrics"] == 1  # подтверждён выбранный лист


def test_csv_upload_is_reviewed_and_answered(client, uploads):
    lines = ["Дата;Канал;Расходы;Лиды"]
    for d in range(1, 29):
        for channel, spend, leads in (("Поиск", 100 + d, 10), ("Email", 50, 5)):
            if channel == "Поиск" and d > 14:
                spend, leads = spend * 2, 6
            lines.append(f"{d:02d}.03.2026;{channel};{str(spend).replace('.', ',')};{leads}")
    content = "\n".join(lines).encode("cp1251")
    body = upload(client, content, "выгрузка.csv").json()
    assert body["mode"] == "review" and body["profile"]["kind"] == "csv" and body["profile"]["rows"] == 56
    assert body["profile"]["grain"] == "day"
    assert confirm(client, body["upload_id"], body["profile"]["spec"]).json()["dataset"]["kind"] == "flat"
    assert (uploads.parent / "active.data.csv").exists()
    ask = client.post("/api/ask", json={"question": "Что происходит с расходами?"}).json()
    assert ask["status"] == "проверен" and ask["memo"]["diagnosis"]["key_segments"] == ["Поиск"]


def test_reset_returns_to_demo_data_and_removes_files(client, uploads, tmp_path):
    body = upload(client, sample_bytes(tmp_path, "branches"), "branches.xlsx").json()
    confirm(client, body["upload_id"], body["profile"]["spec"])
    assert client.get("/api/status").json()["dataset"]["kind"] == "flat"
    r = client.delete("/api/upload")
    assert r.status_code == 200 and r.json()["dataset"]["is_upload"] is False and r.json()["dataset"]["kind"] == "strategic"
    assert len(r.json()["questions"]) == 4
    assert not (uploads.parent / "active.spec.json").exists() and not list(uploads.parent.glob("active.data.*"))
    assert client.post("/api/ask", json={"question": Q}).json()["status"] == "проверен"


def test_replacing_one_kind_of_report_with_another_leaves_no_leftovers(client, uploads, tmp_path):
    upload(client, xlsx_bytes())  # стратегический
    assert uploads.exists()
    body = upload(client, sample_bytes(tmp_path, "support"), "support.xlsx").json()
    confirm(client, body["upload_id"], body["profile"]["spec"])
    assert not uploads.exists() and client.get("/api/status").json()["dataset"]["kind"] == "flat"
    upload(client, xlsx_bytes())  # снова стратегический
    assert uploads.exists() and not (uploads.parent / "active.spec.json").exists()
    assert not list(uploads.parent.glob("active.data.*"))
    assert client.get("/api/status").json()["dataset"]["kind"] == "strategic"


def test_flat_upload_errors_are_explained(client, uploads):
    r = upload(client, b"a;b\n1;2\n", "mini.csv")
    assert r.status_code == 422 and "слишком мало данных" in r.json()["errors"][0]
    no_table = upload(client, xlsx_book({"Список": [["Имя"], ["Анна"]]}), "list.xlsx")
    assert no_table.status_code == 422 and "не найдено ни одной таблицы" in no_table.json()["errors"][0]


def test_strategic_workbook_is_detected_by_sheet_names_only(client, uploads, tmp_path):
    """Плоская таблица, в которой случайно есть лист «facts», не должна считаться стратегической."""
    rows = [["Дата", "Канал", "Клики"]] + [[f"2026-03-0{d}", c, d + i] for d in range(1, 6) for i, c in enumerate(("А", "Б"))]
    body = upload(client, xlsx_book({"facts": rows}), "facts.xlsx").json()
    assert body["mode"] == "review"
