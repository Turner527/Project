"""Показ сводки перед первой отправкой и согласие: что уйдёт в модель, пользователь видит и подтверждает заранее."""
import copy
import hashlib
import json
import re

import pytest
from fastapi.testclient import TestClient

from agent.llm import GigaChatConfig, GigaChatLLM, llm_sends_data_out
from agent.prompts import DIGEST_CLOSE, DIGEST_OPEN, catalog_text
from agent.stub_llm import ScriptedLLM
from agent.tool_adapter import make_adapter
from app.consent import KEEP, ConsentGate
from app.main import create_app
from data import generate_samples
from logs.store import LogStore
from tests.conftest import DEMO_XLSX
from tests.helpers import FakeLLM
from tools import dashboard as D
from tools import datastore

DEMO_QUESTION = "Почему мы отстаём по цели роста розничного кредитования и что делать?"
MARKETING_QUESTION = "Почему выросли расходы в канале Поиск и что делать?"
STALE = "Сводка изменилась, откройте её заново"


class SendingStub(ScriptedLLM):
    """Заглушка, которая считается отправляющей данные наружу, и запоминает всё, что ей «отправили»."""

    sends_data_out = True

    def __init__(self):
        self.sent: list[list[dict]] = []

    def chat(self, messages, functions=None, **kwargs):
        self.sent.append(copy.deepcopy(messages))
        return super().chat(messages, functions, **kwargs)


@pytest.fixture
def store(tmp_path):
    return LogStore(tmp_path / "pilot.db")


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    """Загрузки уходят во временную папку; активным отчётом становится «загруженный, если есть»."""
    monkeypatch.delenv("KPI_DATA_PATH", raising=False)
    monkeypatch.setattr(datastore, "DEFAULT_PATH", DEMO_XLSX)
    datastore._cache.clear()
    return datastore.UPLOAD_PATH


def app_client(store, llm) -> TestClient:
    return TestClient(create_app(llm=llm, adapter=make_adapter("direct"), store=store, load_env_file=False))


def sample_bytes(tmp_path, name) -> bytes:
    path = tmp_path / f"{name}.xlsx"
    generate_samples.write_sample(name, path)
    return path.read_bytes()


def load_report(client, content: bytes, name: str, edit=None) -> dict:
    body = client.post("/api/upload", files={"file": (name, content, "application/vnd.ms-excel")}).json()
    spec = body["profile"]["spec"]
    if edit:
        edit(spec)
    r = client.post("/api/datasets/confirm", json={"upload_id": body["upload_id"], "spec": spec})
    assert r.status_code == 200, r.text
    return r.json()


def exclude_campaigns(spec):
    spec["dimensions"] = [d for d in spec["dimensions"] if d != "Кампания"]
    spec["excluded"] = ["Кампания"]


def consent(client) -> dict:
    preview = client.get("/api/preview").json()
    r = client.post("/api/consent", json={"token": preview["token"]})
    assert r.status_code == 200 and r.json() == {"ok": True, "confirmed": True}, r.text
    return preview


def assert_consent_required(r) -> None:
    assert r.status_code == 428
    body = r.json()
    assert body["ok"] is False and body["consent_required"] is True and "GigaChat" in body["detail"]


# ======================================================================================================================
# Кому нужно согласие
# ======================================================================================================================


def test_clients_declare_whether_they_send_data_out(tmp_path):
    assert GigaChatLLM.sends_data_out is True
    assert GigaChatLLM(GigaChatConfig(credentials="k"), client=object()).sends_data_out is True
    assert llm_sends_data_out(GigaChatLLM(GigaChatConfig(credentials="k"), client=object())) is True
    assert llm_sends_data_out(ScriptedLLM()) is False and llm_sends_data_out(FakeLLM([])) is False and llm_sends_data_out(None) is False
    assert llm_sends_data_out(SendingStub()) is True


def test_force_consent_env_makes_any_client_ask(monkeypatch):
    for value in ("1", "true", "YES", " on "):
        monkeypatch.setenv("PRISM_FORCE_CONSENT", value)
        assert llm_sends_data_out(ScriptedLLM()) is True and llm_sends_data_out(FakeLLM([])) is True
        assert llm_sends_data_out(None) is False  # модели нет — отправлять нечего
    for value in ("", "0", "no", "off"):
        monkeypatch.setenv("PRISM_FORCE_CONSENT", value)
        assert llm_sends_data_out(ScriptedLLM()) is False


def test_the_offline_stub_never_needs_consent(store):
    with app_client(store, ScriptedLLM()) as client:
        status = client.get("/api/status").json()
        assert status["sends_data_out"] is False and status["consent_required"] is False and status["offline"] is True
        preview = client.get("/api/preview").json()
        assert preview["required"] is False and preview["confirmed"] is False
        assert client.post("/api/dashboard").status_code == 200
        assert client.post("/api/ask", json={"question": DEMO_QUESTION}).status_code == 200


def test_forced_consent_applies_to_the_stub_and_is_checked_on_every_request(store, monkeypatch):
    with app_client(store, ScriptedLLM()) as client:
        monkeypatch.setenv("PRISM_FORCE_CONSENT", "1")
        assert client.get("/api/status").json()["consent_required"] is True
        assert_consent_required(client.post("/api/ask", json={"question": DEMO_QUESTION}))
        preview = client.get("/api/preview").json()
        assert preview["required"] is True and preview["llm"] == "stub"
        consent(client)
        assert client.get("/api/status").json()["consent_required"] is False
        assert client.post("/api/ask", json={"question": DEMO_QUESTION}).status_code == 200
        monkeypatch.delenv("PRISM_FORCE_CONSENT")
        assert client.get("/api/status").json()["consent_required"] is False


def test_connected_gigachat_asks_before_the_first_send(tmp_path, monkeypatch):
    """Настоящий клиент GigaChat (с подставным SDK): до согласия сеть не трогается вовсе."""
    for name in ("GIGACHAT_CREDENTIALS", "GIGACHAT_SCOPE", "GIGACHAT_CA_BUNDLE", "GIGACHAT_MODEL"):
        monkeypatch.delenv(name, raising=False)

    class NoNetwork:
        def chat(self, payload):
            raise AssertionError("до согласия в GigaChat ничего отправлять нельзя")

    llm = GigaChatLLM(GigaChatConfig(credentials="k", model="GigaChat-2"), client=NoNetwork())
    with app_client(LogStore(tmp_path / "l.db"), llm) as client:
        status = client.get("/api/status").json()
        assert status["sends_data_out"] is True and status["consent_required"] is True and status["offline"] is False
        assert_consent_required(client.post("/api/dashboard"))
        assert_consent_required(client.post("/api/ask", json={"question": DEMO_QUESTION}))
        assert client.get("/api/preview").json()["required"] is True


# ======================================================================================================================
# Поток: отказ, предпросмотр, согласие, отправка
# ======================================================================================================================


def test_nothing_is_sent_and_nothing_is_saved_before_consent(store):
    llm = SendingStub()
    with app_client(store, llm) as client:
        assert_consent_required(client.post("/api/dashboard"))
        assert_consent_required(client.post("/api/ask", json={"question": DEMO_QUESTION}))
        assert llm.sent == []
        assert client.get("/api/dashboard").json() == {"dashboard": None}
        assert client.get("/api/status").json()["consent_required"] is True


def test_preview_describes_exactly_what_will_be_sent(store):
    with app_client(store, SendingStub()) as client:
        p = client.get("/api/preview").json()
    assert {"required", "confirmed", "digest_text", "digest_pretty", "catalog", "stats", "excluded_columns", "file_name_sent_as",
            "token"} <= set(p)
    assert p["required"] is True and p["confirmed"] is False and p["excluded_columns"] == []
    digest = json.loads(p["digest_text"])
    assert p["digest_text"] == json.dumps(digest, ensure_ascii=False, separators=(",", ":"))  # компактный текст, как уходит в модель
    assert json.loads(p["digest_pretty"]) == digest and "\n" in p["digest_pretty"] and "\n" not in p["digest_text"]
    assert p["catalog"] == catalog_text(datastore.get_table())
    assert p["stats"] == {"chars": len(p["digest_text"]), "metrics": len(digest["metrics"]), "dimensions": len(digest["dimensions"])}
    assert p["token"] == hashlib.sha256((p["digest_text"] + p["catalog"]).encode("utf-8")).hexdigest()
    assert p["file_name_sent_as"] == "отчёт.xlsx" and digest["report"]["source"] == "отчёт.xlsx"
    assert "kpi.xlsx" not in p["digest_text"] + p["catalog"]
    # простыми словами: что уйдёт позже и что не уйдёт никогда
    about = " ".join(p["about"])
    assert "текст вопроса" in about and "результаты расчётов" in about and "названия показателей и значений разрезов" in about
    assert "Строки таблицы и заметки из файла не отправляются никогда" in about and "«отчёт.xlsx»" in about
    assert "стратегическом формате" in about and "нельзя" in about  # у демо-витрины столбцы не помечаются
    assert p["report_kind"] == "strategic"


def test_the_summary_is_stable_between_openings(store):
    with app_client(store, SendingStub()) as client:
        first, second = client.get("/api/preview").json(), client.get("/api/preview").json()
    assert first["token"] == second["token"] and first["digest_text"] == second["digest_text"]


def test_consent_requires_the_current_token(store):
    llm = SendingStub()
    with app_client(store, llm) as client:
        preview = client.get("/api/preview").json()
        for token in ("0" * 64, "чужой-токен", preview["token"][:-1] + ("0" if preview["token"][-1] != "0" else "1")):
            r = client.post("/api/consent", json={"token": token})
            assert r.status_code == 409 and r.json() == {"ok": False, "detail": STALE}
        assert client.post("/api/consent", json={}).status_code == 422
        assert client.post("/api/consent", json={"token": ""}).status_code == 422
        assert_consent_required(client.post("/api/dashboard"))
        assert llm.sent == []


def test_consent_without_a_preview_is_rejected(store):
    """Подтвердить можно только то, что показано: токен без предпросмотра не принимается."""
    with app_client(store, SendingStub()) as client:
        assert client.post("/api/consent", json={"token": "a" * 64}).status_code == 409


def test_after_consent_the_dashboard_sends_exactly_the_previewed_text(store):
    llm = SendingStub()
    with app_client(store, llm) as client:
        preview = consent(client)
        status = client.get("/api/status").json()
        assert status["consent_required"] is False and client.get("/api/preview").json()["confirmed"] is True
        r = client.post("/api/dashboard")
        assert r.status_code == 200 and r.json()["plan_source"] == "model"
        user = next(m["content"] for m in llm.sent[0] if m["role"] == "user")
        assert user.split(DIGEST_OPEN, 1)[1].split(DIGEST_CLOSE, 1)[0].strip() == preview["digest_text"]
        assert r.json()["sent_to_model"]["digest"] == json.loads(preview["digest_text"])
        assert r.json()["sent_to_model"]["chars"] == len(preview["digest_text"])
        assert client.post("/api/ask", json={"question": DEMO_QUESTION}).status_code == 200
        system = next(m["content"] for m in llm.sent[-1] if m["role"] == "system")
        assert preview["catalog"] in system  # каталог в промпте чата — тот, что был в предпросмотре


def test_the_confirmed_summary_is_reused_even_if_a_rebuild_would_differ(store, monkeypatch):
    """Сборка сводки ограничена по времени и может дать другой результат: дашборд строится по подтверждённой, а не по новой."""
    built = []
    real = D.build_digest

    def unstable(table, **kwargs):
        digest = real(table, **kwargs)
        digest["report"]["build_no"] = len(built)
        built.append(1)
        return digest

    monkeypatch.setattr(D, "build_digest", unstable)
    llm = SendingStub()
    with app_client(store, llm) as client:
        preview = consent(client)
        assert '"build_no":0' in preview["digest_text"] and len(built) == 1
        for _ in range(2):  # и построение, и «Построить заново»
            assert client.post("/api/dashboard").status_code == 200
        assert len(built) == 1
        for messages in llm.sent:
            user = next(m["content"] for m in messages if m["role"] == "user")
            assert preview["digest_text"] in user and '"build_no":1' not in user


def test_a_digest_that_would_change_asks_again(store, monkeypatch):
    """Если сводка пересобрана и отличается, старое подтверждение не действует (токен считается по тексту)."""
    gate = ConsentGate()
    table, fingerprint = datastore.snapshot()
    preview = gate.preview(table, fingerprint, required=True, kind="strategic", llm_name="x")
    assert gate.consent(table, fingerprint, preview["token"]) and gate.is_confirmed(table, fingerprint)
    gate._digests[fingerprint]["report"]["extra"] = 1  # сводка под тем же отпечатком изменилась
    assert not gate.is_confirmed(table, fingerprint) and gate.approved_digest(table, fingerprint) is None
    assert not gate.consent(table, fingerprint, preview["token"])


def test_the_gate_forgets_old_reports_and_stays_safe(monkeypatch):
    monkeypatch.setattr(D, "build_digest", lambda table, **kw: {"report": {}, "metrics": [], "dimensions": []})
    gate = ConsentGate()
    table = datastore.get_table()
    for i in range(KEEP + 2):
        preview = gate.preview(table, f"fp{i}", required=True, kind="flat", llm_name=None)
        assert gate.consent(table, f"fp{i}", preview["token"])
    assert not gate.is_confirmed(table, "fp0")  # вытеснен: спросим снова, а не отправим молча
    assert gate.is_confirmed(table, f"fp{KEEP + 1}")
    assert len(gate._digests) <= KEEP and len(gate._confirmed) <= KEEP


# ======================================================================================================================
# Когда согласие перестаёт действовать
# ======================================================================================================================


def test_consent_is_lost_when_the_exclusions_change(uploads, tmp_path, store):
    data = sample_bytes(tmp_path, "marketing")
    llm = SendingStub()
    with app_client(store, llm) as client:
        load_report(client, data, "marketing.xlsx")
        first = consent(client)
        assert client.post("/api/dashboard").status_code == 200 and "Кампания" in first["digest_text"]
        sent = len(llm.sent)
        # тот же файл и те же роли: отчёт тот же, подтверждение действует
        load_report(client, data, "marketing.xlsx")
        assert client.get("/api/status").json()["consent_required"] is False
        # тот же файл, но один из столбцов помечен «Не отправлять»: другая сводка и другое подтверждение
        load_report(client, data, "marketing.xlsx", edit=exclude_campaigns)
        assert client.get("/api/status").json()["consent_required"] is True
        assert_consent_required(client.post("/api/dashboard"))
        assert_consent_required(client.post("/api/ask", json={"question": MARKETING_QUESTION}))
        assert len(llm.sent) == sent
        second = client.get("/api/preview").json()
        assert second["token"] != first["token"] and second["excluded_columns"] == ["Кампания"]
        assert "Кампания" not in second["digest_text"] + second["catalog"]
        assert "Столбцы с пометкой" in " ".join(second["about"])
        consent(client)
        assert client.post("/api/dashboard").status_code == 200 and client.post("/api/ask", json={"question": MARKETING_QUESTION}).status_code == 200
        # возврат к прежним ролям: первое подтверждение ещё помнится
        load_report(client, data, "marketing.xlsx")
        assert client.get("/api/status").json()["consent_required"] is False


def test_consent_is_lost_when_another_report_is_loaded(uploads, tmp_path, store):
    with app_client(store, SendingStub()) as client:
        load_report(client, sample_bytes(tmp_path, "marketing"), "marketing.xlsx")
        stale = consent(client)
        load_report(client, sample_bytes(tmp_path, "support"), "support.xlsx")
        assert client.get("/api/status").json()["consent_required"] is True
        assert_consent_required(client.post("/api/dashboard"))
        # токен прежнего отчёта к новому не подходит, даже если предпросмотр нового ещё не открывали
        assert client.post("/api/consent", json={"token": stale["token"]}).status_code == 409
        fresh = consent(client)
        assert fresh["token"] != stale["token"] and "Очередь" in fresh["digest_text"]
        assert client.post("/api/dashboard").status_code == 200


def test_consent_is_lost_when_the_demo_is_restored(uploads, tmp_path, store):
    with app_client(store, SendingStub()) as client:
        load_report(client, sample_bytes(tmp_path, "marketing"), "marketing.xlsx")
        consent(client)
        client.delete("/api/upload")  # вернулись к демо-витрине: это другой отчёт
        assert client.get("/api/status").json()["consent_required"] is True
        assert client.get("/api/preview").json()["report_kind"] == "strategic"


def test_consent_lives_only_while_the_server_runs(store):
    llm = SendingStub()
    with app_client(store, llm) as client:
        consent(client)
        assert client.get("/api/status").json()["consent_required"] is False
    with app_client(store, llm) as restarted:
        assert restarted.get("/api/status").json()["consent_required"] is True
        assert_consent_required(restarted.post("/api/ask", json={"question": DEMO_QUESTION}))


def test_excluded_columns_are_listed_locally_but_absent_from_what_is_sent(uploads, tmp_path, store):
    llm = SendingStub()
    with app_client(store, llm) as client:
        load_report(client, sample_bytes(tmp_path, "marketing"), "marketing.xlsx", edit=exclude_campaigns)
        preview = consent(client)
        assert preview["excluded_columns"] == ["Кампания"] and preview["file_name_sent_as"] == "отчёт.xlsx"
        assert client.post("/api/dashboard").status_code == 200
        assert client.post("/api/ask", json={"question": MARKETING_QUESTION}).status_code == 200
    everything = json.dumps(llm.sent, ensure_ascii=False)
    assert "Кампания" not in everything and "marketing" not in everything
    user = next(m["content"] for m in llm.sent[0] if m["role"] == "user")
    assert preview["digest_text"] in user  # и дашборд ушёл именно с тем текстом, который был показан


def test_preview_dialog_is_wired_into_the_page():
    html = (DEMO_XLSX.parent.parent / "app" / "static" / "index.html").read_text("utf-8")
    for fragment in ("Что уйдёт в GigaChat", "/api/preview", "/api/consent", "res.status === 428", "consent_required",
                     "Полная сводка (JSON)", "Каталог отчёта в запросах чата", "Отправить", "Отмена", "не отправляется"):
        assert fragment in html, fragment
    assert re.search(r"el\(\"pre\", \{\}, p\.digest_pretty\)", html)  # сводка показывается текстом, а не разметкой
    assert "prefers-color-scheme" not in html and "https://" not in html
