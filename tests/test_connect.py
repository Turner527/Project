import os
from pathlib import Path

import certifi
import pytest
from fastapi.testclient import TestClient

from agent.llm import LLMError
from agent.stub_llm import ScriptedLLM
from agent.tool_adapter import make_adapter
from app.connect import write_env
from app.main import create_app
from logs.store import LogStore

KEY = "dGVzdC1rZXktZm9yLXRlc3RzLW9ubHktMTIzNDU2Nzg5MA=="


@pytest.fixture
def setup(tmp_path, monkeypatch):
    for name in ("GIGACHAT_CREDENTIALS", "GIGACHAT_SCOPE", "GIGACHAT_CA_BUNDLE", "GIGACHAT_MODEL", "GIGACHAT_VERIFY_SSL_CERTS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_BACKEND", "stub")
    env_file, certs = tmp_path / ".env", tmp_path / "certs"
    calls = []

    def make(verify):
        app = create_app(llm=ScriptedLLM(), adapter=make_adapter("direct"), store=LogStore(tmp_path / "l.db"),
                         load_env_file=False, env_path=env_file, certs_dir=certs, verify=verify)
        return TestClient(app)

    return make, env_file, certs, calls


def post(client, key=KEY, cert=True, **extra):
    files = {"ca_file": ("ca.pem", open(certifi.where(), "rb").read())} if cert else None
    return client.post("/api/connect", data={"credentials": key, "scope": "GIGACHAT_API_PERS", **extra}, files=files)


def test_successful_connect_saves_env_and_switches_model(setup, monkeypatch):
    make, env_file, certs, _ = setup
    env_file.write_text("# мои настройки\nPORT=9000\nGIGACHAT_MODEL=old\n")
    client = make(lambda llm: None)
    assert client.get("/api/status").json()["offline"] is True
    r = post(client, model="GigaChat-2-Max")
    assert r.status_code == 200 and r.json() == {"ok": True, "llm": "gigachat:GigaChat-2-Max"}
    status = client.get("/api/status").json()
    assert status["offline"] is False and status["llm"] == "gigachat:GigaChat-2-Max" and KEY not in str(status)
    text = env_file.read_text()
    assert f"GIGACHAT_CREDENTIALS={KEY}" in text and "GIGACHAT_MODEL=GigaChat-2-Max" in text
    assert "# мои настройки" in text and "PORT=9000" in text and "old" not in text  # прочее сохранено
    assert (certs / "russian_trusted_root_ca.pem").exists() and "GIGACHAT_CA_BUNDLE=" in text
    assert oct(env_file.stat().st_mode & 0o777) == "0o600"
    assert os.environ["GIGACHAT_CREDENTIALS"] == KEY
    for name in ("GIGACHAT_CREDENTIALS", "GIGACHAT_SCOPE", "GIGACHAT_CA_BUNDLE", "GIGACHAT_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_bad_key_returns_explanation_and_saves_nothing(setup):
    make, env_file, certs, _ = setup

    def reject(llm):
        raise LLMError("Ключ авторизации отклонён (401).")

    client = make(reject)
    r = post(client)
    assert r.status_code == 422 and "401" in r.json()["detail"]
    assert not env_file.exists() and not list(certs.glob("*")) and client.get("/api/status").json()["offline"] is True


def test_invalid_input_is_rejected(setup):
    make, env_file, certs, _ = setup
    client = make(lambda llm: None)
    assert "пробелы" in post(client, key="два слова").json()["detail"]
    bad = client.post("/api/connect", data={"credentials": KEY}, files={"ca_file": ("ca.pem", b"not a cert")})
    assert bad.status_code == 422 and "PEM" in bad.json()["detail"]
    assert post(client, cert=False).status_code == 200  # сертификат необязателен (может быть задан иначе)
    assert not list(certs.glob("*.tmp"))


def test_connect_is_refused_from_non_local_address(setup):
    make, *_ = setup
    client = make(lambda llm: None)
    remote = TestClient(client.app, client=("203.0.113.5", 5000))
    assert remote.post("/api/connect", data={"credentials": KEY}).status_code == 403


def test_write_env_updates_in_place(tmp_path):
    path = tmp_path / ".env"
    path.write_text("A=1\n# B=commented\nB=2\n")
    write_env(path, {"B": "new", "C": "3"})
    assert path.read_text() == "A=1\n# B=commented\nB=new\nC=3\n"
    with pytest.raises(ValueError):
        write_env(path, {"X": "a\nb"})


def test_client_uses_the_saved_certificate_not_the_deleted_temp_file(tmp_path, monkeypatch):
    """Регрессия: после подключения клиент SDK ссылался на удалённый .tmp и падал с FileNotFoundError."""
    from app import connect as connect_mod

    for name in ("GIGACHAT_CREDENTIALS", "GIGACHAT_CA_BUNDLE", "GIGACHAT_VERIFY_SSL_CERTS"):
        monkeypatch.delenv(name, raising=False)
    certs = tmp_path / "certs"
    llm = connect_mod.connect(KEY, "GIGACHAT_API_PERS", "", open(certifi.where(), "rb").read(),
                              tmp_path / ".env", certs, verify=lambda l: None)
    used = Path(llm._client._settings.ca_bundle_file)
    assert used == certs / "russian_trusted_root_ca.pem" and used.exists()
    assert not list(certs.glob("*.tmp"))
    for name in ("GIGACHAT_CREDENTIALS", "GIGACHAT_SCOPE", "GIGACHAT_CA_BUNDLE", "GIGACHAT_MODEL"):
        monkeypatch.delenv(name, raising=False)


# ---------- выбор модели ----------


class FakeSDK:
    def __init__(self, ids=("GigaChat-2", "GigaChat-2-Pro", "GigaChat-3-Ultra", "EmbeddingsGigaR"), fail=False):
        self.ids, self.fail = ids, fail

    def get_models(self):
        if self.fail:
            raise OSError("нет сети")
        from types import SimpleNamespace
        return SimpleNamespace(data=[SimpleNamespace(id_=i) for i in self.ids])


def gigachat_client(tmp_path, monkeypatch, sdk=None, model="GigaChat-2-Pro"):
    from agent.llm import GigaChatConfig, GigaChatLLM

    for name in ("GIGACHAT_CREDENTIALS", "GIGACHAT_SCOPE", "GIGACHAT_CA_BUNDLE", "GIGACHAT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    llm = GigaChatLLM(GigaChatConfig(credentials=KEY, model=model), client=sdk or FakeSDK())
    app = create_app(llm=llm, adapter=make_adapter("direct"), store=LogStore(tmp_path / "l.db"), load_env_file=False,
                     env_path=env_file, certs_dir=tmp_path / "certs")
    return TestClient(app), env_file


def test_models_come_from_the_account_without_embeddings(tmp_path, monkeypatch):
    client, _ = gigachat_client(tmp_path, monkeypatch)
    body = client.get("/api/models").json()
    assert body == {"current": "GigaChat-2-Pro", "models": ["GigaChat-2", "GigaChat-2-Pro", "GigaChat-3-Ultra"],
                    "source": "api", "switchable": True}


def test_models_fall_back_to_the_known_list_when_the_account_does_not_answer(tmp_path, monkeypatch):
    client, _ = gigachat_client(tmp_path, monkeypatch, FakeSDK(fail=True), model="My-model")
    body = client.get("/api/models").json()
    assert body["source"] == "known" and body["models"][0] == "My-model"
    assert "GigaChat-3-Ultra" in body["models"] and body["current"] == "My-model"


def test_offline_stub_lists_known_models_but_cannot_switch(setup):
    make, env_file, *_ = setup
    client = make(lambda llm: None)
    body = client.get("/api/models").json()
    assert body["current"] is None and body["switchable"] is False and "GigaChat-3-Ultra" in body["models"]
    r = client.post("/api/model", json={"model": "GigaChat-3-Ultra"})
    assert r.status_code == 409 and "Сначала подключите GigaChat" in r.json()["detail"] and not env_file.exists()


def test_switching_the_model_saves_the_choice_and_keeps_the_connection(tmp_path, monkeypatch):
    client, env_file = gigachat_client(tmp_path, monkeypatch)
    env_file.write_text("# мои настройки\nGIGACHAT_MODEL=GigaChat-2-Pro\nPORT=9000\n")
    r = client.post("/api/model", json={"model": "GigaChat-3-Ultra"})
    assert r.status_code == 200 and r.json() == {"ok": True, "llm": "gigachat:GigaChat-3-Ultra"}
    assert client.get("/api/status").json()["llm"] == "gigachat:GigaChat-3-Ultra"
    assert client.get("/api/models").json()["current"] == "GigaChat-3-Ultra"
    text = env_file.read_text()
    assert "GIGACHAT_MODEL=GigaChat-3-Ultra" in text and "# мои настройки" in text and "PORT=9000" in text
    assert os.environ["GIGACHAT_MODEL"] == "GigaChat-3-Ultra"
    monkeypatch.delenv("GIGACHAT_MODEL")


def test_a_model_outside_the_account_list_or_with_a_bad_name_is_rejected(tmp_path, monkeypatch):
    client, env_file = gigachat_client(tmp_path, monkeypatch)
    r = client.post("/api/model", json={"model": "GigaChat-9-Mega"})
    assert r.status_code == 422 and "нет в списке вашего аккаунта" in r.json()["detail"] and "GigaChat-3-Ultra" in r.json()["detail"]
    r = client.post("/api/model", json={"model": "плохо; rm -rf"})
    assert r.status_code == 422 and "латинские буквы" in r.json()["detail"]
    assert client.post("/api/model", json={"model": ""}).status_code == 422
    assert not env_file.exists() and client.get("/api/status").json()["llm"] == "gigachat:GigaChat-2-Pro"


def test_any_well_formed_name_is_accepted_when_the_account_list_is_unavailable(tmp_path, monkeypatch):
    client, _ = gigachat_client(tmp_path, monkeypatch, FakeSDK(fail=True))
    assert client.post("/api/model", json={"model": "GigaChat-Next"}).status_code == 200
    monkeypatch.delenv("GIGACHAT_MODEL", raising=False)


def test_switching_the_model_is_local_only(tmp_path, monkeypatch):
    client, _ = gigachat_client(tmp_path, monkeypatch)
    remote = TestClient(client.app, client=("203.0.113.5", 5000))
    assert remote.post("/api/model", json={"model": "GigaChat-3-Ultra"}).status_code == 403


# ---------- запись .env и параллельные запросы ----------


def test_write_env_is_atomic_and_keeps_private_permissions(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("GIGACHAT_CREDENTIALS=secret\n")
    write_env(path, {"GIGACHAT_MODEL": "GigaChat-2"})
    assert oct(path.stat().st_mode & 0o777) == "0o600" and not list(tmp_path.glob(".env.*.tmp"))

    def boom(src, dst):
        raise OSError("диск переполнен")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        write_env(path, {"GIGACHAT_MODEL": "GigaChat-3-Ultra"})
    assert path.read_text() == "GIGACHAT_CREDENTIALS=secret\nGIGACHAT_MODEL=GigaChat-2\n"  # прежний файл цел
    assert not list(tmp_path.glob(".env.*.tmp"))


def test_parallel_writers_do_not_lose_each_others_lines(tmp_path):
    import threading

    path = tmp_path / ".env"
    path.write_text("GIGACHAT_CREDENTIALS=secret\n")
    threads = [threading.Thread(target=write_env, args=(path, {f"KEY_{i}": str(i)})) for i in range(24)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    lines = path.read_text().splitlines()
    assert lines[0] == "GIGACHAT_CREDENTIALS=secret" and sorted(lines[1:]) == sorted(f"KEY_{i}={i}" for i in range(24))


def test_a_failed_save_to_env_is_a_json_error_and_keeps_the_previous_model(tmp_path, monkeypatch):
    client, env_file = gigachat_client(tmp_path, monkeypatch)
    monkeypatch.setattr(os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("диск только для чтения")))
    r = client.post("/api/model", json={"model": "GigaChat-3-Ultra"})
    assert r.status_code == 500 and r.json()["ok"] is False and "Не удалось сохранить" in r.json()["detail"]
    assert client.get("/api/status").json()["llm"] == "gigachat:GigaChat-2-Pro" and not env_file.exists()
    monkeypatch.delenv("GIGACHAT_MODEL", raising=False)


def test_a_broken_gigachat_setup_is_named_in_the_409(tmp_path, monkeypatch):
    for name in ("GIGACHAT_CREDENTIALS", "GIGACHAT_SCOPE", "GIGACHAT_CA_BUNDLE", "GIGACHAT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_BACKEND", "gigachat")
    app = create_app(llm=None, adapter=make_adapter("direct"), store=LogStore(tmp_path / "l.db"), load_env_file=False,
                     env_path=tmp_path / ".env", certs_dir=tmp_path / "certs")
    client = TestClient(app)
    error = client.get("/api/status").json()["llm_error"]
    assert error
    r = client.post("/api/model", json={"model": "GigaChat-3-Ultra"})
    assert r.status_code == 409 and r.json()["detail"] == f"GigaChat не настроен: {error}"


def test_model_switch_waits_for_a_connection_change_in_progress(tmp_path, monkeypatch):
    import threading

    from app import connect as connect_mod

    client, _ = gigachat_client(tmp_path, monkeypatch)
    result = {}
    done = threading.Event()

    def switch():
        result["r"] = client.post("/api/model", json={"model": "GigaChat-3-Ultra"})
        done.set()

    with connect_mod.ENV_LOCK:  # идёт /api/connect: смена модели ждёт его окончания
        worker = threading.Thread(target=switch)
        worker.start()
        assert not done.wait(0.4)
    assert done.wait(5) and result["r"].status_code == 200
    worker.join()
    monkeypatch.delenv("GIGACHAT_MODEL", raising=False)
