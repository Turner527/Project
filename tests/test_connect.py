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
