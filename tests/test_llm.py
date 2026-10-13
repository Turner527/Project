import ssl
from pathlib import Path

import httpx
import pytest
from gigachat.exceptions import (
    AuthenticationError,
    BadRequestError,
    ForbiddenError,
    RateLimitError,
    ResponseError,
    ServerError,
)
from gigachat.models import Chat, ChatCompletion

from agent.llm import (
    ConfigError,
    GigaChatConfig,
    GigaChatLLM,
    LLMError,
    explain_error,
    make_llm,
)
from agent.stub_llm import ScriptedLLM

URL = "https://example.test/api"


def err(cls, status, body=b""):
    return cls(URL, status, body, httpx.Headers())


@pytest.fixture
def ca_file(tmp_path):
    path = tmp_path / "ca.pem"
    path.write_text("-----BEGIN CERTIFICATE-----\nZmFrZQ==\n-----END CERTIFICATE-----\n")
    return path


@pytest.fixture
def env(monkeypatch, ca_file):
    for name in ("GIGACHAT_CREDENTIALS", "GIGACHAT_SCOPE", "GIGACHAT_CA_BUNDLE", "GIGACHAT_MODEL",
                 "GIGACHAT_VERIFY_SSL_CERTS", "LLM_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "dGVzdDp0ZXN0")
    monkeypatch.setenv("GIGACHAT_CA_BUNDLE", str(ca_file))
    return monkeypatch


# ---------- настройки и SSL ----------


def test_config_defaults_and_reading_from_env(env):
    env.setenv("GIGACHAT_SCOPE", "GIGACHAT_API_CORP")
    env.setenv("GIGACHAT_MODEL", "GigaChat-2-Max")
    cfg = GigaChatConfig.from_env()
    assert (cfg.scope, cfg.model, cfg.credentials) == ("GIGACHAT_API_CORP", "GigaChat-2-Max", "dGVzdDp0ZXN0")
    env.delenv("GIGACHAT_SCOPE")
    env.delenv("GIGACHAT_MODEL")
    assert GigaChatConfig.from_env().scope == "GIGACHAT_API_PERS"


def test_missing_credentials_and_missing_ca_file_are_reported(env):
    env.delenv("GIGACHAT_CREDENTIALS")
    with pytest.raises(ConfigError, match="GIGACHAT_CREDENTIALS"):
        GigaChatLLM()
    env.setenv("GIGACHAT_CREDENTIALS", "x")
    env.setenv("GIGACHAT_CA_BUNDLE", "/no/such/file.pem")
    with pytest.raises(ConfigError, match="не найден"):
        GigaChatLLM()


@pytest.mark.parametrize("value", ["false", "0", "False", "off", "no"])
def test_disabling_ssl_verification_is_refused(env, value):
    env.setenv("GIGACHAT_VERIFY_SSL_CERTS", value)
    with pytest.raises(ConfigError, match="запрещено"):
        GigaChatLLM()


def test_real_sdk_client_always_verifies_ssl_and_uses_ca_bundle(env, ca_file, monkeypatch):
    llm = GigaChatLLM()
    settings = llm._client._settings
    assert settings.verify_ssl_certs is True
    assert settings.ca_bundle_file == str(ca_file) and settings.credentials == "dGVzdDp0ZXN0"
    assert settings.scope == "GIGACHAT_API_PERS" and llm.name == "gigachat:GigaChat-2-Pro"
    assert settings.ssl_context is None
    # даже если бы защита конфигурации пропустила переменную, явный аргумент SDK сильнее окружения
    monkeypatch.setenv("GIGACHAT_VERIFY_SSL_CERTS", "false")
    monkeypatch.setattr(GigaChatConfig, "problems", lambda self: [])
    assert GigaChatLLM()._client._settings.verify_ssl_certs is True


def test_source_code_never_disables_ssl():
    root = Path(__file__).resolve().parent.parent
    for path in list(root.glob("**/*.py")):
        if ".venv" in path.parts or path.name == "test_llm.py":
            continue
        source = path.read_text("utf-8")
        assert "verify_ssl_certs=False" not in source and "verify=False" not in source, path
        assert "CERT_NONE" not in source and "_create_unverified_context" not in source, path


def test_make_llm_selection(env):
    assert isinstance(make_llm("stub"), ScriptedLLM)
    assert isinstance(make_llm("gigachat"), GigaChatLLM)
    assert isinstance(make_llm("auto"), GigaChatLLM)  # ключ задан
    env.delenv("GIGACHAT_CREDENTIALS")
    assert isinstance(make_llm("auto"), ScriptedLLM)
    env.setenv("LLM_BACKEND", "stub")
    assert isinstance(make_llm(), ScriptedLLM)
    with pytest.raises(ConfigError):
        make_llm("openai")


# ---------- обмен с SDK: формат запроса и разбор ответа ----------


class FakeSDKClient:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.payloads = reply, exc, []

    def chat(self, payload):
        self.payloads.append(payload)
        if self.exc:
            raise self.exc
        return ChatCompletion.model_validate(self.reply)


def completion(message, finish="stop"):
    return {
        "choices": [{"message": {"role": "assistant", **message}, "index": 0, "finish_reason": finish}],
        "created": 1,
        "model": "GigaChat-2-Pro",
        "object": "chat.completion",
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


FUNCTIONS = [{"name": "get_deviation", "description": "d",
              "parameters": {"type": "object", "properties": {"kpi": {"type": "string", "description": "k"}},
                             "required": ["kpi"]}}]


def test_request_payload_is_valid_for_the_sdk_and_response_is_parsed(env):
    sdk = FakeSDKClient(completion(
        {"content": "", "function_call": {"name": "get_deviation", "arguments": {"kpi": "NPS"}},
         "functions_state_id": "st-1"}, "function_call"))
    llm = GigaChatLLM(client=sdk)
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "function_call": {"name": "get_deviation", "arguments": {"kpi": "NPS"}},
         "functions_state_id": None},
        {"role": "function", "name": "get_deviation", "content": '{"x": 1}'},
    ]
    resp = llm.chat(messages, FUNCTIONS, function_call="auto", temperature=0.01, max_tokens=500)

    payload = sdk.payloads[0]
    chat = Chat.model_validate(payload)  # SDK принимает именно такой запрос
    assert chat.temperature == 0.01 and chat.max_tokens == 500 and chat.function_call == "auto"
    assert chat.functions[0].name == "get_deviation" and chat.messages[2].function_call.arguments == {"kpi": "NPS"}
    assert getattr(chat.messages[3].role, "value", chat.messages[3].role) == "function" and chat.messages[3].name == "get_deviation"
    assert "functions_state_id" not in payload["messages"][2]  # None не отправляем

    assert resp.function_call.name == "get_deviation" and resp.function_call.arguments == {"kpi": "NPS"}
    assert resp.functions_state_id == "st-1" and resp.finish_reason == "function_call"
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.usage.total_tokens) == (100, 20, 120)


def test_plain_text_answer_and_function_call_none(env):
    sdk = FakeSDKClient(completion({"content": "{\"a\": 1}"}))
    resp = GigaChatLLM(client=sdk).chat([{"role": "user", "content": "q"}], FUNCTIONS, function_call="none")
    assert resp.content == "{\"a\": 1}" and resp.function_call is None
    assert sdk.payloads[0]["function_call"] == "none"
    no_functions = GigaChatLLM(client=sdk)
    no_functions.chat([{"role": "user", "content": "q"}])
    assert "functions" not in sdk.payloads[1] and "function_call" not in sdk.payloads[1]


def test_sdk_errors_become_llm_errors_with_explanation(env):
    llm = GigaChatLLM(client=FakeSDKClient(exc=err(AuthenticationError, 401, b'{"message":"Credentials"}')))
    with pytest.raises(LLMError, match="Ключ авторизации"):
        llm.chat([{"role": "user", "content": "q"}])


# ---------- понятные причины ошибок ----------


@pytest.mark.parametrize(
    "exc, fragment",
    [
        (err(AuthenticationError, 401), "GIGACHAT_CREDENTIALS"),
        (err(BadRequestError, 400, b'{"message":"scope data format is invalid"}'), "GIGACHAT_SCOPE"),
        (err(BadRequestError, 400, b'{"message":"bad"}'), "Запрос отклонён"),
        (err(ResponseError, 402), "пакет токенов"),
        (err(ForbiddenError, 403), "Доступ запрещён"),
        (err(RateLimitError, 429), "лимит"),
        (err(ServerError, 503), "временно недоступен"),
        (ssl.SSLCertVerificationError("certificate verify failed"), "GIGACHAT_CA_BUNDLE"),
        (httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate"),
         "GIGACHAT_CA_BUNDLE"),
        (httpx.ConnectError("Name or service not known"), "Нет связи"),
        (httpx.ConnectTimeout("timed out"), "Нет связи"),
        (ValueError("что-то странное"), "ValueError"),
    ],
)
def test_explain_error(exc, fragment):
    assert fragment in explain_error(exc)


def test_ssl_error_wrapped_in_transport_error_is_still_recognised():
    try:
        try:
            raise ssl.SSLError("tls failure")
        except ssl.SSLError as inner:
            raise httpx.ConnectError("connect failed") from inner
    except httpx.ConnectError as outer:
        assert "SSL" in explain_error(outer)
