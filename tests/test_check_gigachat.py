import socket
import ssl
from types import SimpleNamespace

import certifi
import httpx
import pytest
from gigachat.exceptions import AuthenticationError, RateLimitError

import check_gigachat as cg
from agent.llm import FunctionCall, GigaChatConfig, LLMError, LLMResponse, Usage
from agent.prompts import DIGEST_OPEN
from agent.stub_llm import ScriptedLLM


@pytest.fixture
def env(monkeypatch, tmp_path):
    for name in ("GIGACHAT_CREDENTIALS", "GIGACHAT_SCOPE", "GIGACHAT_CA_BUNDLE", "GIGACHAT_MODEL",
                 "GIGACHAT_VERIFY_SSL_CERTS", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "dGVzdDp0ZXN0LWtleS1mb3ItdGVzdHMtb25seS0xMjM0NTY3ODk=")
    monkeypatch.setenv("GIGACHAT_CA_BUNDLE", certifi.where())  # настоящий PEM со списком УЦ
    monkeypatch.setattr(cg, "tcp_connect", lambda *a, **k: None)
    monkeypatch.setattr(cg, "tls_handshake", lambda *a, **k: None)
    return monkeypatch


class FakeSDK:
    def __init__(self, token_exc=None, models=("GigaChat-2-Pro",)):
        self.token_exc, self.models = token_exc, models

    def get_token(self):
        if self.token_exc:
            raise self.token_exc
        return SimpleNamespace(expires_at=1_900_000_000_000)

    def get_models(self):
        return SimpleNamespace(data=[SimpleNamespace(id_=m) for m in self.models])


class FakeLLM:
    def __init__(self, sdk=None, chat_exc=None, calls_function=True, dashboard=False):
        self.config = GigaChatConfig.from_env()
        self._client = sdk or FakeSDK()
        self.chat_exc, self.calls_function, self.dashboard = chat_exc, calls_function, dashboard
        self.name = "fake"

    def chat(self, messages, functions=None, **kwargs):
        if self.chat_exc:
            raise self.chat_exc
        if self.dashboard and DIGEST_OPEN in messages[-1]["content"]:  # запрос плана дашборда: отвечает как заглушка
            return ScriptedLLM().chat(messages, None)
        if functions:
            call = FunctionCall("get_change", {"metric": "NPS"}) if self.calls_function else None
            return LLMResponse(content="" if call else "Отклонение небольшое", function_call=call, usage=Usage(5, 5, 10))
        return LLMResponse(content="ок", usage=Usage(5, 1, 6))


def run(env, llm=None, **kwargs):
    env.setattr(cg, "make_client", lambda cfg: llm or FakeLLM())
    lines = []
    results = cg.run(out=lines.append, **kwargs)
    return results, "\n".join(lines)


def statuses(results):
    return {r.title: r.status for r in results}


# ---------- настройки ----------


def test_happy_path_passes_every_step(env):
    results, text = run(env)
    assert all(r.status in (cg.OK, cg.WARN) for r in results), text
    s = statuses(results)
    assert s["Авторизация (получение токена)"] == cg.OK and s["Вызов функций"] == cg.OK
    assert s["Тестовый запрос"] == cg.OK and s["Модель"] == cg.OK
    assert not any(r.status == cg.FAIL for r in results)


def test_missing_credentials_stops_before_network(env):
    env.delenv("GIGACHAT_CREDENTIALS")
    env.setattr(cg, "tcp_connect", lambda *a, **k: pytest.fail("сеть не должна проверяться"))
    results, text = run(env)
    assert statuses(results)["Ключ авторизации"] == cg.FAIL and "GIGACHAT_CREDENTIALS" in text
    assert statuses(results)["Проверки сети"] == cg.SKIP


def test_disabled_ssl_is_a_failure_with_explanation(env):
    env.setenv("GIGACHAT_VERIFY_SSL_CERTS", "false")
    results, text = run(env)
    assert statuses(results)["Проверка SSL"] == cg.FAIL and "GIGACHAT_CA_BUNDLE" in text


def test_suspicious_credentials_and_scope_are_warnings(env):
    env.setenv("GIGACHAT_CREDENTIALS", "client_id:client_secret")
    env.setenv("GIGACHAT_SCOPE", "SOMETHING")
    s = statuses(run(env)[0])
    assert s["Ключ авторизации"] == cg.WARN and s["Scope"] == cg.WARN


def test_certificate_file_problems(env, tmp_path):
    env.setenv("GIGACHAT_CA_BUNDLE", str(tmp_path / "нет.pem"))
    assert statuses(run(env)[0])["Сертификат Минцифры"] == cg.FAIL
    garbage = tmp_path / "bad.pem"
    garbage.write_text("это не сертификат")
    env.setenv("GIGACHAT_CA_BUNDLE", str(garbage))
    results, text = run(env)
    assert statuses(results)["Сертификат Минцифры"] == cg.FAIL and "openssl x509 -inform DER" in text


def test_unset_ca_bundle_is_only_a_warning(env):
    env.delenv("GIGACHAT_CA_BUNDLE")
    results, _ = run(env)
    assert statuses(results)["Сертификат Минцифры"] == cg.WARN


def test_offline_mode_checks_only_configuration(env):
    env.setattr(cg, "make_client", lambda cfg: pytest.fail("клиент не нужен"))
    env.setattr(cg, "tcp_connect", lambda *a, **k: pytest.fail("сеть не нужна"))
    results = cg.run(offline=True, out=lambda _: None)
    assert {r.title for r in results} == {"Файл .env", "Проверка SSL", "Ключ авторизации", "Scope", "Сертификат Минцифры"}


# ---------- сеть и сертификат ----------


def test_dns_failure_is_reported_as_network(env):
    def boom(*a, **k):
        raise socket.gaierror(-2, "Name or service not known")

    env.setattr(cg, "tcp_connect", boom)
    results, text = run(env)
    net = [r for r in results if r.title.startswith("Сеть:")]
    assert net and all(r.status == cg.FAIL and "DNS" in r.fix for r in net)
    assert statuses(results)["Авторизация, модель, вызов функций"] == cg.SKIP


def test_timeout_points_to_firewall(env):
    def boom(*a, **k):
        raise TimeoutError()

    env.setattr(cg, "tcp_connect", boom)
    _, text = run(env)
    assert "файрволом" in text and "9443" in text


def test_certificate_error_names_the_ca_bundle(env):
    def boom(*a, **k):
        raise ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate")

    env.setattr(cg, "tls_handshake", boom)
    results, text = run(env)
    tls = [r for r in results if r.title.startswith("TLS:")]
    assert tls and all(r.status == cg.FAIL for r in tls)
    assert "GIGACHAT_CA_BUNDLE" in text and "отключать нельзя" in text and "sub_ca" in text
    assert statuses(results)["Авторизация, модель, вызов функций"] == cg.SKIP


def test_proxy_skips_direct_socket_checks(env):
    env.setenv("HTTPS_PROXY", "http://user:secret@proxy.corp:3128")
    env.setattr(cg, "tcp_connect", lambda *a, **k: pytest.fail("мимо прокси ходить нельзя"))
    results, text = run(env)
    assert "proxy.corp" in text and "secret" not in text  # пароль из адреса не печатается
    assert statuses(results)["Авторизация (получение токена)"] == cg.OK


# ---------- ошибки API ----------


def api_error(cls, status, body=b""):
    return cls("https://x.test", status, body, httpx.Headers())


@pytest.mark.parametrize(
    "exc, fragment",
    [
        (api_error(AuthenticationError, 401), "Ключ авторизации отклонён"),
        (api_error(RateLimitError, 429), "лимит"),
        (httpx.ConnectError("Connection refused"), "Нет связи"),
        (ssl.SSLError("tls"), "GIGACHAT_CA_BUNDLE"),
    ],
)
def test_auth_errors_are_explained_and_later_steps_skipped(env, exc, fragment):
    results, text = run(env, FakeLLM(FakeSDK(token_exc=exc)))
    assert statuses(results)["Авторизация (получение токена)"] == cg.FAIL and fragment in text
    assert statuses(results)["Модель, тестовый запрос, вызов функций"] == cg.SKIP


def test_unknown_model_is_reported_with_available_list(env):
    env.setenv("GIGACHAT_MODEL", "GigaChat-Fake")
    results, text = run(env, FakeLLM(FakeSDK(models=("GigaChat-2-Pro", "GigaChat-2-Max"))))
    assert statuses(results)["Модель"] == cg.FAIL and "GigaChat-2-Max" in text


def test_chat_failure_is_explained(env):
    results, text = run(env, FakeLLM(chat_exc=LLMError("Закончился пакет токенов или нужна оплата (402).")))
    assert statuses(results)["Тестовый запрос"] == cg.FAIL and "402" in text


def test_model_that_ignores_functions_gets_a_warning_not_a_failure(env):
    results, text = run(env, FakeLLM(calls_function=False))
    assert statuses(results)["Вызов функций"] == cg.WARN and "GigaChat-2-Max" in text


def test_dashboard_step_passes_when_the_model_returns_a_good_plan(env):
    results, text = run(env, FakeLLM(dashboard=True))
    step = next(r for r in results if r.title.startswith("План дашборда"))
    assert step.status == cg.OK and "блоков 12" in step.detail and "чисел без источника 0" in step.detail, text


def test_dashboard_step_warns_but_does_not_fail_when_the_plan_is_unusable(env):
    results, text = run(env)  # модель отвечает «ок» на всё: плана нет
    step = next(r for r in results if r.title.startswith("План дашборда"))
    assert step.status == cg.WARN and "по умолчанию" in step.detail + step.fix and "GigaChat-2-Max" in step.fix, text
    assert not any(r.status == cg.FAIL for r in results)


def test_dashboard_step_is_skipped_when_the_chat_check_fails(env):
    results, _ = run(env, FakeLLM(chat_exc=LLMError("Закончился пакет токенов или нужна оплата (402).")))
    assert not any(r.title.startswith("План дашборда") for r in results)


# ---------- код возврата ----------


def test_main_exit_codes(env, capsys):
    env.setattr(cg, "load_env", lambda: None)
    env.setattr(cg, "make_client", lambda cfg: FakeLLM())
    env.setattr("sys.argv", ["check_gigachat.py"])
    assert cg.main() == 0
    assert "можно запускать приложение" in capsys.readouterr().out
    env.delenv("GIGACHAT_CREDENTIALS")
    assert cg.main() == 1
    assert "есть ошибки" in capsys.readouterr().out
