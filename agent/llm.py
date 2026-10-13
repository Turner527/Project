"""Интерфейс LLM-клиента и реализация для GigaChat.

Остальной код зависит только от LLMClient, поэтому в тестах и офлайн-режиме клиент подменяется заглушкой.
Проверка SSL не отключается нигде и ни при каких настройках.
"""
import os
import socket
import ssl
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# «Около нуля»: чистый 0 в некоторых API трактуется как «значение по умолчанию», поэтому 0.01
TEMPERATURE = 0.01
DEFAULT_SCOPE = "GIGACHAT_API_PERS"
DEFAULT_MODEL = "GigaChat-2-Pro"
# Модели, которые предлагаются в интерфейсе, пока список не получен из API (или он недоступен). Идентификаторы сверьте
# со списком, который вернёт ваш аккаунт: в интерфейсе он подгружается через get_models().
KNOWN_MODELS = ("GigaChat-2", "GigaChat-2-Pro", "GigaChat-2-Max", "GigaChat-3-Ultra")
REQUEST_TIMEOUT_S = 90


class LLMError(Exception):
    """Ошибка обращения к модели; сообщение уже на русском и подсказывает, что делать."""


class ConfigError(LLMError):
    pass


@dataclass
class FunctionCall:
    name: str
    arguments: dict


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
        )


@dataclass
class LLMResponse:
    content: str = ""
    function_call: FunctionCall | None = None
    usage: Usage = field(default_factory=Usage)
    functions_state_id: str | None = None
    finish_reason: str | None = None


class LLMClient(Protocol):
    """Сообщения — словари {role, content, [name], [function_call], [functions_state_id]}."""

    name: str

    def chat(
        self,
        messages: list[dict],
        functions: list[dict] | None = None,
        *,
        function_call: str = "auto",
        temperature: float = TEMPERATURE,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...


def load_env() -> None:
    """Подхватывает .env из корня проекта; уже заданные переменные окружения не перезаписывает."""
    load_dotenv(ROOT / ".env", override=False)


# ---------- настройки GigaChat ----------

_FALSE = {"0", "false", "no", "off", "нет"}


@dataclass
class GigaChatConfig:
    credentials: str = ""
    scope: str = DEFAULT_SCOPE
    ca_bundle: str = ""
    model: str = DEFAULT_MODEL

    @classmethod
    def from_env(cls) -> "GigaChatConfig":
        env = os.environ
        return cls(
            credentials=env.get("GIGACHAT_CREDENTIALS", "").strip(),
            scope=env.get("GIGACHAT_SCOPE", "").strip() or DEFAULT_SCOPE,
            ca_bundle=env.get("GIGACHAT_CA_BUNDLE", "").strip(),
            model=env.get("GIGACHAT_MODEL", "").strip() or DEFAULT_MODEL,
        )

    def problems(self) -> list[str]:
        """Что не так с настройками (без обращения к сети)."""
        out = []
        if os.environ.get("GIGACHAT_VERIFY_SSL_CERTS", "").strip().lower() in _FALSE:
            out.append(
                "Отключение проверки SSL запрещено: уберите GIGACHAT_VERIFY_SSL_CERTS из окружения и "
                "укажите сертификат Минцифры в GIGACHAT_CA_BUNDLE."
            )
        if not self.credentials:
            out.append("Не задан GIGACHAT_CREDENTIALS (ключ авторизации из личного кабинета GigaChat API).")
        if self.ca_bundle:
            path = Path(self.ca_bundle_path())
            if not path.is_file():
                out.append(f"Файл сертификата из GIGACHAT_CA_BUNDLE не найден: {path}")
        return out

    def warnings(self) -> list[str]:
        if self.ca_bundle:
            return []
        return [
            "GIGACHAT_CA_BUNDLE не задан: без корневого сертификата Минцифры соединение, скорее всего, не пройдёт "
            "проверку SSL (допустимо только если сертификат подключён иначе, например через SSL_CERT_FILE)."
        ]

    def ca_bundle_path(self) -> str | None:
        if not self.ca_bundle:
            return None
        path = Path(self.ca_bundle)
        return str(path if path.is_absolute() else ROOT / path)


# ---------- человекочитаемые причины ошибок ----------


def _chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def explain_error(exc: BaseException) -> str:
    """Переводит исключение SDK/сети в понятное сообщение: ключ, scope, сертификат, лимиты или сеть."""
    import httpx
    from gigachat.exceptions import ResponseError

    for e in _chain(exc):
        text = str(e).lower()
        if isinstance(e, ssl.SSLError) or "certificate_verify_failed" in text or "certificate verify failed" in text \
                or "self-signed certificate" in text or "unable to get local issuer" in text:
            return (
                "Ошибка проверки SSL-сертификата сервера. Установите корневой сертификат Минцифры и укажите "
                "путь к нему в GIGACHAT_CA_BUNDLE (проверку SSL отключать нельзя)."
            )
    for e in _chain(exc):
        if isinstance(e, ResponseError):
            status = e.status_code
            body = (e.content or b"").decode("utf-8", "replace").lower()
            if status == 401:
                return (
                    "Ключ авторизации отклонён (401). Проверьте GIGACHAT_CREDENTIALS: нужен «Ключ авторизации» "
                    "из личного кабинета (не Client Secret); возможно, ключ перевыпущен. Если ключ верный — "
                    "проверьте GIGACHAT_SCOPE."
                )
            if status == 400 and "scope" in body:
                return (
                    f"Неверный scope: проверьте GIGACHAT_SCOPE (GIGACHAT_API_PERS, GIGACHAT_API_B2B или "
                    f"GIGACHAT_API_CORP — по типу вашего аккаунта). Ответ сервиса: {body[:200]}"
                )
            if status == 400:
                return f"Запрос отклонён сервисом (400): {body[:300]}"
            if status == 402:
                return "Закончился пакет токенов или нужна оплата (402). Проверьте баланс в личном кабинете GigaChat."
            if status == 403:
                return (
                    "Доступ запрещён (403): у ключа нет прав на эту модель или scope не соответствует типу "
                    "аккаунта. Проверьте GIGACHAT_SCOPE и GIGACHAT_MODEL."
                )
            if status == 404:
                return f"Не найдено (404): проверьте GIGACHAT_MODEL. Ответ сервиса: {body[:200]}"
            if status == 429:
                return "Превышен лимит запросов (429). Подождите и повторите; при частых отказах проверьте квоты тарифа."
            if status >= 500:
                return f"Сервис GigaChat временно недоступен ({status}). Повторите позже."
            return f"Сервис GigaChat вернул ошибку {status}: {body[:300]}"
    for e in _chain(exc):
        if isinstance(e, (httpx.TransportError, socket.gaierror, TimeoutError, ConnectionError)):
            return (
                "Нет связи с сервисом GigaChat: проверьте интернет, прокси и DNS, а также доступность "
                "хостов ngw.devices.sberbank.ru:9443 и api.giga.chat. "
                f"Техническая причина: {type(e).__name__}: {e}"
            )
    return f"Ошибка обращения к GigaChat: {type(exc).__name__}: {exc}"


# ---------- клиент GigaChat ----------


def _sdk_message(m: dict) -> dict:
    out = {k: v for k, v in m.items() if v is not None}
    out.setdefault("content", "")
    return out


class GigaChatLLM:
    """Обёртка над официальным пакетом gigachat. Проверка SSL включена всегда."""

    def __init__(self, config: GigaChatConfig | None = None, client=None):
        self.config = config or GigaChatConfig.from_env()
        problems = self.config.problems()
        if problems and client is None:
            raise ConfigError(" ".join(problems))
        self.name = f"gigachat:{self.config.model}"
        if client is None:
            from gigachat import GigaChat

            client = GigaChat(
                credentials=self.config.credentials,
                scope=self.config.scope,
                model=self.config.model,
                ca_bundle_file=self.config.ca_bundle_path(),
                verify_ssl_certs=True,  # явно: перебивает любую переменную окружения
                timeout=REQUEST_TIMEOUT_S,
            )
        self._client = client

    def chat(
        self,
        messages: list[dict],
        functions: list[dict] | None = None,
        *,
        function_call: str = "auto",
        temperature: float = TEMPERATURE,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload: dict = {
            "model": self.config.model,
            "messages": [_sdk_message(m) for m in messages],
            "temperature": temperature,
        }
        if functions:
            payload["functions"] = functions
            payload["function_call"] = function_call
        if max_tokens:
            payload["max_tokens"] = max_tokens
        try:
            completion = self._client.chat(payload)
        except Exception as exc:
            raise LLMError(explain_error(exc)) from exc
        choice = completion.choices[0]
        message = choice.message
        call = None
        if message.function_call is not None:
            call = FunctionCall(message.function_call.name, dict(message.function_call.arguments or {}))
        usage = completion.usage
        return LLMResponse(
            content=message.content or "",
            function_call=call,
            usage=Usage(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) if usage else Usage(),
            functions_state_id=message.functions_state_id,
            finish_reason=choice.finish_reason,
        )


def available_models(llm) -> tuple[list[str], str]:
    """Модели для выбора: (идентификаторы, источник). Источник «api» — список аккаунта, «known» — встроенный запасной."""
    ids: list[str] = []
    try:
        ids = [m.id_ for m in llm._client.get_models().data]
    except Exception:  # noqa: BLE001 — нет сети или прав: покажем встроенный список
        pass
    ids = [i for i in ids if i.lower().startswith("gigachat") and "embedding" not in i.lower()]
    if ids:
        return ids, "api"
    return list(KNOWN_MODELS), "known"


def make_llm(backend: str | None = None) -> LLMClient:
    """backend: 'gigachat', 'stub' или 'auto' (по умолчанию: gigachat, если задан ключ, иначе stub)."""
    backend = (backend or os.getenv("LLM_BACKEND") or "auto").strip().lower()
    if backend == "auto":
        backend = "gigachat" if os.getenv("GIGACHAT_CREDENTIALS", "").strip() else "stub"
    if backend == "gigachat":
        return GigaChatLLM()
    if backend == "stub":
        from agent.stub_llm import ScriptedLLM

        return ScriptedLLM()
    raise ConfigError(f"LLM_BACKEND должен быть gigachat, stub или auto, получено «{backend}»")
