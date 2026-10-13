"""Проверка связи с GigaChat: по шагам и с понятным объяснением, что именно не так.

Запуск:  python check_gigachat.py            (полная проверка)
         python check_gigachat.py --offline  (только настройки и сертификат, без сети)

Шаги идут по цепочке «настройки -> сеть -> TLS с вашим сертификатом -> ключ -> модель -> вызов функции»:
если упал ранний шаг, зависящие от него пропускаются, а причина названа явно (ключ, scope, сертификат,
лимиты или сеть). Проверка SSL здесь тоже не отключается.
"""
import argparse
import os
import re
import socket
import ssl
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from agent.llm import (
    DEFAULT_SCOPE,
    ROOT,
    GigaChatConfig,
    GigaChatLLM,
    LLMError,
    explain_error,
    load_env,
)

KNOWN_SCOPES = ("GIGACHAT_API_PERS", "GIGACHAT_API_B2B", "GIGACHAT_API_CORP")
OK, FAIL, WARN, SKIP = "OK", "ОШИБКА", "ВНИМАНИЕ", "ПРОПУЩЕНО"


@dataclass
class Result:
    status: str
    title: str
    detail: str = ""
    fix: str = ""


# ---------- шаги без сети ----------


def check_env_file() -> Result:
    path = ROOT / ".env"
    if path.is_file():
        return Result(OK, "Файл .env", f"найден: {path}")
    return Result(WARN, "Файл .env", "не найден; настройки берутся только из переменных окружения.",
                  "Скопируйте .env.example в .env и заполните GIGACHAT_CREDENTIALS.")


def check_ssl_guard() -> Result:
    if os.environ.get("GIGACHAT_VERIFY_SSL_CERTS", "").strip().lower() in {"0", "false", "no", "off", "нет"}:
        return Result(FAIL, "Проверка SSL", "в окружении задано GIGACHAT_VERIFY_SSL_CERTS=false.",
                      "Удалите эту переменную: проект не работает с отключённой проверкой SSL. "
                      "Вместо этого укажите сертификат Минцифры в GIGACHAT_CA_BUNDLE.")
    return Result(OK, "Проверка SSL", "включена (отключить её в этом проекте нельзя)")


def check_credentials(cfg: GigaChatConfig) -> Result:
    key = cfg.credentials
    if not key:
        return Result(FAIL, "Ключ авторизации", "GIGACHAT_CREDENTIALS не задан.",
                      "Получите ключ в личном кабинете GigaChat API и впишите его в .env (см. README, шаг 3).")
    if ":" in key or " " in key:
        return Result(WARN, "Ключ авторизации", "в значении есть «:» или пробел — похоже на «client_id:client_secret».",
                      "Нужен именно «Ключ авторизации» (Authorization Key, строка в Base64) из личного кабинета.")
    if not re.fullmatch(r"[A-Za-z0-9+/=_-]{20,}", key):
        return Result(WARN, "Ключ авторизации", "значение не похоже на ключ в Base64 (слишком короткое или с лишними символами).",
                      "Скопируйте ключ целиком, без кавычек и пробелов.")
    return Result(OK, "Ключ авторизации", f"задан ({len(key)} символов, значение не показывается)")


def check_scope(cfg: GigaChatConfig) -> Result:
    if cfg.scope in KNOWN_SCOPES:
        return Result(OK, "Scope", cfg.scope)
    return Result(WARN, "Scope", f"«{cfg.scope}» — нестандартное значение.",
                  f"Допустимо: {', '.join(KNOWN_SCOPES)} (по типу аккаунта; для физлиц — {DEFAULT_SCOPE}).")


def check_ca_bundle(cfg: GigaChatConfig) -> tuple[Result, str | None]:
    if not cfg.ca_bundle:
        return Result(WARN, "Сертификат Минцифры", "GIGACHAT_CA_BUNDLE не задан.",
                      "Без корневого сертификата Минцифры проверка SSL, скорее всего, не пройдёт. "
                      "Скачайте сертификат и укажите путь в .env (см. README, шаг 4)."), None
    path = Path(cfg.ca_bundle_path())
    if not path.is_file():
        return Result(FAIL, "Сертификат Минцифры", f"файл не найден: {path}",
                      "Проверьте путь в GIGACHAT_CA_BUNDLE (относительный путь считается от корня проекта)."), None
    try:
        stats = ssl.create_default_context(cafile=str(path)).cert_store_stats()
    except (ssl.SSLError, OSError) as exc:
        return Result(FAIL, "Сертификат Минцифры", f"файл {path.name} не удалось прочитать как PEM: {exc}",
                      "Если у вас .cer в формате DER, конвертируйте: "
                      "openssl x509 -inform DER -in файл.cer -out certs/russian_trusted_root_ca.pem"), None
    if not stats.get("x509_ca"):
        return Result(FAIL, "Сертификат Минцифры", f"в {path.name} нет ни одного сертификата удостоверяющего центра.",
                      "Нужен корневой сертификат НУЦ Минцифры в формате PEM."), None
    return Result(OK, "Сертификат Минцифры", f"{path} (сертификатов УЦ: {stats['x509_ca']})"), str(path)


# ---------- шаги с сетью ----------


def tcp_connect(host: str, port: int, timeout: float = 8.0) -> None:
    socket.create_connection((host, port), timeout=timeout).close()


def tls_handshake(host: str, port: int, ca_path: str | None, timeout: float = 8.0) -> None:
    context = ssl.create_default_context(cafile=ca_path)  # проверка сертификата и имени хоста включена
    with socket.create_connection((host, port), timeout=timeout) as sock:
        with context.wrap_socket(sock, server_hostname=host):
            pass


def _hosts() -> list[tuple[str, str, int]]:
    from gigachat.settings import Settings

    settings = Settings()
    out = []
    for label, url in (("авторизации", settings.auth_url), ("API", settings.base_url)):
        parsed = urlparse(url)
        out.append((label, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)))
    return out


def _proxy_host() -> str | None:
    """Хост прокси из окружения (логин и пароль из адреса не показываем)."""
    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        value = os.environ.get(name)
        if value:
            return urlparse(value if "//" in value else f"//{value}").hostname or "?"
    return None


def check_network(label: str, host: str, port: int) -> Result:
    title = f"Сеть: сервер {label} ({host}:{port})"
    try:
        tcp_connect(host, port)
    except socket.gaierror as exc:
        return Result(FAIL, title, f"имя хоста не найдено ({exc}).", "Проверьте интернет и DNS; в корпоративной сети — прокси.")
    except (TimeoutError, socket.timeout):
        return Result(FAIL, title, "превышено время ожидания соединения.",
                      "Порт, вероятно, закрыт файрволом или требуется прокси/VPN. Для порта 9443 нужен исходящий доступ.")
    except OSError as exc:
        return Result(FAIL, title, f"не удалось соединиться: {exc}",
                      "Проверьте интернет, файрвол и прокси (переменные HTTPS_PROXY/NO_PROXY).")
    return Result(OK, title, "соединение установлено")


def check_tls(label: str, host: str, port: int, ca_path: str | None) -> Result:
    title = f"TLS: сертификат сервера {label}"
    try:
        tls_handshake(host, port, ca_path)
    except ssl.SSLCertVerificationError as exc:
        return Result(FAIL, title, f"сертификат сервера не прошёл проверку: {getattr(exc, 'verify_message', None) or exc}.",
                      "Укажите корневой сертификат Минцифры в GIGACHAT_CA_BUNDLE. Если он указан — добавьте в тот же PEM-файл "
                      "промежуточный сертификат («russian_trusted_sub_ca»). Проверку SSL отключать нельзя.")
    except (ssl.SSLError, OSError) as exc:
        return Result(FAIL, title, f"ошибка TLS-рукопожатия: {exc}", "Проверьте прокси и перехват HTTPS в сети.")
    return Result(OK, title, "проверка пройдена")


def make_client(cfg: GigaChatConfig):
    return GigaChatLLM(cfg)


def check_auth(llm: GigaChatLLM) -> Result:
    try:
        token = llm._client.get_token()
    except Exception as exc:  # noqa: BLE001 — любая причина должна быть объяснена
        return Result(FAIL, "Авторизация (получение токена)", explain_error(exc))
    expires = getattr(token, "expires_at", None)
    return Result(OK, "Авторизация (получение токена)", "токен получен" + (f", действует до {expires}" if expires else ""))


def check_model(llm: GigaChatLLM) -> Result:
    try:
        models = [m.id_ for m in llm._client.get_models().data]
    except Exception as exc:  # noqa: BLE001
        return Result(WARN, "Модель", f"список моделей получить не удалось: {explain_error(exc)}")
    wanted = llm.config.model
    if wanted in models:
        return Result(OK, "Модель", f"{wanted} доступна")
    return Result(FAIL, "Модель", f"«{wanted}» нет в списке доступных: {', '.join(models) or 'пусто'}.",
                  "Исправьте GIGACHAT_MODEL (нужна модель с поддержкой вызова функций, например GigaChat-2-Pro).")


def check_chat(llm: GigaChatLLM) -> Result:
    started = time.perf_counter()
    try:
        resp = llm.chat([{"role": "user", "content": "Ответь одним словом: ок"}], max_tokens=20)
    except LLMError as exc:
        return Result(FAIL, "Тестовый запрос", str(exc))
    return Result(OK, "Тестовый запрос", f"ответ «{resp.content.strip()[:40]}», {resp.usage.total_tokens} токенов, "
                  f"{time.perf_counter() - started:.1f} с")


def check_function_calling(llm: GigaChatLLM) -> Result:
    from agent.tool_adapter import make_adapter

    adapter = make_adapter("direct")
    functions = [f for f in adapter.functions if f["name"] == "get_change"]
    try:
        resp = llm.chat(
            [{"role": "system", "content": "Ты аналитик. Для чисел вызывай функции, сам не считай."},
             {"role": "user", "content": "Каково изменение показателя «NPS» за последний период?"}],
            functions, function_call="auto")
    except LLMError as exc:
        return Result(FAIL, "Вызов функций", str(exc))
    if resp.function_call and resp.function_call.name == "get_change":
        return Result(OK, "Вызов функций", f"модель вызвала get_change с аргументами {resp.function_call.arguments}")
    return Result(WARN, "Вызов функций", "модель ответила текстом вместо вызова функции.",
                  "Агент будет работать нестабильно: возьмите более сильную модель (GigaChat-2-Pro или GigaChat-2-Max).")


# ---------- запуск ----------


def run(offline: bool = False, out=print) -> list[Result]:
    results: list[Result] = []

    def add(result: Result) -> Result:
        results.append(result)
        marker = f"[{result.status}]"
        out(f"{marker:<13} {result.title}" + (f" — {result.detail}" if result.detail else ""))
        if result.fix:
            out(f"{'':<13} Что делать: {result.fix}")
        return result

    cfg = GigaChatConfig.from_env()
    add(check_env_file())
    guard = add(check_ssl_guard())
    creds = add(check_credentials(cfg))
    add(check_scope(cfg))
    ca_result, ca_path = check_ca_bundle(cfg)
    add(ca_result)
    config_ok = guard.status != FAIL and creds.status != FAIL and ca_result.status != FAIL

    if offline:
        out("\nРежим --offline: проверки сети пропущены.")
        return results
    if not config_ok:
        add(Result(SKIP, "Проверки сети", "сначала исправьте ошибки настройки выше."))
        return results

    network_ok = True
    proxy = _proxy_host()
    if proxy:
        # прямой TCP/TLS-тест идёт мимо прокси и дал бы ложную ошибку: проверяем сразу через SDK
        add(Result(WARN, "Сеть и TLS", f"задан прокси ({proxy}): прямые проверки пропущены, связь проверит шаг авторизации."))
    else:
        for label, host, port in _hosts():
            net = add(check_network(label, host, port))
            if net.status == FAIL:
                network_ok = False
                continue
            if add(check_tls(label, host, port, ca_path)).status == FAIL:
                network_ok = False
    if not network_ok:
        add(Result(SKIP, "Авторизация, модель, вызов функций", "сначала нужна рабочая сеть и сертификат."))
        return results

    llm = make_client(cfg)
    if add(check_auth(llm)).status == FAIL:
        add(Result(SKIP, "Модель, тестовый запрос, вызов функций", "авторизация не прошла."))
        return results
    if add(check_model(llm)).status == FAIL:
        add(Result(SKIP, "Тестовый запрос, вызов функций", "модель недоступна."))
        return results
    if add(check_chat(llm)).status != FAIL:
        add(check_function_calling(llm))
    return results


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Проверка связи с GigaChat API")
    parser.add_argument("--offline", action="store_true", help="проверить только настройки и сертификат, без сети")
    args = parser.parse_args()
    load_env()
    print("Проверка подключения к GigaChat API\n")
    results = run(offline=args.offline)
    failed = [r for r in results if r.status == FAIL]
    warned = [r for r in results if r.status == WARN]
    print()
    if failed:
        print(f"Итог: есть ошибки ({len(failed)}). Исправьте их сверху вниз — последующие шаги зависят от предыдущих.")
        return 1
    print("Итог: критичных проблем нет, можно запускать приложение." + (f" Предупреждений: {len(warned)}." if warned else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
