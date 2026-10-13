"""Подключение GigaChat из интерфейса: проверка ключа живым запросом и сохранение настроек в .env."""
import os
import ssl
from pathlib import Path

import dataclasses
import re

from agent.llm import ROOT, GigaChatConfig, GigaChatLLM, LLMError, available_models, explain_error

ENV_PATH = ROOT / ".env"
CERTS_DIR = ROOT / "certs"
CERT_NAME = "russian_trusted_root_ca.pem"
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def verify_token(llm: GigaChatLLM) -> None:
    """Получает токен: так проверяются и ключ, и scope, и сертификат, и сеть."""
    try:
        llm._client.get_token()
    except Exception as exc:  # noqa: BLE001 — любая причина объясняется пользователю
        raise LLMError(explain_error(exc)) from exc


def check_pem(path: Path) -> None:
    try:
        stats = ssl.create_default_context(cafile=str(path)).cert_store_stats()
    except (ssl.SSLError, OSError) as exc:
        raise LLMError("Файл сертификата не удалось прочитать как PEM. Если у вас .cer в формате DER, "
                       f"конвертируйте его командой openssl из README ({exc}).") from exc
    if not stats.get("x509_ca"):
        raise LLMError("В файле нет ни одного сертификата удостоверяющего центра.")


def write_env(path: Path, updates: dict[str, str]) -> None:
    """Обновляет нужные строки .env, остальное сохраняет; права 600, т.к. внутри ключ."""
    if any("\n" in v or "\r" in v for v in updates.values()):
        raise ValueError("значение содержит перевод строки")
    lines = path.read_text("utf-8").splitlines() if path.exists() else []
    seen, out = set(), []
    for line in lines:
        key = line.split("=", 1)[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)
    out += [f"{k}={v}" for k, v in updates.items() if k not in seen]
    path.write_text("\n".join(out) + "\n", "utf-8")
    os.chmod(path, 0o600)


MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def switch_model(llm: GigaChatLLM, model: str, env_path: Path) -> GigaChatLLM:
    """Переключает модель уже подключённого GigaChat без повторной авторизации и запоминает выбор в .env.
    Если аккаунт отдаёт список моделей, названия вне списка отклоняются. Ошибки — LLMError."""
    model = model.strip()
    if not MODEL_RE.match(model):
        raise LLMError("Название модели: латинские буквы, цифры, точка, дефис и подчёркивание, например GigaChat-2-Pro.")
    ids, source = available_models(llm)
    if source == "api" and model not in ids:
        raise LLMError(f"Модели «{model}» нет в списке вашего аккаунта: {', '.join(ids)}.")
    new = GigaChatLLM(dataclasses.replace(llm.config, model=model), client=llm._client)
    write_env(env_path, {"GIGACHAT_MODEL": model})
    os.environ["GIGACHAT_MODEL"] = model
    return new


def connect(credentials: str, scope: str, model: str, cert_bytes: bytes | None, env_path: Path, certs_dir: Path,
            verify=verify_token) -> GigaChatLLM:
    """Проверяет настройки и связь; только при успехе сохраняет .env и сертификат. Ошибки — LLMError."""
    credentials = credentials.strip()
    if not credentials or any(c.isspace() for c in credentials):
        raise LLMError("Ключ авторизации пустой или содержит пробелы: вставьте его целиком, без кавычек.")
    current = GigaChatConfig.from_env()
    final_cert = certs_dir / CERT_NAME
    tmp_cert = None
    ca_bundle = current.ca_bundle
    if cert_bytes:
        certs_dir.mkdir(parents=True, exist_ok=True)
        tmp_cert = certs_dir / (CERT_NAME + ".tmp")
        tmp_cert.write_bytes(cert_bytes)
        try:
            check_pem(tmp_cert)
        except LLMError:
            tmp_cert.unlink(missing_ok=True)
            raise
        ca_bundle = str(tmp_cert)
    config = GigaChatConfig(credentials=credentials, scope=scope.strip() or current.scope,
                            ca_bundle=ca_bundle, model=model.strip() or current.model)
    try:
        llm = GigaChatLLM(config)
        verify(llm)
    except LLMError:
        if tmp_cert:
            tmp_cert.unlink(missing_ok=True)
        raise
    updates = {"GIGACHAT_CREDENTIALS": config.credentials, "GIGACHAT_SCOPE": config.scope,
               "GIGACHAT_MODEL": config.model}
    if tmp_cert:
        tmp_cert.replace(final_cert)
        config.ca_bundle = str(final_cert)
        llm = GigaChatLLM(config)  # клиент должен ссылаться на итоговый файл, а не на удалённый временный
        try:
            updates["GIGACHAT_CA_BUNDLE"] = str(final_cert.relative_to(ROOT))
        except ValueError:
            updates["GIGACHAT_CA_BUNDLE"] = str(final_cert)
    write_env(env_path, updates)
    os.environ.update(updates)
    os.environ.pop("LLM_BACKEND", None)  # иначе при перезапуске осталась бы заглушка
    return llm
