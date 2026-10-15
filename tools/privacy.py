"""Приватность: имя файла не уходит в модель.

Всё, что отправляется в GigaChat (сводка отчёта, каталог в промпте, результаты инструментов), называет файл нейтральной
подписью «отчёт.xlsx» (расширение сохраняется). Локально пользователь видит настоящее имя: подпись в текстах плана и
ответов заменяется обратно, а текст, который снова уходит в модель (выводы дашборда в промпте чата), маскируется заново.
"""
import os
from pathlib import PurePath

MASK_STEM = "отчёт"
KNOWN_EXTENSIONS = (".xlsx", ".csv")
_TRUE = {"1", "true", "yes", "on", "да"}


def public_name(name: str) -> str:
    """Подпись файла для модели: «Продажи Q3 (клиенты).xlsx» -> «отчёт.xlsx»; неизвестное расширение не сохраняется."""
    ext = PurePath(str(name or "")).suffix.lower()
    return MASK_STEM + (ext if ext in KNOWN_EXTENSIONS else "")


def mask_text(text: str, real: str) -> str:
    """Настоящее имя файла -> подпись для модели."""
    return _swap(text, real, public_name(real))


def unmask_text(text: str, real: str) -> str:
    """Подпись для модели -> настоящее имя файла (для показа пользователю)."""
    return _swap(text, public_name(real), real)


def mask(obj, real: str):
    """mask_text для всех строк вложенной структуры (словари, списки); исходный объект не меняется."""
    return _walk(obj, lambda s: mask_text(s, real))


def unmask(obj, real: str):
    """unmask_text для всех строк вложенной структуры (словари, списки); исходный объект не меняется."""
    return _walk(obj, lambda s: unmask_text(s, real))


def _swap(text: str, old: str, new: str) -> str:
    return text.replace(old, new) if isinstance(text, str) and old and new and old != new else text


def _walk(obj, fn):
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _walk(v, fn) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_walk(v, fn) for v in obj]
    return obj


def force_consent() -> bool:
    """PRISM_FORCE_CONSENT=1 — только для проверки и разработки: офлайн-заглушка ведёт себя так, будто данные уходят
    наружу (показывается окно «Что уйдёт в GigaChat»). В обычной работе не задаётся."""
    return os.getenv("PRISM_FORCE_CONSENT", "").strip().lower() in _TRUE
