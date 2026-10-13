"""Хранение построенных дашбордов рядом с отчётом: один файл на отпечаток отчёта (см. datastore.fingerprint).

Дашборд привязан к конкретным данным: если файл отчёта или роли столбцов изменились, прежний дашборд не показывается.
Хранятся последние KEEP дашбордов, чтобы возврат к ранее загруженному отчёту не терял построенное.
"""
import json
import os
import re
from pathlib import Path

from tools import datastore

KEEP = 10
_FP_RE = re.compile(r"^[0-9a-f]{16}$")


def _dir() -> Path:
    return datastore._dir() / "dashboards"


def save(payload: dict) -> None:
    """payload — DashboardResult.to_dict() с заполненным fingerprint."""
    fp = payload.get("fingerprint") or ""
    if not _FP_RE.match(fp):
        raise ValueError("У дашборда нет отпечатка отчёта")
    folder = _dir()
    folder.mkdir(parents=True, exist_ok=True)
    datastore._atomic_write(folder / f"{fp}.json", json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    files = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime)
    for old in files[:-KEEP]:
        old.unlink(missing_ok=True)


def load(fp: str) -> dict | None:
    if not _FP_RE.match(fp or ""):
        return None
    path = _dir() / f"{fp}.json"
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None


def current() -> dict | None:
    """Дашборд активного отчёта, если он построен по этим же данным."""
    return load(datastore.fingerprint())


def discard(fp: str) -> None:
    if _FP_RE.match(fp or ""):
        (_dir() / f"{fp}.json").unlink(missing_ok=True)
