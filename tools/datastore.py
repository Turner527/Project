"""Доступ к активному отчёту. Его читают и FastAPI, и MCP-сервер (отдельный процесс), поэтому «активный» отчёт —
это файлы на диске, а кэш сбрасывается по времени изменения.

Источники (по приоритету): KPI_DATA_PATH (тесты и отладка) > загруженный пользователем отчёт > демо-данные.
Загруженный отчёт бывает двух видов: стратегический xlsx (листы goals/facts/links) или плоская таблица
со спецификацией ролей столбцов (active.spec.json + active.data.xlsx|csv).
"""
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path

from data.model import Spec, Table, build_table
from data.readers import read_file
from data.strategic import to_table
from data.validation import WorkbookError, load_dataset

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "kpi.xlsx"
UPLOAD_PATH = ROOT / "data" / "uploads" / "active.xlsx"  # стратегический формат; рядом лежат остальные файлы
PENDING_KEEP = 10
PENDING_MAX_AGE_S = 24 * 3600
PENDING_ID_RE = re.compile(r"^[0-9a-f]{32}$")

_lock = threading.Lock()
_cache: dict = {}


def _dir() -> Path:
    return UPLOAD_PATH.parent


def _flat_spec_path() -> Path:
    return _dir() / "active.spec.json"


def _pending_dir() -> Path:
    return _dir() / "pending"


def active_source() -> tuple[str, Path]:
    env = os.getenv("KPI_DATA_PATH")
    if env:
        path = Path(env)
        return ("flat" if path.suffix == ".json" else "strategic"), path
    if _flat_spec_path().exists():
        return "flat", _flat_spec_path()
    if UPLOAD_PATH.exists():
        return "strategic", UPLOAD_PATH
    return "strategic", DEFAULT_PATH


def _load_flat(spec_path: Path) -> Table:
    meta = json.loads(spec_path.read_text("utf-8"))
    data_path = spec_path.parent / meta["data_file"]
    raw = read_file(data_path.read_bytes(), meta.get("filename") or data_path.name)
    sheet = meta["spec"].get("sheet") or raw.sheet
    table = build_table(raw.frames[sheet], Spec.from_dict(meta["spec"]), meta.get("filename") or data_path.name,
                        raw.header_rows.get(sheet, 1))
    table.notes = raw.notes
    return table


def _stat(path: Path) -> tuple:
    st = path.stat()
    return str(path), st.st_mtime_ns, st.st_size


def snapshot() -> tuple[Table, str]:
    """Активный отчёт и его отпечаток из одного и того же состояния файлов."""
    kind, path = active_source()
    key = _stat(path)
    if kind == "flat":
        meta = json.loads(path.read_text("utf-8"))
        key += _stat(path.parent / meta["data_file"])
    with _lock:
        if _cache.get("key") != key:
            _cache["table"] = _load_flat(path) if kind == "flat" else to_table(load_dataset(path, name=path.name), path.name)
            _cache["fingerprint"] = _content_hash(kind, path)
            _cache["key"] = key
        return _cache["table"], _cache["fingerprint"]


def get_table() -> Table:
    return snapshot()[0]


def _content_hash(kind: str, path: Path) -> str:
    """Хеш содержимого отчёта: данные и роли столбцов (для плоского) или книга целиком (для стратегического)."""
    digest = hashlib.sha1()
    if kind == "flat":
        meta = json.loads(path.read_text("utf-8"))
        roles = {"filename": meta.get("filename"), "spec": meta["spec"]}
        digest.update(json.dumps(roles, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update((path.parent / meta["data_file"]).read_bytes())
    else:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def fingerprint() -> str:
    """Отпечаток активного отчёта по его содержимому: меняется, когда меняются данные или роли столбцов. По нему
    дашборд привязан к отчёту; тот же отчёт, загруженный заново, даёт тот же отпечаток и находит свой дашборд."""
    return snapshot()[1]


def describe() -> dict:
    table = get_table()
    kind, path = active_source()
    info = table.summary()
    info["is_upload"] = kind == "flat" or path == UPLOAD_PATH
    info["kind"] = kind
    return info


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "wb") as fh:
        fh.write(content)
    os.replace(tmp, path)  # атомарно: MCP-процесс не увидит недописанный файл


def _clear_active() -> None:
    for path in [UPLOAD_PATH, _flat_spec_path(), *_dir().glob("active.data.*")]:
        path.unlink(missing_ok=True)


def save_upload(content: bytes) -> dict:
    """Стратегический xlsx: проверяет и делает активным. При ошибке бросает WorkbookError, ничего не меняя."""
    load_dataset(content, name="upload")
    spec = _flat_spec_path()
    _atomic_write(UPLOAD_PATH, content)
    spec.unlink(missing_ok=True)
    for path in _dir().glob("active.data.*"):
        path.unlink(missing_ok=True)
    return describe()


def save_pending(content: bytes, filename: str, kind: str) -> str:
    """Временно хранит загруженный плоский отчёт, пока пользователь проверяет роли столбцов."""
    pending = _pending_dir()
    pending.mkdir(parents=True, exist_ok=True)
    now = time.time()
    files = sorted(pending.glob("*.json"), key=lambda p: p.stat().st_mtime)
    for old in files[:-PENDING_KEEP + 1] + [p for p in files if now - p.stat().st_mtime > PENDING_MAX_AGE_S]:
        for ext in (".json", ".xlsx", ".csv"):
            old.with_suffix(ext).unlink(missing_ok=True)
    upload_id = uuid.uuid4().hex
    _atomic_write(pending / f"{upload_id}.{kind}", content)
    _atomic_write(pending / f"{upload_id}.json", json.dumps({"filename": filename, "kind": kind}, ensure_ascii=False).encode())
    return upload_id


def load_pending(upload_id: str) -> tuple[bytes, str, str]:
    if not PENDING_ID_RE.match(upload_id or ""):
        raise WorkbookError(["Неверный идентификатор загрузки."])
    meta_path = _pending_dir() / f"{upload_id}.json"
    if not meta_path.exists():
        raise WorkbookError(["Загруженный файл не найден или устарел: загрузите его заново."])
    meta = json.loads(meta_path.read_text("utf-8"))
    return (_pending_dir() / f"{upload_id}.{meta['kind']}").read_bytes(), meta["filename"], meta["kind"]


def save_flat(upload_id: str, spec_dict: dict) -> dict:
    """Проверяет спецификацию на загруженном файле и делает отчёт активным. Бросает WorkbookError."""
    content, filename, kind = load_pending(upload_id)
    raw = read_file(content, filename)
    spec = Spec.from_dict(spec_dict)
    sheet = spec.sheet if spec.sheet in raw.frames else raw.sheet
    spec.sheet = sheet
    build_table(raw.frames[sheet], spec, filename, raw.header_rows.get(sheet, 1))  # WorkbookError наружу
    data_name = f"active.data.{kind}"
    _atomic_write(_dir() / data_name, content)
    UPLOAD_PATH.unlink(missing_ok=True)
    for path in _dir().glob("active.data.*"):
        if path.name != data_name:
            path.unlink(missing_ok=True)
    meta = {"data_file": data_name, "filename": filename, "spec": spec.to_dict()}
    _atomic_write(_flat_spec_path(), json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8"))
    return describe()


def reset() -> dict:
    """Возвращает демо-данные: удаляет загруженный отчёт."""
    _clear_active()
    return describe()
