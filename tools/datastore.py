"""Доступ к активной витрине. Её читают и FastAPI, и MCP-сервер (отдельный процесс),
поэтому «активная» витрина — это файл на диске, а кэш сбрасывается по mtime."""
import os
import tempfile
import threading
from pathlib import Path

from data.schema import Dataset
from data.validation import load_dataset

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "kpi.xlsx"
UPLOAD_PATH = ROOT / "data" / "uploads" / "active.xlsx"

_lock = threading.Lock()
_cache: dict = {}


def active_path() -> Path:
    """KPI_DATA_PATH (для тестов и отладки) > загруженный пользователем файл > демо-данные."""
    env = os.getenv("KPI_DATA_PATH")
    if env:
        return Path(env)
    return UPLOAD_PATH if UPLOAD_PATH.exists() else DEFAULT_PATH


def get_dataset() -> Dataset:
    path = active_path()
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    with _lock:
        if _cache.get("key") != key:
            _cache["dataset"] = load_dataset(path, name=path.name)
            _cache["key"] = key
        return _cache["dataset"]


def describe() -> dict:
    ds = get_dataset()
    info = ds.summary()
    info["is_upload"] = active_path() == UPLOAD_PATH
    return info


def save_upload(content: bytes) -> dict:
    """Проверяет xlsx и делает его активным. При ошибке бросает WorkbookError, ничего не меняя."""
    load_dataset(content, name="upload")  # WorkbookError наружу
    UPLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=UPLOAD_PATH.parent, suffix=".xlsx")
    with os.fdopen(fd, "wb") as fh:
        fh.write(content)
    os.replace(tmp, UPLOAD_PATH)  # атомарно: MCP-процесс не увидит недописанный файл
    return describe()
