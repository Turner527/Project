"""FastAPI-приложение: чат с агентом, оценки ответов и загрузка xlsx-витрины.

Запуск:  uvicorn app.main:create_app --factory   (или  python -m app.main)
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from agent.llm import ConfigError, LLMClient, LLMError, load_env, make_llm
from agent.loop import Agent
from agent.prompts import PROMPT_VERSION
from agent.tool_adapter import ToolAdapter, make_adapter
from app import connect as connect_mod
from data.validation import WorkbookError
from logs.store import STATUS_ERROR, LogStore
from tools import datastore

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
XLSX_MAGIC = b"PK\x03\x04"  # xlsx — это zip-архив


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    client_id: str | None = Field(default=None, max_length=64)


class FeedbackRequest(BaseModel):
    request_id: str
    rating: str | None = Field(default=None, pattern="^(useful|not_useful)$")
    comment: str | None = Field(default=None, max_length=2000)


def create_app(
    llm: LLMClient | None = None,
    adapter: ToolAdapter | None = None,
    store: LogStore | None = None,
    load_env_file: bool = True,
    env_path: Path | None = None,
    certs_dir: Path | None = None,
    verify=None,
) -> FastAPI:
    """Все зависимости можно передать снаружи (тесты); иначе они создаются из .env и окружения."""
    if load_env_file:
        load_env()
    llm_error: str | None = None
    if llm is None:
        try:
            llm = make_llm()
        except ConfigError as exc:
            llm_error = str(exc)
    state = {"llm": llm, "error": llm_error}
    adapter = adapter or make_adapter()
    store = store or LogStore()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        adapter.close()

    app = FastAPI(title="AI-аналитик стратегического менеджмента банка", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    def status() -> dict:
        return {
            "llm": state["llm"].name if state["llm"] else None,
            "llm_error": state["error"],
            "offline": bool(state["llm"] and state["llm"].name == "stub"),
            "tools_backend": (os.getenv("TOOLS_BACKEND") or "mcp").lower(),
            "prompt_version": PROMPT_VERSION,
            "dataset": datastore.describe(),
        }

    @app.post("/api/ask")
    def ask(req: AskRequest) -> dict:
        if state["llm"] is None:
            raise HTTPException(503, state["error"] or "Языковая модель не настроена.")
        result = Agent(state["llm"], adapter).run(req.question.strip())
        request_id = store.log_request(result, req.client_id)
        data = result.to_dict()
        data["request_id"] = request_id
        data["status"] = result.verification.status if result.verification else STATUS_ERROR
        return data

    @app.post("/api/feedback")
    def feedback(req: FeedbackRequest) -> dict:
        rating = {"useful": 1, "not_useful": -1, None: None}[req.rating]
        if rating is None and req.comment is None:
            raise HTTPException(422, "Нужна оценка или комментарий.")
        if not store.set_feedback(req.request_id, rating, req.comment):
            raise HTTPException(404, "Запрос с таким идентификатором не найден.")
        return {"ok": True}

    @app.post("/api/connect")
    def connect_gigachat(request: Request, credentials: str = Form(...), scope: str = Form(""), model: str = Form(""),
                         ca_file: UploadFile | None = File(None)) -> JSONResponse:
        """Проверяет ключ живым запросом и сохраняет его в .env. Только с локального адреса."""
        if (request.client.host if request.client else "") not in connect_mod.LOCAL_HOSTS:
            raise HTTPException(403, "Подключение из интерфейса доступно только с локального компьютера.")
        cert = ca_file.file.read(1024 * 1024 + 1) if ca_file and ca_file.filename else None
        if cert is not None and len(cert) > 1024 * 1024:
            return _upload_error("Файл сертификата больше 1 МБ.", 413)
        try:
            new_llm = connect_mod.connect(credentials, scope, model, cert, env_path or connect_mod.ENV_PATH,
                                          certs_dir or connect_mod.CERTS_DIR, verify or connect_mod.verify_token)
        except LLMError as exc:
            return JSONResponse({"ok": False, "detail": str(exc)}, status_code=422)
        state["llm"], state["error"] = new_llm, None
        return JSONResponse({"ok": True, "llm": new_llm.name})

    @app.post("/api/upload")
    def upload(file: UploadFile = File(...)) -> JSONResponse:
        name = file.filename or "файл"
        if not name.lower().endswith(".xlsx"):
            return _upload_error(f"Файл «{name}» не в формате .xlsx. Загрузите книгу Excel (.xlsx).")
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            return _upload_error(f"Файл больше {MAX_UPLOAD_BYTES // 1024 // 1024} МБ.", 413)
        if not content.startswith(XLSX_MAGIC):
            return _upload_error(f"Файл «{name}» не похож на книгу Excel (.xlsx): повреждён или другого формата.")
        try:
            dataset = datastore.save_upload(content)
        except WorkbookError as exc:
            return _upload_error("Файл не прошёл проверку.", errors=exc.errors)
        return JSONResponse({"ok": True, "dataset": dataset})

    return app


def _upload_error(message: str, status_code: int = 422, errors: list[str] | None = None) -> JSONResponse:
    return JSONResponse({"ok": False, "detail": message, "errors": errors or [message]}, status_code=status_code)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:create_app", factory=True, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
