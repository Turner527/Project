"""FastAPI-приложение: чат с агентом, оценки ответов и загрузка xlsx-витрины.

Запуск:  uvicorn app.main:create_app --factory   (или  python -m app.main)
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from agent.dashboard import DashboardAgent, context_text
from agent.llm import (
    ConfigError,
    GigaChatLLM,
    LLMClient,
    LLMError,
    KNOWN_MODELS,
    available_models,
    llm_sends_data_out,
    load_env,
    make_llm,
)
from agent.loop import Agent, DashboardContext
from agent.prompts import PROMPT_VERSION
from agent.tool_adapter import ToolAdapter, make_adapter
from app import connect as connect_mod
from app.consent import CONSENT_DETAIL, STALE_DETAIL, ConsentGate
from data.profile import profile
from data.readers import read_file
from data.validation import WorkbookError, check_zip_size
from logs.store import STATUS_ERROR, LogStore
from tools import dashboard_store, datastore
from tools.engine import suggest_questions
from tools.privacy import mask

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
XLSX_MAGIC = b"PK\x03\x04"  # xlsx — это zip-архив


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    client_id: str | None = Field(default=None, max_length=64)


class DashboardRequest(BaseModel):
    client_id: str | None = Field(default=None, max_length=64)


class ConsentRequest(BaseModel):
    token: str = Field(min_length=1, max_length=128)


class ModelRequest(BaseModel):
    model: str = Field(min_length=1, max_length=64)


class ProfileRequest(BaseModel):
    upload_id: str
    sheet: str | None = None


class ConfirmRequest(BaseModel):
    upload_id: str
    spec: dict


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
    gate = ConsentGate()  # подтверждения «что уйдёт в модель»; живут, пока работает сервер

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        adapter.close()

    app = FastAPI(title="PRISM: анализ отчётов и KPI", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    def status() -> dict:
        table, fingerprint = datastore.snapshot()
        sends = llm_sends_data_out(state["llm"])
        return {
            "llm": state["llm"].name if state["llm"] else None,
            "llm_error": state["error"],
            "offline": bool(state["llm"] and state["llm"].name == "stub"),
            "sends_data_out": sends,
            "consent_required": sends and not gate.is_confirmed(table, fingerprint),
            "tools_backend": (os.getenv("TOOLS_BACKEND") or "mcp").lower(),
            "prompt_version": PROMPT_VERSION,
            "dataset": datastore.describe(),
            "questions": suggest_questions(datastore.get_table()),
            "dashboard": _dashboard_brief(dashboard_store.current()),
        }

    @app.post("/api/ask")
    def ask(req: AskRequest):
        if state["llm"] is None:
            raise HTTPException(503, state["error"] or "Языковая модель не настроена.")
        table, fingerprint = datastore.snapshot()
        if llm_sends_data_out(state["llm"]) and not gate.is_confirmed(table, fingerprint):
            return _consent_required()
        result = Agent(state["llm"], adapter).run(req.question.strip(), _dashboard_context(dashboard_store.current(), table.source))
        request_id = store.log_request(result, req.client_id)
        data = result.to_dict()
        data["request_id"] = request_id
        data["status"] = result.verification.status if result.verification else STATUS_ERROR
        return data

    @app.post("/api/dashboard")
    def build_dashboard(req: DashboardRequest = DashboardRequest()):
        """Отправляет сводку активного отчёта в модель, строит дашборд по её рекомендациям и сохраняет его."""
        if state["llm"] is None:
            raise HTTPException(503, state["error"] or "Языковая модель не настроена.")
        table, fingerprint = datastore.snapshot()
        digest = None
        if llm_sends_data_out(state["llm"]):
            digest = gate.approved_digest(table, fingerprint)  # в модель уходит ровно подтверждённая сводка
            if digest is None:
                return _consent_required()
        result = DashboardAgent(state["llm"]).run(table, digest)
        result.fingerprint = fingerprint
        result.request_id = store.log_dashboard(result, req.client_id)
        payload = result.to_dict()
        if result.plan is not None and result.widgets:
            dashboard_store.save(payload)
        return payload

    @app.get("/api/preview")
    def preview() -> dict:
        """Что уйдёт в модель: сводка отчёта (точный текст), каталог чата, исключённые столбцы. Сводка запоминается:
        дашборд после подтверждения строится именно по ней."""
        table, fingerprint = datastore.snapshot()
        return gate.preview(table, fingerprint, required=llm_sends_data_out(state["llm"]),
                            kind=datastore.active_source()[0], llm_name=state["llm"].name if state["llm"] else None)

    @app.post("/api/consent")
    def consent(req: ConsentRequest) -> JSONResponse:
        """Подтверждает отправку сводки текущего отчёта. Токен берётся из /api/preview: изменилась сводка — он не подойдёт."""
        table, fingerprint = datastore.snapshot()
        if not gate.consent(table, fingerprint, req.token):
            return JSONResponse({"ok": False, "detail": STALE_DETAIL}, status_code=409)
        return JSONResponse({"ok": True, "confirmed": True})

    @app.get("/api/dashboard")
    def get_dashboard() -> dict:
        """Дашборд активного отчёта, если он уже построен по этим данным."""
        return {"dashboard": dashboard_store.current()}

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
        with connect_mod.ENV_LOCK:  # подключение и смена модели не перебивают друг друга
            try:
                new_llm = connect_mod.connect(credentials, scope, model, cert, env_path or connect_mod.ENV_PATH,
                                              certs_dir or connect_mod.CERTS_DIR, verify or connect_mod.verify_token)
            except LLMError as exc:
                return JSONResponse({"ok": False, "detail": str(exc)}, status_code=422)
            state["llm"], state["error"] = new_llm, None
        return JSONResponse({"ok": True, "llm": new_llm.name})

    @app.get("/api/models")
    def models() -> dict:
        """Модели для выбора: список аккаунта GigaChat (если подключён и отвечает) или встроенный."""
        llm = state["llm"]
        if isinstance(llm, GigaChatLLM):
            ids, source = available_models(llm)
            if llm.config.model not in ids:
                ids = [llm.config.model, *ids]
            return {"current": llm.config.model, "models": ids, "source": source, "switchable": True}
        return {"current": None, "models": list(KNOWN_MODELS), "source": "known", "switchable": False}

    @app.post("/api/model")
    def set_model(request: Request, req: ModelRequest) -> JSONResponse:
        """Переключает модель подключённого GigaChat и запоминает выбор в .env. Только с локального адреса."""
        if (request.client.host if request.client else "") not in connect_mod.LOCAL_HOSTS:
            raise HTTPException(403, "Смена модели из интерфейса доступна только с локального компьютера.")
        with connect_mod.ENV_LOCK:  # текущее подключение читается уже под замком: иначе его могла бы заменить /api/connect
            if not isinstance(state["llm"], GigaChatLLM):
                detail = ("Сначала подключите GigaChat." if state["llm"] is not None or state["error"] is None
                          else f"GigaChat не настроен: {state['error']}")
                return JSONResponse({"ok": False, "detail": detail}, status_code=409)
            try:
                state["llm"] = connect_mod.switch_model(state["llm"], req.model, env_path or connect_mod.ENV_PATH)
            except LLMError as exc:
                return JSONResponse({"ok": False, "detail": str(exc)}, status_code=422)
            except OSError as exc:
                return JSONResponse({"ok": False, "detail": f"Не удалось сохранить выбор в .env: {exc}"}, status_code=500)
            return JSONResponse({"ok": True, "llm": state["llm"].name})

    @app.post("/api/upload")
    def upload(file: UploadFile = File(...)) -> JSONResponse:
        """Стратегический xlsx (goals/facts/links) принимается сразу; любой другой отчёт (xlsx, csv) идёт на проверку
        ролей столбцов: в ответе профиль и upload_id для /api/datasets/confirm."""
        name = file.filename or "файл"
        lower = name.lower()
        if lower.endswith(".xls"):
            return _upload_error("Формат .xls устарел: откройте файл в Excel и сохраните как .xlsx или .csv.")
        if not lower.endswith((".xlsx", ".csv")):
            return _upload_error(f"Файл «{name}» не в формате .xlsx или .csv. Загрузите книгу Excel или csv-таблицу.")
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            return _upload_error(f"Файл больше {MAX_UPLOAD_BYTES // 1024 // 1024} МБ.", 413)
        if lower.endswith(".xlsx") and not content.startswith(XLSX_MAGIC):
            return _upload_error(f"Файл «{name}» не похож на книгу Excel (.xlsx): повреждён или другого формата.")
        try:
            if lower.endswith(".xlsx") and _is_strategic(content):
                dataset = datastore.save_upload(content)
                return JSONResponse({"ok": True, "mode": "ready", "dataset": dataset,
                                     "questions": suggest_questions(datastore.get_table())})
            raw = read_file(content, name)
            report = profile(raw)
        except WorkbookError as exc:
            return _upload_error("Файл не прошёл проверку.", errors=exc.errors)
        upload_id = datastore.save_pending(content, name, raw.kind)
        return JSONResponse({"ok": True, "mode": "review", "upload_id": upload_id, "profile": report})

    @app.post("/api/datasets/profile")
    def reprofile(req: ProfileRequest) -> JSONResponse:
        """Профиль другого листа уже загруженного файла."""
        try:
            content, name, _ = datastore.load_pending(req.upload_id)
            return JSONResponse({"ok": True, "profile": profile(read_file(content, name), req.sheet)})
        except WorkbookError as exc:
            return _upload_error("Не удалось прочитать файл.", errors=exc.errors)

    @app.post("/api/datasets/confirm")
    def confirm(req: ConfirmRequest) -> JSONResponse:
        """Проверяет подтверждённые пользователем роли столбцов и делает отчёт активным."""
        try:
            dataset = datastore.save_flat(req.upload_id, req.spec)
        except WorkbookError as exc:
            return _upload_error("Настройки столбцов не прошли проверку.", errors=exc.errors)
        return JSONResponse({"ok": True, "dataset": dataset, "questions": suggest_questions(datastore.get_table())})

    @app.delete("/api/upload")
    def reset_dataset() -> JSONResponse:
        """Вернуться к демо-данным."""
        return JSONResponse({"ok": True, "dataset": datastore.reset(), "questions": suggest_questions(datastore.get_table())})

    return app


def _dashboard_brief(payload: dict | None) -> dict | None:
    if not payload or not payload.get("plan"):
        return None
    return {"title": payload["plan"]["title"], "created_at": payload["created_at"], "plan_source": payload["plan_source"],
            "widgets": len(payload["widgets"])}


def _consent_required() -> JSONResponse:
    return JSONResponse({"ok": False, "consent_required": True, "detail": CONSENT_DETAIL}, status_code=428)


def _dashboard_context(payload: dict | None, source: str) -> DashboardContext | None:
    """Выводы дашборда для чата: после дашборда вопросы задаются с учётом его инсайтов, их числа проверены.
    В сохранённом плане стоит настоящее имя файла, а в модель оно не уходит: в промпт текст идёт с подписью «отчёт.xlsx»."""
    if not payload or not payload.get("plan"):
        return None
    sources = {"сводка": payload["sent_to_model"]["digest"], "блоки": payload["widgets"]}
    unverified = {u["field"] for u in ((payload.get("verification") or {}).get("unverified") or [])}
    return DashboardContext(context_text(mask(payload["plan"], source), unverified=unverified), sources)


def _is_strategic(content: bytes) -> bool:
    """Книга в стратегическом формате: есть как минимум два из листов goals, facts, links."""
    try:
        import io

        import openpyxl

        check_zip_size(content)
        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True)
        names = {n.strip().lower() for n in wb.sheetnames}
        wb.close()
    except Exception:  # noqa: BLE001 — любую проблему чтения подробно опишет read_file
        return False
    return len(names & {"goals", "facts", "links"}) >= 2


def _upload_error(message: str, status_code: int = 422, errors: list[str] | None = None) -> JSONResponse:
    return JSONResponse({"ok": False, "detail": message, "errors": errors or [message]}, status_code=status_code)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:create_app", factory=True, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
