"""Тонкий адаптер между MCP-сервером инструментов и форматом functions GigaChat.

Две реализации бэкенда с одним интерфейсом:
  * McpStdioBackend — запускает tools.mcp_server подпроцессом и говорит с ним по MCP (боевой режим);
  * DirectBackend   — вызывает те же функции из tools.core напрямую (тесты, режим без сети).
Схемы в обоих случаях берутся у MCPServer, поэтому модель видит одинаковые описания.
"""
import asyncio
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

from tools.core import TOOLS
from tools.datastore import ROOT

CALL_TIMEOUT_S = 60
START_TIMEOUT_S = 30


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict


@dataclass
class ToolCallRecord:
    """Один вызов инструмента: что просили, что вернулось и сколько заняло."""

    name: str
    arguments: dict
    result: dict
    call_id: str
    error: str | None = None
    duration_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "tool": self.name,
            "arguments": self.arguments,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


class ToolBackend(Protocol):
    def list_tools(self) -> list[ToolSpec]: ...
    def call_tool(self, name: str, arguments: dict) -> dict: ...
    def close(self) -> None: ...


def _run_sync(coro):
    """Запустить корутину из синхронного кода, даже если в потоке уже есть event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _spec_from_mcp(tool: Any) -> ToolSpec:
    schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
    return ToolSpec(tool.name, " ".join((tool.description or "").split()), schema)


class DirectBackend:
    def __init__(self) -> None:
        self._specs: list[ToolSpec] | None = None

    def list_tools(self) -> list[ToolSpec]:
        if self._specs is None:
            from tools.mcp_server import build_server

            self._specs = [_spec_from_mcp(t) for t in _run_sync(build_server().list_tools())]
        return self._specs

    def call_tool(self, name: str, arguments: dict) -> dict:
        fn = TOOLS.get(name)
        if fn is None:
            return {"error": f"Неизвестный инструмент «{name}». Доступны: {', '.join(TOOLS)}."}
        try:
            return fn(**arguments)
        except TypeError as exc:  # неверные имена или типы аргументов
            return {"error": f"Неверные аргументы вызова {name}: {exc}"}

    def close(self) -> None:
        pass


class McpStdioBackend:
    """Долгоживущий подпроцесс с MCP-сервером; асинхронный клиент живёт в своём потоке."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._start_error: BaseException | None = None
        self._specs: list[ToolSpec] | None = None

    # --- жизненный цикл ---
    def _start(self) -> None:
        self._ready.clear()
        self._start_error = None
        self._thread = threading.Thread(target=self._thread_main, name="mcp-client", daemon=True)
        self._thread.start()
        if not self._ready.wait(START_TIMEOUT_S):
            raise RuntimeError("MCP-сервер не запустился за отведённое время")
        if self._start_error is not None:
            raise RuntimeError(f"MCP-сервер не запустился: {self._start_error}") from self._start_error

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        from mcp import ClientSession, StdioServerParameters, stdio_client

        env = {"PYTHONUTF8": "1"}
        if os.getenv("KPI_DATA_PATH"):
            env["KPI_DATA_PATH"] = os.environ["KPI_DATA_PATH"]
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "tools.mcp_server"], env=env, cwd=str(ROOT)
        )
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session, self._stop = session, asyncio.Event()
                    self._ready.set()
                    await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001 — любая причина должна дойти до вызывающего
            self._start_error = exc
        finally:
            self._session = None
            self._ready.set()

    def _ensure_started(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive() or self._session is None:
                self._start()

    def _submit(self, coro):
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(CALL_TIMEOUT_S)

    def close(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive() and self._loop and self._stop:
                self._loop.call_soon_threadsafe(self._stop.set)
                self._thread.join(timeout=10)
            self._thread, self._session = None, None

    # --- интерфейс бэкенда ---
    def list_tools(self) -> list[ToolSpec]:
        if self._specs is None:
            self._ensure_started()
            self._specs = [_spec_from_mcp(t) for t in self._submit(self._session.list_tools()).tools]
        return self._specs

    def call_tool(self, name: str, arguments: dict) -> dict:
        self._ensure_started()
        try:
            res = self._submit(self._session.call_tool(name, arguments))
        except Exception as exc:  # таймаут или упавший подпроцесс: в следующий раз перезапустим
            self.close()
            return {"error": f"Сбой вызова MCP-инструмента {name}: {type(exc).__name__}: {exc}"}
        text = "".join(getattr(c, "text", "") for c in res.content)
        if getattr(res, "is_error", False):
            return {"error": text or f"MCP-инструмент {name} вернул ошибку"}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {"error": f"MCP-инструмент {name} вернул не JSON: {text[:200]}"}
        return payload if isinstance(payload, dict) else {"error": "Ответ инструмента не объект JSON"}


# ---------- MCP-схема -> functions GigaChat ----------


def _simplify_property(prop: dict) -> dict:
    """Оставляет только type/description/enum/items — то, что понимает GigaChat."""
    if "anyOf" in prop:  # Optional[X] приходит как anyOf [X, null]
        options = [o for o in prop["anyOf"] if o.get("type") != "null"]
        merged = {**(options[0] if options else {"type": "string"})}
        if "description" in prop:
            merged["description"] = prop["description"]
        prop = merged
    out: dict[str, Any] = {"type": prop.get("type", "string")}
    if prop.get("description"):
        out["description"] = " ".join(prop["description"].split())
    if prop.get("enum"):
        out["enum"] = [str(v) for v in prop["enum"]]
    if out["type"] == "array":
        out["items"] = _simplify_property(prop.get("items", {"type": "string"}))
    return out


def to_gigachat_function(spec: ToolSpec) -> dict:
    schema = spec.input_schema or {}
    properties = {name: _simplify_property(p) for name, p in schema.get("properties", {}).items()}
    return {
        "name": spec.name,
        "description": spec.description,
        "parameters": {"type": "object", "properties": properties, "required": list(schema.get("required", []))},
    }


@dataclass
class ToolAdapter:
    backend: ToolBackend
    _functions: list[dict] = field(default_factory=list, init=False, repr=False)

    @property
    def functions(self) -> list[dict]:
        if not self._functions:
            self._functions = [to_gigachat_function(s) for s in self.backend.list_tools()]
        return self._functions

    def call(self, name: str, arguments: dict | None) -> ToolCallRecord:
        arguments = arguments or {}
        started = time.perf_counter()
        result = self.backend.call_tool(name, arguments)
        duration = int((time.perf_counter() - started) * 1000)
        return ToolCallRecord(
            name=name,
            arguments=arguments,
            result=result,
            call_id=result.get("call_id") or f"c_local_{int(started * 1000) % 10**8:08d}",
            error=result.get("error"),
            duration_ms=duration,
        )

    def close(self) -> None:
        self.backend.close()


def make_adapter(mode: str | None = None) -> ToolAdapter:
    """mode: 'mcp' (по умолчанию) или 'direct'. Берётся из TOOLS_BACKEND, если не передан."""
    mode = (mode or os.getenv("TOOLS_BACKEND") or "mcp").strip().lower()
    if mode == "direct":
        return ToolAdapter(DirectBackend())
    if mode == "mcp":
        return ToolAdapter(McpStdioBackend())
    raise ValueError(f"TOOLS_BACKEND должен быть 'mcp' или 'direct', получено «{mode}»")
