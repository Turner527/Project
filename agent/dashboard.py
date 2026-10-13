"""Дашборд по отчёту: GigaChat рекомендует блоки и пишет инсайты, код строит блоки и проверяет числа в текстах.

Что уходит в модель: сводка отчёта (tools/dashboard.build_digest) — названия показателей и измерений, итоги к базе,
ряды по периодам, вклады значений. Построчных данных в ней нет. Что возвращает модель: план (какие блоки показать,
инсайты, шаги, вопросы). Что делает код: сверяет план с отчётом, строит данные блоков, проверяет числа в текстах.
"""
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace

from pydantic import ValidationError

from agent.llm import TEMPERATURE, LLMClient, LLMError, Usage
from agent.loop import extract_json, format_validation_error
from agent.prompts import (
    CONTEXT_HEADER,
    DASHBOARD_INSTRUCTION,
    DASHBOARD_PROMPT_VERSION,
    DASHBOARD_SYSTEM_PROMPT,
    DIGEST_CLOSE,
    DIGEST_OPEN,
)
from agent.schemas import DashboardPlan
from agent.verifier import Verification, verify
from data.model import Table
from logs.store import DASHBOARD_PREFIX
from tools import dashboard as D

MAX_TOKENS = 3500
SEVERITY_WORD = {"problem": "проблема", "risk": "риск", "opportunity": "возможность", "info": "наблюдение"}


def plan_texts(plan: dict) -> dict[str, str]:
    """Все тексты очищенного плана: сюда смотрит верификатор чисел."""
    out = {"title": plan["title"], "summary": plan["summary"]}
    for i, w in enumerate(plan["widgets"], 1):
        out[f"widgets[{i}].title"] = w["title"]
        if w.get("note"):
            out[f"widgets[{i}].note"] = w["note"]
    for i, ins in enumerate(plan["insights"], 1):
        out[f"insights[{i}].title"] = ins["title"]
        out[f"insights[{i}].text"] = ins["text"]
    for i, step in enumerate(plan["next_steps"], 1):
        out[f"next_steps[{i}]"] = step
    for i, q in enumerate(plan["questions"], 1):
        out[f"questions[{i}]"] = q
    return out


def context_text(plan: dict, limit: int = 4500, unverified=frozenset()) -> str:
    """Выводы дашборда для системного промпта чата: после дашборда вопросы задаются с этим контекстом.
    unverified — поля плана (как в plan_texts), где верификатор не нашёл число: такие тексты в контекст не попадают,
    потому что промпт обещает модели, что числа в нём посчитаны кодом."""
    def skip(*keys: str) -> bool:
        return any(k in unverified for k in keys)

    lines = [CONTEXT_HEADER + " по этому же отчёту; числа в нём посчитаны кодом, их можно использовать):"]
    if not skip("title"):
        lines.append(f"Название: {plan['title']}")
    if not skip("summary"):
        lines.append(f"Кратко: {plan['summary']}")
    insights = [f"- [{SEVERITY_WORD.get(ins['severity'], 'наблюдение')}] {ins['title']} | {ins['text']}"
                for i, ins in enumerate(plan["insights"], 1) if not skip(f"insights[{i}].title", f"insights[{i}].text")]
    if insights:
        lines += ["Инсайты:", *insights]
    steps = [step for i, step in enumerate(plan["next_steps"], 1) if not skip(f"next_steps[{i}]")]
    if steps:
        lines.append("Рекомендованные шаги: " + " ".join(f"{i}) {step}" for i, step in enumerate(steps, 1)))
    return "\n".join(lines)[:limit]


@dataclass
class DashboardResult:
    source: str
    plan: dict | None
    widgets: list[dict]
    notes: list[str]
    verification: Verification | None
    plan_source: str  # "model" — план дала модель; "default" — план по умолчанию из сводки (модель не ответила или ошиблась)
    raw_text: str
    error: str | None
    usage: Usage
    latency_ms: int
    prompt_version: str
    llm: str
    digest: dict
    digest_chars: int
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    request_id: str | None = None  # id записи в журнале: по нему приходит оценка «полезно / не полезно»
    fingerprint: str | None = None  # отпечаток отчёта, по которому построен дашборд

    def to_dict(self) -> dict:
        return {
            "source": self.source, "created_at": self.created_at, "plan_source": self.plan_source, "plan": self.plan,
            "widgets": self.widgets, "notes": self.notes,
            "verification": self.verification.to_dict() if self.verification else None,
            "error": self.error, "raw_text": self.raw_text if self.plan_source != "model" else "",
            "llm": self.llm, "prompt_version": self.prompt_version,
            "tokens": {"prompt": self.usage.prompt_tokens, "completion": self.usage.completion_tokens,
                       "total": self.usage.total_tokens},
            "latency_ms": self.latency_ms,
            "sent_to_model": {"chars": self.digest_chars, "metrics": len(self.digest["metrics"]),
                              "dimensions": len(self.digest["dimensions"]), "digest": self.digest},
            "request_id": self.request_id, "fingerprint": self.fingerprint,
        }

    def log_view(self) -> dict:
        """Запись для журнала пилота в том же виде, что и у вопросов (см. logs/store.py)."""
        ver = self.verification.to_dict() if self.verification else None
        return {"question": DASHBOARD_PREFIX + self.source, "tool_calls": [], "memo": self.plan, "verification": ver,
                "latency_ms": self.latency_ms,
                "tokens": {"prompt": self.usage.prompt_tokens, "completion": self.usage.completion_tokens,
                           "total": self.usage.total_tokens},
                "prompt_version": self.prompt_version, "llm": self.llm, "error": self.error}


class DashboardAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, table: Table) -> DashboardResult:
        started = time.perf_counter()
        digest = D.build_digest(table)
        text = D.digest_text(digest)
        usage, raw, plan, error = Usage(), "", None, None
        try:
            plan, raw, usage = self._ask(text)
            if plan is None:
                error = "GigaChat не вернул план дашборда в нужном формате, показан дашборд по умолчанию. " + raw_problem(raw)
        except LLMError as exc:
            error = f"{exc} Показан дашборд по умолчанию, без рекомендаций модели."
        except Exception as exc:  # noqa: BLE001 — любой сбой виден пользователю и попадает в журнал
            error = f"Внутренняя ошибка при обращении к модели: {type(exc).__name__}: {exc}"
        plan_source = "model" if plan is not None else "default"
        plan_dict = plan.model_dump() if plan is not None else D.fallback_plan(digest)
        notes: list[str] = []
        widgets: list[dict] = []
        clean = None
        try:
            built = D.assemble(table, plan_dict, digest)
            clean, widgets, notes = built["plan"], built["widgets"], built["notes"]
            if not widgets:
                error = (error + " " if error else "") + "Ни один блок не удалось построить по этому отчёту."
        except Exception as exc:  # noqa: BLE001
            error = (error + " " if error else "") + f"Не удалось построить дашборд: {type(exc).__name__}: {exc}"
        verification = None
        if clean is not None:
            sources = [SimpleNamespace(call_id="сводка", result=digest)] + [SimpleNamespace(call_id=w["id"], result=w) for w in widgets]
            verification = verify(plan_texts(clean), sources)
        return DashboardResult(
            source=table.source, plan=clean, widgets=widgets, notes=notes, verification=verification, plan_source=plan_source,
            raw_text=raw, error=error, usage=usage, latency_ms=int((time.perf_counter() - started) * 1000),
            prompt_version=DASHBOARD_PROMPT_VERSION, llm=self.llm.name, digest=digest, digest_chars=len(text))

    def _ask(self, digest_text: str):
        """Один вызов модели со строгим JSON и одним ретраем с текстом ошибки схемы: (план | None, сырой ответ, токены)."""
        messages = [
            {"role": "system", "content": DASHBOARD_SYSTEM_PROMPT},
            {"role": "user", "content": f"{DASHBOARD_INSTRUCTION}\n\n{DIGEST_OPEN}\n{digest_text}\n{DIGEST_CLOSE}"},
        ]
        usage, raw, problem = Usage(), "", ""
        for attempt in range(2):
            resp = self.llm.chat(messages, None, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
            usage, raw = usage + resp.usage, resp.content or ""
            try:
                return DashboardPlan.model_validate_json(extract_json(raw)), raw, usage
            except ValidationError as exc:
                problem = format_validation_error(exc)
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Ответ не прошёл проверку схемы: {problem}. "
                                                "Верни только исправленный JSON-объект той же структуры, без пояснений."},
                ]
        return None, f"{problem}\n{raw}", usage


def raw_problem(raw: str) -> str:
    """Первая строка сырого ответа при ошибке — причина (остальное — сам ответ модели)."""
    return raw.split("\n", 1)[0][:400] if raw else ""
