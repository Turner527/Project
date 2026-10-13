"""Цикл агента: модель выбирает инструменты (до 6 шагов), затем отдельным вызовом возвращает JSON-записку."""
import json
import re
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from agent.llm import TEMPERATURE, LLMClient, LLMError, Usage
from agent.prompts import FINAL_INSTRUCTION, PROMPT_VERSION, build_system_prompt
from agent.schemas import Memo
from agent.tool_adapter import ToolAdapter, ToolCallRecord
from agent.verifier import Verification, verify
from tools import datastore

MAX_STEPS = 6
FINAL_MAX_TOKENS = 2000


@dataclass
class AgentResult:
    question: str
    memo: Memo | None
    verification: Verification | None
    tool_calls: list[ToolCallRecord]
    raw_text: str
    error: str | None
    usage: Usage
    latency_ms: int
    steps: int
    prompt_version: str
    llm: str
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "memo": self.memo.model_dump() if self.memo else None,
            "verification": self.verification.to_dict() if self.verification else None,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "raw_text": self.raw_text,
            "error": self.error,
            "tokens": {
                "prompt": self.usage.prompt_tokens,
                "completion": self.usage.completion_tokens,
                "total": self.usage.total_tokens,
            },
            "latency_ms": self.latency_ms,
            "steps": self.steps,
            "prompt_version": self.prompt_version,
            "llm": self.llm,
        }


def extract_json(text: str) -> str:
    """Достаёт JSON-объект из ответа, даже если модель обернула его в ```json или добавила пояснение."""
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
    if fence:
        body = fence.group(1).strip()
    start, end = body.find("{"), body.rfind("}")
    return body[start: end + 1] if start != -1 and end > start else body


def format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:6]:
        where = ".".join(str(p) for p in err["loc"]) or "ответ"
        parts.append(f"{where}: {'невалидный JSON' if err['type'] == 'json_invalid' else err['msg']}")
    return "; ".join(parts)


class Agent:
    def __init__(self, llm: LLMClient, adapter: ToolAdapter, max_steps: int = MAX_STEPS):
        self.llm = llm
        self.adapter = adapter
        self.max_steps = max_steps

    def run(self, question: str) -> AgentResult:
        started = time.perf_counter()
        calls: list[ToolCallRecord] = []
        usage, steps, raw = Usage(), 0, ""
        memo: Memo | None = None
        error: str | None = None
        try:
            functions = self.adapter.functions
            messages = [
                {"role": "system", "content": build_system_prompt(datastore.get_dataset().summary())},
                {"role": "user", "content": question},
            ]
            for _ in range(self.max_steps):
                resp = self.llm.chat(messages, functions, function_call="auto", temperature=TEMPERATURE)
                usage, steps = usage + resp.usage, steps + 1
                if resp.function_call is None:  # модель считает, что данных достаточно
                    if resp.content:
                        messages.append({"role": "assistant", "content": resp.content})
                    break
                rec = self.adapter.call(resp.function_call.name, resp.function_call.arguments)
                calls.append(rec)
                messages.append(
                    {
                        "role": "assistant",
                        "content": resp.content or "",
                        "function_call": {"name": resp.function_call.name, "arguments": rec.arguments},
                        "functions_state_id": resp.functions_state_id,
                    }
                )
                messages.append(
                    {"role": "function", "name": rec.name, "content": json.dumps(rec.result, ensure_ascii=False)}
                )
            memo, raw, error, usage = self._final(messages, functions, usage)
        except LLMError as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 — сбой инструментов или MCP тоже должен быть виден и залогирован
            error = f"Внутренняя ошибка агента: {type(exc).__name__}: {exc}"

        verification = None
        if memo is not None:
            verification = verify(memo.texts(), calls, question, memo.used_tool_calls)
            real = {c.call_id for c in calls}
            cited = [c for c in memo.used_tool_calls if c in real]
            derived = [cid for ids in verification.sources.values() for cid in ids]
            memo.used_tool_calls = list(dict.fromkeys(cited + derived))
        return AgentResult(
            question=question,
            memo=memo,
            verification=verification,
            tool_calls=calls,
            raw_text=raw,
            error=error,
            usage=usage,
            latency_ms=int((time.perf_counter() - started) * 1000),
            steps=steps,
            prompt_version=PROMPT_VERSION,
            llm=self.llm.name,
        )

    def _final(self, messages: list[dict], functions: list[dict], usage: Usage):
        """Второй вызов модели: строго JSON по схеме Memo; при невалидном ответе — один ретрай с текстом ошибки."""
        msgs = messages + [{"role": "user", "content": FINAL_INSTRUCTION}]
        text = ""
        for attempt in range(2):
            resp = self.llm.chat(
                msgs, functions, function_call="none", temperature=TEMPERATURE, max_tokens=FINAL_MAX_TOKENS
            )
            usage, text = usage + resp.usage, resp.content or ""
            try:
                return Memo.model_validate_json(extract_json(text)), text, None, usage
            except ValidationError as exc:
                problem = format_validation_error(exc)
            if attempt == 0:
                msgs = msgs + [
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": f"Ответ не прошёл проверку схемы: {problem}. "
                        "Верни только исправленный JSON-объект той же структуры, без пояснений.",
                    },
                ]
        return None, text, f"Модель дважды вернула JSON, не соответствующий схеме: {problem}", usage
