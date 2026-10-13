import copy
import json

from agent.llm import FunctionCall, LLMResponse, Usage


class FakeLLM:
    """Отдаёт заранее заданные ответы по очереди и запоминает, что ему присылали."""

    name = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, functions=None, *, function_call="auto", temperature=0.0, max_tokens=None):
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "functions": functions,
                "function_call": function_call,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def call(name, **arguments) -> LLMResponse:
    return LLMResponse(function_call=FunctionCall(name, arguments), usage=Usage(10, 5, 15), finish_reason="function_call")


def text(content) -> LLMResponse:
    return LLMResponse(content=content, usage=Usage(20, 10, 30), finish_reason="stop")


def memo_json(summary="Отклонение небольшое.", scenarios=None, checks=None, used=None) -> str:
    return json.dumps(
        {
            "diagnosis": {"summary": summary, "root_cause": "Причина", "key_segments": ["Молодёжь"]},
            "scenarios": scenarios or [],
            "manual_checks": checks or ["Проверить вручную."],
            "used_tool_calls": used or [],
        },
        ensure_ascii=False,
    )
