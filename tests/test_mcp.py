import re

import pytest
from gigachat.models import Function

from agent.tool_adapter import DirectBackend, McpStdioBackend, ToolAdapter, make_adapter, to_gigachat_function
from tools.core import TOOLS

EXPECTED_TOOLS = {
    "describe_dataset",
    "find_metrics",
    "get_change",
    "break_down",
    "get_related_metrics",
    "get_goal_info",
    "diagnose_change",
    "get_structure",
    "what_if",
    "forecast_metric",
}


def strip_volatile(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "call_id"}


@pytest.fixture(scope="module")
def mcp_adapter():
    adapter = ToolAdapter(McpStdioBackend())
    yield adapter
    adapter.close()


@pytest.fixture(scope="module")
def direct_adapter():
    return ToolAdapter(DirectBackend())


def test_server_exposes_expected_tools_with_descriptions(direct_adapter):
    specs = {s.name: s for s in direct_adapter.backend.list_tools()}
    assert set(specs) == EXPECTED_TOOLS == set(TOOLS)
    for spec in specs.values():
        assert len(spec.description) > 40 and re.search("[а-я]", spec.description)
        assert spec.input_schema["type"] == "object"


def test_schemas_become_valid_gigachat_functions(direct_adapter):
    functions = direct_adapter.functions
    assert {f["name"] for f in functions} == EXPECTED_TOOLS
    for f in functions:
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", f["name"])
        Function.model_validate(f)  # формат принимает сам SDK GigaChat
        for prop in f["parameters"]["properties"].values():
            assert set(prop) <= {"type", "description", "enum", "items"}  # без anyOf/title/default
            assert prop["type"] in {"string", "number", "integer", "boolean", "array", "object"}
    diag = next(f for f in functions if f["name"] == "diagnose_change")
    assert diag["parameters"]["required"] == ["metric"]
    assert diag["parameters"]["properties"]["period_from"]["type"] == "string"
    assert "2026-09" in diag["parameters"]["properties"]["period_from"]["description"]


def test_optional_and_nullable_schema_is_flattened():
    spec_schema = {
        "properties": {
            "a": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None, "description": "А", "title": "A"},
            "b": {"type": "array", "items": {"type": "integer", "title": "x"}, "title": "B"},
        },
        "required": ["b"],
        "type": "object",
        "title": "Args",
    }
    from agent.tool_adapter import ToolSpec

    fn = to_gigachat_function(ToolSpec("t", "описание", spec_schema))
    assert fn["parameters"]["properties"]["a"] == {"type": "string", "description": "А"}
    assert fn["parameters"]["properties"]["b"] == {"type": "array", "items": {"type": "integer"}}
    assert fn["parameters"]["required"] == ["b"]


def test_mcp_and_direct_schemas_are_identical(mcp_adapter, direct_adapter):
    assert mcp_adapter.functions == direct_adapter.functions


@pytest.mark.parametrize(
    "name, args",
    [
        ("find_metrics", {"query": "розничное кредитование"}),
        ("diagnose_change", {"metric": "Выдачи розничных кредитов"}),
        ("break_down", {"metric": "Комиссионный доход", "period_from": "2026-05", "period_to": "2026-09"}),
        ("get_change", {"metric": "Несуществующий KPI"}),
    ],
)
def test_mcp_result_equals_direct_result(mcp_adapter, direct_adapter, name, args):
    via_mcp, direct = mcp_adapter.call(name, args), direct_adapter.call(name, args)
    assert strip_volatile(via_mcp.result) == strip_volatile(direct.result)
    assert via_mcp.call_id != direct.call_id and via_mcp.call_id.startswith("c_")
    assert via_mcp.name == name and via_mcp.arguments == args


def test_error_results_are_recorded_not_raised(direct_adapter):
    rec = direct_adapter.call("get_change", {"metric": "Совсем левый показатель"})
    assert rec.error and "не найден" in rec.error
    assert direct_adapter.call("no_such_tool", {}).error.startswith("Неизвестный инструмент")
    assert "Неверные аргументы" in direct_adapter.call("get_change", {"wrong": 1}).error


def test_mcp_bad_arguments_do_not_crash_the_session(mcp_adapter):
    bad = mcp_adapter.call("get_change", {"wrong": 1})
    assert bad.error
    good = mcp_adapter.call("get_change", {"metric": "NPS"})
    assert good.error is None and good.result["metric"] == "NPS"


def test_mcp_backend_restarts_after_close():
    adapter = ToolAdapter(McpStdioBackend())
    try:
        assert adapter.call("get_change", {"metric": "NPS"}).error is None
        adapter.backend.close()
        assert adapter.call("get_change", {"metric": "NPS"}).error is None
    finally:
        adapter.close()


def test_make_adapter_modes(monkeypatch):
    assert isinstance(make_adapter("direct").backend, DirectBackend)
    assert isinstance(make_adapter("mcp").backend, McpStdioBackend)
    monkeypatch.setenv("TOOLS_BACKEND", "direct")
    assert isinstance(make_adapter().backend, DirectBackend)
    with pytest.raises(ValueError):
        make_adapter("grpc")
