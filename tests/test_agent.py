import json

import pytest

from agent.llm import LLMError
from agent.loop import MAX_STEPS, Agent, extract_json
from agent.prompts import FINAL_INSTRUCTION, PROMPT_VERSION, SYSTEM_PROMPT
from agent.stub_llm import ScriptedLLM
from agent.tool_adapter import McpStdioBackend, ToolAdapter, make_adapter
from agent.verifier import STATUS_OK, STATUS_UNVERIFIED
from tests.helpers import FakeLLM, call, memo_json, text

QUESTION = "Почему мы отстаём по цели роста розничного кредитования и что делать?"


@pytest.fixture
def adapter():
    return make_adapter("direct")


# ---------- заглушка-сценарий: весь конвейер без сети ----------


def test_stub_pipeline_end_to_end(adapter, anomalies):
    result = Agent(ScriptedLLM(), adapter).run(QUESTION)
    assert result.error is None and result.memo is not None
    assert [c.name for c in result.tool_calls] == ["find_kpis_by_goal", "diagnose_deviation"]
    assert result.verification.status == STATUS_OK and result.verification.checked >= 10

    a1 = next(a for a in anomalies["anomalies"] if a["id"] == "A1")
    exp = a1["checkpoints"]["last_6m"]["expected"]
    summary = result.memo.diagnosis.summary
    assert "Молодёжь" in result.memo.diagnosis.key_segments
    assert str(exp["deviation_pct"]).replace(".", ",").lstrip("-") in summary
    assert str(exp["key_segment_contribution_pct"]).replace(".", ",") in summary
    assert "Доля одобрения заявок" in result.memo.diagnosis.root_cause

    real_ids = {c.call_id for c in result.tool_calls}
    assert result.memo.used_tool_calls and set(result.memo.used_tool_calls) <= real_ids
    assert 2 <= len(result.memo.scenarios) <= 3 and result.memo.manual_checks
    assert result.usage.total_tokens > 0 and result.latency_ms >= 0 and result.llm == "stub"
    assert result.prompt_version == PROMPT_VERSION


def test_stub_respects_period_in_question(adapter):
    result = Agent(ScriptedLLM(), adapter).run("Что с выдачами розничных кредитов за последние 3 месяца?")
    diag = result.tool_calls[-1]
    assert diag.arguments["last_months"] == 3 and diag.result["period"]["months_count"] == 3
    assert result.verification.ok


def test_stub_on_healthy_kpi_says_no_deviation(adapter):
    result = Agent(ScriptedLLM(), adapter).run("Есть ли проблемы с показателем «Cost/Income»?")
    assert result.memo.diagnosis.key_segments == [] and result.memo.scenarios == []
    assert "Значимого отклонения нет" in result.memo.diagnosis.root_cause
    assert result.verification.ok


def test_stub_on_unknown_topic_admits_lack_of_data(adapter):
    result = Agent(ScriptedLLM(), adapter).run("Какая погода на Марсе?")
    assert "данных недостаточно" in result.memo.diagnosis.root_cause.lower()
    assert result.memo.scenarios == [] and result.verification.ok


def test_agent_works_through_real_mcp_server():
    mcp = ToolAdapter(McpStdioBackend())
    try:
        result = Agent(ScriptedLLM(), mcp).run(QUESTION)
        assert result.error is None and result.verification.ok
        assert [c.name for c in result.tool_calls] == ["find_kpis_by_goal", "diagnose_deviation"]
    finally:
        mcp.close()


# ---------- цикл с управляемой моделью ----------


def test_messages_follow_gigachat_function_protocol(adapter):
    llm = FakeLLM([call("get_deviation", kpi="NPS"), text("Хватит."), text(memo_json())])
    Agent(llm, adapter).run("Как дела с NPS?")
    second = llm.calls[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "function"]
    assistant, function = second[2], second[3]
    assert assistant["function_call"] == {"name": "get_deviation", "arguments": {"kpi": "NPS"}}
    assert function["name"] == "get_deviation"
    payload = json.loads(function["content"])  # результат инструмента — JSON-строка
    assert payload["kpi"] == "NPS" and payload["call_id"].startswith("c_")
    assert llm.calls[0]["functions"] and llm.calls[0]["function_call"] == "auto"


def test_temperature_is_near_zero_and_final_call_forbids_functions(adapter):
    llm = FakeLLM([text("Данных достаточно."), text(memo_json())])
    Agent(llm, adapter).run("вопрос")
    assert all(c["temperature"] <= 0.05 for c in llm.calls)
    final = llm.calls[-1]
    assert final["function_call"] == "none" and final["max_tokens"]
    assert final["messages"][-1] == {"role": "user", "content": FINAL_INSTRUCTION}


def test_system_prompt_states_the_main_rules(adapter):
    llm = FakeLLM([text("ок"), text(memo_json())])
    Agent(llm, adapter).run("вопрос")
    system = llm.calls[0]["messages"][0]["content"]
    assert system.startswith(SYSTEM_PROMPT)
    assert "НЕ считаешь числа" in system and "дословно" in system and "данных нет" in system
    assert "2025-04" in system and "2026-09" in system and "Молодёжь" in system  # справка о данных


def test_step_limit_is_six_function_calls_then_final(adapter):
    endless = [call("get_deviation", kpi="NPS") for _ in range(10)]
    llm = FakeLLM(endless[:MAX_STEPS] + [text(memo_json())])
    result = Agent(llm, adapter).run("вопрос")
    assert MAX_STEPS == 6 and result.steps == 6 and len(result.tool_calls) == 6
    assert len(llm.calls) == 7  # 6 шагов + итоговый вызов
    assert result.memo is not None and result.error is None


def test_invalid_json_gets_exactly_one_retry_with_validation_error(adapter):
    llm = FakeLLM([text("ок"), text("это не JSON"), text(memo_json())])
    result = Agent(llm, adapter).run("вопрос")
    assert result.memo is not None and result.error is None
    assert len(llm.calls) == 3
    retry_msgs = llm.calls[2]["messages"]
    assert retry_msgs[-2] == {"role": "assistant", "content": "это не JSON"}
    assert "не прошёл проверку схемы" in retry_msgs[-1]["content"] and "невалидный JSON" in retry_msgs[-1]["content"]


def test_schema_violation_text_is_passed_to_retry(adapter):
    broken = json.dumps({"diagnosis": {"summary": "x", "root_cause": "y"}, "scenarios": [], "manual_checks": []})
    llm = FakeLLM([text("ок"), text(broken), text(memo_json())])
    result = Agent(llm, adapter).run("вопрос")
    assert result.memo is not None
    assert "manual_checks" in llm.calls[2]["messages"][-1]["content"]


def test_two_invalid_answers_end_with_error_not_exception(adapter):
    llm = FakeLLM([text("ок"), text("мусор"), text("снова мусор")])
    result = Agent(llm, adapter).run("вопрос")
    assert result.memo is None and result.verification is None
    assert "дважды" in result.error and result.raw_text == "снова мусор"
    assert len(llm.calls) == 3


def test_json_inside_markdown_fence_is_accepted(adapter):
    llm = FakeLLM([text("ок"), text("Вот записка:\n```json\n" + memo_json() + "\n```\nГотово.")])
    assert Agent(llm, adapter).run("вопрос").memo is not None
    assert json.loads(extract_json("текст {\"a\": 1} хвост")) == {"a": 1}


def test_fabricated_number_in_memo_is_marked_unverified(adapter):
    memo = memo_json(
        summary="Отклонение составило 777,77 млрд ₽.",
        scenarios=[{"name": "Сценарий", "description": "Описание", "effect_estimate": "Эффект 55,55 млн ₽",
                    "assumptions": ["Допущение"]}],
    )
    llm = FakeLLM([call("get_deviation", kpi="NPS"), text("ок"), text(memo)])
    result = Agent(llm, adapter).run("Как дела с NPS?")
    assert result.verification.status == STATUS_UNVERIFIED
    flagged = {u.value for u in result.verification.unverified}
    assert flagged == {"777,77", "55,55"}


def test_invented_call_ids_are_dropped_from_used_calls(adapter):
    llm = FakeLLM([call("get_deviation", kpi="NPS"), text("ок"), text(memo_json(used=["c_deadbeef"]))])
    result = Agent(llm, adapter).run("Как дела с NPS?")
    assert "c_deadbeef" not in result.memo.used_tool_calls
    assert result.verification.unknown_call_ids == ["c_deadbeef"]


def test_unknown_tool_requested_by_model_is_reported_back_to_it(adapter):
    llm = FakeLLM([call("hack_database", x=1), text("ок"), text(memo_json())])
    result = Agent(llm, adapter).run("вопрос")
    assert result.tool_calls[0].error.startswith("Неизвестный инструмент")
    tool_reply = json.loads(llm.calls[1]["messages"][-1]["content"])
    assert "error" in tool_reply


def test_llm_failure_is_returned_as_error(adapter):
    llm = FakeLLM([LLMError("Ключ авторизации отклонён (401).")])
    result = Agent(llm, adapter).run("вопрос")
    assert result.memo is None and result.error == "Ключ авторизации отклонён (401)."
    assert result.to_dict()["error"] and result.to_dict()["memo"] is None


def test_tokens_are_summed_over_all_calls(adapter):
    llm = FakeLLM([call("get_deviation", kpi="NPS"), text("ок"), text("мусор"), text(memo_json())])
    result = Agent(llm, adapter).run("вопрос")
    assert result.usage.total_tokens == 15 + 30 + 30 + 30


def test_prompt_version_is_a_hash_of_the_prompt_text():
    import hashlib

    from agent import prompts

    digest = hashlib.sha1((prompts.SYSTEM_PROMPT + prompts.FINAL_INSTRUCTION).encode("utf-8")).hexdigest()[:6]
    assert prompts.PROMPT_VERSION == f"{prompts.PROMPT_LABEL}-{digest}"
