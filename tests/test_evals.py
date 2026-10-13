import csv
import json

import pytest

from agent.loop import Agent
from agent.schemas import Diagnosis, Memo
from agent.stub_llm import ScriptedLLM
from agent.tool_adapter import make_adapter
from evals import run_evals
from evals.build_questions import OUT as QUESTIONS_PATH
from evals.build_questions import build
from evals.scoring import aggregate, evaluate, mentioned, score_cause, score_numbers
from tests.conftest import ANOMALIES_JSON
from tests.helpers import FakeLLM, call, memo_json, text
from tools import core


def memo(summary="", root="", segments=(), ):
    return Memo(diagnosis=Diagnosis(summary=summary or "-", root_cause=root or "-", key_segments=list(segments)),
                scenarios=[], manual_checks=["проверить"], used_tool_calls=[])


# ---------- набор вопросов ----------


def test_questions_are_in_sync_with_anomalies_and_well_formed():
    anomalies = json.loads(ANOMALIES_JSON.read_text("utf-8"))
    committed = json.loads(QUESTIONS_PATH.read_text("utf-8"))
    assert committed["questions"] == json.loads(json.dumps(build(anomalies), ensure_ascii=False))
    questions = committed["questions"]
    assert 30 <= len(questions) <= 40 and len({q["id"] for q in questions}) == len(questions)
    assert {q["anomaly"] for q in questions} == {"A1", "A2", "A3", "A4", "C1", "C2", "C3", "C4"}
    for q in questions:
        assert q["expect"]["numbers"], q["id"]
        if q["type"] == "control":
            assert q["expect"]["key_segment"] is None and "проблемы" in q["question"]
        else:
            assert q["expect"]["key_segment"] and q["expect"]["cause_label"]


def test_expected_numbers_come_from_the_reference_not_from_the_tools(anomalies):
    """Эталон посчитан генератором независимо; инструменты обязаны с ним совпасть."""
    questions = {q["id"]: q for q in json.loads(QUESTIONS_PATH.read_text("utf-8"))["questions"]}
    for a in anomalies["anomalies"]:
        q = questions[f"{a['id']}-explain_kpi"]
        window = q["expect"]["window"]
        dev = core.diagnose_change(a["primary_kpi"], window["from"], window["to"])
        by_name = {n["name"]: n["value"] for n in q["expect"]["numbers"]}
        assert dev["deviation"]["deviation_pct"] == by_name["deviation_pct"]
        assert dev["dimensions"][0]["top"][0]["contribution_pct"] == by_name["key_segment_contribution_pct"]


# ---------- метрика «причина названа верно» ----------


def test_cause_requires_segment_and_driver_when_driver_expected():
    expect = {"key_segment": "Молодёжь", "driver_kpis_any": ["Доля одобрения заявок"]}
    assert score_cause(memo("Упало одобрение заявок в Молодёжь", "Одобрение заявок", ["Молодёжь"]), expect)
    assert score_cause(memo("...", "Падение доли одобрения заявок в сегменте «Молодёжь»"), expect)
    assert not score_cause(memo("Одобрение заявок упало", "Одобрение заявок", ["Премиум"]), expect)  # не тот сегмент
    assert not score_cause(memo("Просто отставание в сегменте", "Сегмент", ["Молодёжь"]), expect)  # нет драйвера
    assert not score_cause(None, expect)


def test_cause_does_not_require_driver_when_question_is_only_about_segment():
    expect = {"key_segment": "Молодёжь", "driver_kpis_any": []}
    assert score_cause(memo("-", "-", ["Молодёжь"]), expect)


def test_control_question_expects_no_problem_segments():
    expect = {"key_segment": None, "driver_kpis_any": []}
    assert score_cause(memo("Отклонения нет", "Значимого отклонения нет"), expect)
    assert not score_cause(memo("Проблема в ИП", "Сегмент ИП", ["Самозанятые и ИП"]), expect)


def test_kpi_mention_tolerates_paraphrase_but_not_other_kpis():
    assert mentioned("Доля одобрения заявок", "падение одобрения заявок")
    assert mentioned("Остатки на счетах и вкладах физлиц", "снижение остатков на вкладах и счетах физлиц")
    assert not mentioned("Оборот по картам", "упал комиссионный доход")
    assert mentioned("NPS", "NPS снизился")


# ---------- метрика «числа совпадают» ----------


def test_numbers_all_found_missing_and_rounding():
    expect = {"numbers": [{"name": "deviation_pct", "value": -9.2}, {"name": "share", "value": 89.7}]}
    ok, missing = score_numbers(memo("Отклонение −9,2%, вклад 89,7%"), expect)
    assert ok is True and missing == []
    ok, missing = score_numbers(memo("Отклонение −9,2%, вклад около 85%"), expect)
    assert ok is False and missing == ["share"]
    assert score_numbers(memo("Отклонение 9%, вклад 90%"), expect)[0] is True  # целые допускают округление до 0,5
    assert score_numbers(memo("Отклонение 9,2%, вклад 89,7%"), expect)[0] is True  # знак не важен


def test_numbers_any_of_and_edge_cases():
    expect = {"numbers": [{"name": "driver", "any_of": [-4.1, -5.0]}]}
    assert score_numbers(memo("Остатки −5,0%"), expect)[0] is True
    assert score_numbers(memo("Остатки −7,5%"), expect)[0] is False
    assert score_numbers(memo("x"), {"numbers": []}) == (None, [])
    assert score_numbers(None, expect) == (False, ["driver"])


# ---------- метрики умеют падать ----------


QUESTION = {"id": "T", "anomaly": "A1", "type": "t", "question": "Что с «Выдачи розничных кредитов»?",
            "expect": {"key_segment": "Молодёжь", "driver_kpis_any": [], "numbers": [{"name": "d", "value": -9.2}]}}


def run_fake(*responses):
    return Agent(FakeLLM(list(responses)), make_adapter("direct")).run(QUESTION["question"])


def test_evaluate_scores_wrong_and_fabricated_answers_as_failures():
    wrong = memo_json(summary="Отклонение 12,34% в сегменте Премиум.")
    wrong = json.loads(wrong)
    wrong["diagnosis"]["key_segments"] = ["Премиум"]
    result = run_fake(call("get_change", metric="Выдачи розничных кредитов"), text("ок"), text(json.dumps(wrong)))
    item = evaluate(result, QUESTION)
    assert item["cause_ok"] is False and item["numbers_ok"] is False
    assert item["status"] == "есть непроверенные числа" and "12,34" in item["unverified"]


def test_evaluate_handles_failed_runs():
    from agent.llm import LLMError

    item = evaluate(run_fake(LLMError("сбой")), QUESTION)
    assert item["cause_ok"] is False and item["numbers_ok"] is False and item["status"] == "ошибка"
    assert item["error"] == "сбой"


def test_aggregate_shares_and_groups():
    def item(anomaly, cause, numbers, status="проверен"):
        return {"anomaly": anomaly, "cause_ok": cause, "numbers_ok": numbers, "status": status,
                "latency_ms": 1000, "tokens": 100}

    items = [item("A1", True, True), item("A1", False, False, "есть непроверенные числа"),
             item("A2", True, None), item("A2", False, False, "ошибка")]
    s = aggregate(items)
    assert s["n"] == 4 and s["errors"] == 1
    assert s["cause_accuracy"] == 0.5 and s["numbers_accuracy"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["unverified_share"] == pytest.approx(1 / 3, abs=1e-3)  # ошибка в знаменатель не входит
    assert s["by_group"]["A1"] == {"n": 2, "cause_accuracy": 0.5, "numbers_accuracy": 0.5}
    assert s["by_group"]["A2"]["numbers_accuracy"] == 0.0


# ---------- прогонщик целиком ----------


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(run_evals, "load_env", lambda: None)
    return ["--backend", "stub", "--tools", "direct", "--results", str(tmp_path / "r.csv"),
            "--details-dir", str(tmp_path / "details")], tmp_path


def test_stub_reaches_perfect_score_on_all_questions(paths):
    """Заглушка — эталонный сценарий: на ней проверяется согласованность инструментов, верификатора и эталонов."""
    args, tmp = paths
    questions = run_evals.load_questions()
    adapter = make_adapter("direct")
    items = run_evals.run_all(Agent(ScriptedLLM(), adapter), questions, progress=lambda *_: None)
    summary = aggregate(items)
    failed = [(i["id"], i["missing_numbers"], i["summary"][:100]) for i in items if not i["cause_ok"] or i["numbers_ok"] is False]
    assert not failed, failed
    assert summary["cause_accuracy"] == 1.0 and summary["numbers_accuracy"] == 1.0
    assert summary["unverified_share"] == 0.0 and summary["errors"] == 0


def test_runner_writes_table_and_details_and_compares_with_previous_run(paths, capsys):
    args, tmp = paths
    assert run_evals.main(args + ["--limit", "4", "--note", "первый"]) == 0
    assert run_evals.main(args + ["--ids", "A1-why_goal,A2-why_goal", "--note", "второй"]) == 0
    out = capsys.readouterr().out
    assert "Предыдущих прогонов этой модели нет" in out and "Δ" in out and "п.п. к прогону" in out

    with (tmp / "r.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["note"] for r in rows] == ["первый", "второй"] and [r["n"] for r in rows] == ["4", "2"]
    first = rows[0]
    assert first["llm"] == "stub" and first["prompt_version"].startswith("v2-") and first["date"].endswith("+00:00")
    assert first["dataset"] == "kpi.xlsx" and float(first["cause_accuracy"]) == 1.0
    assert list(first) == run_evals.COLUMNS

    details = json.loads(next((tmp / "details").glob("*.json")).read_text("utf-8"))
    assert len(details["items"]) in (2, 4) and details["summary"]["by_group"]


def test_runner_reports_config_problems(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(run_evals, "load_env", lambda: None)
    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    code = run_evals.main(["--backend", "gigachat", "--results", str(tmp_path / "r.csv")])
    assert code == 2 and "GIGACHAT_CREDENTIALS" in capsys.readouterr().err
    assert not (tmp_path / "r.csv").exists()
