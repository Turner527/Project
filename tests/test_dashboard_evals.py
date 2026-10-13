import csv
import json

import pytest

from agent.llm import LLMError
from agent.stub_llm import ScriptedLLM
from evals import run_dashboard_evals as R
from tests.helpers import FakeLLM, text


def quiet(*_):
    return None


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("KPI_DATA_PATH", "будет заменён прогоном и восстановлен после теста")

    def go(llm, names):
        return R.run_sets(llm, names, tmp_path, progress=quiet)

    return go


BRANCHES_PLAN = {
    "title": "Сеть отделений", "summary": "Расходы растут в отделении «Новосибирск».",
    "widgets": [{"id": "w1", "type": "kpi", "metric": "Расходы", "title": "Расходы"},
                {"id": "w2", "type": "trend", "metric": "Расходы", "title": "Динамика расходов", "dimension": "Отделение"},
                {"id": "w3", "type": "breakdown", "metric": "Расходы", "title": "Расходы по отделениям", "dimension": "Отделение"},
                {"id": "w4", "type": "kpi", "metric": "Несуществующий показатель", "title": "Лишнее"}],
    "insights": [{"title": "Новосибирск", "text": "Расходы растут в отделении «Новосибирск».", "severity": "problem", "widgets": ["w1"]}],
    "next_steps": ["Проверить отделение."], "questions": [],
}


def test_expected_causes_come_from_the_planted_anomalies():
    assert R.expected_segments("demo") == ["Молодёжь", "Самозанятые и ИП", "Премиум", "Зарплатные"]
    assert R.expected_segments("marketing") == ["Поиск"] and "Москва Север" in R.expected_segments("branches")


def test_stub_dashboards_pass_every_check_on_all_sets(run):
    items = run(ScriptedLLM(), R.SETS["all"])
    assert [i["dataset"] for i in items] == ["demo", "marketing", "support", "branches"]
    for item in items:
        assert item["plan_valid"] and item["causes_named"] == 1.0 and item["unverified"] == 0 and item["all_kinds"], item
        assert item["blocks_kept"] == 1.0 and item["missing"] == []
    summary = R.aggregate(items)
    assert summary["n"] == 4 and summary["plan_valid"] == 1.0 and summary["unverified_share"] == 0.0
    assert summary["causes_named"] == 1.0 and summary["all_kinds"] == 1.0 and summary["avg_tokens"] > 0


def test_a_model_plan_is_scored_on_causes_dropped_blocks_and_kinds(run):
    item = run(FakeLLM([text(json.dumps(BRANCHES_PLAN, ensure_ascii=False))]), ["branches"])[0]
    assert item["plan_valid"] and item["proposed"] == 4 and item["widgets"] == 3 and item["blocks_kept"] == 0.75  # лишний блок отброшен
    assert item["named"] == ["Новосибирск"] and item["missing"] == ["Москва Север", "Екатеринбург"]
    assert item["causes_named"] == pytest.approx(1 / 3) and item["all_kinds"] and item["kinds"] == ["breakdown", "kpi", "trend"]
    assert any("Несуществующий показатель" in n for n in item["notes"])


def test_invented_numbers_are_counted(run):
    plan = {**BRANCHES_PLAN, "summary": "Расходы выросли на 77,7% в отделении «Новосибирск»."}
    item = run(FakeLLM([text(json.dumps(plan, ensure_ascii=False))]), ["branches"])[0]
    assert item["unverified"] == 1 and item["checked"] >= 1
    assert R.aggregate([item])["unverified_share"] == item["unverified"] / item["checked"] > 0


def test_a_plan_of_tiles_only_fails_the_kinds_check(run):
    tiles = {**BRANCHES_PLAN, "widgets": [{"id": f"w{i}", "type": "kpi", "metric": m, "title": m}
                                          for i, m in enumerate(("Расходы", "Прибыль", "Сделки"), 1)]}
    item = run(FakeLLM([text(json.dumps(tiles, ensure_ascii=False))]), ["branches"])[0]
    assert item["plan_valid"] and item["widgets"] == 3 and item["kinds"] == ["kpi"] and item["all_kinds"] is False


def test_a_short_plan_is_topped_up_by_defaults_and_scored_on_what_the_model_proposed(run):
    short = {**BRANCHES_PLAN, "widgets": [BRANCHES_PLAN["widgets"][0]]}
    item = run(FakeLLM([text(json.dumps(short, ensure_ascii=False))]), ["branches"])[0]
    assert item["plan_valid"] and item["proposed"] == 1 and item["widgets"] >= 3 and item["blocks_kept"] == 1.0
    assert any("дополнен блоками по умолчанию" in n for n in item["notes"])


def test_a_failed_model_scores_zero_even_though_a_default_dashboard_is_shown(run):
    for llm in (FakeLLM([text("не JSON"), text("опять не JSON")]), FakeLLM([LLMError("Превышен лимит запросов (429).")])):
        item = run(llm, ["marketing"])[0]
        assert item["plan_valid"] is False and item["error"] and item["widgets"] >= 3  # пользователь без пустого экрана
        assert item["causes_named"] == 0.0 and item["blocks_kept"] == 0.0 and item["all_kinds"] is False
        assert item["unverified"] == 0 and item["named"] == [] and item["missing"] == ["Поиск"]
    summary = R.aggregate([item])
    assert summary["plan_valid"] == 0.0 and summary["causes_named"] == 0.0


def test_aggregate_ignores_metrics_that_do_not_apply():
    items = [{"plan_valid": True, "blocks_kept": None, "unverified": 0, "checked": 0, "causes_named": 0.5, "all_kinds": True,
              "latency_ms": 1000, "tokens": 100},
             {"plan_valid": True, "blocks_kept": 0.5, "unverified": 1, "checked": 4, "causes_named": 1.0, "all_kinds": False,
              "latency_ms": 3000, "tokens": 300}]
    s = R.aggregate(items)
    assert s["blocks_kept"] == 0.5 and s["unverified_share"] == 0.25 and s["causes_named"] == 0.75 and s["all_kinds"] == 0.5
    assert s["avg_latency_ms"] == 2000 and s["avg_tokens"] == 200 and R.aggregate([])["plan_valid"] is None


def test_cli_writes_a_row_and_compares_with_the_previous_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KPI_DATA_PATH", "значение до прогона")
    args = ["--backend", "stub", "--set", "flat", "--results", str(tmp_path / "r.csv"), "--details-dir", str(tmp_path / "d")]
    assert R.main(args + ["--note", "первый"]) == 0
    first = capsys.readouterr().out
    assert "Предыдущих прогонов этой модели на этих наборах нет" in first and "заложенные причины названы" in first
    assert R.main(args + ["--note", "второй"]) == 0
    second = capsys.readouterr().out
    assert "Δ +0.0 п.п. к прогону" in second
    rows = list(csv.DictReader((tmp_path / "r.csv").open(encoding="utf-8")))
    assert [r["note"] for r in rows] == ["первый", "второй"] and rows[0]["llm"] == "stub" and rows[0]["set"] == "flat"
    assert rows[0]["plan_valid"] == "1.0" and rows[0]["causes_named"] == "1.0" and rows[0]["n"] == "3"
    details = sorted((tmp_path / "d").glob("dashboards-*.json"))
    assert len(details) >= 1 and json.loads(details[0].read_text("utf-8"))["items"][0]["dataset"] == "marketing"
    assert __import__("os").environ["KPI_DATA_PATH"] == "значение до прогона"  # окружение после прогона восстановлено


def test_cli_explains_a_missing_model(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LLM_BACKEND", "gigachat")
    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    assert R.main(["--results", str(tmp_path / "r.csv"), "--details-dir", str(tmp_path / "d")]) == 2
    assert "GIGACHAT_CREDENTIALS" in capsys.readouterr().err and not (tmp_path / "r.csv").exists()
