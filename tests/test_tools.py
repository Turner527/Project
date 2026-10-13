import json

import pandas as pd
import pytest

from data import generate
from tools import core

PRIMARY_WINDOWS = ("full", "last_6m", "last_3m")


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ---------- поиск ----------


@pytest.mark.parametrize(
    "query, goal",
    [
        ("почему мы отстаём по цели роста розничного кредитования и что делать", "Рост розничного кредитования"),
        ("Рост розничного кредитования", "Рост розничного кредитования"),
        ("как у нас с комиссионными доходами", "Рост комиссионных доходов"),
        ("качество кредитного портфеля", "Качество кредитного портфеля"),
        ("удержание клиентов", "Привлечение и удержание клиентов"),
    ],
)
def test_find_goal_in_free_text(query, goal):
    res = core.find_kpis_by_goal(query)
    assert res["goal_matches"][0]["goal"] == goal
    assert res["goal_matches"][0]["kpis"], "у цели должны быть KPI"


def test_find_kpi_by_name_and_no_match():
    res = core.find_kpis_by_goal("просрочка 30+")
    assert res["kpi_matches"][0]["kpi"] == "Доля просрочки 30+ в рознице"
    none = core.find_kpis_by_goal("погода на марсе")
    assert none["goal_matches"] == [] and none["available_goals"] and "note" in none


# ---------- отклонение и разбивка против независимого эталона из генератора ----------


@pytest.mark.parametrize("window", PRIMARY_WINDOWS)
def test_deviation_matches_reference(anomalies, window):
    for a in anomalies["anomalies"]:
        cp = a["checkpoints"][window]
        res = core.get_deviation(a["primary_kpi"], cp["from"], cp["to"])
        assert "error" not in res
        assert res["plan"] == cp["plan"] and res["fact"] == cp["fact"]
        assert res["deviation_abs"] == cp["expected"]["deviation_abs"]
        assert res["deviation_pct"] == cp["expected"]["deviation_pct"]
        assert res["unfavorable"] is True and res["assessment"] == "хуже плана"


@pytest.mark.parametrize("window", PRIMARY_WINDOWS)
def test_breakdown_matches_reference_and_names_key_segment(anomalies, window):
    for a in anomalies["anomalies"]:
        cp = a["checkpoints"][window]
        res = core.break_down_by_segment(a["primary_kpi"], cp["from"], cp["to"])
        assert res["top_segment"] == a["key_segment"]
        assert res["top_segment_contribution_pct"] == cp["expected"]["key_segment_contribution_pct"]
        top = res["segments"][0]
        assert top["deviation_pct"] == cp["expected"]["key_segment_deviation_pct"]
        assert sum(s["contribution_pct"] for s in res["segments"]) == pytest.approx(100, abs=0.5)


def test_default_period_is_last_six_months():
    res = core.get_deviation("Выдачи розничных кредитов")
    assert res["period"] == {"from": "2026-04", "to": "2026-09", "months_count": 6}


def test_related_kpis_reveal_the_driver(anomalies):
    for a in anomalies["anomalies"]:
        cp = a["checkpoints"]["last_6m"]
        res = core.get_related_kpis(a["primary_kpi"], cp["from"], cp["to"])
        by_kpi = {r["kpi"]: r for r in res["related"]}
        for driver, exp in cp["drivers"].items():
            assert driver in by_kpi, f"{a['id']}: нет связанного KPI {driver}"
            r = by_kpi[driver]
            assert r["deviation_pct"] == exp["deviation_pct"]
            assert r["most_deviating_segment"] == a["key_segment"]
            assert r["most_deviating_segment_pct"] == exp["key_segment_deviation_pct"]
            assert r["consistent_with_link"] is True, f"{a['id']}: {driver} должен подтверждать связь"


def test_goal_info():
    res = core.get_goal_info("Рост розничного кредитования")
    assert res["owner"] == "Директор департамента розничного бизнеса"
    assert res["parent_goal"] == "Рост прибыли и капитала" and res["parent_owner"] == "Финансовый директор"
    by_kpi = core.get_goal_info("Выдачи розничных кредитов")
    assert by_kpi["goal"] == "Рост розничного кредитования"
    root = core.get_goal_info("Рост прибыли и капитала")
    assert root["parent_goal"] is None and len(root["child_goals"]) >= 4


# ---------- диагностика ----------


@pytest.mark.parametrize("window", PRIMARY_WINDOWS)
def test_diagnose_matches_reference_and_scenarios(anomalies, window):
    for a in anomalies["anomalies"]:
        cp = a["checkpoints"][window]
        res = core.diagnose_deviation(a["primary_kpi"], cp["from"], cp["to"])
        exp = cp["expected"]
        assert res["deviation"]["deviation_pct"] == exp["deviation_pct"]
        assert res["breakdown"]["top_segment"] == a["key_segment"]
        assert res["owner"] == a["owner"] and res["goal"] == a["goal"]
        s1 = res["recovery_scenarios"][0]
        assert s1["scenario_id"] == "S1" and s1["segments"] == [a["key_segment"]]
        assert s1["effect_abs"] == pytest.approx(exp["s1_effect_abs"], abs=1e-6)
        s2 = res["recovery_scenarios"][1]
        assert s2["effect_abs"] == pytest.approx(s1["effect_abs"] / 2, abs=0.011)
        for s in res["recovery_scenarios"]:
            assert approx(s["effect_abs"] + s["remaining_gap_abs"], abs(res["deviation"]["deviation_abs"]), 0.011)


def test_diagnose_reports_onset_of_the_problem():
    res = core.diagnose_deviation("Выдачи розничных кредитов")
    assert res["off_plan_since"] == "2026-04" and res["months_off_plan_in_row"] == 6
    healthy = core.diagnose_deviation("Cost/Income")
    assert healthy["off_plan_since"] is None


def test_diagnose_is_compact_and_has_no_tables():
    res = core.diagnose_deviation("Выдачи розничных кредитов")
    text = json.dumps(res, ensure_ascii=False)
    assert len(text) < 6000

    def longest_list(x):
        if isinstance(x, list):
            return max([len(x)] + [longest_list(i) for i in x])
        if isinstance(x, dict):
            return max([0] + [longest_list(v) for v in x.values()])
        return 0

    assert longest_list(res) <= 10


@pytest.mark.parametrize("kpi", generate.CONTROL_KPIS)
def test_healthy_kpi_gets_no_recovery_scenarios(kpi):
    res = core.diagnose_deviation(kpi)
    assert res["deviation"]["significant"] is False
    assert res["recovery_scenarios"] == [] and "scenarios_note" in res


def test_every_tool_returns_unique_call_id():
    calls = [
        core.find_kpis_by_goal("кредитование"),
        core.get_deviation("NPS"),
        core.break_down_by_segment("NPS"),
        core.get_related_kpis("NPS"),
        core.get_goal_info("NPS"),
        core.diagnose_deviation("NPS"),
        core.get_deviation("NPS"),
    ]
    ids = [c["call_id"] for c in calls]
    assert all(i.startswith("c_") for i in ids) and len(set(ids)) == len(ids)
    assert [c["tool"] for c in calls[:6]] == [
        "find_kpis_by_goal", "get_deviation", "break_down_by_segment", "get_related_kpis", "get_goal_info",
        "diagnose_deviation",
    ]


# ---------- ошибки ----------


def test_unknown_kpi_returns_error_json_with_suggestions():
    res = core.get_deviation("Совсем левый показатель")
    assert "error" in res and "call_id" in res and "find_kpis_by_goal" in res["error"]
    near = core.get_deviation("Оборот карт")  # похоже на KPI, но не однозначно
    assert "error" in near or near["kpi"] == "Оборот по картам"


def test_kpi_name_is_forgiving_about_case_and_small_slips():
    assert core.get_deviation("выдачи розничных кредитов")["kpi"] == "Выдачи розничных кредитов"
    assert core.get_deviation("Выдачи розничных кредитов  ")["kpi"] == "Выдачи розничных кредитов"


def test_bad_periods():
    assert "ГГГГ-ММ" in core.get_deviation("NPS", "апрель")["error"]
    assert "данных нет" in core.get_deviation("NPS", "2030-01", "2030-06")["error"]
    assert "позже конца" in core.get_deviation("NPS", "2026-09", "2026-01")["error"]
    clipped = core.get_deviation("NPS", "2024-01", "2026-09")
    assert clipped["period"]["from"] == "2025-04" and "обрезан" in clipped["note"]


# ---------- семантика агрегации на крошечной витрине с числами, посчитанными вручную ----------


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    goals = pd.DataFrame(
        [
            ["Цель", None, "Босс", "Поток", "шт", "поток", "больше"],
            ["Цель", None, "Босс", "Остаток", "шт", "остаток", "больше"],
            ["Цель", None, "Босс", "Доля", "%", "доля", "меньше"],
        ],
        columns=list(generate.GOALS_COLUMNS),
    )
    rows = []
    data = {  # (kpi, segment): (план, факт) по трём месяцам
        ("Поток", "A"): ([10, 10, 10], [8, 9, 10]),
        ("Поток", "B"): ([5, 5, 5], [5, 5, 6]),
        ("Остаток", "A"): ([100, 100, 100], [90, 95, 90]),
        ("Остаток", "B"): ([50, 50, 50], [50, 50, 50]),
        ("Доля", "A"): ([4, 4, 4], [5, 5, 5]),
        ("Доля", "B"): ([2, 2, 2], [2, 2, 2]),
    }
    for (kpi, seg), (plan, fact) in data.items():
        for i, (p, f) in enumerate(zip(plan, fact)):
            rows.append([kpi, pd.Timestamp(2026, i + 1, 1), seg, p, f])
    facts = pd.DataFrame(rows, columns=list(generate.FACTS_COLUMNS))
    links = pd.DataFrame([["Поток", "Остаток", "положительно"]], columns=list(generate.LINKS_COLUMNS))
    path = tmp_path / "tiny.xlsx"
    generate.write_workbook(path, goals, facts, links)
    monkeypatch.setenv("KPI_DATA_PATH", str(path))


def test_flow_is_summed(tiny):
    r = core.get_deviation("Поток", "2026-01", "2026-03")
    assert (r["plan"], r["fact"], r["deviation_abs"], r["deviation_pct"]) == (45.0, 43.0, -2.0, -4.4)
    b = core.break_down_by_segment("Поток", "2026-01", "2026-03")
    a, bb = b["segments"][0], b["segments"][1]
    assert (a["segment"], a["deviation_abs"], a["contribution_pct"]) == ("A", -3.0, 150.0)
    assert (bb["segment"], bb["deviation_abs"], bb["contribution_pct"]) == ("B", 1.0, -50.0)


def test_stock_takes_last_month_of_period(tiny):
    r = core.get_deviation("Остаток", "2026-01", "2026-03")
    assert (r["plan"], r["fact"], r["deviation_abs"], r["deviation_pct"]) == (150.0, 140.0, -10.0, -6.7)
    r2 = core.get_deviation("Остаток", "2026-01", "2026-02")
    assert (r2["plan"], r2["fact"]) == (150.0, 145.0)


def test_ratio_is_mean_and_lower_is_better(tiny):
    r = core.get_deviation("Доля", "2026-01", "2026-03")
    assert (r["plan"], r["fact"], r["deviation_abs"], r["deviation_pct"]) == (3.0, 3.5, 0.5, 16.7)
    assert r["unfavorable"] is True and r["assessment"] == "хуже плана"
    b = core.break_down_by_segment("Доля", "2026-01", "2026-03")
    assert b["segments"][0]["segment"] == "A" and b["segments"][0]["contribution_abs"] == 0.5
    assert "среднее" in b["aggregation_note"]


def test_related_in_tiny_dataset(tiny):
    r = core.get_related_kpis("Поток", "2026-01", "2026-03")
    item = r["related"][0]
    assert item["kpi"] == "Остаток" and item["relation"] == "зависит от выбранного KPI"
    assert item["consistent_with_link"] is True  # оба хуже плана, связь положительная
