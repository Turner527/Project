import json

import pandas as pd
import pytest

from data import generate, generate_samples
from tests.helpers import activate_flat
from tools import core

PRIMARY_WINDOWS = ("full", "last_6m", "last_3m")


def strip(result):
    return {k: v for k, v in result.items() if k != "call_id"}


def longest_list(x):
    if isinstance(x, list):
        return max([len(x)] + [longest_list(i) for i in x])
    if isinstance(x, dict):
        return max([0] + [longest_list(v) for v in x.values()])
    return 0


def by_value(rows):
    return {row["value"]: row for row in rows}


# ======================================================================================================================
# Стратегический формат (три листа): сверка с независимым эталоном из генератора данных
# ======================================================================================================================


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
    res = core.find_metrics(query)
    assert res["goal_matches"][0]["goal"] == goal
    assert res["goal_matches"][0]["metrics"], "у цели должны быть показатели"


def test_find_metric_by_name_value_and_no_match():
    res = core.find_metrics("просрочка 30+")
    assert res["metric_matches"][0]["metric"] == "Доля просрочки 30+ в рознице"
    assert core.find_metrics("почему мы отстаём по Молодёжи")["value_matches"] == [{"dimension": "Сегмент", "value": "Молодёжь"}]
    none = core.find_metrics("погода на марсе")
    assert none["metric_matches"] == none["goal_matches"] == none["value_matches"] == []
    assert "Выдачи розничных кредитов" in none["available_metrics"] and "note" in none


def test_first_metric_of_each_goal_is_its_headline_metric(anomalies):
    for a in anomalies["anomalies"]:
        goal = core.find_metrics(a["goal"])["goal_matches"][0]
        assert goal["metrics"][0]["metric"] == a["primary_kpi"], f"{a['id']}: главный показатель цели должен идти первым"


def test_full_metric_name_inside_a_long_question_is_an_exact_match():
    """Регрессия из evals: посторонние слова вопроса («вклад» ~ «вкладах») не должны ронять точное совпадение."""
    q = "Какой сегмент сильнее всего влияет на отклонение показателя «Доля просрочки 30+ в рознице» и каков его вклад?"
    hit = core.find_metrics(q)["metric_matches"][0]
    assert hit["metric"] == "Доля просрочки 30+ в рознице" and hit["match_score"] == 1.0


@pytest.mark.parametrize("window", PRIMARY_WINDOWS)
def test_change_matches_reference(anomalies, window):
    for a in anomalies["anomalies"]:
        cp = a["checkpoints"][window]
        res = core.get_change(a["primary_kpi"], period_from=cp["from"], period_to=cp["to"])
        assert "error" not in res
        assert res["baseline"]["type"] == "plan"
        assert res["base"] == cp["plan"] and res["fact"] == cp["fact"]
        assert res["deviation_abs"] == cp["expected"]["deviation_abs"]
        assert res["deviation_pct"] == cp["expected"]["deviation_pct"]
        assert res["unfavorable"] is True and res["assessment"] == "хуже плана"


@pytest.mark.parametrize("window", PRIMARY_WINDOWS)
def test_breakdown_matches_reference_and_names_key_segment(anomalies, window):
    for a in anomalies["anomalies"]:
        cp = a["checkpoints"][window]
        res = core.break_down(a["primary_kpi"], period_from=cp["from"], period_to=cp["to"])
        assert res["dimension"] == "Сегмент"  # единственное измерение выбирается само
        assert res["top_value"] == a["key_segment"]
        assert res["top_value_share_pct"] == cp["expected"]["key_segment_contribution_pct"]
        top = res["values"][0]
        assert top["change_pct"] == cp["expected"]["key_segment_deviation_pct"]
        assert sum(v["contribution_pct"] for v in res["values"]) == pytest.approx(100, abs=0.5)


def test_default_period_is_last_six_months():
    res = core.get_change("Выдачи розничных кредитов")
    assert res["period"] == {"from": "2026-04", "to": "2026-09", "periods": 6}


def test_related_metrics_reveal_the_driver(anomalies):
    for a in anomalies["anomalies"]:
        cp = a["checkpoints"]["last_6m"]
        res = core.get_related_metrics(a["primary_kpi"], period_from=cp["from"], period_to=cp["to"])
        by_metric = {x["metric"]: x for x in res["related"]}
        for driver, exp in cp["drivers"].items():
            assert driver in by_metric, f"{a['id']}: нет связанного показателя {driver}"
            x = by_metric[driver]
            assert x["deviation_pct"] == exp["deviation_pct"]
            assert x["top_value"] == a["key_segment"]
            assert x["top_value_change_pct"] == exp["key_segment_deviation_pct"]
            assert x["consistent_with_link"] is True, f"{a['id']}: {driver} должен подтверждать связь"


def test_goal_info():
    res = core.get_goal_info("Рост розничного кредитования")
    assert res["available"] and res["owner"] == "Директор департамента розничного бизнеса"
    assert res["parent_goal"] == "Рост прибыли и капитала" and res["parent_owner"] == "Финансовый директор"
    by_metric = core.get_goal_info("Выдачи розничных кредитов")
    assert by_metric["goal"] == "Рост розничного кредитования"
    root = core.get_goal_info("Рост прибыли и капитала")
    assert root["parent_goal"] is None and len(root["child_goals"]) >= 4
    assert "не найдены" in core.get_goal_info("Совсем левая цель")["error"]


@pytest.mark.parametrize("window", PRIMARY_WINDOWS)
def test_diagnose_matches_reference_and_scenarios(anomalies, window):
    for a in anomalies["anomalies"]:
        cp = a["checkpoints"][window]
        res = core.diagnose_change(a["primary_kpi"], period_from=cp["from"], period_to=cp["to"])
        exp = cp["expected"]
        assert res["deviation"]["deviation_pct"] == exp["deviation_pct"]
        assert res["primary_dimension"] == "Сегмент"
        assert res["dimensions"][0]["top"][0]["value"] == a["key_segment"]
        assert res["dimensions"][0]["top"][0]["contribution_pct"] == exp["key_segment_contribution_pct"]
        assert res["owner"] == a["owner"] and res["goal"] == a["goal"]
        s1, s2 = res["recovery_scenarios"][:2]
        assert s1["scenario_id"] == "S1" and s1["values"] == [a["key_segment"]]
        assert s1["effect_abs"] == pytest.approx(exp["s1_effect_abs"], abs=1e-6)
        assert s2["effect_abs"] == pytest.approx(s1["effect_abs"] / 2, abs=0.011)
        for s in res["recovery_scenarios"]:
            assert s["effect_abs"] + s["remaining_gap_abs"] == pytest.approx(abs(res["deviation"]["deviation_abs"]), abs=0.011)


def test_diagnose_reports_onset_of_the_problem():
    res = core.diagnose_change("Выдачи розничных кредитов")
    assert res["unfavorable_since"] == "2026-04" and res["unfavorable_periods_in_row"] == 6 and res["interval"] == "мес."
    healthy = core.diagnose_change("Cost/Income")
    assert healthy["unfavorable_since"] is None


@pytest.mark.parametrize("kpi", generate.CONTROL_KPIS)
def test_healthy_metric_gets_no_recovery_scenarios(kpi):
    res = core.diagnose_change(kpi)
    assert res["deviation"]["significant"] is False
    assert res["recovery_scenarios"] == [] and "scenarios_note" in res


def test_diagnose_is_compact_and_has_no_tables():
    res = core.diagnose_change("Выдачи розничных кредитов")
    assert len(json.dumps(res, ensure_ascii=False)) < 6000
    assert longest_list(res) <= 10


def test_every_tool_returns_unique_call_id():
    calls = [core.describe_dataset(), core.find_metrics("кредитование"), core.get_change("NPS"), core.break_down("NPS"),
             core.get_related_metrics("NPS"), core.get_goal_info("NPS"), core.diagnose_change("NPS"), core.get_change("NPS")]
    ids = [c["call_id"] for c in calls]
    assert all(i.startswith("c_") and len(i) == 10 for i in ids) and len(set(ids)) == len(ids)
    assert [c["tool"] for c in calls[:7]] == ["describe_dataset", "find_metrics", "get_change", "break_down",
                                              "get_related_metrics", "get_goal_info", "diagnose_change"]


def test_describe_strategic_dataset():
    res = core.describe_dataset()
    assert res["grain"] == "month" and res["period"] == {"from": "2025-04", "to": "2026-09"}
    assert res["default_window"] == {"from": "2026-04", "to": "2026-09", "periods": 6}
    assert res["baselines_available"] == {"plan": True, "prev": True, "yoy": True}
    assert res["dimensions"][0]["name"] == "Сегмент" and "Молодёжь" in res["dimensions"][0]["top_values"]
    assert res["goals"] and any(m["name"] == "NPS" and m["has_plan"] for m in res["metrics"])


def test_other_baselines_are_available_on_the_strategic_dataset():
    prev = core.get_change("Выдачи розничных кредитов", baseline="prev")
    assert prev["baseline"] == {"type": "prev", "label": "предыдущий период", "from": "2025-10", "to": "2026-03"}
    assert prev["fact"] == 288.0 and prev["base"] != core.get_change("Выдачи розничных кредитов")["base"]
    yoy = core.get_change("Выдачи розничных кредитов", baseline="yoy")
    assert yoy["baseline"]["from"] == "2025-04" and yoy["baseline"]["to"] == "2025-09"


# ---------- ошибки ----------


def test_unknown_metric_returns_error_json_with_hint():
    res = core.get_change("Совсем левый показатель")
    assert "call_id" in res and "не найден" in res["error"] and "find_metrics" in res["error"]
    assert "metric" not in res


def test_metric_name_is_forgiving_about_case_and_small_slips():
    assert core.get_change("выдачи розничных кредитов")["metric"] == "Выдачи розничных кредитов"
    assert core.get_change("Выдачи розничных кредитов  ")["metric"] == "Выдачи розничных кредитов"
    assert core.get_change("Оборот карт")["metric"] == "Оборот по картам"


def test_bad_periods():
    assert "2026-09-15" in core.get_change("NPS", period_from="апрель")["error"]
    assert "данных нет" in core.get_change("NPS", period_from="2030-01", period_to="2030-06")["error"]
    assert "позже конца" in core.get_change("NPS", period_from="2026-09", period_to="2026-01")["error"]
    clipped = core.get_change("NPS", period_from="2024-01", period_to="2026-09")
    assert clipped["period"]["from"] == "2025-04" and "обрезан" in clipped["note"]


def test_unknown_dimension_and_value_come_with_suggestions():
    bad_dim = core.break_down("NPS", by="Регион")
    assert "Измерения «Регион» нет" in bad_dim["error"] and bad_dim["dimensions"] == ["Сегмент"]
    bad_value = core.get_change("NPS", where_dimension="Сегмент", where_value="Пенсионеры")
    assert "нет в измерении" in bad_value["error"] and "Молодёжь" in bad_value["suggestions"]
    assert "оба параметра" in core.get_change("NPS", where_dimension="Сегмент")["error"]
    assert core.get_change("NPS", where_dimension="сегмент", where_value="молодёжь")["filter"] == {"Сегмент": "Молодёжь"}


def test_unknown_baseline_is_an_error():
    assert "Неизвестная база" in core.get_change("NPS", baseline="вчера")["error"]


def test_last_months_shortcut():
    three = core.get_change("Выдачи розничных кредитов", last_months=3)
    assert three["period"] == {"from": "2026-07", "to": "2026-09", "periods": 3}
    explicit = core.get_change("Выдачи розничных кредитов", period_from="2026-07", period_to="2026-09")
    assert strip(three) == strip(explicit)
    assert core.get_change("NPS", period_from="2026-08", last_months=3)["period"]["from"] == "2026-08"  # явный период главнее
    assert core.diagnose_change("NPS", last_months=2)["period"]["periods"] == 2
    assert "last_days" in core.get_change("NPS", last_days=14)["error"]


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
    data = {  # (показатель, сегмент): (план, факт) по трём месяцам
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
    r = core.get_change("Поток", period_from="2026-01", period_to="2026-03")
    assert (r["base"], r["fact"], r["deviation_abs"], r["deviation_pct"]) == (45.0, 43.0, -2.0, -4.4)
    rows = by_value(core.break_down("Поток", period_from="2026-01", period_to="2026-03")["values"])
    assert (rows["A"]["change_abs"], rows["A"]["contribution_pct"]) == (-3.0, 150.0)
    assert (rows["B"]["change_abs"], rows["B"]["contribution_pct"]) == (1.0, -50.0)


def test_stock_takes_last_month_of_period(tiny):
    r = core.get_change("Остаток", period_from="2026-01", period_to="2026-03")
    assert (r["base"], r["fact"], r["deviation_abs"], r["deviation_pct"]) == (150.0, 140.0, -10.0, -6.7)
    r2 = core.get_change("Остаток", period_from="2026-01", period_to="2026-02")
    assert (r2["base"], r2["fact"]) == (150.0, 145.0)


def test_ratio_is_mean_and_lower_is_better(tiny):
    r = core.get_change("Доля", period_from="2026-01", period_to="2026-03")
    assert (r["base"], r["fact"], r["deviation_abs"], r["deviation_pct"]) == (3.0, 3.5, 0.5, 16.7)
    assert r["unfavorable"] is True and r["assessment"] == "хуже плана"
    b = core.break_down("Доля", period_from="2026-01", period_to="2026-03")
    assert b["values"][0]["value"] == "A" and b["values"][0]["contribution_abs"] == 0.5
    assert "ставка и структура" in b["aggregation_note"]


def test_related_in_tiny_dataset(tiny):
    r = core.get_related_metrics("Поток", period_from="2026-01", period_to="2026-03")
    item = r["related"][0]
    assert item["metric"] == "Остаток" and item["relation"] == "зависит от выбранного показателя"
    assert item["consistent_with_link"] is True  # оба хуже плана, связь положительная


def test_zero_plan_never_produces_non_finite_numbers(tmp_path, monkeypatch):
    """Регрессия из обзора кода: нулевой план в сегменте не должен давать Infinity/NaN в JSON."""
    goals = pd.DataFrame(
        [["Цель", None, "Босс", "А", "шт", "поток", "больше"], ["Цель", None, "Босс", "Б", "шт", "поток", "больше"]],
        columns=list(generate.GOALS_COLUMNS),
    )
    rows = []
    for kpi in ("А", "Б"):
        for seg, plan, fact in (("X", 10, 8), ("Y", 0, 3)):  # у сегмента Y план равен нулю
            for i in range(3):
                rows.append([kpi, pd.Timestamp(2026, i + 1, 1), seg, plan, fact])
    facts = pd.DataFrame(rows, columns=list(generate.FACTS_COLUMNS))
    links = pd.DataFrame([["А", "Б", "положительно"]], columns=list(generate.LINKS_COLUMNS))
    path = tmp_path / "zero.xlsx"
    generate.write_workbook(path, goals, facts, links)
    monkeypatch.setenv("KPI_DATA_PATH", str(path))
    for result in (core.diagnose_change("А"), core.break_down("А"), core.get_related_metrics("А"), core.get_change("А")):
        json.dumps(result, allow_nan=False)  # бросит ValueError, если есть inf/NaN
    rows_by_value = by_value(core.break_down("А")["values"])
    assert rows_by_value["Y"]["change_pct"] is None  # процент от нулевого плана не определён


# ======================================================================================================================
# Плоские отчёты: образцы, проходящие весь путь «xlsx -> профиль -> таблица -> движок»; ожидания считает pandas
# ======================================================================================================================


@pytest.fixture
def marketing(tmp_path, monkeypatch):
    activate_flat(tmp_path, monkeypatch, "marketing")
    df = generate_samples.marketing()  # независимый источник ожиданий: тот же генератор, минуя чтение файла и движок
    df["Дата"] = pd.to_datetime(df["Дата"])
    return df


@pytest.fixture
def support(tmp_path, monkeypatch):
    activate_flat(tmp_path, monkeypatch, "support")
    df = generate_samples.support()
    df["Дата"] = pd.to_datetime(df["Дата"])
    return df


@pytest.fixture
def branches(tmp_path, monkeypatch):
    activate_flat(tmp_path, monkeypatch, "branches")
    df = generate_samples.branches()
    df["Период"] = pd.to_datetime(df["Период"])
    return df


def days(df, start, end):
    return df[(df["Дата"] >= start) & (df["Дата"] <= end)]


def test_describe_flat_daily_report(marketing):
    res = core.describe_dataset()
    assert (res["grain"], res["rows"], res["periods"]) == ("day", 488, 61)
    assert res["period"] == {"from": "2026-08-01", "to": "2026-09-30"}
    assert res["default_window"] == {"from": "2026-09-01", "to": "2026-09-30", "periods": 30}
    assert res["baselines_available"] == {"plan": False, "prev": True, "yoy": False}
    assert [d["name"] for d in res["dimensions"]] == ["Канал", "Кампания"]
    metrics = {m["name"]: m for m in res["metrics"]}
    assert metrics["CTR"]["definition"] == "Клики / Показы × 100" and metrics["Расходы"]["better"] == "меньше"
    assert "goals" not in res and "filters" not in res


def test_sum_and_ratio_match_pandas(marketing):
    fact, base = days(marketing, "2026-09-01", "2026-09-30"), days(marketing, "2026-08-02", "2026-08-31")
    spend = core.get_change("Расходы")
    assert spend["baseline"] == {"type": "prev", "label": "предыдущий период", "from": "2026-08-02", "to": "2026-08-31"}
    assert spend["fact"] == fact["Расходы, руб"].sum() and spend["base"] == base["Расходы, руб"].sum()
    assert spend["deviation_pct"] == pytest.approx((spend["fact"] / spend["base"] - 1) * 100, abs=0.06)
    assert spend["unfavorable"] is True and spend["assessment"] == "хуже, чем в предыдущем периоде"
    conv = core.get_change("Конверсия в лиды")  # отношение сумм, а не среднее дневных процентов
    assert conv["fact"] == pytest.approx(100 * fact["Лиды"].sum() / fact["Клики"].sum(), abs=0.006)
    assert conv["base"] == pytest.approx(100 * base["Лиды"].sum() / base["Клики"].sum(), abs=0.006)
    cost = core.get_change("Стоимость лида")
    assert cost["fact"] == pytest.approx(fact["Расходы, руб"].sum() / fact["Лиды"].sum(), abs=0.006)
    assert core.get_change("ROAS")["fact"] == pytest.approx(fact["Выручка, руб"].sum() / fact["Расходы, руб"].sum(), abs=0.006)


def test_windows_for_daily_data(marketing):
    last14 = core.get_change("Расходы", last_days=14)
    assert last14["period"] == {"from": "2026-09-17", "to": "2026-09-30", "periods": 14}
    assert last14["baseline"]["from"] == "2026-09-03" and last14["baseline"]["to"] == "2026-09-16"
    assert last14["fact"] == days(marketing, "2026-09-17", "2026-09-30")["Расходы, руб"].sum()
    explicit = core.get_change("Расходы", period_from="2026-09-15", period_to="2026-09-30")
    assert explicit["period"]["periods"] == 16 and explicit["baseline"]["from"] == "2026-08-30"
    assert core.get_change("Расходы", last_months=1)["period"] == {"from": "2026-09-01", "to": "2026-09-30", "periods": 30}
    clipped = core.get_change("Расходы", period_from="2026-09-25", period_to="2026-12-31")
    assert clipped["period"]["to"] == "2026-09-30" and "обрезан" in clipped["note"]
    assert "данных нет" in core.get_change("Расходы", period_from="2027-01", period_to="2027-02")["error"]


def test_unavailable_baselines_explain_why(marketing):
    assert "нет плана" in core.get_change("Расходы", baseline="plan")["error"]
    yoy = core.get_change("Расходы", baseline="yoy")
    assert "Не с чем сравнивать" in yoy["error"] and "год назад" in yoy["error"]


def test_where_filter_narrows_every_number(marketing):
    res = core.get_change("Расходы", where_dimension="канал", where_value="поиск")
    search = marketing[marketing["Канал"] == "Поиск"]
    assert res["filter"] == {"Канал": "Поиск"}
    assert res["fact"] == days(search, "2026-09-01", "2026-09-30")["Расходы, руб"].sum()
    assert res["base"] == days(search, "2026-08-02", "2026-08-31")["Расходы, руб"].sum()
    diag = core.diagnose_change("Расходы", where_dimension="Канал", where_value="Поиск")
    assert diag["filter"] == {"Канал": "Поиск"}
    assert [d["dimension"] for d in diag["dimensions"]] == ["Кампания"]  # измерение фильтра не раскладывается
    assert diag["primary_dimension"] == "Кампания"
    assert core.get_change("Расходы", where_dimension="Канал", where_value="Телевидение")["suggestions"][0] == "Поиск"


def test_ratio_contributions_sum_to_the_total_change(marketing):
    res = core.break_down("Конверсия в лиды", by="Канал")
    total = res["total"]["deviation_abs"]
    assert sum(v["contribution_abs"] for v in res["values"]) == pytest.approx(total, abs=0.011)
    rows = by_value(res["values"])
    assert res["top_value"] == "Поиск" and rows["Поиск"]["contribution_pct"] > 90
    # ставка у канала не менялась: сдвиг доли в объёме не должен выдаваться за причину
    assert abs(rows["Ретаргетинг"]["change_pct"]) < 1 and abs(rows["Ретаргетинг"]["contribution_pct"]) < 5
    assert "ставка и структура" in res["aggregation_note"]


def test_diagnose_finds_the_search_channel_in_marketing(marketing):
    res = core.diagnose_change("Расходы")
    assert res["deviation"]["significant"] and res["deviation"]["unfavorable"]
    assert res["primary_dimension"] == "Канал"
    top = res["dimensions"][0]["top"][0]
    assert top["value"] == "Поиск" and top["contribution_pct"] > 95 and top["change_pct"] > 20
    assert res["unfavorable_since"] == "2026-09-10" and res["interval"] == "нед."  # семидневки, отсчитанные от конца окна
    assert res["drill_down"] and res["drill_down"][0]["within"] == {"Канал": "Поиск"}
    assert res["drill_down"][0]["dimension"] == "Кампания"
    s1, s2 = res["recovery_scenarios"][:2]
    assert s1["values"] == ["Поиск"] and s1["effect_abs"] == pytest.approx(top["contribution_abs"], abs=0.5)
    assert s2["effect_abs"] == pytest.approx(s1["effect_abs"] / 2, abs=1)
    json.dumps(res, allow_nan=False)


def test_related_metrics_point_at_numerator_and_denominator(marketing):
    res = core.diagnose_change("Конверсия в лиды")
    related = {x["metric"]: x for x in res["related_metrics"]}
    assert related["Лиды"]["relation"] == "числитель этого показателя" and related["Лиды"]["significant"]
    assert related["Лиды"]["top_value"] == "Поиск"
    assert related["Клики"]["relation"] == "знаменатель этого показателя" and not related["Клики"]["significant"]
    # показатели без установленной связи называются совпадением по времени, а не причиной
    assert all(x["relation"].startswith("изменился в тот же период") for x in res["related_metrics"]
               if x["metric"] not in ("Лиды", "Клики"))
    spend = {x["metric"]: x["relation"] for x in core.diagnose_change("Расходы")["related_metrics"]}
    assert spend["Стоимость лида"] == "этот показатель — числитель для «Стоимость лида»"


def test_support_report_points_at_technical_queue(support):
    for metric in ("Выполнение SLA", "CSAT", "Среднее время обработки"):
        res = core.diagnose_change(metric)
        assert res["deviation"]["unfavorable"] and res["primary_dimension"] == "Очередь", metric
        assert res["dimensions"][0]["top"][0]["value"] == "Техническая поддержка", metric
        assert res["dimensions"][0]["top"][0]["contribution_pct"] > 90, metric
    fact = days(support, "2026-09-01", "2026-09-30")
    assert core.get_change("Выполнение SLA")["fact"] == pytest.approx(
        100 * fact["SLA выполнено"].sum() / fact["Решено"].sum(), abs=0.006)
    assert core.get_change("Среднее время обработки")["fact"] == pytest.approx(
        fact["Минуты обработки"].sum() / fact["Решено"].sum(), abs=0.006)
    assert core.get_change("Новые обращения")["assessment"] in ("в пределах допуска", "выше предыдущего периода",
                                                                 "ниже предыдущего периода")


def test_branches_filter_plan_and_planted_signals(branches):
    res = core.describe_dataset()
    assert res["period"] == {"from": "2026-01", "to": "2026-09"}  # неполный октябрь отфильтрован
    assert res["filters"] == [{"column": "Период закрыт", "kept_values": ["Да"]}]
    assert res["rows_excluded_by_filters"] == 36
    assert res["baselines_available"] == {"plan": True, "prev": True, "yoy": False}
    closed = branches[branches["Период"] < "2026-10-01"]
    income = core.get_change("Доход", period_from="2026-06", period_to="2026-09")
    assert income["baseline"] == {"type": "plan", "label": "план"}
    window = closed[closed["Период"] >= "2026-06-01"]
    assert income["fact"] == window["Доход"].sum() and income["base"] == window["План дохода"].sum()
    prev = core.get_change("Доход", period_from="2026-06", period_to="2026-09", baseline="prev")
    assert prev["base"] == closed[(closed["Период"] >= "2026-02-01") & (closed["Период"] <= "2026-05-01")]["Доход"].sum()
    profit = core.get_change("Прибыль", period_from="2026-06", period_to="2026-09")
    assert profit["fact"] == (window["Доход"] - window["Расходы"]).sum()  # выражение «Доход - Расходы»
    spend = core.diagnose_change("Расходы")
    assert spend["dimensions"][0]["top"][0]["value"] == "Новосибирск" and spend["unfavorable_since"] == "2026-07"
    conv = core.diagnose_change("Конверсия сделок", last_months=2)
    top2 = [x["value"] for x in conv["dimensions"][0]["top"][:2]]
    assert conv["primary_dimension"] == "Отделение" and top2 == ["Москва Север", "Екатеринбург"]
    assert [s["values"] for s in conv["recovery_scenarios"]][-1] == ["Москва Север", "Екатеринбург"]  # сценарий S3


def test_when_nothing_is_wrong_no_scenarios_are_built(branches):
    res = core.diagnose_change("Доход", period_from="2026-06", period_to="2026-09")
    assert not res["deviation"]["significant"]
    assert res["recovery_scenarios"] == [] and "scenarios_note" in res


def test_goal_tool_is_honest_when_the_report_has_no_goals(marketing):
    res = core.get_goal_info("Расходы")
    assert res["available"] is False and "нет информации" in res["note"]


def test_find_metrics_in_flat_report(marketing):
    res = core.find_metrics("конверсия")
    assert [m["metric"] for m in res["metric_matches"]][:2] == ["Конверсия в лиды", "Конверсия лидов в заказы"]
    assert res["goal_matches"] == []
    assert core.find_metrics("что с каналом поиск")["value_matches"] == [{"dimension": "Канал", "value": "Поиск"}]


@pytest.mark.parametrize("sample_fixture", ["marketing", "support", "branches"])
def test_every_tool_is_json_safe_and_compact_on_flat_reports(sample_fixture, request):
    request.getfixturevalue(sample_fixture)
    names = [m["name"] for m in core.describe_dataset()["metrics"]]
    for name in names:
        for result in (core.get_change(name), core.break_down(name), core.get_related_metrics(name),
                       core.diagnose_change(name)):
            text = json.dumps(result, ensure_ascii=False, allow_nan=False)
            assert "error" not in result, (name, result)
            assert len(text) < 9000 and longest_list(result) <= 10, (name, len(text))
