import numpy as np
import pandas as pd
import pytest

from data.model import Metric, Spec, build_table
from tests.helpers import table_from
from tools import engine as E
from tools.engine import ToolError


def monthly(rows, metrics, columns=("Месяц", "Сегмент", "Значение"), dims=("Сегмент",), **kw):
    return table_from(rows, list(columns), dict(date=columns[0], dimensions=list(dims), metrics=metrics, **kw))


def run(t, metric, by="Сегмент", baseline=None, **window):
    m = t.metric(metric)
    win, _ = E.resolve_window(t, **window)
    return E.compare(t, m, win, E.resolve_baseline(t, m, win, baseline), by=by), win


REVENUE = [{"name": "Выручка", "kind": "sum", "column": "Значение", "better": "up"}]
SEGMENTS = [("2026-01", "A", 100), ("2026-01", "B", 50), ("2026-02", "A", 100), ("2026-02", "B", 50),
            ("2026-03", "A", 80), ("2026-03", "B", 50), ("2026-04", "A", 70), ("2026-04", "B", 55)]


# ---------- сумма и предыдущий период ----------


def test_sum_against_previous_period_and_contributions():
    t = monthly(SEGMENTS, REVENUE)
    cmp, win = run(t, "Выручка", period_from="2026-03", period_to="2026-04")
    assert (cmp.v1, cmp.v0, cmp.diff, round(cmp.pct, 6)) == (255, 300, -45, -15.0)
    assert cmp.base.kind == E.PREV and (t.label(cmp.base.win.a), t.label(cmp.base.win.b)) == ("2026-01", "2026-02")
    assert cmp.significant and cmp.unfavorable and cmp.assessment == "хуже, чем в предыдущем периоде"
    g = cmp.groups
    assert list(g.index) == ["A", "B"]  # сначала значение, тянущее в сторону общего изменения
    assert (g.loc["A", "v1"], g.loc["A", "v0"], g.loc["A", "contrib"]) == (150, 200, -50)
    assert (g.loc["B", "v1"], g.loc["B", "v0"], g.loc["B", "contrib"]) == (105, 100, 5)
    assert g.loc["A", "share"] == pytest.approx(111.111, abs=1e-3) and g.loc["B", "share"] == pytest.approx(-11.111, abs=1e-3)
    assert g["contrib"].sum() == cmp.diff


def test_direction_and_neutral_metrics():
    down = monthly(SEGMENTS, [{"name": "Выручка", "kind": "sum", "column": "Значение", "better": "down"}])
    cmp, _ = run(down, "Выручка", period_from="2026-03", period_to="2026-04")
    assert cmp.diff == -45 and not cmp.unfavorable and cmp.assessment == "лучше, чем в предыдущем периоде"
    neutral = monthly(SEGMENTS, [{"name": "Выручка", "kind": "sum", "column": "Значение", "better": "none"}])
    cmp, _ = run(neutral, "Выручка", period_from="2026-03", period_to="2026-04")
    assert not cmp.unfavorable and cmp.significant and cmp.assessment == "ниже предыдущего периода"
    assert E.scenarios(cmp, "") == []  # для нейтрального показателя сценарии не строятся


def test_year_over_year_baseline_differs_from_previous_period():
    rows = [(f"{2025 + (m - 1) // 12}-{(m - 1) % 12 + 1:02d}", "A", v)
            for m, v in zip(range(1, 15), [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5, 5, 12, 12])]
    t = monthly(rows, REVENUE)
    yoy, _ = run(t, "Выручка", by=None, baseline="yoy", period_from="2026-01", period_to="2026-02")
    prev, _ = run(t, "Выручка", by=None, baseline="prev", period_from="2026-01", period_to="2026-02")
    assert (yoy.v1, yoy.v0, round(yoy.pct, 6)) == (24, 20, 20.0) and yoy.assessment == "лучше, чем год назад"
    assert (prev.v0, round(prev.pct, 6)) == (10, 140.0)


# ---------- отношения: сумма к сумме, «ставка и структура» ----------


def conversion_table(rows):
    metrics = [{"name": "Конверсия", "kind": "ratio", "num": "Сделки", "den": "Обращения", "scale": 100, "unit": "%"}]
    return table_from(rows, ["Месяц", "Сегмент", "Обращения", "Сделки"],
                      dict(date="Месяц", dimensions=["Сегмент"], metrics=metrics))


def test_ratio_is_sum_over_sum_not_average_of_rates():
    t = conversion_table([("2026-01", "X", 100, 50), ("2026-01", "Y", 100, 10), ("2026-02", "X", 100, 50), ("2026-02", "Y", 300, 30)])
    cmp, _ = run(t, "Конверсия", by=None, period_from="2026-02", period_to="2026-02")
    assert cmp.v1 == 20.0 and cmp.v0 == 30.0  # (50+30)/(100+300), а не среднее из 50% и 10%
    assert cmp.assessment == "хуже, чем в предыдущем периоде" and cmp.diff == -10


def test_mix_shift_with_unchanged_rates_is_visible_as_zero_own_change():
    """Ставки не менялись, доля объёма сместилась к менее конверсионному значению: вклады есть, собственное изменение 0%."""
    t = conversion_table([("2026-01", "X", 100, 50), ("2026-01", "Y", 100, 10), ("2026-02", "X", 100, 50), ("2026-02", "Y", 300, 30)])
    cmp, _ = run(t, "Конверсия", period_from="2026-02", period_to="2026-02")
    g = cmp.groups
    assert g["contrib"].tolist() == pytest.approx([-5.0, -5.0]) and g["contrib"].sum() == pytest.approx(cmp.diff)
    assert g["own_pct"].tolist() == [0.0, 0.0]


def test_rate_change_in_one_value_takes_the_whole_contribution():
    t = conversion_table([("2026-01", "X", 100, 50), ("2026-01", "Y", 100, 10), ("2026-02", "X", 100, 25), ("2026-02", "Y", 100, 10)])
    cmp, _ = run(t, "Конверсия", period_from="2026-02", period_to="2026-02")
    g = cmp.groups
    assert cmp.diff == pytest.approx(-12.5) and g.index[0] == "X"
    assert g.loc["X", "contrib"] == pytest.approx(-12.5) and g.loc["Y", "contrib"] == pytest.approx(0)
    assert g.loc["X", "share"] == pytest.approx(100.0) and g.loc["X", "own_pct"] == pytest.approx(-50.0)


def test_ratio_with_zero_denominator_is_an_error_not_infinity():
    t = conversion_table([("2026-01", "X", 100, 50), ("2026-02", "X", 0, 0)])
    with pytest.raises(ToolError, match="Знаменатель"):
        run(t, "Конверсия", by=None, period_from="2026-02", period_to="2026-02")


def test_expression_in_numerator():
    metrics = [{"name": "Маржа", "kind": "ratio", "num": "Доход - Расходы", "den": "Доход", "scale": 100, "unit": "%"}]
    t = table_from([("2026-01", "A", 200, 100), ("2026-02", "A", 200, 150)], ["Месяц", "Сегмент", "Доход", "Расходы"],
                   dict(date="Месяц", dimensions=["Сегмент"], metrics=metrics))
    cmp, _ = run(t, "Маржа", by=None, period_from="2026-02", period_to="2026-02")
    assert (cmp.v0, cmp.v1) == (50.0, 25.0)


# ---------- остаток и среднее ----------


def test_last_kind_takes_value_at_the_end_of_period_and_plan_baseline():
    rows = [("2026-01", "A", 10, 12), ("2026-01", "B", 20, 20), ("2026-02", "A", 12, 13), ("2026-02", "B", 18, 20),
            ("2026-03", "A", 14, 15), ("2026-03", "B", 16, 20)]
    metrics = [{"name": "Остаток", "kind": "last", "column": "Факт", "plan_column": "План", "better": "up"}]
    t = table_from(rows, ["Месяц", "Сегмент", "Факт", "План"], dict(date="Месяц", dimensions=["Сегмент"], metrics=metrics))
    cmp, _ = run(t, "Остаток", period_from="2026-02", period_to="2026-03")
    assert cmp.base.kind == E.PLAN and (cmp.v1, cmp.v0, cmp.diff) == (30, 35, -5)  # только март: 14+16 против 15+20
    assert cmp.groups["contrib"].to_dict() == {"B": -4.0, "A": -1.0}


def test_mean_kind_ignores_empty_values_and_plan_gaps_exclude_rows():
    rows = [("2026-01", "A", 4, 6), ("2026-01", "B", None, 100), ("2026-01", "C", 5, None), ("2026-01", "D", 8, 8)]
    metrics = [{"name": "Оценка", "kind": "mean", "column": "Факт", "plan_column": "План"}]
    t = table_from(rows, ["Месяц", "Сегмент", "Факт", "План"], dict(date="Месяц", dimensions=["Сегмент"], metrics=metrics))
    cmp, _ = run(t, "Оценка", by="Сегмент")
    # с планом сравниваются только строки, где есть и план, и факт: A и D. План без факта (B) и факт без плана (C) не участвуют
    assert cmp.base.kind == E.PLAN and cmp.n_rows == 2
    assert (cmp.v1, cmp.v0, cmp.diff) == (6, 7, -1)
    assert set(cmp.groups.index) == {"A", "D"}
    assert cmp.groups["contrib"].sum() == pytest.approx(cmp.diff)


def test_zero_baseline_gives_no_percent_and_is_not_significant():
    t = monthly([("2026-01", "A", 0), ("2026-02", "A", 10)], REVENUE)
    cmp, _ = run(t, "Выручка", by=None, period_from="2026-02", period_to="2026-02")
    assert cmp.pct is None and not cmp.significant and "база равна нулю" in cmp.assessment


# ---------- окна периодов ----------


def daily():
    days = pd.date_range("2026-08-01", "2026-09-30", freq="D")
    return table_from([(d.strftime("%Y-%m-%d"), "A", 10) for d in days], ["Дата", "Сегмент", "Значение"],
                      dict(date="Дата", dimensions=["Сегмент"], metrics=REVENUE))


def test_default_window_for_daily_data_is_last_30_days_and_baseline_is_equal_length():
    t = daily()
    win, note = E.resolve_window(t)
    assert t.grain == "day" and (t.label(win.a), t.label(win.b), win.n, note) == ("2026-09-01", "2026-09-30", 30, None)
    m = t.metric("Выручка")
    base = E.resolve_baseline(t, m, win)
    assert (t.label(base.win.a), t.label(base.win.b), base.win.n) == ("2026-08-02", "2026-08-31", 30)
    cmp = E.compare(t, m, win, base)
    assert (cmp.v1, cmp.v0, cmp.diff) == (300, 300, 0)  # одинаковая длина: нет «лишнего дня» в августе


def test_explicit_periods_months_quarters_last_days_and_last_months():
    t = daily()
    for kwargs, expect in [({"last_days": 7}, ("2026-09-24", "2026-09-30", 7)),
                           ({"period_from": "2026-09", "period_to": "2026-09"}, ("2026-09-01", "2026-09-30", 30)),
                           ({"period_from": "2026-09-10"}, ("2026-09-10", "2026-09-30", 21)),
                           ({"last_months": 2}, ("2026-08-01", "2026-09-30", 61)),
                           ({"period_from": "2026-08-15", "period_to": "2026-08-20"}, ("2026-08-15", "2026-08-20", 6))]:
        win, _ = E.resolve_window(t, **kwargs)
        assert (t.label(win.a), t.label(win.b), win.n) == expect, kwargs
    q = E.resolve_window(t, period_from="2026-Q3", period_to="2026-Q3")[0]
    assert (t.label(q.a), t.label(q.b)) == ("2026-08-01", "2026-09-30")  # третий квартал обрезан границами данных


def test_window_errors_and_notes():
    t = daily()
    with pytest.raises(ToolError, match="данных нет"):
        E.resolve_window(t, period_from="2027-01", period_to="2027-02")
    with pytest.raises(ToolError, match="позже конца"):
        E.resolve_window(t, period_from="2026-09-20", period_to="2026-09-10")
    with pytest.raises(ToolError, match="Не удалось понять дату"):
        E.resolve_window(t, period_from="вчера")
    win, note = E.resolve_window(t, period_from="2026-07-01", period_to="2026-09-30")
    assert "обрезан" in note and t.label(win.a) == "2026-08-01"
    monthly_t = monthly(SEGMENTS, REVENUE)
    with pytest.raises(ToolError, match="last_days"):
        E.resolve_window(monthly_t, last_days=7)


def test_baseline_unavailable_explains_why_and_auto_falls_back():
    t = daily()
    m = t.metric("Выручка")
    win, _ = E.resolve_window(t, last_months=2)  # всё окно данных: предыдущего периода нет
    with pytest.raises(ToolError, match="Не с чем сравнивать"):
        E.resolve_baseline(t, m, win)
    with pytest.raises(ToolError, match="нет плана"):
        E.resolve_baseline(t, m, win, "plan")
    with pytest.raises(ToolError, match="Неизвестная база"):
        E.resolve_baseline(t, m, win, "вчера")
    assert E.resolve_baseline(t, m, E.resolve_window(t, last_days=10)[0], "auto").kind == E.PREV


# ---------- с какого периода держится отклонение ----------


def test_trailing_run_counts_consecutive_unfavorable_periods():
    plan = [100] * 8
    fact = [100, 100, 100, 100, 100, 95, 94, 93]
    rows = [(f"2026-{i + 1:02d}", "A", f, p) for i, (f, p) in enumerate(zip(fact, plan))]
    metrics = [{"name": "Выручка", "kind": "sum", "column": "Факт", "plan_column": "План"}]
    t = table_from(rows, ["Месяц", "Сегмент", "Факт", "План"], dict(date="Месяц", dimensions=["Сегмент"], metrics=metrics))
    cmp, win = run(t, "Выручка", by=None, period_from="2026-06", period_to="2026-08")
    since, run_len, unit = E.trailing_run(t, t.metric("Выручка"), win, cmp.base, cmp)
    assert (since, run_len, unit) == ("2026-06", 3, "мес.")
    # если последний месяц в норме, серия прерывается
    rows[-1] = ("2026-08", "A", 99, 100)
    t2 = table_from(rows, ["Месяц", "Сегмент", "Факт", "План"], dict(date="Месяц", dimensions=["Сегмент"], metrics=metrics))
    cmp2, win2 = run(t2, "Выручка", by=None, period_from="2026-06", period_to="2026-08")
    assert E.trailing_run(t2, t2.metric("Выручка"), win2, cmp2.base, cmp2)[:2] == (None, 0)


def test_trailing_run_for_daily_data_uses_weeks_anchored_at_the_end():
    days = pd.date_range("2026-08-01", "2026-09-30", freq="D")
    rows = [(d.strftime("%Y-%m-%d"), "A", 10 if d < pd.Timestamp("2026-09-10") else 7) for d in days]
    t = table_from(rows, ["Дата", "Сегмент", "Значение"], dict(date="Дата", dimensions=["Сегмент"], metrics=REVENUE))
    cmp, win = run(t, "Выручка", by=None)
    since, run_len, unit = E.trailing_run(t, t.metric("Выручка"), win, cmp.base, cmp)
    assert unit == "нед." and run_len == 3 and since == "2026-09-10"  # недели 24–30, 17–23 и 10–16 сентября


# ---------- сценарии ----------


def test_scenarios_arithmetic():
    rows = [("2026-01", "A", 100), ("2026-01", "B", 100), ("2026-01", "C", 100), ("2026-02", "A", 70), ("2026-02", "B", 80),
            ("2026-02", "C", 100)]
    t = monthly(rows, REVENUE)
    cmp, _ = run(t, "Выручка", period_from="2026-02", period_to="2026-02")
    sc = E.scenarios(cmp, "руб")
    assert [s["scenario_id"] for s in sc] == ["S1", "S2", "S3"]
    s1, s2, s3 = sc
    assert (s1["values"], s1["effect_abs"], s1["gap_closed_pct"], s1["remaining_gap_abs"]) == (["A"], 30, 60.0, 20)
    assert (s2["close_share_pct"], s2["effect_abs"], s2["remaining_gap_abs"]) == (50, 15, 35)
    assert (s3["values"], s3["effect_abs"], s3["gap_closed_pct"], s3["remaining_gap_abs"]) == (["A", "B"], 50, 100.0, 0)


def test_no_second_scenario_value_when_it_explains_less_than_ten_percent():
    rows = [("2026-01", "A", 100), ("2026-01", "B", 100), ("2026-02", "A", 50), ("2026-02", "B", 95)]
    cmp, _ = run(monthly(rows, REVENUE), "Выручка", period_from="2026-02", period_to="2026-02")
    assert [s["scenario_id"] for s in E.scenarios(cmp, "")] == ["S1", "S2"]  # B даёт лишь 9,1% разрыва


# ---------- фильтр, поиск значений, связанные показатели ----------


def test_where_filter_and_value_resolution():
    t = monthly(SEGMENTS, REVENUE)
    where = E.resolve_where(t, "сегмент", "a")
    assert where == [("Сегмент", "A")]
    cmp = E.compare(t, t.metric("Выручка"), *(lambda w: (w, E.resolve_baseline(t, t.metric("Выручка"), w)))(
        E.resolve_window(t, period_from="2026-03", period_to="2026-04")[0]), where=where)
    assert (cmp.v1, cmp.v0) == (150, 200)
    with pytest.raises(ToolError, match="Измерения «Регион» нет"):
        E.resolve_where(t, "Регион", "A")
    with pytest.raises(ToolError) as exc:
        E.resolve_where(t, "Сегмент", "Z")
    assert exc.value.extra["suggestions"]
    with pytest.raises(ToolError, match="оба параметра"):
        E.resolve_where(t, "Сегмент", None)
    assert E.resolve_where(t, None, None) == []


def test_find_metric_is_forgiving_but_not_reckless():
    t = monthly(SEGMENTS, REVENUE + [{"name": "Выручка с НДС", "kind": "sum", "column": "Значение"}])
    assert E.find_metric(t, "выручка").name == "Выручка"
    with pytest.raises(ToolError, match="не найден"):
        E.find_metric(t, "Погода на Марсе")


def test_related_metrics_for_ratio_formula_and_same_value_rule():
    rows, days = [], pd.date_range("2026-08-01", "2026-09-30", freq="D")
    for d in days:
        late = d >= pd.Timestamp("2026-09-15")
        for ch in ("Поиск", "Соцсети"):
            clicks = 100 if not (ch == "Поиск" and late) else 120
            rows.append((d.strftime("%Y-%m-%d"), ch, 1000, clicks, 10 if not (ch == "Поиск" and late) else 6))
    metrics = [{"name": "Показы", "kind": "sum", "column": "Показы"}, {"name": "Клики", "kind": "sum", "column": "Клики"},
               {"name": "Лиды", "kind": "sum", "column": "Лиды"},
               {"name": "Конверсия в лиды", "kind": "ratio", "num": "Лиды", "den": "Клики", "scale": 100, "unit": "%"}]
    t = table_from(rows, ["Дата", "Канал", "Показы", "Клики", "Лиды"], dict(date="Дата", dimensions=["Канал"], metrics=metrics))
    cmp, win = run(t, "Конверсия в лиды", by="Канал", last_days=14)
    items = E.related(t, t.metric("Конверсия в лиды"), E.compare(t, t.metric("Конверсия в лиды"), win, cmp.base), (), "Канал", primary_value="Поиск")
    by_name = {i["metric"]: i for i in items}
    assert by_name["Лиды"]["relation"] == "числитель этого показателя" and by_name["Клики"]["relation"] == "знаменатель этого показателя"
    assert by_name["Лиды"]["top_value"] == "Поиск" and by_name["Лиды"]["deviation_pct"] < 0 < by_name["Клики"]["deviation_pct"]
    assert "Показы" not in by_name  # Показы не менялись: «совпадение по времени» не набирается


# ---------- инвариант: сумма вкладов = изменение итога ----------


@pytest.mark.parametrize("seed", range(12))
def test_contributions_always_add_up_to_the_total_change(seed):
    rng = np.random.default_rng(seed)
    months = [f"2026-{m:02d}" for m in range(1, 13)]
    rows = [(m, a, b, rng.integers(50, 500), rng.integers(1, 80), rng.integers(40, 520), rng.integers(1, 5) + rng.random())
            for m in months for a in "WXYZ" for b in ("p", "q", "r") if rng.random() > 0.08]
    metrics = [
        {"name": "Сумма", "kind": "sum", "column": "N", "plan_column": "План"},
        {"name": "Остаток", "kind": "last", "column": "N", "plan_column": "План"},
        {"name": "Среднее", "kind": "mean", "column": "R", "plan_column": "План"},
        {"name": "Отношение", "kind": "ratio", "num": "K", "den": "N", "scale": 100},
    ]
    t = table_from(rows, ["Месяц", "A", "B", "N", "K", "План", "R"], dict(date="Месяц", dimensions=["A", "B"], metrics=metrics))
    for name in ("Сумма", "Остаток", "Среднее", "Отношение"):
        m = t.metric(name)
        win, _ = E.resolve_window(t, last_months=3)
        for kind in ([E.PLAN] if m.has_plan else []) + [E.PREV, E.YOY]:
            try:
                base = E.resolve_baseline(t, m, win, kind)
            except ToolError:
                continue
            for dim in ("A", "B"):
                cmp = E.compare(t, m, win, base, by=dim)
                assert cmp.groups["contrib"].sum() == pytest.approx(cmp.diff, rel=1e-9, abs=1e-9), (name, kind, dim)
                if abs(cmp.diff) > 1e-9:
                    assert cmp.groups["share"].sum() == pytest.approx(100.0, abs=1e-6)
