import numpy as np
import pandas as pd
import pytest

from tests.helpers import table_from
from tools import engine as E
from tools import series as S

METRICS = [
    {"name": "Выручка", "kind": "sum", "column": "Сумма"},
    {"name": "Конверсия", "kind": "ratio", "num": "Сумма", "den": "Объём", "scale": 100.0},
    {"name": "Остаток", "kind": "last", "column": "Сумма"},
    {"name": "Оценка", "kind": "mean", "column": "Балл"},
]


def table(rows, columns=("Дата", "Канал", "Сумма", "Объём", "Балл"), metrics=None):
    return table_from(rows, list(columns), dict(date=columns[0], dimensions=["Канал"], metrics=metrics or METRICS))


def daily(days=100, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=days)
    rows = [(d.strftime("%Y-%m-%d"), ch, int(rng.integers(10, 90)), int(rng.integers(100, 900)), float(rng.integers(1, 6)))
            for d in dates for ch in ("А", "Б", "В")]
    return rows, pd.DataFrame(rows, columns=["Дата", "Канал", "Сумма", "Объём", "Балл"]).assign(
        Дата=lambda x: pd.to_datetime(x["Дата"]))


# ---------- выбор укрупнения ----------


@pytest.mark.parametrize("days, bucket", [(30, "day"), (40, "day"), (41, "week"), (100, "week"), (280, "week"), (300, "month"),
                                          (1200, "month"), (1500, "quarter")])
def test_bucket_is_the_finest_one_that_fits_the_chart(days, bucket):
    rows = [(d.strftime("%Y-%m-%d"), "А", 1, 1, 1.0) for d in pd.date_range("2022-01-01", periods=days)]
    assert S.pick_bucket(table(rows)) == bucket


def test_requested_bucket_is_used_only_when_it_is_not_finer_than_the_data():
    monthly = [(f"2026-{m:02d}-01", "А", 10, 1, 1.0) for m in range(1, 9)]
    t = table(monthly)
    assert S.pick_bucket(t) == "month" and S.pick_bucket(t, requested="day") == "month"  # мельче данных нельзя
    assert S.pick_bucket(t, requested="quarter") == "quarter"


# ---------- значения ряда совпадают с независимым расчётом ----------


def test_weekly_sums_and_ratios_match_pandas():
    rows, df = daily(100)
    t = table(rows)
    s = S.series(t, t.metric("Выручка"))
    assert s["bucket"] == "week" and s["dates"][0] == "2025-12-29" and s["labels"] == s["dates"]
    weekly = df.groupby(df["Дата"].dt.to_period("W").dt.start_time)["Сумма"].sum()
    assert s["series"][0]["values"] == [float(x) for x in weekly.tolist()]
    c = S.series(t, t.metric("Конверсия"))
    g = df.groupby(df["Дата"].dt.to_period("W").dt.start_time)[["Сумма", "Объём"]].sum()
    assert c["series"][0]["values"] == [round(float(v), 2) for v in (100 * g["Сумма"] / g["Объём"]).tolist()]


def test_mean_counts_only_non_empty_observations():
    rows = [("2026-01-01", "А", 1, 1, 4.0), ("2026-01-02", "А", 1, 1, None), ("2026-01-03", "А", 1, 1, 6.0),
            ("2026-02-01", "А", 1, 1, 2.0)]
    t = table(rows)
    s = S.series(t, t.metric("Оценка"), bucket="month")
    assert s["series"][0]["values"] == [5.0, 2.0]  # (4 + 6) / 2, пустое значение не считается нулём


def test_last_takes_the_end_of_each_bucket_and_sums_across_values():
    rows = [(f"2026-{m:02d}-01", ch, 100 * m + (7 if ch == "Б" else 0), 1, 1.0) for m in range(1, 7) for ch in ("А", "Б")]
    t = table(rows)
    q = S.series(t, t.metric("Остаток"), bucket="quarter")
    assert q["labels"] == ["2026-Q1", "2026-Q2"]
    assert q["series"][0]["values"] == [300 + 300 + 7, 600 + 600 + 7]  # март и июнь, по двум значениям


def test_missing_buckets_inside_the_range_become_none():
    rows = [(f"2026-{m:02d}-01", "А", 10 * m, 1, 1.0) for m in (1, 2, 4, 5, 6, 7)]  # марта нет
    t = table(rows)
    s = S.series(t, t.metric("Выручка"))
    assert s["bucket"] == "month" and s["labels"] == [f"2026-0{m}" for m in range(1, 8)]
    assert s["series"][0]["values"] == [10.0, 20.0, None, 40.0, 50.0, 60.0, 70.0]


def test_plan_overlay_follows_the_same_buckets_and_skips_unplanned_ones():
    rows = [("2026-01", "А", 10, 12), ("2026-02", "А", 20, 25), ("2026-03", "А", 30, None), ("2026-04", "А", 40, 45)]
    metrics = [{"name": "Выручка", "kind": "sum", "column": "Факт", "plan_column": "План"}]
    t = table(rows, columns=("Месяц", "Канал", "Факт", "План"), metrics=metrics)
    s = S.series(t, t.metric("Выручка"))
    assert s["series"][0]["values"] == [10.0, 20.0, 30.0, 40.0] and s["plan"] == [12.0, 25.0, None, 45.0]
    no_plan = S.series(table([("2026-01-01", "А", 1, 1, 1.0), ("2026-02-01", "А", 1, 1, 1.0)]), table([("2026-01-01", "А", 1, 1, 1.0)]).metric("Выручка"))
    assert no_plan["plan"] is None


# ---------- разрезы, фильтры, окно ----------


def test_split_keeps_the_heaviest_values_and_counts_the_rest():
    rows, df = daily(60)
    t = table(rows)
    s = S.series(t, t.metric("Выручка"), by="Канал", max_values=2)
    totals = df.groupby("Канал")["Сумма"].sum().sort_values(ascending=False)
    assert [x["name"] for x in s["series"]] == list(totals.index[:2]) and s["others"] == 1
    # по обеим выбранным линиям сумма совпадает с общим рядом минус отброшенное значение
    whole = S.series(t, t.metric("Выручка"))["series"][0]["values"]
    dropped = S.series(t, t.metric("Выручка"), where=[("Канал", totals.index[2])])["series"][0]["values"]
    parts = [sum(v or 0 for v in vals) for vals in zip(*(x["values"] for x in s["series"]), dropped)]
    assert parts == whole


def test_ratio_split_ranks_by_volume_not_by_ratio():
    rows = [("2026-01-01", "Крупный", 10, 1000, 1.0), ("2026-01-01", "Мелкий", 90, 10, 1.0)]
    t = table(rows)
    s = S.series(t, t.metric("Конверсия"), by="Канал", max_values=1)
    assert [x["name"] for x in s["series"]] == ["Крупный"] and s["others"] == 1


def test_where_filter_and_window_restrict_the_rows():
    rows, df = daily(100)
    t = table(rows)
    m = t.metric("Выручка")
    only = S.series(t, m, where=[("Канал", "А")], bucket="month")
    assert only["series"][0]["values"] == [float(x) for x in df[df["Канал"] == "А"].groupby(df["Дата"].dt.to_period("M"))["Сумма"].sum()]
    win, _ = E.resolve_window(t, last_days=14)
    recent = S.series(t, m, window=win)
    assert recent["bucket"] == "day" and len(recent["labels"]) == 14 and recent["labels"][-1] == "2026-04-10"


def test_the_series_agrees_with_the_totals_of_the_engine():
    rows, _ = daily(70)
    t = table(rows)
    for name in ("Выручка", "Конверсия", "Оценка"):
        m = t.metric(name)
        win, _ = E.resolve_window(t, period_from="2026-02-02", period_to="2026-02-22")  # три полные недели (пн–вс)
        cmp = E.compare(t, m, win, E.resolve_baseline(t, m, win, "prev"))
        s = S.series(t, m, window=win, bucket="week")
        assert len(s["labels"]) == 3
        if m.kind == "sum":
            assert sum(s["series"][0]["values"]) == pytest.approx(cmp.v1, abs=0.01)
        else:  # для отношения и среднего согласуется итог за всё окно, а не среднее по неделям
            whole = S.series(t, m, window=win, bucket="year")["series"][0]["values"]
            assert whole == [round(cmp.v1, 2)]


# ---------- неполные корзины ----------


def test_partial_buckets_are_flagged_for_sums_on_daily_data_only():
    rows = [(d.strftime("%Y-%m-%d"), "А", 10, 1, 1.0) for d in pd.date_range("2026-01-07", "2026-02-17")]  # среда … вторник
    t = table(rows)
    s = S.series(t, t.metric("Выручка"), bucket="week")
    assert s["labels"][0] == "2026-01-05" and s["partial"][0] and s["partial"][-1] and not any(s["partial"][1:-1])
    assert s["series"][0]["values"][0] == 50.0 and s["series"][0]["values"][1] == 70.0  # первая неделя неполная: 5 дней из 7
    ratio = S.series(t, t.metric("Конверсия"), bucket="week")
    assert ratio["partial"] == [False] * len(ratio["labels"])  # отношение неполной неделей не искажается
    day = S.series(t, t.metric("Выручка"), bucket="day")
    assert not any(day["partial"])


def test_gaps_in_the_middle_make_the_bucket_partial_too():
    rows = [(d.strftime("%Y-%m-%d"), "А", 10, 1, 1.0) for d in pd.date_range("2026-01-05", "2026-01-25") if d.day != 8]
    s = S.series(table(rows), table(rows).metric("Выручка"), bucket="week")
    assert s["partial"] == [True, False, False]


# ---------- краевые случаи ----------


def test_report_without_dates_has_no_series():
    t = table_from([("А", 5), ("Б", 7)], ["Канал", "Сумма"], dict(dimensions=["Канал"], metrics=[METRICS[0]]))
    assert S.series(t, t.metric("Выручка")) == {"bucket": None, "labels": [], "dates": [], "partial": [], "series": [],
                                                  "plan": None, "others": 0}


def test_sparkline_returns_the_last_points_without_partial_edges():
    rows, df = daily(100)  # 1 января (четверг) – 10 апреля (пятница): первая и последняя недели неполные
    t = table(rows)
    full = S.series(t, t.metric("Выручка"))
    assert full["partial"][0] and full["partial"][-1] and not any(full["partial"][1:-1])
    sp = S.sparkline(t, t.metric("Выручка"), points=5)
    assert sp["values"] == full["series"][0]["values"][-6:-1] and sp["labels"] == full["labels"][-6:-1] and sp["bucket"] == "week"
    whole = S.sparkline(t, t.metric("Выручка"), points=50)
    assert whole["values"] == full["series"][0]["values"][1:-1]  # не показаны ни первая, ни последняя неполная неделя
    assert S.sparkline(t, t.metric("Выручка"), where=[("Канал", "нет такого")]) == {"labels": [], "values": []}


def test_sparkline_hides_partial_weeks_only_while_enough_points_remain(monkeypatch):
    rows = [(d.strftime("%Y-%m-%d"), "А", 10, 1, 1.0) for d in pd.date_range("2026-01-07", "2026-02-18")]  # среда … среда: 7 недель
    t = table(rows)
    sp = S.sparkline(t, t.metric("Выручка"))
    assert sp["bucket"] == "week" and sp["labels"] == ["2026-01-12", "2026-01-19", "2026-01-26", "2026-02-02", "2026-02-09"]
    assert S.sparkline(t, t.metric("Конверсия"))["labels"][0] == "2026-01-05"  # у отношений неполных недель нет: ничего не обрезается
    fake = {"labels": ["a", "b", "c"], "dates": [], "partial": [True, True, True], "series": [{"name": "x", "values": [1.0, 2.0, 3.0]}],
            "plan": None, "others": 0, "bucket": "week"}
    monkeypatch.setattr(S, "series", lambda *args, **kwargs: fake)
    assert S.sparkline(None, None)["values"] == [1.0, 2.0]  # меньше двух точек линия не рисуется, поэтому два остаются
