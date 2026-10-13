import pandas as pd
import pytest

from data.model import Metric, Spec, build_table, parse_expr
from data.validation import WorkbookError
from tests.helpers import table_from


def frame(rows, columns=("Дата", "Канал", "Расходы", "Клики")):
    return pd.DataFrame(rows, columns=list(columns), dtype=object)


ROWS = [["2026-01-01", "Поиск", 100, 10], ["2026-01-02", "Поиск", 120, 12], ["2026-01-03", "Email", 90, 9],
        ["2026-01-04", "Email", 80, 8]]
SPEC = Spec(date="Дата", dimensions=["Канал"], metrics=[Metric("Расходы", "sum", "Расходы", better="down"),
                                                       Metric("Клики", "sum", "Клики")])


def errors_of(raw, spec):
    with pytest.raises(WorkbookError) as exc:
        build_table(raw, spec, "t.xlsx", header_row=1)
    return exc.value.errors


# ---------- спецификация ----------


def test_spec_round_trip_and_defaults():
    spec = Spec(date="Дата", dimensions=["Канал"], filters=[{"column": "Период закрыт", "values": ["Да"]}], sheet="Данные",
                metrics=[Metric("Расходы", "sum", "Расходы", unit="₽", better="down", plan_column="План"),
                         Metric("CTR", "ratio", num="Клики", den="Показы", unit="%", scale=100.0)])
    again = Spec.from_dict(spec.to_dict())
    assert again.to_dict() == spec.to_dict() and again.significant_pct == 3.0
    assert again.metrics[1].definition() == "Клики / Показы × 100" and again.metrics[0].definition() == "сумма «Расходы»"
    assert again.metrics[0].has_plan and not again.metrics[1].has_plan


def test_spec_from_dict_rejects_garbage_with_russian_messages():
    with pytest.raises(WorkbookError, match="неверном формате"):
        Spec.from_dict([1, 2])
    with pytest.raises(WorkbookError, match="не задано название"):
        Spec.from_dict({"metrics": [{"column": "x"}]})
    with pytest.raises(WorkbookError, match="множитель должен быть числом"):
        Spec.from_dict({"metrics": [{"name": "A", "column": "x", "scale": "много"}]})
    with pytest.raises(WorkbookError, match="Порог значимости"):
        Spec.from_dict({"significant_pct": "высокий"})


def test_expression_parsing():
    columns = ["Доход", "Расходы", "Доход, руб"]
    assert parse_expr("Доход", columns) == [(1, "Доход")]
    assert parse_expr("Доход - Расходы", columns) == [(1, "Доход"), (-1, "Расходы")]
    assert parse_expr("Доход + Расходы − Доход, руб", columns) == [(1, "Доход"), (1, "Расходы"), (-1, "Доход, руб")]
    assert parse_expr("Доход - Налоги", columns) is None
    assert parse_expr("Доход -", columns) is None
    assert parse_expr("", columns) is None


def test_spec_errors_are_listed_together_in_russian():
    spec = Spec(date="Дата", dimensions=["Регион", "Дата"], significant_pct=0,
                metrics=[Metric("Расходы", "sum", "Нет такого"), Metric("Расходы", "median", "Расходы"),
                         Metric("Канал", "sum", "Расходы", better="sideways"),
                         Metric("Доля", "ratio", num="Клики", den="Пропало", plan_column="Расходы"),
                         Metric("Ноль", "sum", "Расходы", scale=0)])
    text = "\n".join(errors_of(frame(ROWS), spec))
    for fragment in ("Столбца «Регион» нет в таблице", "не может быть и датой, и измерением", "должен быть больше 0",
                     "столбца «Нет такого» нет в таблице", "повторяется", "неизвестный способ расчёта «median»",
                     "должно быть up, down или none", "знаменатель «Пропало» не найден", "план не поддерживается",
                     "множитель не может быть нулевым"):
        assert fragment in text, fragment
    assert "Не выбран ни один показатель" in "\n".join(errors_of(frame(ROWS), Spec(date="Дата")))
    assert "одновременно и показатель, и измерение" in "\n".join(
        errors_of(frame(ROWS), Spec(dimensions=["Канал"], metrics=[Metric("Канал", "sum", "Расходы")])))


# ---------- построение таблицы ----------


def test_table_basics_and_labels():
    t = build_table(frame(ROWS), SPEC, "t.xlsx")
    assert (t.grain, t.rows_total, t.rows_excluded, len(t.df)) == ("day", 4, 0, 4)
    assert t.label(t.p_min) == "2026-01-01" and t.label(t.p_max) == "2026-01-04"
    assert set(t.dim_values["Канал"]) == {"Поиск", "Email"} and len(t.dim_values["Канал"]) == 2
    s = t.summary()
    assert s["first_period"] == "2026-01-01" and s["periods"] == 4 and s["has_plan"] is False
    assert (s["kpis"], s["first_month"], s["segments"]) == (2, "2026-01-01", t.dim_values["Канал"])


@pytest.mark.parametrize("dates, grain, labels", [
    (["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"], "month", ("2026-01", "2026-04")),
    (["2026-01-01", "2026-04-01", "2026-07-01", "2026-10-01"], "quarter", ("2026-Q1", "2026-Q4")),
    (["2023-01-01", "2024-01-01", "2025-01-01", "2026-01-01"], "year", ("2023", "2026")),
    (["2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26"], "week", ("2026-01-05", "2026-01-26")),
])
def test_grain_and_period_labels(dates, grain, labels):
    t = build_table(frame([[d, "А", 1, 1] for d in dates]), SPEC, "t")
    assert t.grain == grain and (t.label(t.p_min), t.label(t.p_max)) == labels


def test_no_date_column_means_no_periods():
    t = build_table(frame(ROWS), Spec(dimensions=["Канал"], metrics=SPEC.metrics), "t")
    assert t.grain is None and t.p_min is None and t.summary()["first_period"] is None


def test_dates_in_various_formats_are_understood():
    for value in ("15.09.2026", "2026-09-15", "15/09/2026", "2026-09-15 00:00:00"):
        t = build_table(frame([[value, "А", 1, 1], [value, "Б", 2, 2], [value, "В", 3, 3]]), SPEC, "t")
        assert t.label(t.p_min) == "2026-09-15", value


def test_bad_dates_name_the_excel_rows():
    rows = [r[:] for r in ROWS]
    rows[1][0], rows[3][0] = "вчера", "когда-то"
    (msg,) = errors_of(frame(rows), SPEC)
    assert "Столбец даты «Дата»" in msg and "в строках 3, 5" in msg and "вчера" in msg  # заголовок — строка 1


def test_total_rows_are_dropped_with_a_warning():
    rows = ROWS + [["2026-01-05", "Итого", 390, 39]]
    t = build_table(frame(rows), SPEC, "t")
    assert len(t.df) == 4 and any("итоговых строк" in w and "1" in w for w in t.warnings)


def test_blank_rows_are_dropped_silently():
    t = build_table(frame(ROWS + [[None, None, None, None]]), SPEC, "t")
    assert len(t.df) == 4 and t.warnings == []


def test_non_numeric_cells_are_an_error_when_many_and_a_warning_when_few():
    many = [r[:] for r in ROWS]
    many[0][2] = "много"
    msg = "\n".join(errors_of(frame(many), SPEC))
    assert "Столбец «Расходы»: нечисловые значения в строках 2" in msg and "много" in msg and "роль столбца" in msg
    days = pd.date_range("2026-01-01", periods=150).strftime("%Y-%m-%d")
    big = [[d, "А" if i % 2 else "Б", 10 + i, 5] for i, d in enumerate(days)]  # одна плохая ячейка из 150 — меньше 1%
    big[5][2] = "н/д"
    t = build_table(frame(big), SPEC, "t")
    assert any("нечисловые значения" in w and "считаются пустыми" in w for w in t.warnings)
    assert t.df["N::Расходы"].iloc[5] == 0  # пустая ячейка не превращается в мусор


def test_numbers_with_spaces_commas_and_percent_signs_are_parsed():
    rows = [["2026-01-01", "А", "1 234,5", "10"], ["2026-01-02", "А", "2 000", "20"], ["2026-01-03", "Б", "3,5", "30"]]
    t = build_table(frame(rows), SPEC, "t")
    assert t.df["N::Расходы"].tolist() == [1234.5, 2000.0, 3.5]


def test_filters_keep_only_chosen_values_case_insensitively():
    rows = [["2026-01-01", "А", 1, 1, "Да"], ["2026-01-02", "А", 2, 2, "да"], ["2026-01-03", "А", 3, 3, "Нет"],
            ["2026-01-04", "А", 4, 4, " Да "]]
    spec = Spec(date="Дата", dimensions=["Канал"], metrics=SPEC.metrics, filters=[{"column": "Закрыт", "values": ["Да"]}])
    t = build_table(frame(rows, ("Дата", "Канал", "Расходы", "Клики", "Закрыт")), spec, "t")
    assert (t.rows_total, t.rows_excluded, len(t.df)) == (4, 1, 3) and t.label(t.p_max) == "2026-01-04"
    spec.filters = [{"column": "Закрыт", "values": ["Возможно"]}]
    assert "не осталось ни одной строки" in errors_of(frame(rows, ("Дата", "Канал", "Расходы", "Клики", "Закрыт")), spec)[0]


def test_empty_dimension_values_get_a_visible_placeholder():
    rows = [["2026-01-01", None, 1, 1], ["2026-01-02", "А", 2, 2], ["2026-01-03", "А", 3, 3]]
    t = build_table(frame(rows), SPEC, "t")
    assert "(не указано)" in t.dim_values["Канал"]


def test_expressions_use_signs_and_empty_operands_stay_empty():
    rows = [["2026-01-01", "А", 100, 30], ["2026-01-02", "А", 100, None], ["2026-01-03", "А", 100, 20]]
    spec = Spec(date="Дата", dimensions=["Канал"], metrics=[Metric("Разность", "sum", "Расходы - Клики")])
    t = build_table(frame(rows), spec, "t")
    assert t.df["N::Разность"].tolist() == [70.0, 0.0, 80.0]  # без значения «Клики» разность не считается за 100


def test_plan_gaps_are_preserved_but_fact_gaps_are_zero():
    rows = [["2026-01-01", "А", 10, 8], ["2026-01-02", "А", 10, None], ["2026-01-03", "А", None, 7]]
    spec = Spec(date="Дата", dimensions=["Канал"], metrics=[Metric("Клики", "sum", "Клики", plan_column="Расходы")])
    t = build_table(frame(rows), spec, "t")
    assert t.df["N::Клики"].tolist() == [8.0, 0.0, 7.0]
    assert t.df["P::Клики"].isna().tolist() == [False, False, True]


def test_mean_metric_counts_only_non_empty_observations():
    rows = [["2026-01-01", "А", 4, 1], ["2026-01-02", "А", None, 1], ["2026-01-03", "А", 6, 1]]
    spec = Spec(date="Дата", dimensions=["Канал"], metrics=[Metric("Оценка", "mean", "Расходы")])
    t = build_table(frame(rows), spec, "t")
    assert t.df["D::Оценка"].tolist() == [1.0, 0.0, 1.0]


def test_helper_builds_small_tables_for_other_tests():
    t = table_from([("2026-01", "А", 5)], ["Месяц", "Сегмент", "Значение"],
                   dict(date="Месяц", dimensions=["Сегмент"], metrics=[{"name": "Показатель", "kind": "sum", "column": "Значение"}]))
    assert t.metric("Показатель").kind == "sum" and t.metric("Нет") is None and t.dims == ["Сегмент"]
