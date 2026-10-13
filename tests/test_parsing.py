import numpy as np
import pandas as pd
import pytest

from data.parsing import (
    TOTAL_RE,
    bool_like_values,
    date_share,
    detect_grain,
    is_empty,
    parse_date_value,
    to_number,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1 234,5", 1234.5), ("1 234,5", 1234.5), ("1.234,5", 1234.5), ("1,234.5", 1234.5), ("12,5", 12.5),
        ("12%", 12.0), ("1500 ₽", 1500.0), ("1500 руб.", 1500.0), ("−5", -5.0), ("-5,25", -5.25), (7, 7.0), (3.5, 3.5),
        ("1e3", 1000.0),
    ],
)
def test_to_number_understands_report_formats(value, expected):
    assert to_number(value) == expected


@pytest.mark.parametrize("value", ["", "-", "—", "n.a.", "#DIV/0!", "abc", "12abc", None, True, float("nan"), float("inf")])
def test_to_number_rejects_non_numbers(value):
    assert to_number(value) is None


def test_is_empty_treats_placeholders_as_empty():
    assert all(is_empty(v) for v in (None, np.nan, pd.NaT, "", "  ", "-", "n/a", "Нет данных"))
    assert not any(is_empty(v) for v in (0, "0", "Да", "текст"))


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2026-09-15", "2026-09-15"), ("2026-09-15 10:30:00", "2026-09-15"), ("15.09.2026", "2026-09-15"),
        ("15/09/2026", "2026-09-15"), ("15.09.26", "2026-09-15"), ("2026-09", "2026-09-01"), ("09.2026", "2026-09-01"),
        ("сентябрь 2026", "2026-09-01"), ("Сен 2026", "2026-09-01"), ("сентября 2026 г.", "2026-09-01"),
        ("2026-Q3", "2026-07-01"), ("Q4 2026", "2026-10-01"), ("3 кв. 2026", "2026-07-01"),
        (46280, "2026-09-15"),  # серийный номер даты Excel
        (pd.Timestamp("2026-09-15 13:00"), "2026-09-15"),
    ],
)
def test_parse_date_value(value, expected):
    assert str(parse_date_value(value).date()) == expected


@pytest.mark.parametrize("value", ["Итого", "2026", 2026, 1.5, "2026-13-01", "31.02.2026", "hello", "", None, True])
def test_parse_date_value_rejects_non_dates(value):
    assert parse_date_value(value) is None


def test_date_share_distinguishes_dates_from_numbers_and_text():
    assert date_share(pd.Series(["2026-01-01", "2026-01-02", None])) == 1.0
    assert date_share(pd.Series([1, 2, 3])) == 0.0
    assert date_share(pd.Series(["Москва", "Казань"])) == 0.0
    assert 0.5 < date_share(pd.Series(["2026-01-01", "2026-01-02", "xx"])) < 1.0


@pytest.mark.parametrize(
    "dates, grain",
    [
        (pd.date_range("2026-01-01", periods=60, freq="D"), "day"),
        (pd.date_range("2026-01-05", periods=20, freq="W-MON"), "week"),
        (pd.date_range("2026-01-01", periods=12, freq="MS"), "month"),
        (pd.date_range("2026-01-31", periods=12, freq="ME"), "month"),
        (pd.date_range("2025-01-01", periods=8, freq="QS"), "quarter"),
        (pd.date_range("2018-01-01", periods=6, freq="YS"), "year"),
        (pd.bdate_range("2026-01-01", periods=40), "day"),  # только рабочие дни
    ],
)
def test_detect_grain(dates, grain):
    assert detect_grain(pd.Series(dates)) == grain


def test_detect_grain_with_a_single_date():
    assert detect_grain(pd.Series([pd.Timestamp("2026-03-01")])) == "month"
    assert detect_grain(pd.Series([pd.Timestamp("2026-03-17")])) == "day"


def test_bool_like_values_and_totals_regex():
    assert bool_like_values(["Да", "Нет"]) == ("Да", "Нет")
    assert bool_like_values(["yes", "NO"]) == ("yes", "NO")
    assert bool_like_values(["Да", "Нет", "Возможно"]) is None
    assert bool_like_values(["Москва", "Казань"]) is None
    assert all(TOTAL_RE.match(x) for x in ("Итого", "ИТОГО по отделам", "Всего", "Total", "grand total"))
    assert not TOTAL_RE.match("Итоговая сумма отдела")  # слово «итоговая» внутри названия — не строка «Итого»
    assert not TOTAL_RE.match("Поиск")
