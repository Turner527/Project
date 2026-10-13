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


# ---------- быстрые векторные пути дают тот же результат, что поэлементный разбор ----------

import datetime as _dt  # noqa: E402

import numpy as _np  # noqa: E402
import pandas as _pd  # noqa: E402

from data.parsing import empty_mask, to_number, to_number_series  # noqa: E402

TRICKY = ["", " ", "-", "—", "n/a", "N/A", "нет данных", "#Н/Д", "nan", "None", "0", "12", " 12 ", "12.5", "-7", "+5", "1e3", "1E-2",
          "1 234,5", "1 234,5", "1 234", "1.234.567,89", "1,234,567.89", "12,5", "12%", "-12,5 %", "1500 ₽", "1 500 руб.",
          "−5", "1_000", "inf", "-inf", "infinity", "abc", "1,2,3", "--5", "0x1A", "٣", "１２", "12abc", "Итого", "2026-09-15",
          True, False, None, _np.nan, _pd.NaT, _pd.NA, 0, 5, -3, 5.5, 1e20, float("inf"), float("-inf"), _np.int64(7),
          _np.float64(2.5), _dt.datetime(2026, 9, 15), _pd.Timestamp("2026-09-15"), _dt.date(2026, 9, 15), b"12", [1], (2,)]


def _same(a: _pd.Series, b: _pd.Series) -> bool:
    return bool(((a == b) | (a.isna() & b.isna())).all())


def test_empty_mask_matches_the_cellwise_definition():
    from data.parsing import is_empty

    s = _pd.Series(TRICKY, dtype=object)
    assert empty_mask(s).tolist() == s.map(is_empty).tolist()
    assert empty_mask(_pd.Series([1.0, _np.nan, 3.0])).tolist() == [False, True, False]  # числовой столбец
    assert empty_mask(_pd.Series(["a", None, " - "], dtype=object)).tolist() == [False, True, True]
    assert empty_mask(_pd.Series([], dtype=object)).tolist() == []
    shuffled = s.sample(frac=1, random_state=3)  # индекс не по порядку: результат привязан к метке, а не к позиции
    assert empty_mask(shuffled).tolist() == shuffled.map(is_empty).tolist()


def test_to_number_series_matches_the_cellwise_definition_on_tricky_values():
    s = _pd.Series(TRICKY, dtype=object)
    assert _same(to_number_series(s), s.map(to_number).astype(float))
    for chunk in ([1, 2.5, "3"], ["1 234,5", "12%"], [True, 1, None], ["", "abc"], [_dt.datetime(2026, 1, 1), "5"]):
        series = _pd.Series(chunk, dtype=object)
        assert _same(to_number_series(series), series.map(to_number).astype(float)), chunk


def test_to_number_series_matches_on_random_mixed_columns():
    rng = _np.random.default_rng(11)
    pool = [v for v in TRICKY if not isinstance(v, (list, tuple, bytes))]
    for _ in range(25):
        s = _pd.Series([pool[i] for i in rng.integers(0, len(pool), 200)], dtype=object)
        assert _same(to_number_series(s), s.map(to_number).astype(float))


def test_to_number_series_keeps_numeric_dtypes_and_never_returns_infinity():
    assert to_number_series(_pd.Series([1, 2, 3])).tolist() == [1.0, 2.0, 3.0]
    assert to_number_series(_pd.Series([True, False])).dtype == float  # bool-столбец идёт построчным путём: нечисловой
    assert _np.isnan(to_number_series(_pd.Series([True, False])).to_numpy()).all()
    assert _np.isnan(to_number_series(_pd.Series(["inf", float("inf")], dtype=object)).to_numpy()).all()
    assert to_number_series(_pd.Series(["5", "6"], dtype="string")).tolist() == [5.0, 6.0]


def test_vector_date_parsing_matches_the_cellwise_definition_on_big_mixed_columns():
    from data.parsing import parse_date_series, parse_date_value

    pool = ["2026-09-15", "2026-9-5", " 2026-09-15 ", "2026-09-15 12:30:00", "2026-09-15T08:00", "2026-02-30", "2026-13-01",
            "0000-01-01", "15.09.2026", "5.9.2026", "05/09/2026", "15.09.26", "31.02.2026", "15.09.2026 10:00", "15.09.2026\n",
            "2026-09", "09.2026", "сентябрь 2026", "Сен 2026 г.", "2026-Q3", "3 кв. 2026", "вчера", "", "-", "n/a", "12345",
            "2026-09-15\n", None, _np.nan, 45000, 45000.5, -1, True, _dt.datetime(2026, 9, 15, 13, 5), _pd.Timestamp("2026-01-31"),
            _dt.date(2026, 3, 1), _pd.NaT]
    rng = _np.random.default_rng(5)
    s = _pd.Series([pool[i] for i in rng.integers(0, len(pool), 4000)], dtype=object)
    assert len(s) > 1000  # короткие столбцы идут прежним построчным путём
    fast = parse_date_series(s)
    slow = _pd.to_datetime(s.map(parse_date_value), errors="coerce")
    # единица времени в dtype может отличаться (ns / us) — значения и пропуски те же
    assert pd.api.types.is_datetime64_any_dtype(fast) and fast.isna().tolist() == slow.isna().tolist()
    assert (fast.dropna() == slow.dropna()).all()
    shuffled = s.sample(frac=1, random_state=2)  # индекс не по порядку
    assert (parse_date_series(shuffled).fillna(_pd.Timestamp(0)) == _pd.to_datetime(shuffled.map(parse_date_value), errors="coerce").fillna(_pd.Timestamp(0))).all()


def test_vector_date_parsing_handles_a_column_of_only_unparseable_text():
    from data.parsing import parse_date_series

    s = _pd.Series(["вчера", "завтра", "когда-нибудь"] * 500, dtype=object)
    assert parse_date_series(s).isna().all()
    assert parse_date_series(_pd.Series([_dt.datetime(2026, 9, 15)] * 1500, dtype=object)).eq(_pd.Timestamp("2026-09-15")).all()
