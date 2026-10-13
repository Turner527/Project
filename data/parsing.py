"""Разбор значений из отчётов: даты в разных форматах, числа из текста, «пустые» ячейки."""
import datetime as dt
import math
import re

import numpy as np
import pandas as pd

RU_MONTHS = {
    "январь": 1, "января": 1, "янв": 1, "февраль": 2, "февраля": 2, "фев": 2, "март": 3, "марта": 3, "мар": 3,
    "апрель": 4, "апреля": 4, "апр": 4, "май": 5, "мая": 5, "июнь": 6, "июня": 6, "июн": 6, "июль": 7, "июля": 7,
    "июл": 7, "август": 8, "августа": 8, "авг": 8, "сентябрь": 9, "сентября": 9, "сен": 9, "сент": 9,
    "октябрь": 10, "октября": 10, "окт": 10, "ноябрь": 11, "ноября": 11, "ноя": 11, "декабрь": 12, "декабря": 12,
    "дек": 12,
}
PLACEHOLDERS = {"", "-", "—", "–", "n.a.", "n/a", "na", "nan", "none", "null", "#n/a", "#div/0!", "#н/д", "нет данных"}
TOTAL_RE = re.compile(r"^\s*(итого|всего|итог|total|grand total|sum)\b", re.I)
_CURRENCY_TAIL = re.compile(r"(₽|руб\.?|р\.|\$|€|rub|usd|eur)$", re.I)


def is_empty(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if v is pd.NaT or v is pd.NA:
        return True
    return isinstance(v, str) and v.strip().lower() in PLACEHOLDERS


def to_number(v) -> float | None:
    """Число из значения ячейки: поддерживает «1 234,5», «1.234,5», «12%», «1500 ₽». Иначе None."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        return x if math.isfinite(x) else None
    if not isinstance(v, str):
        return None
    t = v.strip().replace(" ", "").replace(" ", "").replace(" ", "").replace("−", "-")
    if t.lower() in PLACEHOLDERS:
        return None
    t = _CURRENCY_TAIL.sub("", t).rstrip("%")
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+,\d+", t):  # 1.234.567,89
        t = t.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", t):  # 1,234,567.89
        t = t.replace(",", "")
    elif re.fullmatch(r"-?\d+,\d+", t):  # 12,5
        t = t.replace(",", ".")
    try:
        x = float(t)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def to_number_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    return s.map(to_number).astype(float)


def parse_date_value(v) -> pd.Timestamp | None:
    """Дата из значения ячейки; None, если это не дата. Формат дд.мм.гггг читается как день.месяц."""
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, pd.Timestamp):
        return v.normalize()
    if isinstance(v, (dt.datetime, dt.date)):
        return pd.Timestamp(v).normalize()
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        if math.isfinite(x) and x == int(x) and 20000 <= x <= 80000:  # серийный номер даты Excel (1954–2119)
            return pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(x))
        return None
    if not isinstance(v, str):
        return None
    t = v.strip()
    if m := re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T].*)?", t):
        return _ymd(m.group(1), m.group(2), m.group(3))
    if m := re.fullmatch(r"(\d{1,2})[./](\d{1,2})[./](\d{4}|\d{2})(?:\s.*)?", t):
        year = int(m.group(3)) + (2000 if len(m.group(3)) == 2 else 0)
        return _ymd(year, m.group(2), m.group(1))
    if m := re.fullmatch(r"(\d{4})[-/.](\d{1,2})", t):
        return _ymd(m.group(1), m.group(2), 1)
    if m := re.fullmatch(r"(\d{1,2})[./-](\d{4})", t):
        return _ymd(m.group(2), m.group(1), 1)
    if m := re.fullmatch(r"(?:(\d{4})[\s-]?q([1-4])|q([1-4])[\s-]?(\d{4})|([1-4])\s*кв\.?\s*(\d{4}))", t, re.I):
        g = m.groups()
        year, quarter = (g[0], g[1]) if g[0] else (g[3], g[2]) if g[2] else (g[5], g[4])
        return _ymd(year, (int(quarter) - 1) * 3 + 1, 1)
    if m := re.fullmatch(r"([а-яё]+)\.?\s*(\d{4})\s*(?:г\.?|года)?", t.lower()):
        month = RU_MONTHS.get(m.group(1))
        return _ymd(m.group(2), month, 1) if month else None
    return None


def _ymd(y, m, d) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(year=int(y), month=int(m), day=int(d))
    except (ValueError, TypeError):
        return None


def parse_date_series(s: pd.Series) -> pd.Series:
    """Серия datetime64 (NaT там, где не удалось разобрать)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.normalize()
    return pd.to_datetime(s.map(parse_date_value), errors="coerce")


def date_share(s: pd.Series) -> float:
    """Доля непустых значений, которые разбираются как даты."""
    values = s[~s.map(is_empty)]
    if values.empty:
        return 0.0
    if pd.api.types.is_datetime64_any_dtype(values):
        return 1.0
    if pd.api.types.is_numeric_dtype(values):  # числа считаем датами только при явном серийном формате — не угадываем
        return 0.0
    return float(values.map(parse_date_value).notna().mean())


def detect_grain(dates: pd.Series) -> str:
    """Детализация ряда по медианному шагу между соседними датами: day/week/month/quarter/year."""
    unique = np.sort(dates.dropna().dt.normalize().unique())
    if len(unique) < 2:
        only = pd.Timestamp(unique[0]) if len(unique) else None
        return "month" if only is not None and only.day == 1 else "day"
    step = float(np.median(np.diff(unique).astype("timedelta64[D]").astype(float)))
    if step <= 1.5:
        return "day"
    if 6 <= step <= 8:
        return "week"
    if 27 <= step <= 32:
        return "month"
    if 85 <= step <= 95:
        return "quarter"
    if 360 <= step <= 370:
        return "year"
    return "day"


TRUE_WORDS = {"да", "yes", "true", "y", "1", "истина", "закрыт", "закрыто"}
FALSE_WORDS = {"нет", "no", "false", "n", "0", "ложь", "открыт"}


def bool_like_values(values: list[str]) -> tuple[str, str] | None:
    """Если у столбца ровно два значения вида да/нет, возвращает (положительное, отрицательное)."""
    low = {str(v).strip().lower(): str(v) for v in values}
    if len(low) != 2:
        return None
    pos = [orig for k, orig in low.items() if k in TRUE_WORDS]
    neg = [orig for k, orig in low.items() if k in FALSE_WORDS]
    return (pos[0], neg[0]) if len(pos) == 1 and len(neg) == 1 else None
