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


_NUM_EU = re.compile(r"-?\d{1,3}(\.\d{3})+,\d+")  # 1.234.567,89
_NUM_US = re.compile(r"-?\d{1,3}(,\d{3})+(\.\d+)?")  # 1,234,567.89
_NUM_COMMA = re.compile(r"-?\d+,\d+")  # 12,5
_FAST_TYPES = {str, int, float, np.int64, np.int32, np.float64, np.float32}
_D_ISO = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T].*)?")
_D_DMY = re.compile(r"(\d{1,2})[./](\d{1,2})[./](\d{4}|\d{2})(?:\s.*)?")
_D_YM = re.compile(r"(\d{4})[-/.](\d{1,2})")
_D_MY = re.compile(r"(\d{1,2})[./-](\d{4})")
_D_Q = re.compile(r"(?:(\d{4})[\s-]?q([1-4])|q([1-4])[\s-]?(\d{4})|([1-4])\s*кв\.?\s*(\d{4}))", re.I)
_D_RU = re.compile(r"([а-яё]+)\.?\s*(\d{4})\s*(?:г\.?|года)?")


def is_empty(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if v is pd.NaT or v is pd.NA:
        return True
    return isinstance(v, str) and v.strip().lower() in PLACEHOLDERS


def empty_mask(s: pd.Series) -> pd.Series:
    """Булева серия «ячейка пуста» — то же, что s.map(is_empty), но без вызова Python на каждую ячейку."""
    mask = s.isna()
    if s.dtype == object:
        strings = s[s.map(type).eq(str)]
        if len(strings):
            mask.loc[strings.index] = strings.str.strip().str.lower().isin(PLACEHOLDERS)
    return mask


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
    if _NUM_EU.fullmatch(t):  # 1.234.567,89
        t = t.replace(".", "").replace(",", ".")
    elif _NUM_US.fullmatch(t):  # 1,234,567.89
        t = t.replace(",", "")
    elif _NUM_COMMA.fullmatch(t):  # 12,5
        t = t.replace(",", ".")
    try:
        x = float(t)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def to_number_series(s: pd.Series) -> pd.Series:
    """Числа из столбца (NaN там, где число не разобралось). Результат тот же, что у s.map(to_number), но настоящие числа и
    простые числовые строки разбираются векторно, а построчный разбор остаётся только для «1 234,5», «12%», «1500 ₽»."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if s.dtype != object:
        return s.map(to_number).astype(float)
    # векторно разбираем только строки и обычные числа; всё остальное (bool, bytes, даты…) — построчно, как раньше
    fast = s.map(type).isin(_FAST_TYPES)
    out = pd.to_numeric(s.where(fast), errors="coerce").astype(float)
    out[~np.isfinite(out)] = np.nan  # бесконечность, как и в to_number, числом не считается
    rest = s[out.isna()]  # не разобралось векторно: пустые ячейки так и остаются NaN, остальное — построчно
    rest = rest[~empty_mask(rest)]
    if len(rest):
        out[rest.index] = rest.map(to_number).astype(float)
    return out


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
    if m := _D_ISO.fullmatch(t):
        return _ymd(m.group(1), m.group(2), m.group(3))
    if m := _D_DMY.fullmatch(t):
        year = int(m.group(3)) + (2000 if len(m.group(3)) == 2 else 0)
        return _ymd(year, m.group(2), m.group(1))
    if m := _D_YM.fullmatch(t):
        return _ymd(m.group(1), m.group(2), 1)
    if m := _D_MY.fullmatch(t):
        return _ymd(m.group(2), m.group(1), 1)
    if m := _D_Q.fullmatch(t):
        g = m.groups()
        year, quarter = (g[0], g[1]) if g[0] else (g[3], g[2]) if g[2] else (g[5], g[4])
        return _ymd(year, (int(quarter) - 1) * 3 + 1, 1)
    if m := _D_RU.fullmatch(t.lower()):
        month = RU_MONTHS.get(m.group(1))
        return _ymd(m.group(2), month, 1) if month else None
    return None


def _ymd(y, m, d) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(year=int(y), month=int(m), day=int(d))
    except (ValueError, TypeError):
        return None


_ISO_VEC = r"\s*(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T].*)?\s*"
_DMY_VEC = r"\s*(\d{1,2})[./](\d{1,2})[./](\d{4}|\d{2})(?:\s.*)?\s*"


def _vector_dates(strings: pd.Series) -> pd.Series:
    """Векторный разбор самых частых текстовых форматов — 2026-09-15 [время] и 15.09.2026; остальное вернётся как NaT."""
    out = pd.Series(pd.NaT, index=strings.index, dtype="datetime64[ns]")
    for pattern, order in ((_ISO_VEC, ("year", "month", "day")), (_DMY_VEC, ("day", "month", "year"))):
        parts = strings.str.extract("^" + pattern + "$", expand=True).dropna()
        if parts.empty:
            continue
        parts.columns = order
        short_year = parts["year"].str.len() == 2  # «15.09.26» — это 2026, как в parse_date_value; «0026» остаётся годом 26
        parts = parts.astype(int)
        parts.loc[short_year, "year"] += 2000
        parsed = pd.to_datetime(parts[["year", "month", "day"]], errors="coerce")
        out.loc[parsed.index] = parsed
    return out


def parse_date_series(s: pd.Series) -> pd.Series:
    """Серия datetime64 (NaT там, где не удалось разобрать). Тот же результат, что у s.map(parse_date_value), но частые
    форматы разбираются векторно, а построчный разбор остаётся только для редких."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.normalize()
    if s.dtype == object and len(s) > 1000:
        strings = s[s.map(type).eq(str)]
        out = _vector_dates(strings).reindex(s.index)
        rest = s[out.isna()]
        if len(rest):
            out.loc[rest.index] = pd.to_datetime(rest.map(parse_date_value), errors="coerce")
        return pd.to_datetime(out, errors="coerce")
    return pd.to_datetime(s.map(parse_date_value), errors="coerce")


def date_share(s: pd.Series) -> float:
    """Доля непустых значений, которые разбираются как даты."""
    values = s[~empty_mask(s)]
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
