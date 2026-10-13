"""Автоопределение ролей столбцов плоского отчёта: дата, измерения, показатели, план, фильтры, отношения.

Всё эвристики по названиям и значениям, без модели: результат пользователь проверяет и правит в интерфейсе
(«Проверьте роли и типы столбцов»), а подтверждённая спецификация сохраняется как есть.
"""
import datetime
import re

import pandas as pd

from data.model import Metric, Spec, build_table
from data.parsing import bool_like_values, date_share, detect_grain, empty_mask, parse_date_series, to_number_series
from data.readers import RawFile
from data.validation import WorkbookError
from tools.text import stems

MAX_DIMENSION_VALUES = 500
MAX_FILTER_VALUES = 40
SAMPLE = 5000
PLAN_WORDS = ("план", "plan", "бюджет", "budget", "target")
DATE_HINTS = ("дата", "date", "период", "period", "месяц", "month", "день", "day", "неделя", "week")
CALENDAR_RE = re.compile(r"^(год|месяц|квартал|неделя|день|year|month|quarter|week|day)$", re.I)
ID_RE = re.compile(r"(^|\W)(id|код|номер|№|артикул|sku|идентификатор|uuid)(\W|$)", re.I)
COMPLETE_HINTS = ("закрыт", "closed", "final", "итог", "полн", "complete", "актуал", "утвержд", "проведен")
CURRENCY_HINTS = ("валют", "currency")
CURRENCY_SYMBOL = {"rub": "₽", "rur": "₽", "руб": "₽", "usd": "$", "eur": "€"}
MONEY_WORDS = ("руб", "доход", "выруч", "расход", "затрат", "стоимост", "цена", "сумм", "прибыл", "бюджет", "оплат",
               "revenue", "income", "cost", "price", "spend", "amount", "sales", "profit", "budget")
RATE_WORDS = ("%", "доля", "процент", "rate", "коэфф", "конверс", "ctr")
UNIT_WORDS = ("руб", "тыс", "млн", "млрд", "%", "шт", "мин", "час", "сек", "дн", "usd", "eur", "₽", "$", "€", "кг",
              "км", "ед.", "чел", "пункт", "балл")
DOWN_WORDS = ("расход", "затрат", "стоимост", "себестоим", "издерж", "отток", "потер", "брак", "просроч", "жалоб",
              "ошибк", "время", "минут", "задерж", "cost", "expense", "spend", "churn", "loss", "defect", "complaint",
              "error", "latency", "cpa", "cpc", "cpl", "cac")
NEUTRAL_WORDS = ("обращени", "всего", "total", "остаток", "запас")
INCOME_WORDS = ("доход", "income")
EXPENSE_WORDS = ("расход", "затрат", "expense", "cost", "spend")

# (название, ключевые слова числителя, знаменателя, множитель, единица, лучше). Единица "" — как у числителя.
RATIO_PATTERNS = [
    ("CTR", ("клик", "click"), ("показ", "impress"), 100, "%", "up"),
    ("Конверсия в лиды", ("лид", "lead"), ("клик", "click"), 100, "%", "up"),
    ("Конверсия лидов в заказы", ("заказ", "order"), ("лид", "lead"), 100, "%", "up"),
    ("Стоимость клика", EXPENSE_WORDS, ("клик", "click"), 1, "", "down"),
    ("Стоимость лида", EXPENSE_WORDS, ("лид", "lead"), 1, "", "down"),
    ("Стоимость заказа", EXPENSE_WORDS, ("заказ", "order"), 1, "", "down"),
    ("ROAS", ("выруч", "revenue", "sales"), EXPENSE_WORDS, 1, "×", "up"),
    ("Средний чек", ("выруч", "revenue", "sales"), ("заказ", "order"), 1, "", "up"),
    ("Выполнение SLA", ("sla",), ("решен", "resolved", "closed"), 100, "%", "up"),
    ("CSAT", ("положит", "positive"), ("оценок", "ratings", "всего оцен"), 100, "%", "up"),
    ("Среднее время обработки", ("минут", "minutes"), ("решен", "resolved"), 1, "мин", "down"),
    ("Доля решённых обращений", ("решен", "resolved"), ("новые обращ", "new ticket", "new_ticket"), 100, "%", "up"),
    ("Конверсия сделок", ("сделк", "deal"), ("обращен", "inquir", "request"), 100, "%", "up"),
]


def split_unit(column: str) -> tuple[str, str]:
    """«Расходы, руб» -> («Расходы», «руб»); «Выручка (тыс. руб.)» -> («Выручка», «тыс. руб.»). Иначе единица пуста."""
    for pattern in (r"(.+?),\s*([^,]{1,20})", r"(.+?)\s*\(([^()]{1,20})\)"):
        m = re.fullmatch(pattern, column.strip())
        if m and any(w in m.group(2).lower() for w in UNIT_WORDS):
            return m.group(1).strip(), m.group(2).strip()
    return column.strip(), ""


def better_by_name(name: str) -> str:
    low = name.lower()
    if any(w in low for w in DOWN_WORDS):
        return "down"
    if any(w in low for w in NEUTRAL_WORDS):
        return "none"
    return "up"


def _unit_from_format(fmt: str) -> tuple[str, float]:
    """(единица, множитель) по формату числа Excel: ₽, $, € или проценты (значения хранятся долями)."""
    if "%" in fmt:
        return "%", 100.0
    for symbol in ("₽", "$", "€"):
        if symbol in fmt:
            return symbol, 1.0
    return "", 1.0


def _kind_of(values: pd.Series) -> str:
    """Тип столбца по его непустым значениям."""
    if values.empty:
        return "empty"
    sample = values.iloc[:: max(1, len(values) // SAMPLE)]
    if pd.api.types.is_datetime64_any_dtype(values):
        return "date"
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        return "number"
    if to_number_series(sample).notna().mean() >= 0.95:
        return "number"
    if date_share(sample) >= 0.9:
        return "date"
    return "text"


def _examples(values: pd.Series, n: int = 3) -> list[str]:
    out = []
    for v in values.drop_duplicates().iloc[:n]:
        if isinstance(v, (pd.Timestamp, datetime.datetime)):  # openpyxl отдаёт даты как datetime
            v = v.date()
        out.append(str(v))
    return out


def _column_stats(name: str, series: pd.Series, fmt: str) -> dict:
    blank = empty_mask(series)  # один раз на столбец: на больших файлах это заметная часть времени
    values = series[~blank]
    kind = _kind_of(values)
    info = {"name": name, "kind": kind, "missing": int(blank.sum()), "n_unique": int(series.dropna().astype(str).nunique()),
            "examples": _examples(values), "format": fmt, "role": "ignore", "reason": ""}
    if kind == "number":
        nums = to_number_series(values).dropna()
        info.update(min=float(nums.min()), max=float(nums.max()),
                    integer=bool((nums == nums.round()).all()), negative=int((nums < 0).sum()))
    return info


def _fill_defaults(c: dict, cur_unit: str) -> bool:
    """Название, единица, множитель, способ агрегации и «что лучше» для числового столбца. Возвращает, доля ли это."""
    name, unit = split_unit(c["name"])
    fmt_unit, scale = _unit_from_format(c["format"])
    unit = unit or fmt_unit or (cur_unit if any(w in c["name"].lower() for w in MONEY_WORDS) else "")
    # проценты и доли суммировать нельзя: берём среднее (точнее — задать отношение двух столбцов)
    rate = unit == "%" or any(w in c["name"].lower() for w in RATE_WORDS)
    if rate and scale == 1.0 and c["max"] <= 1.0:  # значения-доли без процентного формата показываем в процентах
        scale, unit = 100.0, "%"
    c.update(unit=unit, scale=scale, agg="mean" if rate else "sum", better=better_by_name(c["name"]), display=name)
    return rate


def _plan_base(name: str) -> set[str]:
    plan_stems = {w[:4] for w in PLAN_WORDS}
    return {s for s in stems(name, stop=False) if s not in plan_stems}


def _is_plan_name(name: str) -> bool:
    return any(w in name.lower() for w in PLAN_WORDS)


def _pair_plans(metric_cols: list[dict], plan_cols: list[dict]) -> dict[str, str]:
    """Столбец плана -> столбец показателя: по совпадению основ слов («План дохода» ~ «Доход»)."""
    pairs = {}
    for p in plan_cols:
        base = _plan_base(p["name"])
        if not base:
            continue
        best = max(((len(base & stems(m["name"], stop=False)) / len(base | stems(m["name"], stop=False)), m["name"])
                    for m in metric_cols if stems(m["name"], stop=False)), default=(0, ""))
        if best[0] >= 0.5:
            pairs[p["name"]] = best[1]
    return pairs


def _suggest_ratios(metrics: list[Metric]) -> list[dict]:
    names = {m.name.lower(): m for m in metrics if m.kind == "sum"}
    cols = {m.name: m for m in metrics}

    def find(words: tuple[str, ...]) -> Metric | None:
        for low, m in names.items():
            if any(w in low for w in words) and not _is_plan_name(m.name):
                return m
        return None

    taken = {m.name.lower() for m in metrics}
    out = []
    for name, num_w, den_w, scale, unit, better in RATIO_PATTERNS:
        num, den = find(num_w), find(den_w)
        if num and den and num.name != den.name and name.lower() not in taken:
            out.append({"name": name, "num": num.column, "den": den.column, "scale": scale,
                        "unit": unit or num.unit, "better": better, "enabled": True})
    income, expense = find(INCOME_WORDS), find(EXPENSE_WORDS)
    if income and expense and "прибыль" not in taken:
        expr = f"{income.column} - {expense.column}"
        out.append({"name": "Прибыль", "kind": "sum", "column": expr, "unit": income.unit, "better": "up",
                    "scale": 1, "enabled": True})
        out.append({"name": "Маржа", "num": expr, "den": income.column, "scale": 100, "unit": "%", "better": "up",
                    "enabled": True})
    del cols
    return out[:12]


def profile(raw: RawFile, sheet: str | None = None) -> dict:
    """Профиль листа и предлагаемая спецификация. Ничего не меняет в данных."""
    sheet = sheet if sheet in raw.frames else raw.sheet
    df = raw.frames[sheet]
    formats = raw.formats.get(sheet, {})
    cols = [_column_stats(str(c), df[c], formats.get(str(c), "")) for c in df.columns]
    by_name = {c["name"]: c for c in cols}
    warnings: list[str] = []
    n_rows = len(df)

    # --- дата
    date_cols = [c for c in cols if c["kind"] == "date"]
    date_col = None
    if date_cols:
        date_cols.sort(key=lambda c: (not any(h in c["name"].lower() for h in DATE_HINTS), -c["n_unique"]))
        date_col = date_cols[0]
        date_col.update(role="date")
        for extra in date_cols[1:]:
            extra.update(role="ignore", reason="ещё один столбец с датами (основной — «%s»)" % date_col["name"])

    # --- текстовые столбцы
    currency_values: list[str] = []
    for c in cols:
        if c["kind"] == "empty":
            c["reason"] = "столбец пуст"
        elif c["kind"] == "text":
            values = [str(v) for v in df[c["name"]].dropna().astype(str).unique()]
            if len(values) <= MAX_FILTER_VALUES:
                c["values"] = sorted(values)  # интерфейс предложит их как значения фильтра, если пользователь выберет эту роль
            if c["n_unique"] <= 1:
                c.update(role="ignore", reason=f"одно и то же значение ({c['examples'][0] if c['examples'] else '—'})")
                if any(h in c["name"].lower() for h in CURRENCY_HINTS) and values:
                    currency_values = values
                continue
            pair = bool_like_values(values)
            if pair and any(h in c["name"].lower() for h in COMPLETE_HINTS):
                c.update(role="filter", reason="служебный признак: оставить только строки «%s»" % pair[0],
                         values=sorted(values), keep=[pair[0]])
                continue
            if any(h in c["name"].lower() for h in CURRENCY_HINTS):
                currency_values = values
                c.update(role="dimension", reason="")
                warnings.append(f"В столбце «{c['name']}» несколько валют: суммы разных валют нельзя складывать.")
                continue
            if c["n_unique"] > MAX_DIMENSION_VALUES or (n_rows >= 50 and c["n_unique"] > 0.8 * n_rows):
                c.update(role="ignore", reason="слишком много уникальных значений: похоже на идентификатор или комментарий")
            else:
                c.update(role="dimension")
            low = {v.strip().lower() for v in values}
            if any("план" in v for v in low) and any("факт" in v for v in low):
                warnings.append(f"В столбце «{c['name']}» есть значения «план» и «факт»: если план и факт идут отдельными "
                                "строками, разнесите их по разным столбцам — иначе суммы сложатся.")

    # --- числовые столбцы: идентификаторы, календарные признаки, план, показатели
    number_cols = [c for c in cols if c["kind"] == "number"]
    for c in number_cols:
        # идентификатор: «id», «код» в названии или подряд идущие целые числа (суммы тоже уникальны, но не идут подряд)
        sequential = (c["integer"] and n_rows > 20 and c["n_unique"] == n_rows and c["max"] - c["min"] + 1 == n_rows)
        if ID_RE.search(c["name"]) or sequential:
            c.update(role="ignore", reason="похоже на идентификатор")
        elif CALENDAR_RE.match(c["name"].strip()):
            c.update(role="ignore", reason="календарный признак: дата уже есть")
        elif c["n_unique"] <= 1 and not _is_plan_name(c["name"]):
            c.update(role="ignore", reason="одно и то же значение")
        else:
            c["role"] = "metric"
    plan_candidates = [c for c in number_cols if c["role"] == "metric" and _is_plan_name(c["name"])]
    metric_candidates = [c for c in number_cols if c["role"] == "metric" and c not in plan_candidates]
    pairs = _pair_plans(metric_candidates, plan_candidates)
    for p in plan_candidates:
        if p["name"] in pairs:  # постоянный план (одна и та же целевая величина в каждой строке) — это нормально
            p.update(role="plan", plan_for=pairs[p["name"]])
        elif p["n_unique"] <= 1:
            p.update(role="ignore", reason="одно и то же значение")
        else:
            p.update(role="metric")  # план без пары остаётся обычным показателем; пользователь может связать вручную

    cur_unit = next((CURRENCY_SYMBOL.get(v.strip().lower()) for v in currency_values
                     if CURRENCY_SYMBOL.get(v.strip().lower())), "")
    metrics: list[Metric] = []
    for c in cols:
        if c["kind"] != "number":
            continue
        # настройки показателя есть у каждого числового столбца: интерфейс подставляет их, если пользователь сменит роль
        rate = _fill_defaults(c, cur_unit)
        if c["role"] != "metric":
            continue
        if rate:
            warnings.append(f"«{c['name']}» — это доля или процент: берётся простое среднее по строкам. Если известны "
                            "числитель и знаменатель, точнее задать отношение двух столбцов.")
        metrics.append(Metric(name=c["display"], kind=c["agg"], column=c["name"], unit=c["unit"], better=c["better"],
                              scale=c["scale"], plan_column=next((p["name"] for p in cols if p.get("plan_for") == c["name"]), None)))
    names = [m.name for m in metrics]
    for dup in {n for n in names if names.count(n) > 1}:  # «Расходы, руб» и «Расходы, шт» -> оставляем исходные названия
        for m in metrics:
            if m.name == dup:
                m.name = by_name[m.column]["display"] = m.column
    ratios = _suggest_ratios(metrics)

    filters = [{"column": c["name"], "values": c["keep"]} for c in cols if c["role"] == "filter"]
    spec = Spec(date=date_col["name"] if date_col else None,
                dimensions=[c["name"] for c in cols if c["role"] == "dimension"], metrics=list(metrics), filters=filters,
                sheet=sheet)
    for r in ratios:
        spec.metrics.append(Metric(name=r["name"], kind=r.get("kind", "ratio"), column=r.get("column"), num=r.get("num"),
                                   den=r.get("den"), unit=r["unit"], better=r["better"], scale=float(r["scale"])))

    period, grain = None, None
    if date_col:
        dates = parse_date_series(df[date_col["name"]]).dropna()
        if len(dates):
            grain = detect_grain(dates)
            period = {"from": str(dates.min().date()), "to": str(dates.max().date()), "periods": int(dates.nunique())}
    else:
        warnings.append("Нет столбца с датой: можно сравнивать с планом, но не с прошлыми периодами.")
    if not any(m.has_plan for m in spec.metrics) and (period is None or period["periods"] < 2):
        warnings.append("В данных нет плана и только один период: сравнивать не с чем.")
    if not spec.metrics:
        warnings.append("Не найдено ни одного числового столбца для показателей.")

    # --- проверка качества данных (по предлагаемой спецификации)
    key_cols = ([date_col["name"]] if date_col else []) + spec.dimensions
    if key_cols and n_rows:
        dups = int(df.duplicated(subset=key_cols).sum())
        if dups:
            warnings.append(f"{dups} строк повторяют сочетание «{' + '.join(key_cols)}»: значения будут суммироваться.")
    gaps = [c for c in cols if c["role"] == "metric" and c["missing"]]
    if gaps:
        warnings.append("Есть пустые значения: " + ", ".join(f"«{c['name']}» — {c['missing']}" for c in gaps[:4]) + ".")
    neg = [c for c in cols if c["role"] == "metric" and c.get("negative")]
    if neg:
        warnings.append("Есть отрицательные значения: " + ", ".join(f"«{c['name']}» — {c['negative']}" for c in neg[:4]) + ".")
    rows_excluded = 0
    try:
        table = build_table(df, spec, raw.filename, raw.header_rows.get(sheet, 1))
        warnings += table.warnings
        rows_excluded = table.rows_excluded
        if grain and table.p_max is not None:
            span = table.p_max - table.p_min + 1
            present = len(table.periods_present())
            if present < span:
                warnings.append(f"В ряду дат есть пропуски: присутствует {present} из {span} периодов.")
            if rows_excluded:  # период показываем уже после фильтров: неполный месяц в анализ не попадёт
                period = {"from": table.label(table.p_min), "to": table.label(table.p_max), "periods": present}
    except WorkbookError as exc:
        warnings += exc.errors

    return {
        "filename": raw.filename, "kind": raw.kind, "sheet": sheet, "sheets": raw.sheets,
        "header_row": raw.header_rows.get(sheet, 1), "rows": n_rows, "notes": raw.notes[:MAX_NOTES_SHOWN],
        "columns": cols, "ratios": ratios, "grain": grain, "period": period, "rows_excluded": rows_excluded,
        "warnings": warnings,
        "spec": spec.to_dict(),
    }


MAX_NOTES_SHOWN = 30
