"""Универсальная модель отчёта: плоская таблица + спецификация ролей столбцов.

Спецификация (Spec) говорит, какой столбец — дата, какие — измерения (разрезы), какие — показатели и как
показатели агрегируются: sum (сумма), mean (среднее), last (значение на конец периода), ratio (отношение сумм
двух столбцов, например CTR = клики / показы). Table — проверенная таблица, готовая к расчётам.
"""
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.parsing import (
    TOTAL_RE,
    detect_grain,
    empty_mask,
    parse_date_series,
    to_number_series,
)
from data.validation import WorkbookError
from tools.privacy import public_name

FREQ = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}
KINDS = ("sum", "mean", "last", "ratio")
BETTER = ("up", "down", "none")
MISSING_VALUE = "(не указано)"
MAX_EXAMPLES = 5
BAD_SHARE_ERROR = 0.01  # больше 1% нечисловых ячеек в показателе — это ошибка, меньше — предупреждение


@dataclass
class Metric:
    name: str
    kind: str = "sum"
    column: str | None = None  # для sum / mean / last
    num: str | None = None  # для ratio: столбец или выражение «A - B»
    den: str | None = None
    unit: str = ""
    better: str = "up"  # up — больше лучше, down — меньше лучше, none — нейтрально
    scale: float = 1.0  # множитель при показе (100 для процентов из долей)
    plan_column: str | None = None
    goal: str | None = None
    owner: str | None = None

    @property
    def N(self) -> str:
        return f"N::{self.name}"

    @property
    def D(self) -> str:
        return f"D::{self.name}"

    @property
    def P(self) -> str:
        return f"P::{self.name}"

    @property
    def has_plan(self) -> bool:
        return bool(self.plan_column)

    def definition(self) -> str:
        if self.kind == "ratio":
            return f"{self.num} / {self.den}" + (f" × {self.scale:g}" if self.scale != 1 else "")
        return {"sum": "сумма", "mean": "среднее", "last": "значение на конец периода"}[self.kind] + f" «{self.column}»"

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("name", "kind", "column", "num", "den", "unit", "better", "scale",
                                              "plan_column", "goal", "owner")}

    @classmethod
    def from_dict(cls, d: dict) -> "Metric":
        if not isinstance(d, dict) or not str(d.get("name", "")).strip():
            raise WorkbookError(["У показателя не задано название."])
        try:
            scale = float(d.get("scale", 1) or 1)
        except (TypeError, ValueError):
            raise WorkbookError([f"Показатель «{d.get('name')}»: множитель должен быть числом."]) from None
        text = lambda k: (str(d[k]).strip() or None) if d.get(k) not in (None, "") else None  # noqa: E731
        return cls(name=str(d["name"]).strip(), kind=str(d.get("kind", "sum")), column=text("column"),
                   num=text("num"), den=text("den"), unit=str(d.get("unit") or "").strip(),
                   better=str(d.get("better", "up")), scale=scale, plan_column=text("plan_column"),
                   goal=text("goal"), owner=text("owner"))


@dataclass
class Spec:
    date: str | None = None
    dimensions: list[str] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    filters: list[dict] = field(default_factory=list)  # {"column": ..., "values": [...]} — оставить только эти значения
    significant_pct: float = 3.0
    sheet: str | None = None
    # «Не отправлять в модель»: такой столбец не может быть ни датой, ни разрезом, ни показателем (числителем,
    # знаменателем, планом); как фильтр строк (локальная чистка данных) — может, но ни название, ни значения не раскрываются
    excluded: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"sheet": self.sheet, "date": self.date, "dimensions": list(self.dimensions),
                "metrics": [m.to_dict() for m in self.metrics], "filters": self.filters,
                "significant_pct": self.significant_pct, "excluded": list(self.excluded)}

    @classmethod
    def from_dict(cls, d: dict) -> "Spec":
        if not isinstance(d, dict):
            raise WorkbookError(["Настройки столбцов переданы в неверном формате."])
        try:
            pct = float(d.get("significant_pct", 3.0))
        except (TypeError, ValueError):
            raise WorkbookError(["Порог значимости должен быть числом."]) from None
        filters = []
        for f in d.get("filters") or []:
            if isinstance(f, dict) and f.get("column") and isinstance(f.get("values"), list):
                filters.append({"column": str(f["column"]), "values": [str(v) for v in f["values"]]})
        return cls(date=(str(d["date"]) if d.get("date") else None),
                   dimensions=[str(x) for x in d.get("dimensions") or []],
                   metrics=[Metric.from_dict(m) for m in d.get("metrics") or []],
                   filters=filters, significant_pct=pct, sheet=(str(d["sheet"]) if d.get("sheet") else None),
                   excluded=list(dict.fromkeys(str(x) for x in d.get("excluded") or [])))


def parse_expr(expr: str, columns: list[str]) -> list[tuple[int, str]] | None:
    """«Клики» -> [(+1, 'Клики')]; «Доход - Расходы» -> [(+1, 'Доход'), (-1, 'Расходы')]. None — не разобралось."""
    expr = (expr or "").strip()
    if expr in columns:
        return [(1, expr)]
    tokens = re.split(r"\s+([+\-−])\s+", expr)
    if len(tokens) < 3 or len(tokens) % 2 == 0:
        return None
    terms, sign = [], 1
    for i, token in enumerate(tokens):
        if i % 2 == 0:
            if token not in columns:
                return None
            terms.append((sign, token))
        else:
            sign = 1 if token == "+" else -1
    return terms


def _spec_errors(spec: Spec, columns: list[str]) -> list[str]:
    errors = []
    if not spec.metrics:
        errors.append("Не выбран ни один показатель: отметьте хотя бы один числовой столбец.")
    for col in ([spec.date] if spec.date else []) + spec.dimensions:
        if col not in columns:
            errors.append(f"Столбца «{col}» нет в таблице.")
    names = [m.name for m in spec.metrics]
    for name in {n for n in names if names.count(n) > 1}:
        errors.append(f"Название показателя «{name}» повторяется — названия должны быть разными.")
    for name in set(names) & set(spec.dimensions):
        errors.append(f"«{name}» одновременно и показатель, и измерение.")
    if spec.date and spec.date in spec.dimensions:
        errors.append(f"Столбец «{spec.date}» не может быть и датой, и измерением.")
    if not 0 < spec.significant_pct <= 100:
        errors.append("Порог значимости должен быть больше 0 и не больше 100 процентов.")
    for m in spec.metrics:
        label = f"Показатель «{m.name}»"
        if m.kind not in KINDS:
            errors.append(f"{label}: неизвестный способ расчёта «{m.kind}».")
        elif m.better not in BETTER:
            errors.append(f"{label}: «лучше» должно быть up, down или none.")
        elif m.kind == "ratio":
            for part, label_part in ((m.num, "числитель"), (m.den, "знаменатель")):
                if not part or parse_expr(part, columns) is None:
                    errors.append(f"{label}: {label_part} «{part or ''}» не найден среди столбцов "
                                  "(допустимо: столбец или «Столбец1 - Столбец2»).")
        elif not m.column or parse_expr(m.column, columns) is None:
            errors.append(f"{label}: столбца «{m.column}» нет в таблице (допустимо: столбец или «Столбец1 - Столбец2»).")
        if m.plan_column and parse_expr(m.plan_column, columns) is None:
            errors.append(f"{label}: столбца плана «{m.plan_column}» нет в таблице.")
        if m.plan_column and m.kind == "ratio":
            errors.append(f"{label}: для отношения план не поддерживается.")
        if m.scale == 0:
            errors.append(f"{label}: множитель не может быть нулевым.")
    for f in spec.filters:
        if f["column"] not in columns:
            errors.append(f"Фильтр: столбца «{f['column']}» нет в таблице.")
    errors += _excluded_errors(spec, columns)
    return errors


def _excluded_errors(spec: Spec, columns: list[str]) -> list[str]:
    """Столбец с пометкой «Не отправлять в модель» нельзя использовать ни в одной роли, которая доходит до модели.
    Фильтр строк разрешён: он работает локально, а его название и значения модели не раскрываются."""
    if not spec.excluded:
        return []
    errors = []
    banned = set(spec.excluded)
    mark = "помечен «Не отправлять в модель»"
    for col in spec.excluded:
        if col not in columns:
            errors.append(f"Не отправлять: столбца «{col}» нет в таблице.")
    if spec.date in banned:
        errors.append(f"Столбец «{spec.date}» {mark}, поэтому не может быть датой.")
    for col in dict.fromkeys(spec.dimensions):
        if col in banned:
            errors.append(f"Столбец «{col}» {mark}, поэтому не может быть разрезом (измерением).")
    for m in spec.metrics:
        label = f"Показатель «{m.name}»"
        if m.name in banned:
            errors.append(f"{label}: название совпадает со столбцом «{m.name}», который {mark}.")
        parts = ((m.num, "числителем"), (m.den, "знаменателем")) if m.kind == "ratio" else ((m.column, "столбцом показателя"),)
        for expr, role in parts + ((m.plan_column, "столбцом плана"),):
            used = parse_expr(expr, columns) if expr else None
            for _, col in used or []:
                if col in banned:
                    errors.append(f"{label}: столбец «{col}» {mark}, поэтому не может быть {role}.")
    return errors


@dataclass
class Table:
    df: pd.DataFrame
    spec: Spec
    source: str
    grain: str | None
    freq: str | None
    p_min: int | None
    p_max: int | None
    warnings: list[str]
    rows_total: int
    rows_excluded: int
    dim_values: dict[str, list[str]]
    goals: dict = field(default_factory=dict)  # цель -> {"parent": ..., "owner": ...} (стратегический формат)
    links: list = field(default_factory=list)  # [{"cause": ..., "effect": ..., "direction": ...}]
    notes: list[str] = field(default_factory=list)

    def metric(self, name: str) -> Metric | None:
        return next((m for m in self.spec.metrics if m.name == name), None)

    @property
    def dims(self) -> list[str]:
        return self.spec.dimensions

    @property
    def public_name(self) -> str:
        """Подпись файла, под которой отчёт известен модели («отчёт.xlsx»): настоящее имя наружу не уходит."""
        return public_name(self.source)

    def label(self, ordinal: int) -> str:
        """Подпись периода в формате детализации: 2026-09-15, 2026-09, 2026-Q3, 2026."""
        period = pd.Period(ordinal=int(ordinal), freq=self.freq)
        if self.grain == "month":
            return f"{period.year:04d}-{period.month:02d}"
        if self.grain == "quarter":
            return f"{period.year:04d}-Q{period.quarter}"
        if self.grain == "year":
            return f"{period.year:04d}"
        return str(period.start_time.date())

    def periods_present(self) -> np.ndarray:
        return np.sort(self.df["_p"].unique()) if "_p" in self.df else np.array([], dtype=int)

    def summary(self) -> dict:
        info = {
            "source": self.source,
            "rows": int(len(self.df)),
            "metrics": len(self.spec.metrics),
            "dimensions": list(self.spec.dimensions),
            "grain": self.grain,
            "first_period": self.label(self.p_min) if self.p_min is not None else None,
            "last_period": self.label(self.p_max) if self.p_max is not None else None,
            "periods": int(len(self.periods_present())),
            "has_plan": any(m.has_plan for m in self.spec.metrics),
            "goals": len(self.goals),
        }
        # поля, на которые опираются интерфейс и промпт
        info["first_month"], info["last_month"] = info["first_period"], info["last_period"]
        info["kpis"] = info["metrics"]
        info["segments"] = self.dim_values.get(self.spec.dimensions[0], []) if self.spec.dimensions else []
        return info


def _total_mask(col: pd.Series) -> pd.Series:
    """Строки вида «Итого», «Всего» в текстовом столбце."""
    mask = pd.Series(False, index=col.index)
    if col.dtype == object:
        strings = col[col.map(type).eq(str)]
        if len(strings):
            mask.loc[strings.index] = strings.str.match(TOTAL_RE).astype(bool)
    return mask


def _excel_row(idx: int, header_row: int) -> int:
    return int(idx) + header_row + 1


def _rows_text(index, header_row: int) -> str:
    nums = [_excel_row(i, header_row) for i in list(index)[:MAX_EXAMPLES]]
    more = f" и ещё {len(index) - MAX_EXAMPLES}" if len(index) > MAX_EXAMPLES else ""
    return ", ".join(map(str, nums)) + more


def build_table(raw: pd.DataFrame, spec: Spec, source: str = "", header_row: int = 1) -> Table:
    """Проверяет таблицу по спецификации и готовит её к расчётам. Бросает WorkbookError с русскими сообщениями."""
    raw = raw.reset_index(drop=True)
    columns = [str(c) for c in raw.columns]
    raw.columns = columns
    errors = _spec_errors(spec, columns)
    if errors:
        raise WorkbookError(errors)

    warnings: list[str] = []
    df = raw.copy()
    rows_total = len(df)

    # пустые строки и строки «Итого» (иначе суммы удвоятся); маски пустых ячеек считаем один раз и сужаем вместе со строками
    empty = df.apply(empty_mask)
    df = df[~empty.all(axis=1)]
    empty = empty.loc[df.index]
    key_cols = ([spec.date] if spec.date else []) + spec.dimensions
    if key_cols:
        total_rows = df[key_cols].apply(_total_mask).any(axis=1)
        if total_rows.any():
            warnings.append(f"Исключено итоговых строк («Итого», «Всего»): {int(total_rows.sum())}.")
            df = df[~total_rows]
            empty = empty.loc[df.index]

    dates = None
    if spec.date:
        dates = parse_date_series(df[spec.date])
        bad = df[dates.isna()]
        if len(bad):
            raise WorkbookError([
                f"Столбец даты «{spec.date}»: не удалось распознать значения в строках "
                f"{_rows_text(bad.index, header_row)} (например «{bad[spec.date].iloc[0]}»). Поддерживаются даты Excel, "
                "2026-09-15, 15.09.2026, 2026-09, «сентябрь 2026», 2026-Q3."
            ])

    # фильтры строк (например, «Период закрыт = Да»)
    excluded = 0
    for f in spec.filters:
        keep = {v.strip().lower() for v in f["values"]}
        mask = df[f["column"]].astype(str).str.strip().str.lower().isin(keep)
        excluded += int((~mask).sum())
        df = df[mask]
        empty = empty.loc[df.index]
        if dates is not None:
            dates = dates[mask]
    if df.empty:
        raise WorkbookError(["После применения фильтров не осталось ни одной строки. Проверьте выбранные значения."])

    # числовые столбцы и выражения над ними
    def expressions(m: Metric) -> list[str]:
        parts = [m.num, m.den] if m.kind == "ratio" else [m.column]
        return parts + ([m.plan_column] if m.plan_column else [])

    needed = [c for m in spec.metrics for e in expressions(m) for _, c in parse_expr(e, columns)]
    parsed_cols: dict[str, pd.Series] = {}
    for col in dict.fromkeys(needed):
        parsed = to_number_series(df[col])
        bad = df[parsed.isna() & ~empty[col]]
        if len(bad):
            share = len(bad) / len(df)
            msg = (f"Столбец «{col}»: нечисловые значения в строках {_rows_text(bad.index, header_row)} "
                   f"(например «{bad[col].iloc[0]}»), всего {len(bad)}.")
            if share > BAD_SHARE_ERROR:
                raise WorkbookError([msg + " Показатель должен быть числовым — проверьте роль столбца."])
            warnings.append(msg + " Эти ячейки считаются пустыми.")
        parsed_cols[col] = parsed

    def evaluate(expr: str) -> pd.Series:
        # если пуст любой операнд, результат пуст (а не «0 минус число»)
        return sum(sign * parsed_cols[c] for sign, c in parse_expr(expr, columns))

    out = pd.DataFrame(index=df.index)
    if dates is not None:
        grain = detect_grain(dates)
        freq = FREQ[grain]
        out["_p"] = pd.PeriodIndex(dates, freq=freq).asi8
        p_min, p_max = int(out["_p"].min()), int(out["_p"].max())
    else:
        grain = freq = p_min = p_max = None
        out["_p"] = 0
    for d in spec.dimensions:
        out[d] = df[d].astype(str).str.strip().where(~empty[d], MISSING_VALUE)
    for m in spec.metrics:
        if m.kind == "ratio":
            out[m.N] = evaluate(m.num).fillna(0.0)
            out[m.D] = evaluate(m.den).fillna(0.0)
        else:
            series = evaluate(m.column)
            out[m.N] = series.fillna(0.0)
            if m.kind == "mean":
                out[m.D] = series.notna().astype(float)
        if m.plan_column:
            out[m.P] = evaluate(m.plan_column)  # пропуски плана сохраняем: такие строки в сравнении с планом не участвуют

    dim_values = {d: list(out[d].value_counts().index) for d in spec.dimensions}
    return Table(df=out.reset_index(drop=True), spec=spec, source=source, grain=grain, freq=freq, p_min=p_min,
                 p_max=p_max, warnings=warnings, rows_total=rows_total, rows_excluded=excluded, dim_values=dim_values)
