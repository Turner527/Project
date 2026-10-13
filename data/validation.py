"""Загрузка и проверка xlsx-витрины. Все сообщения об ошибках — на русском."""
from __future__ import annotations

import datetime as dt
import difflib
import io
import math
import numbers
import re
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from data.schema import (
    BETTERS,
    DIRECTIONS,
    FACTS_COLUMNS,
    GOALS_COLUMNS,
    KINDS,
    LINKS_COLUMNS,
    OPTIONAL_BLANK,
    SHEET_FACTS,
    SHEET_GOALS,
    SHEET_LINKS,
    SHEETS,
    Dataset,
)

MAX_ERRORS = 15  # дальше пользователю уже не помогает
MAX_EXAMPLES = 5  # сколько номеров строк показывать в одном сообщении


class WorkbookError(Exception):
    """Файл не прошёл проверку; `errors` — список понятных сообщений."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


def _norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _clean_text(value: object) -> object:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return pd.NA
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if text else pd.NA


def _rows(index: pd.Index) -> str:
    """Номера строк Excel (шапка — строка 1) для сообщения."""
    nums = [int(i) + 2 for i in index]
    shown = ", ".join(str(n) for n in nums[:MAX_EXAMPLES])
    more = f" и ещё {len(nums) - MAX_EXAMPLES}" if len(nums) > MAX_EXAMPLES else ""
    return f"{shown}{more}"


def _suggest(name: str, options: list[str]) -> str:
    close = difflib.get_close_matches(name, options, n=1, cutoff=0.6)
    return f" Возможно, имелось в виду «{close[0]}»." if close else ""


def _read_workbook(source: str | Path | bytes | BinaryIO) -> dict[str, pd.DataFrame]:
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    try:
        return pd.read_excel(source, sheet_name=None, dtype=object, engine="openpyxl")
    except Exception as exc:  # битый zip, не xlsx, защищённый паролем и т.п.
        raise WorkbookError(
            [f"Не удалось открыть файл как Excel (.xlsx): {exc}. Загрузите файл в формате .xlsx."]
        ) from exc


def _prepare_sheet(
    raw: dict[str, pd.DataFrame], sheet: str, columns: dict[str, str], errors: list[str]
) -> pd.DataFrame | None:
    """Находит лист и нужные колонки, переименовывает их, чистит текст. None — если есть структурные ошибки."""
    by_norm = {_norm(name): name for name in raw}
    if _norm(sheet) not in by_norm:
        found = ", ".join(f"«{n}»" for n in raw) or "нет"
        errors.append(f"Нет листа «{sheet}». Найденные листы: {found}.")
        return None
    df = raw[by_norm[_norm(sheet)]]
    cols_by_norm = {_norm(c): c for c in df.columns}
    missing = [rus for rus in columns if _norm(rus) not in cols_by_norm]
    if missing:
        have = ", ".join(f"«{c}»" for c in df.columns) or "нет"
        errors.append(
            f"Лист «{sheet}»: нет колонок {', '.join(f'«{m}»' for m in missing)}. Найденные колонки: {have}."
        )
        return None
    df = df[[cols_by_norm[_norm(rus)] for rus in columns]].copy()
    df.columns = list(columns.values())
    df = df.dropna(how="all")
    if df.empty:
        errors.append(f"Лист «{sheet}» пуст: нет ни одной строки с данными.")
        return None
    return df


def _check_blanks(df: pd.DataFrame, sheet: str, columns: dict[str, str], errors: list[str]) -> None:
    rus_by_int = {v: k for k, v in columns.items()}
    for col in df.columns:
        if (sheet, col) in OPTIONAL_BLANK:
            continue
        blank = df[df[col].isna()]
        if len(blank):
            errors.append(
                f"Лист «{sheet}», колонка «{rus_by_int[col]}»: пустые ячейки в строках {_rows(blank.index)}"
                f" (всего {len(blank)})."
            )


def _as_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def _parse_month(value: object) -> str | None:
    if isinstance(value, (dt.datetime, dt.date, pd.Timestamp)):
        return f"{value.year:04d}-{value.month:02d}"
    if isinstance(value, str):
        text = value.strip()
        for pattern, order in (
            (r"^(\d{4})-(\d{1,2})(?:-\d{1,2})?$", "ym"),
            (r"^(\d{1,2})\.(\d{4})$", "my"),
            (r"^\d{1,2}\.(\d{1,2})\.(\d{4})$", "my"),
        ):
            m = re.match(pattern, text)
            if m:
                a, b = (int(x) for x in m.groups())
                year, month = (a, b) if order == "ym" else (b, a)
                if 1 <= month <= 12 and 1900 <= year <= 2200:
                    return f"{year:04d}-{month:02d}"
    return None


def _check_types(facts: pd.DataFrame, errors: list[str]) -> pd.DataFrame:
    """Приводит месяц к 'YYYY-MM', план/факт к float; ошибки типов добавляет в errors."""
    rus_by_int = {v: k for k, v in FACTS_COLUMNS.items()}
    facts = facts.copy()
    for col in ("kpi", "segment"):
        facts[col] = facts[col].map(_clean_text)

    months = facts["month"].map(lambda v: _parse_month(v) if not pd.isna(v) else None)
    bad = facts[months.isna() & facts["month"].notna()]
    if len(bad):
        example = bad["month"].iloc[0]
        errors.append(
            f"Лист «{SHEET_FACTS}», колонка «{rus_by_int['month']}»: не удалось распознать месяц "
            f"(строки {_rows(bad.index)}, например «{example}»). Используйте дату или текст вида 2026-04."
        )
    facts["month"] = months

    for col in ("plan", "fact"):
        numeric = facts[col].map(lambda v: _as_number(v) if not pd.isna(v) else None)
        bad = facts[numeric.isna() & facts[col].notna()]
        if len(bad):
            example = bad[col].iloc[0]
            errors.append(
                f"Лист «{SHEET_FACTS}», колонка «{rus_by_int[col]}»: нужно число, а в строках "
                f"{_rows(bad.index)} другое значение (например «{example}»). Числа, записанные текстом, "
                "не принимаются — переведите ячейки в числовой формат."
            )
        facts[col] = numeric.astype(float)
    return facts


def _find_cycle(parents: dict[str, str]) -> str | None:
    """Первая цель, от которой по цепочке родителей можно вернуться к ней же."""
    for start in parents:
        seen, cur = {start}, parents.get(start)
        while cur is not None:
            if cur in seen:
                return start
            seen.add(cur)
            cur = parents.get(cur)
    return None


def _check_goals(goals: pd.DataFrame, errors: list[str]) -> pd.DataFrame:
    goals = goals.copy()
    for col in goals.columns:
        goals[col] = goals[col].map(_clean_text)
    rus_by_int = {v: k for k, v in GOALS_COLUMNS.items()}

    for col, allowed in (("kind", KINDS), ("better", BETTERS)):
        goals[col] = goals[col].map(lambda v: _norm(v) if not pd.isna(v) else v)
        bad = goals[goals[col].notna() & ~goals[col].isin(allowed)]
        if len(bad):
            errors.append(
                f"Лист «{SHEET_GOALS}», колонка «{rus_by_int[col]}»: недопустимое значение «{bad[col].iloc[0]}» "
                f"(строки {_rows(bad.index)}). Допустимо: {', '.join(allowed)}."
            )

    dup = goals[goals.duplicated("kpi", keep=False) & goals["kpi"].notna()]
    if len(dup):
        errors.append(
            f"Лист «{SHEET_GOALS}»: KPI «{dup['kpi'].iloc[0]}» указан несколько раз (строки {_rows(dup.index)}). "
            "Каждый KPI должен относиться ровно к одной цели."
        )

    for goal, grp in goals.groupby("goal"):
        for col, label in (("owner", "владельцами"), ("parent_goal", "родительскими целями")):
            values = grp[col].dropna().unique()
            if len(values) > 1:
                errors.append(
                    f"Лист «{SHEET_GOALS}»: цель «{goal}» указана с разными {label}: "
                    f"{', '.join(f'«{v}»' for v in values)}."
                )

    names = set(goals["goal"].dropna())
    parents = goals.drop_duplicates("goal").set_index("goal")["parent_goal"].dropna().to_dict()
    for goal, parent in parents.items():
        if parent not in names:
            errors.append(
                f"Лист «{SHEET_GOALS}»: родительская цель «{parent}» (у цели «{goal}») не найдена среди целей."
                + _suggest(parent, sorted(names))
            )
    looped = _find_cycle(parents)
    if looped:
        errors.append(f"Лист «{SHEET_GOALS}»: иерархия целей зациклена вокруг цели «{looped}».")
    elif names and len(parents) == len(names):
        errors.append(f"Лист «{SHEET_GOALS}»: нет корневой цели (у всех целей заполнена «родительская цель»).")
    return goals


def _check_cross(goals: pd.DataFrame, facts: pd.DataFrame, links: pd.DataFrame, errors: list[str]) -> None:
    kpis = sorted(goals["kpi"].dropna().unique())
    kpi_set = set(kpis)

    unknown = facts[~facts["kpi"].isin(kpi_set)]
    for name in list(unknown["kpi"].unique())[:3]:
        rows = unknown[unknown["kpi"] == name]
        errors.append(
            f"Лист «{SHEET_FACTS}»: KPI «{name}» (строки {_rows(rows.index)}) не описан на листе «{SHEET_GOALS}»."
            + _suggest(name, kpis)
        )
    for name in sorted(kpi_set - set(facts["kpi"])):
        errors.append(f"KPI «{name}» описан на листе «{SHEET_GOALS}», но на листе «{SHEET_FACTS}» для него нет данных.")

    dup = facts[facts.duplicated(["kpi", "month", "segment"], keep=False)]
    if len(dup):
        r = dup.iloc[0]
        errors.append(
            f"Лист «{SHEET_FACTS}»: повторяется комбинация «{r['kpi']} / {r['month']} / {r['segment']}» "
            f"(строки {_rows(dup.index)}). На один KPI, месяц и сегмент должна быть одна строка."
        )

    for col, rus in (("cause_kpi", "KPI-причина"), ("effect_kpi", "KPI-следствие")):
        bad = links[~links[col].isin(kpi_set)]
        if len(bad):
            name = bad[col].iloc[0]
            errors.append(
                f"Лист «{SHEET_LINKS}», колонка «{rus}»: KPI «{name}» (строки {_rows(bad.index)}) "
                f"не описан на листе «{SHEET_GOALS}»." + _suggest(name, kpis)
            )
    bad = links[links["direction"].notna() & ~links["direction"].isin(DIRECTIONS)]
    if len(bad):
        errors.append(
            f"Лист «{SHEET_LINKS}», колонка «влияние»: недопустимое значение «{bad['direction'].iloc[0]}» "
            f"(строки {_rows(bad.index)}). Допустимо: {', '.join(DIRECTIONS)}."
        )
    selfs = links[links["cause_kpi"] == links["effect_kpi"]]
    if len(selfs):
        errors.append(f"Лист «{SHEET_LINKS}»: KPI не может влиять сам на себя (строки {_rows(selfs.index)}).")


def load_dataset(source: str | Path | bytes | BinaryIO, name: str = "") -> Dataset:
    """Читает xlsx, проверяет и возвращает Dataset. При проблемах бросает WorkbookError."""
    raw = _read_workbook(source)
    errors: list[str] = []
    frames: dict[str, pd.DataFrame | None] = {}
    for sheet, columns in SHEETS.items():
        df = _prepare_sheet(raw, sheet, columns, errors)
        if df is not None:
            before = len(errors)
            _check_blanks(df, sheet, columns, errors)
            if len(errors) > before:
                df = None
        frames[sheet] = df

    goals, facts, links = frames[SHEET_GOALS], frames[SHEET_FACTS], frames[SHEET_LINKS]
    if goals is not None:
        n = len(errors)
        goals = _check_goals(goals, errors)
        if len(errors) > n:
            goals = None
    if facts is not None:
        n = len(errors)
        facts = _check_types(facts, errors)
        if len(errors) > n:
            facts = None
    if links is not None:
        links = links.copy()
        for col in links.columns:
            links[col] = links[col].map(_clean_text)
        links["direction"] = links["direction"].map(lambda v: _norm(v) if not pd.isna(v) else v)

    if not errors and goals is not None and facts is not None and links is not None:
        _check_cross(goals, facts, links, errors)

    if errors:
        if len(errors) > MAX_ERRORS:
            errors = errors[:MAX_ERRORS] + [f"…и ещё {len(errors) - MAX_ERRORS} замечаний."]
        raise WorkbookError(errors)

    assert goals is not None and facts is not None and links is not None
    facts = facts.sort_values(["kpi", "month", "segment"]).reset_index(drop=True)
    return Dataset(
        goals=goals.reset_index(drop=True),
        facts=facts,
        links=links.reset_index(drop=True),
        source=name or (str(source) if isinstance(source, (str, Path)) else "upload"),
        months=sorted(facts["month"].unique()),
        segments=sorted(facts["segment"].unique()),
    )
