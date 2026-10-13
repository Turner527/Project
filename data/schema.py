"""Контракт xlsx-витрины: листы, колонки, допустимые значения."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

SHEET_GOALS = "goals"
SHEET_FACTS = "facts"
SHEET_LINKS = "links"

# русское имя колонки в xlsx -> внутреннее имя
GOALS_COLUMNS = {
    "цель": "goal",
    "родительская цель": "parent_goal",
    "владелец": "owner",
    "KPI": "kpi",
    "единица": "unit",
    "тип": "kind",
    "лучше": "better",
}
FACTS_COLUMNS = {
    "KPI": "kpi",
    "месяц": "month",
    "сегмент": "segment",
    "план": "plan",
    "факт": "fact",
}
LINKS_COLUMNS = {
    "KPI-причина": "cause_kpi",
    "KPI-следствие": "effect_kpi",
    "влияние": "direction",
}
SHEETS = {
    SHEET_GOALS: GOALS_COLUMNS,
    SHEET_FACTS: FACTS_COLUMNS,
    SHEET_LINKS: LINKS_COLUMNS,
}
# колонка goals.parent_goal может быть пустой (корневая цель)
OPTIONAL_BLANK = {(SHEET_GOALS, "parent_goal")}

# тип KPI определяет, как агрегировать по месяцам и сегментам
KIND_FLOW = "поток"  # сумма по месяцам и сегментам
KIND_STOCK = "остаток"  # значение на конец периода, сумма по сегментам
KIND_RATIO = "доля"  # простое среднее по месяцам и сегментам
KINDS = (KIND_FLOW, KIND_STOCK, KIND_RATIO)

BETTER_UP = "больше"
BETTER_DOWN = "меньше"
BETTERS = (BETTER_UP, BETTER_DOWN)

DIR_POSITIVE = "положительно"
DIR_NEGATIVE = "отрицательно"
DIRECTIONS = (DIR_POSITIVE, DIR_NEGATIVE)


@dataclass
class Dataset:
    """Проверенная витрина во внутреннем формате (английские имена колонок)."""

    goals: pd.DataFrame
    facts: pd.DataFrame  # month — строка 'YYYY-MM'
    links: pd.DataFrame
    source: str = ""
    months: list[str] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)

    def kpi_names(self) -> list[str]:
        return list(self.goals["kpi"])

    def summary(self) -> dict:
        return {
            "source": self.source,
            "first_month": self.months[0] if self.months else None,
            "last_month": self.months[-1] if self.months else None,
            "months": len(self.months),
            "kpis": int(self.goals["kpi"].nunique()),
            "goals": int(self.goals["goal"].nunique()),
            "segments": list(self.segments),
        }
