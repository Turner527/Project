"""Стратегический формат (листы goals / facts / links) -> универсальная таблица.

Так старые демо-данные работают на том же движке, что и любые плоские отчёты:
KPI становятся показателями, сегмент — измерением, план — столбцом плана, а цели, владельцы и связи
сохраняются как дополнительные сведения.
"""
import pandas as pd

from data.model import Metric, Spec, Table, build_table
from data.schema import BETTER_UP, KIND_FLOW, KIND_RATIO, KIND_STOCK, Dataset

DATE_COLUMN = "Месяц"
DIMENSION = "Сегмент"
PLAN_SUFFIX = " · план"
KIND_MAP = {KIND_FLOW: "sum", KIND_STOCK: "last", KIND_RATIO: "mean"}


def to_table(ds: Dataset, source: str = "") -> Table:
    facts = ds.facts
    fact = facts.pivot_table(index=["month", "segment"], columns="kpi", values="fact", aggfunc="sum")
    plan = facts.pivot_table(index=["month", "segment"], columns="kpi", values="plan", aggfunc="sum")
    wide = fact.join(plan.add_suffix(PLAN_SUFFIX)).reset_index()
    wide[DATE_COLUMN] = pd.to_datetime(wide.pop("month") + "-01")
    wide = wide.rename(columns={"segment": DIMENSION})

    metrics = [
        Metric(name=r.kpi, kind=KIND_MAP[r.kind], column=r.kpi, unit=r.unit,
               better="up" if r.better == BETTER_UP else "down", plan_column=r.kpi + PLAN_SUFFIX,
               goal=r.goal, owner=r.owner)
        for r in ds.goals.itertuples()
    ]
    spec = Spec(date=DATE_COLUMN, dimensions=[DIMENSION], metrics=metrics)
    table = build_table(wide, spec, source or ds.source)
    goals = ds.goals.drop_duplicates("goal")
    table.goals = {r.goal: {"parent": (r.parent_goal if pd.notna(r.parent_goal) else None), "owner": r.owner}
                   for r in goals.itertuples()}
    table.links = [{"cause": r.cause_kpi, "effect": r.effect_kpi, "direction": r.direction}
                   for r in ds.links.itertuples()]
    return table
