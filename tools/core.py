"""Инструменты над витриной. Все числа считает этот модуль; модель лишь выбирает вызовы.

Правила расчёта (фиксируются в README):
  * поток   — сумма по месяцам периода и по сегментам;
  * остаток — значение на последний месяц периода, сумма по сегментам;
  * доля    — простое среднее по месяцам и сегментам.
Вклад сегмента = его отклонение / сумма отклонений сегментов (может быть больше 100% или
отрицательным, если другие сегменты отклоняются в противоположную сторону).
"""
import functools
import math
import re
import uuid
from collections import Counter
from typing import Annotated, Callable

import pandas as pd
from pydantic import Field

from data.schema import BETTER_UP, DIR_POSITIVE, KIND_FLOW, KIND_RATIO, KIND_STOCK
from tools import datastore

SIGNIFICANT_PCT = 3.0  # |отклонение| от плана в процентах, начиная с которого оно значимо
DEFAULT_WINDOW = 6  # месяцев, если период не задан
MAX_SEGMENTS = 8
MAX_RELATED = 6
MIN_MATCH_SCORE = 0.5
MIN_SECOND_SEGMENT_PCT = 10.0  # сценарий «два сегмента» строится, только если второй даёт не меньше 10% разрыва

KpiArg = Annotated[str, Field(description="Точное название KPI (берите из результата find_kpis_by_goal)")]
FromArg = Annotated[
    str | None,
    Field(description="Первый месяц периода, формат ГГГГ-ММ. Не указан — берётся 6 месяцев до конца периода"),
]
ToArg = Annotated[
    str | None, Field(description="Последний месяц периода, формат ГГГГ-ММ. Не указан — последний месяц в данных")
]


LastArg = Annotated[
    int | None,
    Field(description="Только последние N месяцев до конца данных (для «за последние 3 месяца»): "
          "тогда period_from указывать не нужно", ge=1, le=60),
]


class ToolError(Exception):
    def __init__(self, message: str, **extra):
        super().__init__(message)
        self.extra = extra


def _new_call_id() -> str:
    return "c_" + uuid.uuid4().hex[:8]


def _tool(fn: Callable) -> Callable:
    """Добавляет call_id и имя инструмента; ошибки возвращает как JSON, а не исключением."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> dict:
        try:
            return {"call_id": _new_call_id(), "tool": fn.__name__, **fn(*args, **kwargs)}
        except ToolError as exc:
            return {"call_id": _new_call_id(), "tool": fn.__name__, "error": str(exc), **exc.extra}

    return wrapper


def _r(x, nd: int = 2):
    """Округление; None, NaN и бесконечности превращаются в None (иначе JSON получился бы невалидным)."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


# ---------- сопоставление названий (лёгкая «стемминг»-нормализация по первым 4 буквам) ----------

_STOP = (
    "почему мы нам что как какие какой какая делать делаем отстаем отстаём отстаём цели цель целью показатель "
    "показателя показатели показателю kpi кпи это для при про над или его они там где стоит нужно надо можно период "
    "месяц месяцев последние последних последний три шесть расскажи объясни покажи"
).split()
_STOP_STEMS = {w.replace("ё", "е")[:4] for w in _STOP}


def _stems(text: str) -> set[str]:
    words = re.findall(r"[a-zа-я0-9]+", text.lower().replace("ё", "е"))
    return {w[:4] for w in words if len(w) > 2 and w[:4] not in _STOP_STEMS}


def _scores(query: str, docs: dict[str, str]) -> list[tuple[str, float]]:
    """Совпадение запроса с названиями: среднее двух долей (с весами IDF) — какая часть названия
    найдена в запросе и какая часть запроса объяснена названием. Лучшие — первыми."""
    doc_stems = {k: _stems(v) for k, v in docs.items()}
    df = Counter(s for st in doc_stems.values() for s in st)
    idf = {s: math.log(1 + len(docs) / n) for s, n in df.items()}
    q = {s for s in _stems(query) if s in idf}  # слова, которых нет ни в одном названии, не штрафуют
    q_total = sum(idf[s] for s in q)
    out = []
    for key, st in doc_stems.items():
        total = sum(idf[s] for s in st)
        if total and q_total:
            hit = sum(idf[s] for s in st & q)
            full_mention = st <= q  # название целиком в запросе, посторонние слова не в счёт
            out.append((key, 1.0 if full_mention else round(0.5 * hit / total + 0.5 * hit / q_total, 3)))
    return sorted(out, key=lambda kv: (-kv[1], kv[0]))


def _resolve_kpi(name: str) -> str:
    """Название KPI -> каноническое. Допускает регистр и мелкие неточности, иначе бросает ToolError."""
    ds = datastore.get_dataset()
    kpis = ds.kpi_names()
    for k in kpis:
        if k.casefold() == name.strip().casefold():
            return k
    ranked = _scores(name, {k: k for k in kpis})
    if ranked and ranked[0][1] >= 0.75 and (len(ranked) < 2 or ranked[1][1] < ranked[0][1] - 0.15):
        return ranked[0][0]
    raise ToolError(
        f"KPI «{name}» не найден. Возьмите точное название из find_kpis_by_goal.",
        suggestions=[k for k, s in ranked[:3] if s > 0.3],
    )


# ---------- периоды и агрегация ----------

_MONTH_RE = re.compile(r"^\s*(\d{4})-(\d{1,2})(?:-\d{1,2})?\s*$")


def _parse_month(value: str) -> str:
    m = _MONTH_RE.match(value)
    if not m or not 1 <= int(m.group(2)) <= 12:
        raise ToolError(f"Неверный формат месяца «{value}», нужен ГГГГ-ММ, например 2026-04.")
    return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"


def _shift(month: str, n: int) -> str:
    return str(pd.Period(month, "M") + n)


def _kpi_frame(kpi: str) -> tuple[pd.DataFrame, dict]:
    ds = datastore.get_dataset()
    meta = ds.goals[ds.goals["kpi"] == kpi].iloc[0].to_dict()
    return ds.facts[ds.facts["kpi"] == kpi], meta


def _period(
    dfk: pd.DataFrame, period_from: str | None, period_to: str | None, last_months: int | None = None
) -> tuple[str, str, str | None]:
    months = sorted(dfk["month"].unique())
    first, last = months[0], months[-1]
    end = _parse_month(period_to) if period_to else last
    if period_from:
        start = _parse_month(period_from)
    else:
        start = _shift(end, -((last_months or DEFAULT_WINDOW) - 1))
    if start > end:
        raise ToolError(f"Начало периода {start} позже конца {end}.")
    if end < first or start > last:
        raise ToolError(f"В запрошенном периоде {start}..{end} данных нет. Данные есть с {first} по {last}.")
    note = None
    if end > last or start < first:
        note = f"Период {start}..{end} обрезан по границам данных: {max(start, first)}..{min(end, last)}."
    return max(start, first), min(end, last), note


def _seg_table(dfk: pd.DataFrame, kind: str, start: str, end: str) -> pd.DataFrame:
    sub = dfk[(dfk["month"] >= start) & (dfk["month"] <= end)]
    if kind == KIND_FLOW:
        return sub.groupby("segment", sort=False)[["plan", "fact"]].sum()
    if kind == KIND_STOCK:
        return sub[sub["month"] == sub["month"].max()].set_index("segment")[["plan", "fact"]]
    return sub.groupby("segment", sort=False)[["plan", "fact"]].mean()


def _totals(seg: pd.DataFrame, kind: str) -> tuple[float, float]:
    agg = seg.mean() if kind == KIND_RATIO else seg.sum()
    return float(agg["plan"]), float(agg["fact"])


def _deviation(plan: float, fact: float, better: str) -> dict:
    dev = fact - plan
    pct = dev / plan * 100 if plan else None
    unfavorable = dev < 0 if better == BETTER_UP else dev > 0
    significant = pct is not None and abs(pct) >= SIGNIFICANT_PCT
    if pct is None:
        assessment = "план равен нулю, процент отклонения не определён"
    elif not significant:
        assessment = "в пределах допуска"
    else:
        assessment = "хуже плана" if unfavorable else "лучше плана"
    return {
        "plan": _r(plan),
        "fact": _r(fact),
        "deviation_abs": _r(dev),
        "deviation_pct": _r(pct, 1),
        "unfavorable": bool(unfavorable and dev != 0),
        "significant": bool(significant),
        "assessment": assessment,
    }


def _kpi_deviation(kpi: str, start: str, end: str) -> tuple[dict, dict]:
    dfk, meta = _kpi_frame(kpi)
    seg = _seg_table(dfk, meta["kind"], start, end)
    plan, fact = _totals(seg, meta["kind"])
    return _deviation(plan, fact, meta["better"]), meta


def _breakdown(dfk: pd.DataFrame, meta: dict, start: str, end: str) -> dict:
    kind, better = meta["kind"], meta["better"]
    seg = _seg_table(dfk, kind, start, end)
    plan, fact = _totals(seg, kind)
    total_dev = fact - plan
    dev = seg["fact"] - seg["plan"]
    denom = float(dev.sum())
    divisor = len(seg) if kind == KIND_RATIO else 1
    rows = []
    for name in seg.index:
        d = float(dev[name])
        p = float(seg["plan"][name])
        rows.append(
            {
                "segment": name,
                "plan": _r(p),
                "fact": _r(seg["fact"][name]),
                "deviation_abs": _r(d),
                "deviation_pct": _r(d / p * 100 if p else None, 1),
                "contribution_abs": _r(d / divisor),
                "contribution_pct": _r(d / denom * 100 if abs(denom) > 1e-12 else None, 1),
                "unfavorable": bool((d < 0) if better == BETTER_UP else (d > 0)) and d != 0,
            }
        )
    sign = 1 if total_dev >= 0 else -1  # сначала сегменты, тянущие в сторону общего отклонения
    rows.sort(key=lambda r: -(r["contribution_abs"] or 0) * sign)
    top = rows[0] if rows and abs(total_dev) > 1e-12 else None
    return {
        "total": _deviation(plan, fact, better),
        "segments": rows[:MAX_SEGMENTS],
        "segments_total": len(rows),
        "top_segment": top["segment"] if top else None,
        "top_segment_contribution_pct": top["contribution_pct"] if top else None,
        "all_rows": rows,  # для сценариев; наружу не отдаётся
    }


def _public_breakdown(b: dict) -> dict:
    return {k: v for k, v in b.items() if k != "all_rows"}


def _related(kpi: str, main_dev: dict, start: str, end: str) -> list[dict]:
    ds = datastore.get_dataset()
    links = ds.links
    length = len(pd.period_range(start, end, freq="M"))
    out = []
    for _, link in links[(links["cause_kpi"] == kpi) | (links["effect_kpi"] == kpi)].iterrows():
        is_cause = link["effect_kpi"] == kpi  # связанный KPI влияет на выбранный
        other = link["cause_kpi"] if is_cause else link["effect_kpi"]
        dfk, meta = _kpi_frame(other)
        months = sorted(dfk["month"].unique())
        o_start, o_end = max(start, months[0]), min(end, months[-1])
        if o_start > o_end:
            continue
        seg = _seg_table(dfk, meta["kind"], o_start, o_end)
        plan, fact = _totals(seg, meta["kind"])
        dev = _deviation(plan, fact, meta["better"])

        prev_start, prev_end = _shift(start, -length), _shift(start, -1)
        change = None
        if prev_start >= months[0]:
            prev_plan, prev_fact = _totals(_seg_table(dfk, meta["kind"], prev_start, prev_end), meta["kind"])
            change = _r((fact / prev_fact - 1) * 100, 1) if prev_fact else None

        seg_dev = ((seg["fact"] - seg["plan"]) / seg["plan"].replace(0, float("nan")) * 100).dropna()
        worst = seg_dev.abs().idxmax() if len(seg_dev) else None
        worst_pct = float(seg_dev[worst]) if worst is not None else None
        consistent = None
        if dev["significant"] and main_dev["significant"]:
            same_sign = (dev["deviation_abs"] > 0) == (main_dev["deviation_abs"] > 0)
            consistent = same_sign if link["direction"] == DIR_POSITIVE else not same_sign
        out.append(
            {
                "kpi": other,
                "relation": "влияет на выбранный KPI" if is_cause else "зависит от выбранного KPI",
                "link_direction": link["direction"],
                "unit": meta["unit"],
                **dev,
                "change_vs_previous_period_pct": change,
                "most_deviating_segment": worst if worst_pct is not None and abs(worst_pct) >= SIGNIFICANT_PCT else None,
                "most_deviating_segment_pct": _r(worst_pct, 1)
                if worst_pct is not None and abs(worst_pct) >= SIGNIFICANT_PCT
                else None,
                "consistent_with_link": consistent,
            }
        )
    out.sort(key=lambda r: -abs(r["deviation_pct"] or 0))
    return out[:MAX_RELATED]


def _onset(dfk: pd.DataFrame, meta: dict, end: str) -> tuple[str | None, int]:
    """Сколько последних месяцев подряд KPI значимо хуже плана и с какого они начались."""
    sub = dfk[dfk["month"] <= end]
    grouped = sub.groupby("month")[["plan", "fact"]]
    monthly = grouped.mean() if meta["kind"] == KIND_RATIO else grouped.sum()
    run, first = 0, None
    for month in sorted(monthly.index, reverse=True):
        dev = _deviation(float(monthly.loc[month, "plan"]), float(monthly.loc[month, "fact"]), meta["better"])
        if dev["significant"] and dev["unfavorable"]:
            run, first = run + 1, month
        else:
            break
    return (first if run >= 2 else None), run


def _scenarios(rows: list[dict], total_dev: float, unit: str) -> list[dict]:
    """Арифметика «что если закрыть отставание в сегменте»; без допущений о причинах."""
    gap = abs(total_dev)
    pushing = [r for r in rows if (r["contribution_abs"] or 0) * total_dev > 0]
    plan = []
    if pushing:
        plan += [("S1", pushing[:1], 100), ("S2", pushing[:1], 50)]
    if len(pushing) >= 2 and (pushing[1]["contribution_pct"] or 0) >= MIN_SECOND_SEGMENT_PCT:
        plan.append(("S3", pushing[:2], 100))
    out = []
    for sid, segs, share in plan:
        effect = sum(abs(r["contribution_abs"]) for r in segs) * share / 100
        out.append(
            {
                "scenario_id": sid,
                "segments": [r["segment"] for r in segs],
                "close_share_pct": share,
                "effect_abs": _r(effect),
                "unit": unit,
                "gap_closed_pct": _r(effect / gap * 100, 1),
                "remaining_gap_abs": _r(gap - effect),
            }
        )
    return out


# ---------- публичные инструменты ----------


@_tool
def find_kpis_by_goal(
    query: Annotated[str, Field(description="Название цели или KPI своими словами, например «рост розничного кредитования»")],
) -> dict:
    """Найти цели банка и их KPI по названию. Вызывайте первым, если в вопросе названа цель или показатель."""
    ds = datastore.get_dataset()
    goals = ds.goals.drop_duplicates("goal")
    goal_hits = [(g, s) for g, s in _scores(query, {g: g for g in goals["goal"]}) if s >= MIN_MATCH_SCORE][:3]
    kpi_hits = [(k, s) for k, s in _scores(query, {k: k for k in ds.kpi_names()}) if s >= MIN_MATCH_SCORE][:4]
    matches = []
    for goal, score in goal_hits:
        rows = ds.goals[ds.goals["goal"] == goal]
        matches.append(
            {
                "goal": goal,
                "match_score": score,
                "parent_goal": rows["parent_goal"].iloc[0] if pd.notna(rows["parent_goal"].iloc[0]) else None,
                "kpis": [
                    {"kpi": r.kpi, "unit": r.unit, "kind": r.kind, "better": r.better} for r in rows.itertuples()
                ],
            }
        )
    kpi_matches = []
    for kpi, score in kpi_hits:
        r = ds.goals[ds.goals["kpi"] == kpi].iloc[0]
        kpi_matches.append({"kpi": kpi, "goal": r["goal"], "unit": r["unit"], "match_score": score})
    result = {"query": query, "goal_matches": matches, "kpi_matches": kpi_matches}
    if not matches and not kpi_matches:
        result["note"] = "Совпадений нет. Уточните название; список целей приведён в available_goals."
        result["available_goals"] = list(goals["goal"])
    return result


@_tool
def get_deviation(kpi: KpiArg, period_from: FromArg = None, period_to: ToArg = None, last_months: LastArg = None) -> dict:
    """Отклонение факта от плана по KPI за период: план, факт, разница, процент, хуже или лучше плана."""
    name = _resolve_kpi(kpi)
    dfk, meta = _kpi_frame(name)
    start, end, note = _period(dfk, period_from, period_to, last_months)
    dev, _ = _kpi_deviation(name, start, end)
    result = {
        "kpi": name,
        "unit": meta["unit"],
        "kind": meta["kind"],
        "better": meta["better"],
        "period": {"from": start, "to": end, "months_count": len(pd.period_range(start, end, freq="M"))},
        **dev,
    }
    if note:
        result["note"] = note
    return result


@_tool
def break_down_by_segment(kpi: KpiArg, period_from: FromArg = None, period_to: ToArg = None, last_months: LastArg = None) -> dict:
    """Разложить отклонение KPI по сегментам клиентов: план, факт и вклад каждого сегмента в процентах."""
    name = _resolve_kpi(kpi)
    dfk, meta = _kpi_frame(name)
    start, end, note = _period(dfk, period_from, period_to, last_months)
    b = _breakdown(dfk, meta, start, end)
    result = {
        "kpi": name,
        "unit": meta["unit"],
        "period": {"from": start, "to": end},
        **_public_breakdown(b),
    }
    if meta["kind"] == KIND_RATIO:
        result["aggregation_note"] = "Для относительного показателя итог — простое среднее по сегментам."
    if not b["total"]["significant"]:
        result["interpretation_note"] = "Общее отклонение незначимо: вклады сегментов трактовать осторожно."
    if note:
        result["note"] = note
    return result


@_tool
def get_related_kpis(kpi: KpiArg, period_from: FromArg = None, period_to: ToArg = None, last_months: LastArg = None) -> dict:
    """Связанные KPI из таблицы влияний и их динамика за тот же период: отклонение от плана, изменение к
    предыдущему периоду, сегмент с наибольшим отклонением и согласуется ли это с направлением связи."""
    name = _resolve_kpi(kpi)
    dfk, meta = _kpi_frame(name)
    start, end, note = _period(dfk, period_from, period_to, last_months)
    main_dev, _ = _kpi_deviation(name, start, end)
    result = {"kpi": name, "period": {"from": start, "to": end}, "related": _related(name, main_dev, start, end)}
    if not result["related"]:
        result["note"] = "Для этого KPI в таблице связей нет записей."
    if note:
        result["note_period"] = note
    return result


@_tool
def get_goal_info(
    name: Annotated[str, Field(description="Название цели или KPI")],
) -> dict:
    """Владелец цели, родительская цель (и её владелец), дочерние цели и KPI цели."""
    ds = datastore.get_dataset()
    goals = ds.goals.drop_duplicates("goal").set_index("goal")
    ranked = _scores(name, {g: g for g in goals.index})
    kpi_ranked = _scores(name, {k: k for k in ds.kpi_names()})
    if kpi_ranked and kpi_ranked[0][1] >= MIN_MATCH_SCORE and kpi_ranked[0][1] > (ranked[0][1] if ranked else 0):
        goal = ds.goals[ds.goals["kpi"] == kpi_ranked[0][0]].iloc[0]["goal"]
    elif ranked and ranked[0][1] >= MIN_MATCH_SCORE:
        goal = ranked[0][0]
    else:
        raise ToolError(f"Цель или KPI «{name}» не найдены.", available_goals=list(goals.index))
    parent = goals.loc[goal, "parent_goal"]
    parent = parent if pd.notna(parent) else None
    children = [
        {"goal": g, "owner": r["owner"]} for g, r in goals.iterrows() if r["parent_goal"] == goal
    ]
    return {
        "goal": goal,
        "owner": goals.loc[goal, "owner"],
        "parent_goal": parent,
        "parent_owner": goals.loc[parent, "owner"] if parent else None,
        "child_goals": children,
        "kpis": [{"kpi": r.kpi, "unit": r.unit} for r in ds.goals[ds.goals["goal"] == goal].itertuples()],
    }


@_tool
def diagnose_deviation(kpi: KpiArg, period_from: FromArg = None, period_to: ToArg = None, last_months: LastArg = None) -> dict:
    """Главный инструмент для вопросов «почему отстаём»: за один вызов даёт отклонение KPI, момент начала
    отставания, разбивку по сегментам с вкладами, связанные KPI и арифметику сценариев закрытия разрыва.
    Все числа для итоговой записки берите из его результата."""
    name = _resolve_kpi(kpi)
    dfk, meta = _kpi_frame(name)
    start, end, note = _period(dfk, period_from, period_to, last_months)
    b = _breakdown(dfk, meta, start, end)
    total = b["total"]
    onset, run = _onset(dfk, meta, end)
    ds = datastore.get_dataset()
    goals = ds.goals.drop_duplicates("goal").set_index("goal")
    parent = goals.loc[meta["goal"], "parent_goal"]
    result = {
        "kpi": name,
        "unit": meta["unit"],
        "goal": meta["goal"],
        "owner": meta["owner"],
        "parent_goal": parent if pd.notna(parent) else None,
        "period": {"from": start, "to": end, "months_count": len(pd.period_range(start, end, freq="M"))},
        "deviation": total,
        "months_off_plan_in_row": run,
        "off_plan_since": onset,
        "breakdown": _public_breakdown(b),
        "related_kpis": _related(name, total, start, end),
    }
    if total["significant"] and total["unfavorable"]:
        result["recovery_scenarios"] = _scenarios(b["all_rows"], total["deviation_abs"], meta["unit"])
    else:
        result["recovery_scenarios"] = []
        result["scenarios_note"] = "Отклонение незначимо или в лучшую сторону: сценарии закрытия разрыва не строились."
    if meta["kind"] == KIND_RATIO:
        result["aggregation_note"] = "Для относительного показателя итог — простое среднее по сегментам."
    if note:
        result["note"] = note
    return result


TOOLS: dict[str, Callable[..., dict]] = {
    fn.__name__: fn
    for fn in (find_kpis_by_goal, get_deviation, break_down_by_segment, get_related_kpis, get_goal_info, diagnose_deviation)
}
