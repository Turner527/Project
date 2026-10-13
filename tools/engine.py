"""Расчёты по универсальной таблице: окна периодов, базы сравнения, вклады по измерениям.

Правила (описаны и в README):
  * показатель: sum — сумма, mean — простое среднее, last — значение на конец периода, ratio — отношение сумм
    двух столбцов (CTR = сумма кликов / сумма показов), а не среднее готовых процентов;
  * база сравнения: план (столбец плана), предыдущий период той же длины, тот же период год назад;
  * вклад значения измерения: для сумм — его изменение; для отношений N/D — разложение «ставка и структура»:
    w1·(r1 − R0) − w0·(r0 − R0), где w — доля значения в объёме (знаменателе), r — его ставка, R0 — базовое среднее.
    Сумма вкладов точно равна изменению итога; значение, у которого ставка не менялась, а сдвинулась лишь доля
    в объёме, почти ничего не получает;
  * вклад в процентах = вклад / изменение итога (может быть больше 100% или отрицательным, если другие значения
    идут в противоположную сторону).
"""
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.model import Metric, Table, parse_expr
from data.parsing import parse_date_value
from tools.text import scores, stems

PLAN, PREV, YOY = "plan", "prev", "yoy"
BASE_LABEL = {PLAN: "план", PREV: "предыдущий период", YOY: "тот же период год назад"}
BASE_GEN = {PLAN: "плана", PREV: "предыдущего периода", YOY: "того же периода год назад"}
BASE_ALIASES = {"plan": PLAN, "план": PLAN, "prev": PREV, "previous": PREV, "предыдущий": PREV, "yoy": YOY,
                "год": YOY, "year": YOY}
DEFAULT_CAP = {"day": 30, "week": 8, "month": 6, "quarter": 4, "year": 2}
YOY_SHIFT = {"day": 365, "week": 52, "month": 12, "quarter": 4, "year": 1}
UNIT_LABEL = {"day": "нед.", "week": "нед.", "month": "мес.", "quarter": "кв.", "year": "г."}
GRAIN_LABEL = {"day": "дни", "week": "недели", "month": "месяцы", "quarter": "кварталы", "year": "годы"}
MIN_COVERAGE = 0.9  # доля периодов, которые должны присутствовать в базовом окне
MIN_SECOND_SHARE = 10.0  # сценарий «два значения» строится, если второе даёт не меньше 10% изменения
MAX_DRILL_VALUE_SHARE = 40.0  # углубляемся в значение, если оно даёт не меньше 40% изменения
MAX_SCAN_DIMENSIONS = 6
MAX_POINTS_PER_CHART = 40  # ряд для графика укрупняется до такого числа точек (день → неделя → месяц …)
# «локальные отклонения»: заметное ухудшение внутри значения измерения при нормальном итоге
LOCAL_MIN_OWN_PCT = 10.0  # собственное изменение значения, %
LOCAL_MIN_WEIGHT_PCT = 2.0  # доля значения в базе (в объёме для отношений), %
LOCAL_MIN_EFFECT_PCT = 0.5  # вклад значения в итог, % от базы итога
MAX_LOCAL = 3


class ToolError(Exception):
    def __init__(self, message: str, **extra):
        super().__init__(message)
        self.extra = extra


def r(x, nd: int = 2):
    """Округление для ответов инструментов; None, NaN и бесконечности превращаются в None."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    if nd == 2 and 0 < abs(x) < 0.1:
        nd = 4
    return round(x, nd)


@dataclass
class Window:
    a: int
    b: int

    @property
    def n(self) -> int:
        return self.b - self.a + 1


@dataclass
class Baseline:
    kind: str
    win: Window | None = None

    @property
    def label(self) -> str:
        return BASE_LABEL[self.kind]


# ---------- периоды ----------


def _user_ts(text, end: bool) -> pd.Timestamp:
    t = str(text).strip()
    try:
        if m := re.fullmatch(r"(\d{4})-(\d{1,2})", t):
            p = pd.Period(f"{int(m[1]):04d}-{int(m[2]):02d}", "M")
        elif m := re.fullmatch(r"(\d{4})-?[qQ]([1-4])", t):
            p = pd.Period(f"{m[1]}Q{m[2]}", "Q")
        elif re.fullmatch(r"\d{4}", t):
            p = pd.Period(t, "Y")
        else:
            ts = parse_date_value(t)
            if ts is None:
                raise ValueError
            return ts
        return (p.end_time if end else p.start_time).normalize()
    except (ValueError, TypeError):
        raise ToolError(f"Не удалось понять дату «{text}». Допустимо: 2026-09-15, 2026-09, 2026-Q3, 2026.") from None


def _ord(t: Table, ts: pd.Timestamp) -> int:
    return pd.Period(ts, freq=t.freq).ordinal


def _coverage(t: Table, w: Window) -> float:
    present = t.periods_present()
    return (np.searchsorted(present, w.b, "right") - np.searchsorted(present, w.a, "left")) / w.n


def resolve_window(t: Table, period_from=None, period_to=None, last_months=None, last_days=None):
    """Окно анализа: явные даты > последние N дней/месяцев > окно по умолчанию. Возвращает (Window, примечание)."""
    if t.grain is None:
        return Window(0, 0), None
    span = t.p_max - t.p_min + 1
    length = max(1, min(DEFAULT_CAP[t.grain], span // 2))
    end = _ord(t, _user_ts(period_to, True)) if period_to else t.p_max
    start = None
    if last_days:
        if t.grain not in ("day", "week"):
            raise ToolError(f"Данные по периодам «{GRAIN_LABEL[t.grain]}»: параметр last_days не подходит, используйте last_months.")
        length = last_days if t.grain == "day" else math.ceil(last_days / 7)
    elif last_months:
        if t.grain == "month":
            length = last_months
        elif t.grain == "quarter":
            length = math.ceil(last_months / 3)
        elif t.grain == "year":
            raise ToolError("Данные годовые: используйте period_from и period_to.")
        else:  # дни и недели: последние N календарных месяцев до конца окна
            end_ts = pd.Period(ordinal=end, freq=t.freq).end_time
            start = _ord(t, (end_ts.to_period("M") - (last_months - 1)).start_time)
    if period_from:
        start = _ord(t, _user_ts(period_from, False))
    elif start is None:
        start = end - length + 1
    if start > end:
        raise ToolError(f"Начало периода {t.label(start)} позже конца {t.label(end)}.")
    if end < t.p_min or start > t.p_max:
        raise ToolError(f"В запрошенном периоде {t.label(start)}..{t.label(end)} данных нет. "
                        f"Данные есть с {t.label(t.p_min)} по {t.label(t.p_max)}.")
    note = None
    if start < t.p_min or end > t.p_max:
        note = (f"Период {t.label(start)}..{t.label(end)} обрезан по границам данных: "
                f"{t.label(max(start, t.p_min))}..{t.label(min(end, t.p_max))}.")
    win = Window(max(start, t.p_min), min(end, t.p_max))
    cov = _coverage(t, win)
    if cov < MIN_COVERAGE:
        gap = f"В выбранном периоде есть данные лишь за {round(cov * win.n)} из {win.n} периодов."
        note = f"{note} {gap}" if note else gap
    return win, note


def resolve_baseline(t: Table, m: Metric, win: Window, requested: str | None = None) -> Baseline:
    """База сравнения: план, предыдущий период или год назад. «auto» берёт первую доступную в этом порядке."""
    req = (requested or "auto").strip().lower()
    if req in ("auto", "авто", ""):
        order = [PLAN, PREV, YOY]
    elif req in BASE_ALIASES:
        order = [BASE_ALIASES[req]]
    else:
        raise ToolError(f"Неизвестная база сравнения «{requested}». Допустимо: plan, prev, yoy, auto.")
    reasons = []
    for kind in order:
        if kind == PLAN:
            if m.has_plan:
                return Baseline(PLAN)
            reasons.append("у показателя нет плана")
            continue
        if t.grain is None:
            reasons.append("в данных нет дат")
            continue
        shift = win.n if kind == PREV else YOY_SHIFT[t.grain]
        w = Window(win.a - shift, win.b - shift)
        cov = _coverage(t, w)
        if cov >= MIN_COVERAGE:
            return Baseline(kind, w)
        present = round(cov * w.n)
        reasons.append(f"для «{BASE_LABEL[kind]}» ({t.label(w.a)}..{t.label(w.b)}) в данных только {present} из {w.n} периодов")
    raise ToolError("Не с чем сравнивать: " + "; ".join(reasons) + ". Укажите более короткий период или загрузите отчёт с планом.")


# ---------- расчёт ----------


def _where_rows(df: pd.DataFrame, where) -> pd.DataFrame:
    for dim, value in where:
        df = df[df[dim] == value]
    return df


def _rows(t: Table, win: Window, where) -> pd.DataFrame:
    df = t.df
    if t.grain is not None:
        df = df[(df["_p"] >= win.a) & (df["_p"] <= win.b)]
    return _where_rows(df, where)


def judge(diff, pct, better: str, sig_pct: float, base_kind: str) -> tuple[bool, bool, str]:
    """(значимо, неблагоприятно, словесная оценка)."""
    gen = BASE_GEN[base_kind]
    unfavorable = better != "none" and diff is not None and diff != 0 and ((diff < 0) if better == "up" else (diff > 0))
    if pct is None:
        return False, False, "база равна нулю, процент отклонения не определён"
    significant = abs(pct) >= sig_pct
    if not significant:
        return False, False, "в пределах допуска"
    if better == "none":
        return True, False, f"выше {gen}" if diff > 0 else f"ниже {gen}"
    prefix = {PLAN: "", PREV: ", чем в предыдущем периоде", YOY: ", чем год назад"}[base_kind]
    word = "хуже" if unfavorable else "лучше"
    return True, unfavorable, f"{word} плана" if base_kind == PLAN else f"{word}{prefix}"


@dataclass
class Comparison:
    metric: Metric
    win: Window
    base: Baseline
    v1: float | None
    v0: float | None
    diff: float | None
    pct: float | None
    significant: bool
    unfavorable: bool
    assessment: str
    groups: pd.DataFrame | None = None  # индекс — значения измерения; v1, v0, contrib, share, own_pct
    n_rows: int = 0
    extra: dict = field(default_factory=dict)


def _with_plan(df: pd.DataFrame, m: Metric) -> pd.DataFrame:
    """Строки, по которым есть и план, и (для среднего) факт: план без факта в среднем не участвует, иначе числитель
    плана и число наблюдений относились бы к разным строкам."""
    df = df[df[m.P].notna()]
    return df[df[m.D] > 0] if m.kind == "mean" else df


def compare(t: Table, m: Metric, win: Window, base: Baseline, by: str | None = None, where=()) -> Comparison:
    ratio = m.kind in ("mean", "ratio")
    F = _rows(t, win, where)
    if base.kind == PLAN:
        F = _with_plan(F, m)
        B = F
    else:
        B = _rows(t, base.win, where)
    if m.kind == "last":
        if len(F):
            F = F[F["_p"] == F["_p"].max()]
        B = F if base.kind == PLAN else (B[B["_p"] == B["_p"].max()] if len(B) else B)
    if F.empty:
        raise ToolError("В выбранном периоде (с учётом фильтров) нет данных.")
    n0 = m.P if base.kind == PLAN else m.N
    N1, N0 = float(F[m.N].sum()), float(B[n0].sum())
    D1 = float(F[m.D].sum()) if ratio else None
    D0 = (D1 if base.kind == PLAN else float(B[m.D].sum())) if ratio else None
    if ratio and (not D1 or not D0):
        raise ToolError("Знаменатель показателя в одном из периодов равен нулю: отношение не определено.")
    s = m.scale
    v1 = s * N1 / D1 if ratio else s * N1
    v0 = s * N0 / D0 if ratio else s * N0
    diff = v1 - v0
    pct = diff / abs(v0) * 100 if v0 else None
    sig, unfav, text = judge(diff, pct, m.better, t.spec.significant_pct, base.kind)
    cmp = Comparison(m, win, base, v1, v0, diff, pct, sig, unfav, text, n_rows=len(F))
    if by:
        cols = [m.N] + ([m.D] if ratio else [])
        g1 = F.groupby(by, sort=False)[cols].sum()
        if base.kind == PLAN:
            g0 = F.groupby(by, sort=False)[[m.P]].sum().rename(columns={m.P: m.N})
            if ratio:
                g0[m.D] = g1[m.D]
        else:
            g0 = B.groupby(by, sort=False)[cols].sum()
        index = g1.index.union(g0.index)
        g1, g0 = g1.reindex(index, fill_value=0.0), g0.reindex(index, fill_value=0.0)
        if ratio:
            # «ставка и структура»: взвешенное превышение над базовым средним сейчас минус тогда. Сумма вкладов равна
            # изменению итога, а сдвиг доли объёма при неизменной ставке почти ничего не даёт
            R0 = N0 / D0
            contrib = s * (g1[m.N] / D1 - g0[m.N] / D0 - R0 * (g1[m.D] / D1 - g0[m.D] / D0))
            v1g = s * g1[m.N] / g1[m.D].replace(0, np.nan)
            v0g = s * g0[m.N] / g0[m.D].replace(0, np.nan)
        else:
            v1g, v0g = s * g1[m.N], s * g0[m.N]
            contrib = v1g - v0g
        # доля значения в базе: в объёме (знаменателе) для отношений, в сумме модулей для сумм
        total_abs = float(v0g.abs().sum())
        w0 = g0[m.D] / D0 * 100 if ratio else (v0g.abs() / total_abs * 100 if total_abs else np.nan)
        groups = pd.DataFrame({"v1": v1g, "v0": v0g, "contrib": contrib, "w0": w0})
        groups["own_pct"] = (groups["v1"] - groups["v0"]) / groups["v0"].abs().replace(0, np.nan) * 100
        groups["share"] = groups["contrib"] / diff * 100 if abs(diff) > 1e-12 else np.nan
        groups = groups[(groups["v1"].fillna(0) != 0) | (groups["v0"].fillna(0) != 0) | (groups["contrib"] != 0)]
        sign = 1 if diff >= 0 else -1
        cmp.groups = groups.iloc[np.argsort(-(groups["contrib"].to_numpy() * sign), kind="stable")]
    return cmp


def _window_rows(t: Table, m: Metric, win: Window, where=()) -> pd.DataFrame:
    """Строки окна для показателя; для «на конец периода» — только последний период окна."""
    F = _rows(t, win, where)
    if m.kind == "last" and len(F):
        F = F[F["_p"] == F["_p"].max()]
    if F.empty:
        raise ToolError("В выбранном периоде (с учётом фильтров) нет данных.")
    return F


def value_only(t: Table, m: Metric, win: Window, where=()) -> float:
    """Значение показателя за окно без сравнения: для отчётов без плана и без прошлых периодов."""
    F = _window_rows(t, m, win, where)
    n = float(F[m.N].sum())
    if m.kind in ("mean", "ratio"):
        d = float(F[m.D].sum())
        if not d:
            raise ToolError("Знаменатель показателя равен нулю: отношение не определено.")
        return m.scale * n / d
    return m.scale * n


def structure(t: Table, m: Metric, win: Window, by: str, where=(), top: int = 8) -> list[dict]:
    """Из чего состоит показатель: значение и доля каждого значения измерения, без сравнения с базой.
    Доля — в сумме модулей значений (для отношений — в объёме, то есть в знаменателе)."""
    F = _window_rows(t, m, win, where)
    ratio = m.kind in ("mean", "ratio")
    g = F.groupby(by, sort=False)[[m.N] + ([m.D] if ratio else [])].sum()
    if ratio:
        value = m.scale * g[m.N] / g[m.D].replace(0, np.nan)
        weight = g[m.D]
    else:
        value = m.scale * g[m.N]
        weight = value.abs()
    total = float(weight.sum())
    order = weight.sort_values(ascending=False, kind="stable").index
    rows = [{"value": str(v), "fact": r(value[v]), "share_pct": r(weight[v] / total * 100, 1) if total else None}
            for v in order[:top]]
    if len(order) > top:
        rest = weight[order[top:]]
        rows.append({"value": f"прочие ({len(rest)})", "fact": None, "share_pct": r(float(rest.sum()) / total * 100, 1) if total else None})
    return rows


def group_rows(cmp: Comparison, top: int, dimension: str | None = None) -> list[dict]:
    """Строки для ответа: значения измерения с вкладом; остальные сворачиваются в «прочие»."""
    g = cmp.groups
    rows = []
    for value, row in g.head(top).iterrows():
        rows.append({"value": str(value), "base": r(row["v0"]), "fact": r(row["v1"]), "change_abs": r(row["v1"] - row["v0"]),
                     "change_pct": r(row["own_pct"], 1), "contribution_abs": r(row["contrib"]),
                     "contribution_pct": r(row["share"], 1),
                     "unfavorable": bool(_group_unfavorable(cmp.metric, row))})
    rest = g.iloc[top:]
    if len(rest):
        rows.append({"value": f"прочие ({len(rest)})", "base": None, "fact": None, "change_abs": None, "change_pct": None,
                     "contribution_abs": r(rest["contrib"].sum()), "contribution_pct": r(rest["share"].sum(), 1),
                     "unfavorable": None})
    return rows


def _group_unfavorable(m: Metric, row) -> bool:
    d = row["v1"] - row["v0"]
    if m.better == "none" or d is None or not math.isfinite(d) or d == 0:
        return False
    return d < 0 if m.better == "up" else d > 0


# ---------- с какого периода держится отклонение ----------


def trailing_run(t: Table, m: Metric, win: Window, base: Baseline, total: Comparison, where=()):
    """Сколько последних интервалов подряд показатель значимо хуже базы и с какого они начались.
    Интервалы: месяцы, кварталы, недели; для дневных данных — семидневки, отсчитанные от конца окна."""
    if m.better == "none" or t.grain is None:
        return None, 0, ""
    size = 7 if t.grain == "day" else 1
    ratio = m.kind in ("mean", "ratio")
    df = _where_rows(t.df, where)
    df = df[df["_p"] <= win.b]
    if base.kind == PLAN:
        df = _with_plan(df, m)
    if df.empty:
        return None, 0, ""
    df = df.assign(_b=(win.b - df["_p"]) // size)
    if m.kind == "last":
        df = df[df["_p"] == df.groupby("_b")["_p"].transform("max")]
    cols = [m.N] + ([m.D] if ratio else []) + ([m.P] if base.kind == PLAN else [])
    g = df.groupby("_b")[cols].sum()
    present = df.groupby("_b")["_p"].nunique()
    complete = present >= max(1, 0.6 * size)
    s = m.scale
    flagged_run, k = 0, 0
    while k in g.index and bool(complete.get(k, False)):
        row = g.loc[k]
        if ratio:
            if not row[m.D]:
                break
            value = s * row[m.N] / row[m.D]
        else:
            value = s * row[m.N]
        if base.kind == PLAN:
            if ratio:
                expected = s * row[m.P] / row[m.D]
            else:
                expected = s * row[m.P]
        elif m.kind in ("sum",):
            expected = total.v0 * size / win.n
        else:
            expected = total.v0
        diff = value - expected
        pct = diff / abs(expected) * 100 if expected else None
        sig, unfav, _ = judge(diff, pct, m.better, t.spec.significant_pct, base.kind)
        if not (sig and unfav):
            break
        flagged_run, k = flagged_run + 1, k + 1
    if flagged_run < 2:
        return None, flagged_run, UNIT_LABEL[t.grain]
    return t.label(win.b - size * flagged_run + 1), flagged_run, UNIT_LABEL[t.grain]


# ---------- перебор измерений, углубление, сценарии ----------


def concentration(contrib: np.ndarray, diff: float) -> float:
    """Насколько изменение сосредоточено в немногих значениях измерения: индекс Херфиндаля по вкладам в сторону общего
    изменения, из которого вычтено равномерное распределение (0 — вклад размазан поровну, 1 — всё в одном значении).
    В отличие от доли главного значения, не завышает измерения из двух-трёх значений: 46% среди трёх — почти норма,
    44% среди двенадцати — явный всплеск."""
    k = len(contrib)
    if k < 2 or not math.isfinite(diff) or abs(diff) < 1e-12:
        return 0.0
    push = np.clip(np.nan_to_num(contrib) * (1 if diff > 0 else -1), 0, None)
    total = float(push.sum())
    if total <= 0:
        return 0.0
    hhi = float(((push / total) ** 2).sum())
    return (hhi - 1 / k) / (1 - 1 / k)


def scan_dimensions(t: Table, m: Metric, win: Window, base: Baseline, where=()) -> list[dict]:
    """Вклад значений каждого измерения в изменение; измерения упорядочены по концентрации изменения в немногих значениях."""
    skip = {d for d, _ in where}
    out = []
    for dim in [d for d in t.dims if d not in skip][:MAX_SCAN_DIMENSIONS]:
        cmp = compare(t, m, win, base, by=dim, where=where)
        if cmp.groups is None or len(cmp.groups) < 2:
            continue
        g = cmp.groups
        shares = g["share"].to_numpy()
        top_share = float(shares[0]) if len(shares) and np.isfinite(shares[0]) else None
        cumulative = np.cumsum(np.nan_to_num(shares))
        n80 = int(np.argmax(cumulative >= 80) + 1) if len(cumulative) and (cumulative >= 80).any() else None
        out.append({"dimension": dim, "values_count": int(len(g)), "top_share_pct": r(top_share, 1),
                    "values_for_80pct": n80, "top": group_rows(cmp, 3), "_cmp": cmp,
                    "_score": concentration(g["contrib"].to_numpy(), cmp.diff)})
    out.sort(key=lambda d: (-d["_score"], d["values_count"]))
    for d in out:
        del d["_score"]
    return out


def local_deviations(m: Metric, total: Comparison, scan: list[dict], limit: int = MAX_LOCAL) -> list[dict]:
    """Заметные ухудшения внутри значений измерений, когда общий итог их не показывает: итог в пределах допуска или
    лучше базы, а отдельное значение (отделение, канал, очередь) просело на LOCAL_MIN_OWN_PCT и больше. Мелкие значения
    (доля в базе меньше LOCAL_MIN_WEIGHT_PCT) и значения, почти не двигающие итог, не считаются: это шум."""
    if m.better == "none" or total.v0 in (None, 0) or (total.significant and total.unfavorable):
        return []
    found = []
    for d in scan:
        g = d["_cmp"].groups
        if g is None:
            continue
        for value, row in g.iterrows():
            own, weight = row["own_pct"], row["w0"]
            effect = abs(row["contrib"]) / abs(total.v0) * 100
            if (_group_unfavorable(m, row) and math.isfinite(own) and abs(own) >= LOCAL_MIN_OWN_PCT
                    and math.isfinite(weight) and weight >= LOCAL_MIN_WEIGHT_PCT and effect >= LOCAL_MIN_EFFECT_PCT):
                found.append({"dimension": d["dimension"], "value": str(value), "base": r(row["v0"]), "fact": r(row["v1"]),
                              "change_pct": r(own, 1), "base_weight_pct": r(weight, 1),
                              "effect_on_total_pct": r(row["contrib"] / total.v0 * 100, 1)})
    found.sort(key=lambda x: -abs(x["effect_on_total_pct"]))
    return found[:limit]


def drill_down(t: Table, m: Metric, win: Window, base: Baseline, where, primary: dict) -> list[dict]:
    """Углубление: что внутри главного значения главного измерения (по остальным измерениям)."""
    cmp: Comparison = primary["_cmp"]
    if cmp.groups is None or not len(cmp.groups):
        return []
    top_value = cmp.groups.index[0]
    share = cmp.groups["share"].iloc[0]
    if not np.isfinite(share) or abs(share) < MAX_DRILL_VALUE_SHARE:
        return []
    inner_where = list(where) + [(primary["dimension"], top_value)]
    out = []
    for dim in [d for d in t.dims if d != primary["dimension"] and d not in {w[0] for w in where}][:3]:
        try:
            inner = compare(t, m, win, base, by=dim, where=inner_where)
        except ToolError:
            continue
        if inner.groups is None or len(inner.groups) < 2:
            continue
        out.append({"within": {primary["dimension"]: str(top_value)}, "dimension": dim,
                    "within_change_abs": r(inner.diff), "within_change_pct": r(inner.pct, 1),
                    "top": group_rows(inner, 2)})
    return out


def scenarios(cmp: Comparison, unit: str) -> list[dict]:
    """Арифметика «что если вернуть значение к базе»; без допущений о причинах."""
    g = cmp.groups
    if g is None or not cmp.significant or not cmp.unfavorable:
        return []
    gap = abs(cmp.diff)
    pushing = g[g["contrib"] * cmp.diff > 0]
    plan = []
    if len(pushing):
        plan += [("S1", pushing.iloc[:1], 100), ("S2", pushing.iloc[:1], 50)]
    if len(pushing) >= 2 and pushing["share"].iloc[1] >= MIN_SECOND_SHARE:
        plan.append(("S3", pushing.iloc[:2], 100))
    out = []
    for sid, part, share in plan:
        effect = float(part["contrib"].abs().sum()) * share / 100
        out.append({"scenario_id": sid, "values": [str(v) for v in part.index], "close_share_pct": share,
                    "effect_abs": r(effect), "unit": unit, "gap_closed_pct": r(effect / gap * 100, 1),
                    "remaining_gap_abs": r(gap - effect)})
    return out


# ---------- связанные показатели ----------


def _single_column(expr: str | None, columns: list[str]) -> str | None:
    terms = parse_expr(expr or "", columns)
    return terms[0][1] if terms and len(terms) == 1 else None


def related(t: Table, m: Metric, main: Comparison, where=(), primary_dim: str | None = None, limit: int = 6,
            primary_value: str | None = None) -> list[dict]:
    """Связанные показатели и их изменение за тот же период (по формуле, по таблице связей и «совпавшие по времени»)."""
    columns = sorted({c for x in t.spec.metrics for c in (x.column, x.num, x.den) if c}, key=len, reverse=True)
    candidates: dict[str, dict] = {}

    def add(name: str, relation: str, **extra):
        if name != m.name and name not in candidates and t.metric(name):
            candidates[name] = {"relation": relation, **extra}

    if m.kind == "ratio":
        for part, label in ((m.num, "числитель"), (m.den, "знаменатель")):
            col = _single_column(part, columns)
            for x in t.spec.metrics:
                if x.kind == "sum" and x.column == col:
                    add(x.name, f"{label} этого показателя")
    for x in t.spec.metrics:
        if x.kind == "ratio" and m.kind in ("sum", "last", "mean"):
            for part, label in ((x.num, "числитель"), (x.den, "знаменатель")):
                if _single_column(part, columns) == m.column:
                    add(x.name, f"этот показатель — {label} для «{x.name}»")
    for link in t.links:
        if link["effect"] == m.name:
            add(link["cause"], "влияет на выбранный показатель", link_direction=link["direction"])
        elif link["cause"] == m.name:
            add(link["effect"], "зависит от выбранного показателя", link_direction=link["direction"])

    def evaluate(name: str, info: dict) -> dict | None:
        cm = t.metric(name)
        try:
            base = resolve_baseline(t, cm, main.win, main.base.kind)
        except ToolError:
            try:
                base = resolve_baseline(t, cm, main.win, "auto")
            except ToolError:
                return None
        try:
            cmp = compare(t, cm, main.win, base, by=primary_dim, where=where) if primary_dim and primary_dim not in {w[0] for w in where} \
                else compare(t, cm, main.win, base, where=where)
        except ToolError:
            return None
        item = {"metric": name, **info, "unit": cm.unit, "baseline": base.label, "fact": r(cmp.v1), "base": r(cmp.v0),
                "deviation_abs": r(cmp.diff), "deviation_pct": r(cmp.pct, 1), "unfavorable": bool(cmp.unfavorable),
                "significant": bool(cmp.significant), "assessment": cmp.assessment}
        top = None
        if cmp.groups is not None and cmp.significant and len(cmp.groups):
            share = cmp.groups["share"].iloc[0]
            top = {"top_value": str(cmp.groups.index[0]), "top_value_share_pct": r(share, 1),
                   "top_value_change_pct": r(cmp.groups["own_pct"].iloc[0], 1)}
        item.update(top or {"top_value": None, "top_value_share_pct": None, "top_value_change_pct": None})
        consistent = None
        if "link_direction" in info and cmp.significant and main.significant:
            same = (cmp.diff > 0) == (main.diff > 0)
            consistent = same if info["link_direction"] == "положительно" else not same
        item["consistent_with_link"] = consistent
        trend = None
        if base.kind != PREV:
            try:
                prev = resolve_baseline(t, cm, main.win, PREV)
                trend = r(compare(t, cm, main.win, prev, where=where).pct, 1)
            except ToolError:
                pass
        item["change_vs_previous_period_pct"] = trend
        return item

    items = [x for x in (evaluate(n, i) for n, i in candidates.items()) if x]
    items.sort(key=lambda x: -abs(x["deviation_pct"] or 0))
    # «совпавшие по времени»: другие показатели со значимым изменением, связь с выбранным не установлена; берём только
    # те, у кого изменение сосредоточено в том же значении измерения (иначе это чужая история)
    extras = []
    if len([x for x in items if x["significant"]]) < 3:
        for other in t.spec.metrics:
            if other.name == m.name or other.name in candidates:
                continue
            item = evaluate(other.name, {"relation": "изменился в тот же период (связь не установлена)"})
            if item and item["significant"] and (primary_value is None or item["top_value"] == primary_value):
                extras.append(item)
        extras.sort(key=lambda x: -abs(x["deviation_pct"] or 0))
    return (items + extras[:3])[:limit]


# ---------- служебное ----------


def resolve_where(t: Table, dimension: str | None, value: str | None) -> list[tuple[str, str]]:
    """Фильтр «измерение = значение» с поиском без учёта регистра и подсказками при ошибке."""
    if not dimension and not value:
        return []
    if not dimension or not value:
        raise ToolError("Для фильтра нужны оба параметра: where_dimension и where_value.")
    dims = {d.casefold(): d for d in t.dims}
    dim = dims.get(dimension.strip().casefold())
    if dim is None:
        raise ToolError(f"Измерения «{dimension}» нет. Доступны: {', '.join(t.dims) or 'нет измерений'}.", dimensions=list(t.dims))
    values = t.dim_values[dim]
    exact = [v for v in values if v.casefold() == value.strip().casefold()]
    if exact:
        return [(dim, exact[0])]
    ranked = [(v, s) for v, s in scores(value, {v: v for v in values}) if s >= 0.75]
    if len(ranked) == 1 or (ranked and ranked[0][1] > ranked[1][1] + 0.15):
        return [(dim, ranked[0][0])]
    raise ToolError(f"Значения «{value}» нет в измерении «{dim}».", suggestions=[v for v, _ in ranked[:5]] or values[:5])


def find_metric(t: Table, name: str) -> Metric:
    """Название -> показатель: точное (без регистра), иначе единственное уверенное совпадение."""
    for m in t.spec.metrics:
        if m.name.casefold() == str(name).strip().casefold():
            return m
    ranked = scores(str(name), {m.name: m.name for m in t.spec.metrics})
    if ranked and ranked[0][1] >= 0.75 and (len(ranked) < 2 or ranked[1][1] < ranked[0][1] - 0.15):
        return t.metric(ranked[0][0])
    raise ToolError(f"Показатель «{name}» не найден. Возьмите точное название из describe_dataset или find_metrics.",
                    suggestions=[k for k, s in ranked[:3] if s > 0.3])


def top_findings(t: Table, limit: int = 3) -> list[tuple[str, float]]:
    """Показатели с самым заметным неблагоприятным изменением за окно по умолчанию (для подсказок вопросов)."""
    try:
        win, _ = resolve_window(t)
    except ToolError:
        return []
    out = []
    for m in t.spec.metrics:
        try:
            cmp = compare(t, m, win, resolve_baseline(t, m, win))
        except ToolError:
            continue
        if cmp.significant and cmp.unfavorable and cmp.pct is not None:
            out.append((m.name, abs(cmp.pct)))
    return sorted(out, key=lambda x: -x[1])[:limit]


def suggest_questions(t: Table) -> list[str]:
    """Вопросы-подсказки по структуре данных: названия берутся из самих данных, числа не используются."""
    if t.goals:  # стратегическая витрина: вопросы подобраны вручную
        return ["Почему мы отстаём по цели роста розничного кредитования и что делать?",
                "Почему растёт просрочка и в каком сегменте?",
                "Что происходит с комиссионными доходами за последние 3 месяца?",
                "Есть ли проблемы с оттоком клиентов?"]
    findings = [name for name, _ in top_findings(t, 2)]
    names = findings or [m.name for m in t.spec.metrics[:2]]
    questions = [f"Что изменилось с показателем «{n}» за последний период и почему?" for n in names]
    if t.dims:
        questions.append(f"Какие значения измерения «{t.dims[0]}» дали основной вклад в изменение «{names[0]}»?")
    others = [m.name for m in t.spec.metrics if m.name not in names][:1]
    if others:
        questions.append(f"Есть ли проблемы с показателем «{others[0]}»?")
    return questions[:4]


__all__ = ["ToolError", "Window", "Baseline", "Comparison", "resolve_window", "resolve_baseline", "compare", "r",
           "group_rows", "trailing_run", "scan_dimensions", "local_deviations", "drill_down", "scenarios", "related",
           "value_only", "structure",
           "resolve_where",
           "find_metric", "top_findings", "suggest_questions"]
