"""Ряды показателя по периодам для графиков дашборда: все значения считает код, как и в остальных инструментах.

Ряд строится по тем же правилам, что и итог в tools/engine.py: сумма, среднее непустых, значение на конец периода
или отношение сумм (клики / показы), поэтому последняя точка ряда согласуется с итогом за период.
"""
import math

import numpy as np
import pandas as pd

from data.model import Metric, Table
from tools.engine import MAX_POINTS_PER_CHART, Window, _where_rows, r

BUCKETS = ("day", "week", "month", "quarter", "year")
BUCKET_DAYS = {"day": 1, "week": 7, "month": 30.44, "quarter": 91.3, "year": 365.25}
PANDAS_FREQ = {"day": "D", "week": "W-MON", "month": "MS", "quarter": "QS", "year": "YS"}
PERIOD_FREQ = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}


def pick_bucket(t: Table, window: Window | None = None, requested: str | None = None) -> str:
    """Укрупнение ряда: исходная детализация, пока точек не больше MAX_POINTS_PER_CHART, иначе неделя → месяц → квартал → год.
    Явно запрошенное укрупнение учитывается, если оно не мельче исходной детализации."""
    start, end = _span(t, window)
    base = BUCKETS.index(t.grain)
    if requested in BUCKETS and BUCKETS.index(requested) >= base:
        return requested
    days = max((end - start).days + 1, 1)
    for bucket in BUCKETS[base:]:
        if math.ceil(days / BUCKET_DAYS[bucket]) <= MAX_POINTS_PER_CHART:
            return bucket
    return BUCKETS[-1]


def _span(t: Table, window: Window | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    lo, hi = (window.a, window.b) if window else (t.p_min, t.p_max)
    return (pd.Period(ordinal=int(lo), freq=t.freq).start_time, pd.Period(ordinal=int(hi), freq=t.freq).end_time.normalize())


def _label(ts: pd.Timestamp, bucket: str) -> str:
    if bucket == "month":
        return f"{ts.year:04d}-{ts.month:02d}"
    if bucket == "quarter":
        return f"{ts.year:04d}-Q{(ts.month - 1) // 3 + 1}"
    if bucket == "year":
        return f"{ts.year:04d}"
    return str(ts.date())


def _bucket_start(ts: pd.Timestamp, bucket: str) -> pd.Timestamp:
    return ts if bucket == "day" else ts.to_period(PERIOD_FREQ[bucket]).start_time.normalize()


def _aggregate(df: pd.DataFrame, m: Metric) -> pd.DataFrame:
    """Итоги по корзинам времени: числитель N, знаменатель D (для отношений и средних), план P."""
    ratio = m.kind in ("mean", "ratio")
    if m.kind == "last":  # значение на конец корзины: берём последний исходный период корзины
        df = df[df["_p"] == df.groupby("_b")["_p"].transform("max")]
    cols = {"N": df.groupby("_b")[m.N].sum()}
    if ratio:
        cols["D"] = df.groupby("_b")[m.D].sum()
    if m.has_plan:
        have = df[df[m.P].notna()]
        cols["P"] = have.groupby("_b")[m.P].sum()
        if ratio:
            cols["PD"] = have.groupby("_b")[m.D].sum()
    return pd.DataFrame(cols)


def _values(agg: pd.DataFrame, m: Metric, index: pd.DatetimeIndex) -> list[float | None]:
    agg = agg.reindex(index)
    if m.kind in ("mean", "ratio"):
        d = agg["D"].replace(0, np.nan)
        v = m.scale * agg["N"] / d
    else:
        v = m.scale * agg["N"]
    return [None if pd.isna(x) else r(x) for x in v]


def _plan_values(agg: pd.DataFrame, m: Metric, index: pd.DatetimeIndex) -> list[float | None] | None:
    if "P" not in agg or agg["P"].dropna().empty:
        return None
    agg = agg.reindex(index)
    if m.kind in ("mean", "ratio"):
        v = m.scale * agg["P"] / agg["PD"].replace(0, np.nan)
    else:
        v = m.scale * agg["P"]
    return [None if pd.isna(x) else r(x) for x in v]


def _partial(df: pd.DataFrame, m: Metric, t: Table, bucket: str, index: pd.DatetimeIndex) -> list[bool]:
    """Корзины, в которых есть данные не за все дни: сумма за такую корзину занижена, на графике её рисуют иначе.
    Определяется только для сумм по дневным данным; для отношений, средних и остатков неполная корзина не искажает значение."""
    if m.kind != "sum" or t.grain != "day" or bucket == "day":
        return [False] * len(index)
    end = index[-1] + pd.tseries.frequencies.to_offset(PANDAS_FREQ[bucket])
    expected = pd.Series([(b - a).days for a, b in zip(index, list(index[1:]) + [end])], index=index)
    present = df.groupby("_b")["_p"].nunique().reindex(index).fillna(0)
    return [bool(p < e) for p, e in zip(present, expected)]


def series(t: Table, m: Metric, *, where=(), by: str | None = None, window: Window | None = None,
           bucket: str | None = None, max_values: int = 5) -> dict:
    """Ряд показателя по корзинам времени.

    Возвращает {"bucket", "labels", "dates", "partial", "series": [{"name", "values"}], "plan", "others"}; пропущенные
    корзины — None; partial отмечает корзины с данными не за все дни (для сумм по дневным данным).
    by — разрезать по значениям измерения (до max_values самых весомых; остальные считаются в "others").
    """
    if t.grain is None:
        return {"bucket": None, "labels": [], "dates": [], "partial": [], "series": [], "plan": None, "others": 0}
    df = _where_rows(t.df, where)
    if window is not None:
        df = df[(df["_p"] >= window.a) & (df["_p"] <= window.b)]
    if df.empty:
        return {"bucket": None, "labels": [], "dates": [], "partial": [], "series": [], "plan": None, "others": 0}
    bucket = pick_bucket(t, window, bucket)
    starts = {int(o): _bucket_start(pd.Period(ordinal=int(o), freq=t.freq).start_time, bucket) for o in df["_p"].unique()}
    df = df.assign(_b=df["_p"].map(starts))
    first, last = df["_b"].min(), df["_b"].max()
    index = pd.date_range(first, last, freq=PANDAS_FREQ[bucket])
    if len(index) == 0 or index[0] != first:  # недельные корзины, начинающиеся не с понедельника, и т. п.
        index = pd.DatetimeIndex(sorted(df["_b"].unique()))
    out = {"bucket": bucket, "labels": [_label(ts, bucket) for ts in index], "dates": [str(ts.date()) for ts in index],
           "partial": _partial(df, m, t, bucket, index), "plan": None, "others": 0}
    if by is None:
        agg = _aggregate(df, m)
        out["series"] = [{"name": m.name, "values": _values(agg, m, index)}]
        out["plan"] = _plan_values(agg, m, index)
        return out
    weight = m.D if m.kind in ("mean", "ratio") else m.N
    ranked = df.groupby(by)[weight].apply(lambda s: float(s.abs().sum())).sort_values(ascending=False, kind="stable")
    keep = list(ranked.index[:max_values])
    out["others"] = max(len(ranked) - len(keep), 0)
    out["series"] = [{"name": str(v), "values": _values(_aggregate(df[df[by] == v], m), m, index)} for v in keep]
    return out


def sparkline(t: Table, m: Metric, *, where=(), points: int = 12, window: Window | None = None) -> dict:
    """Короткий ряд для плитки KPI: последние points корзин."""
    s = series(t, m, where=where, window=window)
    if not s["series"]:
        return {"labels": [], "values": []}
    return {"labels": s["labels"][-points:], "values": s["series"][0]["values"][-points:], "bucket": s["bucket"]}
