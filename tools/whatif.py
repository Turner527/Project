"""Сценарии «что если» и прогноз по активному отчёту. Все числа считает код, модель лишь выбирает параметры.

Ответ собирается из трёх слоёв, и каждый назван в результате своим словом, чтобы не выдать оценку за арифметику:
  * арифметика по окну (точно): значение показателя в срезе меняется (на N процентов, на N единиц или до N), итог и все
    показатели, у которых изменённый столбец входит в формулу (в том числе в выражение «Доход - Расходы»), пересчитываются
    из тех же сумм, что и в остальных инструментах. Объёмы (знаменатели) при изменении ставки не меняются: это допущение,
    оно названо в ответе;
  * оценка по истории (приблизительно): для показателя, не связанного формулой, берётся чувствительность к изменению
    драйвера по прошлым периодам (регрессия процентных изменений) с интервалом 80% и оговоркой «это корреляция»;
    если истории мало или связь слабая, оценка не строится и причина названа;
  * прогноз (tools/forecast.py): ряд на горизонт «если ничего не менять» и тот же ряд с эффектом сценария. Эффект в
    будущих периодах принимается равным относительному эффекту в окне: допущение тоже названо.

Проценты везде считаются от модуля базы: изменение на +10% — это прибавка 10% от |значения|, поэтому у отрицательных
значений (убыток) «+10%» тоже значит «стало лучше на 10% от величины», а не «убыток вырос».
"""
import math
import re
from statistics import NormalDist, median

import numpy as np
import pandas as pd

from data.model import Metric, Table
from tools import series as S
from tools.engine import ToolError, Window, _where_rows, _window_rows, find_metric, r

CHANGE_ALIASES = {
    "pct": "pct", "percent": "pct", "percentage": "pct", "relative": "pct", "процент": "pct", "проценты": "pct", "%": "pct",
    "abs": "abs", "absolute": "abs", "delta": "abs", "add": "abs", "plus": "abs", "единицы": "abs",
    "set": "set", "value": "set", "target": "set", "level": "set", "значение": "set",
}
MAX_EFFECTS = 6  # показателей-следствий в ответе
MIN_HISTORY_DIFFS = 8  # столько пар изменений нужно, чтобы оценивать чувствительность по истории
MIN_CORR = 0.5  # слабее связь — по истории не оцениваем
LEVEL_PCT = 80
Z = NormalDist().inv_cdf(0.5 + LEVEL_PCT / 200)
BUCKET_WORD = {"day": "дн.", "week": "нед.", "month": "мес.", "quarter": "кв.", "year": "г."}
DEFAULT_HORIZON = {"day": 7, "week": 4, "month": 3, "quarter": 2, "year": 1}
SEASON = {"day": 7, "week": 52, "month": 12, "quarter": 4, "year": None}
HISTORY_POINTS = 12  # сколько последних фактических точек возвращается вместе с прогнозом (для графика)
# пока в исходной детализации не больше стольких периодов, прогноз и оценка идут по ней, без укрупнения
MAX_NATIVE_POINTS = {"day": 120, "week": 260, "month": 240, "quarter": 80, "year": 40}
MIN_DAY_COVERAGE = 0.9  # доля календарных дней с данными; меньше — в данных регулярные пропуски (выходные), ряд укрупняется до недель
FILL_NOTE_SHARE = 0.10  # доля интерполированных точек, с которой надёжность прогноза понижается
MAX_HORIZON = 24
EDGE_PAD = 400  # запас периодов вокруг данных, чтобы посчитать календарную длину крайних корзин


def normalize_change_type(value) -> str:
    key = str(value or "").strip().lower()
    if key not in CHANGE_ALIASES:
        raise ToolError("Тип изменения change_type: pct — на N процентов (−20 — снизить на 20%), abs — прибавить N единиц "
                        "показателя, set — задать значение N.")
    return CHANGE_ALIASES[key]


def normalize_horizon(value) -> int | None:
    """Горизонт прогноза: целое от 1 до 24 или None (подберётся по детализации)."""
    if value is None:
        return None
    try:
        h = float(value)
    except (TypeError, ValueError):
        raise ToolError(f"horizon должен быть целым числом от 1 до {MAX_HORIZON} (периодов вперёд).") from None
    if not h.is_integer() or not 1 <= h <= MAX_HORIZON:
        raise ToolError(f"horizon должен быть целым числом от 1 до {MAX_HORIZON} (периодов вперёд), получено {value}.")
    return int(h)


def normalize_names(value) -> list[str]:
    """effect_on: список названий; одна строка — это один показатель, а не набор букв."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    try:
        return [str(v) for v in value if str(v).strip()]
    except TypeError:
        return [str(value)]


def verdict(delta: float | None, pct: float | None, better: str, sig_pct: float) -> str:
    """Слово об изменении для руководителя: лучше, хуже или в пределах допуска (порог значимости из настроек отчёта)."""
    if delta is None or abs(delta) < 1e-12:
        return "без изменений"
    if pct is not None and abs(pct) < sig_pct:
        return "в пределах допуска"
    if better == "none":
        return "выше" if delta > 0 else "ниже"
    return "лучше" if (delta > 0) == (better == "up") else "хуже"


def shift(value: float, pct: float) -> float:
    """Значение, сдвинутое на pct процентов от его модуля: у отрицательных чисел «+10%» тоже означает рост значения."""
    return value + abs(value) * pct / 100


def _line(before: float | None, after: float | None, m: Metric, sig_pct: float) -> dict:
    """before / after / delta_abs / delta_pct / verdict — одинаковые поля у любого показателя ответа.
    Ключи с подчёркиванием — неокруглённые числа для дальнейших расчётов; из ответа они убираются."""
    delta = None if before is None or after is None else after - before
    pct = None if delta is None or not before else delta / abs(before) * 100
    return {"before": r(before), "after": r(after), "delta_abs": r(delta), "delta_pct": r(pct, 1),
            "verdict": verdict(delta, pct, m.better, sig_pct), "_pct": pct}


def _strip_private(obj):
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip_private(v) for v in obj]
    return obj


# ---------- арифметика по окну ----------


def _value(m: Metric, n: float, d: float | None) -> float | None:
    if m.kind in ("mean", "ratio"):
        return m.scale * n / d if d else None
    return m.scale * n


def changed_expression(m: Metric) -> str | None:
    """Выражение, числитель которого меняется вместе со значением показателя: сумма и среднее — их столбец, отношение — его
    числитель (объём-знаменатель остаётся прежним). «На конец периода» не связывается с суммами окна."""
    if m.kind in ("sum", "mean"):
        return (m.column or "").strip() or None
    if m.kind == "ratio":
        return (m.num or "").strip() or None
    return None


def apply_change(t: Table, m: Metric, win: Window, where, change_type: str, change_value: float) -> dict:
    """Меняет значение показателя в срезе и пересчитывает итог. Возвращает суммы до и после (в единицах столбца)."""
    if not math.isfinite(change_value):
        raise ToolError("change_value должно быть конечным числом.")
    if change_type == "pct" and change_value < -100:
        raise ToolError("Снизить значение больше чем на 100% нельзя: минимум −100 (до нуля).")
    F = _window_rows(t, m, win)
    S_ = _where_rows(F, where)
    if S_.empty:
        raise ToolError("В выбранном срезе за период нет данных.")
    ratio = m.kind in ("mean", "ratio")
    n_tot, n_s = float(F[m.N].sum()), float(S_[m.N].sum())
    d_tot = float(F[m.D].sum()) if ratio else None
    d_s = float(S_[m.D].sum()) if ratio else None
    if ratio and (not d_tot or not d_s):
        raise ToolError("Знаменатель показателя в срезе или в целом равен нулю: отношение не определено.")
    v_s = _value(m, n_s, d_s)
    v_tot = _value(m, n_tot, d_tot)
    if change_type == "pct":
        v_s2 = shift(v_s, change_value)
    elif change_type == "abs":
        v_s2 = v_s + change_value
    else:
        v_s2 = change_value
    if change_type != "pct" and v_s2 < 0 <= v_s and bool((t.df[m.N] >= 0).all()):
        raise ToolError(f"Результат был бы отрицательным ({r(v_s2)}), а «{m.name}» во всех периодах не меньше нуля: так изменить нельзя.")
    # изменение числителя в единицах столбца; у отношений и средних объём (знаменатель) остаётся прежним
    dn = (v_s2 - v_s) / m.scale * (d_s if ratio else 1.0)
    v_tot2 = _value(m, n_tot + dn, d_tot)
    return {"selection_before": v_s, "selection_after": v_s2, "total_before": v_tot, "total_after": v_tot2, "dn": dn,
            "n_rows": len(S_), "selection_is_total": len(S_) == len(F)}


_SPLIT = re.compile(r"\s+([+\-−])\s+")


def _terms(expr: str | None) -> list[tuple[int, str]]:
    """«Доход - Расходы» -> [(+1, «Доход»), (−1, «Расходы»)]; один столбец — [(+1, столбец)]."""
    expr = (expr or "").strip()
    if not expr:
        return []
    parts = _SPLIT.split(expr)
    if len(parts) < 3 or len(parts) % 2 == 0:
        return [(1, expr)]
    out, sign = [], 1
    for i, part in enumerate(parts):
        if i % 2 == 0:
            out.append((sign, part))
        else:
            sign = 1 if part == "+" else -1
    return out


def _coefficient(expr: str | None, changed: str) -> int:
    """С каким весом изменённый столбец входит в выражение: «Доход - Расходы» и изменённый «Доход» → +1, «Расходы» → −1, нет → 0.
    Если изменено само выражение (несколько слагаемых), связаны только те, где оно записано точно так же."""
    if not expr or not changed:
        return 0
    own = _terms(changed)
    if len(own) == 1 and own[0][0] == 1:
        return sum(sign for sign, col in _terms(expr) if col == own[0][1])
    return 1 if expr.strip() == changed.strip() else 0


def linked_metrics(t: Table, m: Metric) -> list[dict]:
    """Показатели, значение которых точно пересчитывается при изменении m: суммы и отношения, в формулу которых входит
    изменённый столбец (с учётом знака в выражении). Каждый элемент — {metric, part, s_num, s_den}."""
    changed = changed_expression(m)
    if not changed:
        return []
    out = []
    for x in t.spec.metrics:
        if x.name == m.name:
            continue
        if x.kind == "sum":
            s = _coefficient(x.column, changed)
            if s:
                out.append({"metric": x, "part": "сумма", "s_num": s, "s_den": 0})
        elif x.kind == "ratio":
            s_n, s_d = _coefficient(x.num, changed), _coefficient(x.den, changed)
            if s_n or s_d:
                part = "числитель и знаменатель" if s_n and s_d else ("числитель" if s_n else "знаменатель")
                out.append({"metric": x, "part": part, "s_num": s_n, "s_den": s_d})
    return out


def formula_effect(t: Table, link: dict, win: Window, dn: float, other: dict | None = None) -> dict | None:
    """Эффект на связанный показатель, если сумма изменённого столбца изменилась на dn.
    Основная строка — механический пересчёт: остальные столбцы формулы не меняются. Если вторая часть формулы оценена по
    истории (other = {"metric", "pct", "low", "high"}), добавляется вариант with_estimate с её реакцией."""
    x, s_n, s_d = link["metric"], link["s_num"], link["s_den"]
    F = _window_rows(t, x, win)
    n = float(F[x.N].sum())
    d = float(F[x.D].sum()) if x.kind == "ratio" else None
    n2, d2 = n + s_n * dn, (d + s_d * dn if d is not None else None)
    if d is not None and (not d or not d2):
        return None
    sig = t.spec.significant_pct
    line = _line(_value(x, n, d), _value(x, n2, d2), x, sig)
    if other is not None and x.kind == "ratio" and bool(s_n) != bool(s_d):
        def with_other(pct: float) -> float | None:
            nn, dd = (n2, shift(d, pct)) if s_n else (shift(n, pct), d2)
            return _value(x, nn, dd) if dd else None

        est = _line(_value(x, n, d), with_other(other["pct"]), x, sig)
        lo, hi = with_other(other["low"]), with_other(other["high"])
        if est["after"] is not None and lo is not None and hi is not None:
            est.update(after_low=r(min(lo, hi)), after_high=r(max(lo, hi)), level_pct=LEVEL_PCT, based_on=other["metric"])
            line["with_estimate"] = est
    return line


def _other_part_metric(t: Table, link: dict) -> str | None:
    """Показатель-сумма, чей столбец — вторая часть формулы отношения (знаменатель, если изменён числитель, и наоборот)."""
    x = link["metric"]
    if x.kind != "ratio" or bool(link["s_num"]) == bool(link["s_den"]):
        return None
    col = ((x.den if link["s_num"] else x.num) or "").strip()
    return next((y.name for y in t.spec.metrics if y.kind == "sum" and (y.column or "").strip() == col), None)


# ---------- ряды по корзинам времени: общие правила для оценки по истории и прогноза ----------


def series_bucket(t: Table) -> str | None:
    """Корзина времени для прогноза и оценки по истории: исходная детализация, если периодов немного, иначе то укрупнение,
    которое выбирает график. Дневные данные с регулярными пропусками (выходные) укрупняются до недель: интерполировать
    выходные и прогнозировать их как рабочие дни нельзя."""
    if t.grain is None:
        return None
    n_base = t.p_max - t.p_min + 1
    if t.grain == "day" and len(t.periods_present()) / n_base < MIN_DAY_COVERAGE:
        return S.pick_bucket(t, None, "week")
    if n_base <= MAX_NATIVE_POINTS[t.grain]:
        return t.grain
    return S.pick_bucket(t)


def bucket_reason(t: Table) -> str | None:
    """Почему ряд укрупнён, если это не исходная детализация (для пояснения в ответе)."""
    bucket = series_bucket(t)
    if bucket is None or bucket == t.grain:
        return None
    if t.grain == "day" and len(t.periods_present()) / (t.p_max - t.p_min + 1) < MIN_DAY_COVERAGE:
        return "В данных есть регулярные пропуски дней (например выходные), поэтому ряд взят по неделям."
    return "Периодов в исходной детализации слишком много, поэтому ряд укрупнён."


def _complete_buckets(t: Table, bucket: str) -> set:
    """Начала корзин, в которых данные есть за все периоды, какие в них обычно бывают. Неполной может быть только крайняя
    корзина: у данных только по будням неделя из пяти дней полная, а месяц, оборванный на второй неделе, нет."""
    cache = t.__dict__.setdefault("_complete_cache", {})
    if bucket in cache:
        return cache[bucket]
    ords = np.arange(t.p_min - EDGE_PAD, t.p_max + EDGE_PAD + 1)
    stamps = pd.PeriodIndex.from_ordinals(ords, freq=t.freq).start_time
    starts = stamps.normalize() if bucket == "day" else stamps.to_period(S.PERIOD_FREQ[bucket]).start_time.normalize()
    present = {int(o) for o in t.periods_present()}
    frame = pd.DataFrame({"b": starts, "got": [int(o) in present for o in ords]})
    per = frame.groupby("b")["got"].agg(["size", "sum"])
    first = starts[EDGE_PAD]  # корзина, в которой начинаются данные
    last = starts[len(ords) - 1 - EDGE_PAD]  # и в которой они кончаются
    inner = [row["sum"] / row["size"] for b, row in per.iterrows() if first < b < last]
    density = median(inner) if len(inner) >= 3 else 1.0  # доля заполненных периодов в обычной корзине (5/7 у будней)
    complete = {b for b, row in per.iterrows() if first < b < last}  # внутри диапазона корзины не отбрасываются
    complete |= {b for b, row in per.iterrows() if b in (first, last) and row["sum"] >= math.ceil(row["size"] * density - 1e-6)}
    cache[bucket] = complete
    return complete


def _series(t: Table, m: Metric, where=()) -> dict | None:
    """Неокруглённый ряд показателя без неполных корзин по краям: {bucket, labels, dates, values}."""
    bucket = series_bucket(t)
    s = S.series(t, m, where=where, bucket=bucket, exact=True)
    if not s["series"]:
        return None
    complete = _complete_buckets(t, s["bucket"])
    keep = [i for i, d in enumerate(s["dates"]) if pd.Timestamp(d) in complete]
    if not keep:
        return None
    lo, hi = keep[0], keep[-1] + 1
    return {"bucket": s["bucket"], "labels": s["labels"][lo:hi], "dates": s["dates"][lo:hi], "values": s["series"][0]["values"][lo:hi]}


# ---------- оценка по истории ----------


def history_effect(t: Table, driver: Metric, target: Metric, win: Window, x0_pct: float | None) -> tuple[dict | None, dict | None]:
    """Оценка эффекта на target, если driver в целом изменился на x0_pct процентов: регрессия процентных изменений по
    прошлым периодам. Возвращает (строка эффекта, None) либо (None, причина отказа). Это корреляция, а не причинность."""
    def refuse(reason: str, **numbers) -> tuple[None, dict]:
        return None, {"metric": target.name, "reason": reason, **numbers}

    if t.grain is None:
        return refuse("В данных нет дат: по истории изменений оценить нечем.")
    if x0_pct is None:
        return refuse("Изменение драйвера в процентах не определено (в окне он равен нулю).")
    sd, st = _series(t, driver), _series(t, target)
    if sd is None or st is None or sd["dates"] != st["dates"]:
        return refuse("Ряды показателей не удалось сопоставить по периодам.")
    xs, ys = sd["values"], st["values"]
    gx, gy = [], []
    for i in range(1, len(xs)):  # только соседние корзины с данными у обоих показателей
        if None in (xs[i], xs[i - 1], ys[i], ys[i - 1]) or xs[i - 1] == 0 or ys[i - 1] == 0:
            continue
        gx.append((xs[i] - xs[i - 1]) / abs(xs[i - 1]) * 100)
        gy.append((ys[i] - ys[i - 1]) / abs(ys[i - 1]) * 100)
    n = len(gx)
    if n < MIN_HISTORY_DIFFS:
        return refuse("Прошлых периодов слишком мало, чтобы оценить связь по истории.", n_periods=n, min_periods=MIN_HISTORY_DIFFS)
    gx, gy = np.array(gx), np.array(gy)
    xm, ym = gx.mean(), gy.mean()
    sxx = float(((gx - xm) ** 2).sum())
    syy = float(((gy - ym) ** 2).sum())
    if sxx < 1e-9 or syy < 1e-9:
        return refuse("В прошлом драйвер или показатель почти не менялись: связь не оценить.", n_periods=n)
    sxy = float(((gx - xm) * (gy - ym)).sum())
    slope = sxy / sxx
    corr = sxy / math.sqrt(sxx * syy)
    if abs(corr) < MIN_CORR:
        return refuse("По истории связь слабая: изменения этих показателей почти не совпадали, прогноз эффекта был бы случайным.",
                      n_periods=n, correlation=r(corr), min_correlation=MIN_CORR)
    resid = gy - (ym + slope * (gx - xm))
    se_slope = math.sqrt(float((resid ** 2).sum()) / (n - 2) / sxx)
    effect = slope * x0_pct
    half = Z * se_slope * abs(x0_pct)
    before = _value(target, *_sums(t, target, win))
    if before is None:
        return refuse("Значение показателя в окне не определено.")
    line = _line(before, shift(before, effect), target, t.spec.significant_pct)
    line.update(
        metric=target.name, unit=target.unit, relation="оценка по истории",
        estimate={"effect_pct": r(effect, 1), "effect_low_pct": r(effect - half, 1), "effect_high_pct": r(effect + half, 1),
                  "level_pct": LEVEL_PCT, "sensitivity": r(slope), "correlation": r(corr), "r2": r(corr ** 2), "n_periods": n,
                  "beyond_history": bool(abs(x0_pct) > 2 * float(np.abs(gx).max()))},
        after_low=r(shift(before, effect - half)), after_high=r(shift(before, effect + half)),
        caveat="Оценка по прошлым изменениям (корреляция), а не причинно-следственная связь: она верна, только если связь "
               "между показателями останется такой же. Интервал учитывает только неточность оценки связи. Это ориентир, "
               "а не обещание результата.",
        _effect=effect, _low=effect - half, _high=effect + half)
    return line, None


def _sums(t: Table, m: Metric, win: Window) -> tuple[float, float | None]:
    F = _window_rows(t, m, win)
    ratio = m.kind in ("mean", "ratio")
    return float(F[m.N].sum()), (float(F[m.D].sum()) if ratio else None)


# ---------- прогноз ----------


def _next_periods(inp: dict, horizon: int) -> tuple[list[str], list[str]]:
    """Подписи и даты начала следующих корзин после последней корзины ряда."""
    last = pd.Timestamp(inp["dates"][-1])
    step = pd.tseries.frequencies.to_offset(S.PANDAS_FREQ[inp["bucket"]])
    stamps = [last + step * k for k in range(1, horizon + 1)]
    return [S._label(ts, inp["bucket"]) for ts in stamps], [str(ts.date()) for ts in stamps]


def _fill_note(item: dict) -> None:
    """Интерполированные пропуски делают ряд менее надёжным: уровень прогноза понижается, причина добавляется к остальным."""
    n, filled = item["history_n"], item["filled"]
    share = filled / n if n else 0.0
    if share <= FILL_NOTE_SHARE:
        return
    rel = item["reliability"]
    rel["reasons"] = [*rel["reasons"], f"В истории {filled} из {n} точек заполнены интерполяцией: часть периодов в данных пропущена."]
    if share > 0.25 or rel["level"] == "высокая":
        rel["level"] = "низкая" if share > 0.25 else "средняя"
    rel["numbers"] = [*rel.get("numbers", []), float(filled), float(n)]


def forecast_block(t: Table, items: list[tuple[Metric, float | None]], horizon: int | None, where=()) -> dict:
    """Базовый прогноз («если ничего не менять») и прогноз с эффектом сценария для каждого показателя из items.
    items — пары (показатель, относительный эффект в процентах от модуля или None, если сценария нет). Все показатели ответа
    берутся на одной сетке периодов (по первому), поэтому подписи и факт у них совпадают."""
    from tools.forecast import forecast_series

    if t.grain is None:
        return {"note": "В данных нет дат: прогноз по времени не строится."}
    first = _series(t, items[0][0], where)
    if first is None:
        return {"note": "В данных нет ряда для прогноза."}
    bucket = first["bucket"]
    horizon = horizon or DEFAULT_HORIZON[bucket]
    labels, dates = _next_periods(first, horizon)
    out = {"bucket": bucket, "unit_word": BUCKET_WORD[bucket], "horizon": horizon, "level_pct": LEVEL_PCT, "labels": labels, "dates": dates,
           "metrics": [], "not_forecast": []}
    if reason := bucket_reason(t):
        out["bucket_note"] = reason
    for m, effect_pct in items:
        inp = _series(t, m, where)
        if inp is None:
            out["not_forecast"].append({"metric": m.name, "reason": "Нет ряда по периодам."})
            continue
        by_date = dict(zip(inp["dates"], inp["values"]))
        values = [by_date.get(d) for d in first["dates"]]  # одна сетка периодов у всех показателей
        known = [v for v in values if v is not None]
        fc = forecast_series(values, horizon, SEASON[bucket], nonnegative=bool(known) and min(known) >= 0)
        if not fc["ok"]:
            out["not_forecast"].append({"metric": m.name, "reason": fc["reason"]})
            continue
        base = fc["point"]
        scen = [None if b is None else shift(b, effect_pct or 0) for b in base]
        tail = slice(-HISTORY_POINTS, None)
        item = {"metric": m.name, "unit": m.unit, "last_actual": r(known[-1]) if known else None,
                "history": {"labels": first["labels"][tail], "dates": first["dates"][tail], "values": [r(v) for v in values[tail]]},
                "baseline": base, "low": fc["low"], "high": fc["high"],
                "method": fc["method_label"], "trend": fc["trend"], "reliability": fc["reliability"], "backtest": fc["backtest"],
                "history_n": fc["history_n"], "filled": fc["filled"]}
        _fill_note(item)
        if effect_pct is not None:
            item["effect_pct"] = r(effect_pct, 1)
            item["scenario"] = [r(v) for v in scen]
            item["delta_abs"] = [r(None if b is None else s_ - b) for b, s_ in zip(base, scen)]
        if m.kind == "sum" and all(v is not None for v in base):  # суммы можно сложить за весь горизонт
            item["horizon_total_baseline"] = r(sum(base))
            # границы периодов складываются: ошибки соседних периодов связаны, поэтому интервал суммы консервативный (шире 80%)
            item["horizon_total_low"], item["horizon_total_high"] = r(sum(fc["low"])), r(sum(fc["high"]))
            if effect_pct is not None:
                item["horizon_total_scenario"] = r(sum(scen))
                item["horizon_total_delta"] = r(sum(scen) - sum(base))
        out["metrics"].append(item)
    if not out["metrics"]:
        out["note"] = out["not_forecast"][0]["reason"] if out["not_forecast"] else "Прогноз не построен."
    if effect_pct_any(items):
        out["assumption"] = ("Эффект сценария в будущих периодах принят равным его относительному эффекту в окне анализа; "
                             "прогноз «если ничего не менять» — продолжение прошлой динамики, а не обещание.")
    return out


def effect_pct_any(items) -> bool:
    return any(e is not None for _, e in items)


# ---------- сценарий целиком ----------


def build_scenario(t: Table, m: Metric, win: Window, where, change_type: str, change_value: float,
                   effect_on: list[str] | None, horizon: int | None, with_forecast: bool = True) -> dict:
    """Всё, что нужно для ответа what_if (без шапки с периодом: её добавляет инструмент)."""
    sig = t.spec.significant_pct
    res = apply_change(t, m, win, where, change_type, change_value)
    total = _line(res["total_before"], res["total_after"], m, sig)
    out = {
        "change": {"type": change_type, "value": r(change_value), "selection": ({d: v for d, v in where} if where else None)},
        "selection_value": {"before": r(res["selection_before"]), "after": r(res["selection_after"])},
        "total": total,
    }
    x0 = total["_pct"]

    links = linked_metrics(t, m)
    linked_names = {link["metric"].name for link in links}
    asked, refused = [], []
    for name in normalize_names(effect_on):
        try:
            target = find_metric(t, name)
        except ToolError as exc:
            refused.append({"metric": str(name), "reason": str(exc)})
            continue
        if target.name == m.name or target.name in linked_names or any(a.name == target.name for a in asked):
            continue
        asked.append(target)
    history: dict[str, dict] = {}  # показатель -> строка эффекта по истории
    for target in asked[:MAX_EFFECTS]:
        line, why = history_effect(t, m, target, win, x0)
        if line is None:
            refused.append(why)
        else:
            history[target.name] = line

    effects, relative = [], []
    for link in links[:MAX_EFFECTS]:
        x = link["metric"]
        other_name = _other_part_metric(t, link)
        est = history.get(other_name) if other_name else None
        other = None if est is None else {"metric": other_name, "pct": est["_effect"], "low": est["_low"], "high": est["_high"]}
        line = formula_effect(t, link, win, res["dn"], other)
        if line is None:
            refused.append({"metric": x.name, "reason": "Знаменатель после изменения равен нулю."})
            continue
        effects.append({"metric": x.name, "unit": x.unit, "relation": f"формула: {m.name} входит в {link['part']} показателя", **line})
        main = line.get("with_estimate") or line
        relative.append((x, main["_pct"]))
    for name, line in history.items():
        effects.append(line)
        relative.append((t.metric(name), line["_effect"]))
    if effects:
        asked_first = {str(n).strip().casefold(): i for i, n in enumerate(normalize_names(effect_on))}
        effects.sort(key=lambda e: asked_first.get(e["metric"].casefold(), len(asked_first)))  # названные в вопросе — первыми
        out["effects"] = effects
    omitted = max(0, len(links) - MAX_EFFECTS) + max(0, len(asked) - MAX_EFFECTS)
    if omitted:
        out["effects_omitted"] = omitted
    if refused:
        out["not_estimated"] = refused
    out["assumptions"] = _assumptions(m, bool(where), bool(links), bool(asked), bool(omitted))
    if with_forecast:
        items = [(m, x0 if x0 is not None else 0.0)] + relative
        out["forecast"] = forecast_block(t, items, horizon)
    return _strip_private(out)


def _assumptions(m: Metric, sliced: bool, linked: bool, historical: bool, truncated: bool) -> list[str]:
    notes = ["Остальные значения и периоды остаются как в окне анализа: меняется только названное."]
    if m.kind in ("mean", "ratio"):
        notes.append("Меняется ставка (значение показателя) в срезе; объём, на который она приходится, остаётся прежним.")
    if sliced:
        notes.append("Изменение применено только к названному срезу; итог пересчитан по всему отчёту.")
    if linked:
        notes.append("Связанные показатели пересчитаны по формуле из тех же сумм. Основной расчёт механический: остальные столбцы "
                     "формулы (например, объём результата) приняты неизменными, хотя в жизни они тоже реагируют. Реакцию можно "
                     "оценить по истории: попросите оценить влияние на такой показатель, тогда появится вариант с учётом реакции.")
    if historical:
        notes.append("Эффект на показатели без формулы связи — оценка по истории, с интервалом и оговоркой.")
    if truncated:
        notes.append("Показаны не все связанные показатели: в ответ попадают самые близкие по формуле, остальные не рассчитывались.")
    return notes


def build_forecast(t: Table, m: Metric, horizon: int | None, where=()) -> dict:
    """Прогноз показателя «если ничего не менять» (для forecast_metric)."""
    return forecast_block(t, [(m, None)], horizon, where)
