"""Сценарии «что если» и прогноз по активному отчёту. Все числа считает код, модель лишь выбирает параметры.

Ответ собирается из трёх слоёв, и каждый назван в результате своим словом, чтобы не выдать оценку за арифметику:
  * арифметика по окну (точно): значение показателя в срезе меняется (на N процентов, на N единиц или до N), итог и все
    показатели-отношения, у которых этот столбец числитель или знаменатель, пересчитываются из тех же сумм, что и
    в остальных инструментах. Объёмы (знаменатели) при изменении ставки не меняются: это допущение, оно названо в ответе;
  * оценка по истории (приблизительно): для показателя, не связанного формулой, берётся чувствительность к изменению
    драйвера по прошлым периодам (регрессия процентных изменений) с интервалом 80% и оговоркой «это корреляция»;
    если истории мало или связь слабая, оценка не строится и причина названа;
  * прогноз (tools/forecast.py): ряд на горизонт «если ничего не менять» и тот же ряд с эффектом сценария. Эффект в
    будущих периодах принимается равным относительному эффекту в окне: допущение тоже названо.
"""
import math
from statistics import NormalDist

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
MAX_NATIVE_POINTS = 120  # пока в исходной детализации не больше стольких периодов, прогноз и оценка идут по ней, без укрупнения


def series_bucket(t: Table) -> str | None:
    """Корзина времени для прогноза и оценки по истории: исходная детализация, если периодов немного (60 дней — дни, с учётом
    дня недели), иначе то укрупнение, которое выбирает график (неделя, месяц, …)."""
    if t.grain is None:
        return None
    if t.p_max - t.p_min + 1 <= MAX_NATIVE_POINTS:
        return t.grain
    return S.pick_bucket(t)


def normalize_change_type(value) -> str:
    key = str(value or "").strip().lower()
    if key not in CHANGE_ALIASES:
        raise ToolError("Тип изменения change_type: pct — на N процентов (−20 — снизить на 20%), abs — прибавить N единиц "
                        "показателя, set — задать значение N.")
    return CHANGE_ALIASES[key]


def verdict(delta: float | None, pct: float | None, better: str, sig_pct: float) -> str:
    """Слово об изменении для руководителя: лучше, хуже или в пределах допуска (порог значимости из настроек отчёта)."""
    if delta is None or abs(delta) < 1e-12:
        return "без изменений"
    if pct is not None and abs(pct) < sig_pct:
        return "в пределах допуска"
    if better == "none":
        return "выше" if delta > 0 else "ниже"
    return "лучше" if (delta > 0) == (better == "up") else "хуже"


def _line(before: float | None, after: float | None, m: Metric, sig_pct: float) -> dict:
    """before / after / delta_abs / delta_pct / verdict — одинаковые поля у любого показателя ответа."""
    delta = None if before is None or after is None else after - before
    pct = None if delta is None or not before else delta / abs(before) * 100
    return {"before": r(before), "after": r(after), "delta_abs": r(delta), "delta_pct": r(pct, 1),
            "verdict": verdict(delta, pct, m.better, sig_pct)}


# ---------- арифметика по окну ----------


def _value(m: Metric, n: float, d: float | None) -> float | None:
    if m.kind in ("mean", "ratio"):
        return m.scale * n / d if d else None
    return m.scale * n


def apply_change(t: Table, m: Metric, win: Window, where, change_type: str, change_value: float) -> dict:
    """Меняет значение показателя в срезе и пересчитывает итог. Возвращает суммы до и после (в единицах столбца)."""
    if not math.isfinite(change_value):
        raise ToolError("change_value должно быть конечным числом.")
    if change_type == "pct" and change_value <= -100:
        raise ToolError("Снизить значение больше чем на 100% нельзя: максимум −100 (до нуля).")
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
        v_s2 = v_s * (1 + change_value / 100)
    elif change_type == "abs":
        v_s2 = v_s + change_value
    else:
        v_s2 = change_value
    # изменение числителя в единицах столбца; у отношений и средних объём (знаменатель) остаётся прежним
    dn = (v_s2 - v_s) / m.scale * (d_s if ratio else 1.0)
    v_tot2 = _value(m, n_tot + dn, d_tot)
    return {"selection_before": v_s, "selection_after": v_s2, "total_before": v_tot, "total_after": v_tot2, "dn": dn,
            "n_rows": len(S_), "selection_is_total": len(S_) == len(F)}


def linked_ratios(t: Table, m: Metric) -> list[tuple[Metric, str]]:
    """Показатели-отношения, у которых столбец показателя m (сумма) — числитель или знаменатель: их пересчёт точен."""
    if m.kind != "sum" or not m.column:
        return []
    col = m.column.strip()
    out = []
    for x in t.spec.metrics:
        if x.kind != "ratio" or x.name == m.name:
            continue
        if (x.num or "").strip() == col:
            out.append((x, "числитель"))
        elif (x.den or "").strip() == col:
            out.append((x, "знаменатель"))
    return out


def formula_effect(t: Table, x: Metric, part: str, win: Window, dn: float, other: dict | None = None) -> dict | None:
    """Эффект на показатель-отношение x, если сумма его числителя или знаменателя изменилась на dn.
    Основная строка — механический пересчёт: вторая часть формулы не меняется. Если вторая часть оценена по истории
    (other = {"metric", "effect_pct", "low_pct", "high_pct"}), добавляется вариант with_estimate с её реакцией."""
    F = _window_rows(t, x, win)
    n, d = float(F[x.N].sum()), float(F[x.D].sum())
    n2, d2 = (n + dn, d) if part == "числитель" else (n, d + dn)
    if not d or not d2:
        return None
    sig = t.spec.significant_pct
    line = _line(_value(x, n, d), _value(x, n2, d2), x, sig)
    if other is not None:
        def with_other(pct: float) -> float | None:
            k = 1 + pct / 100
            nn, dd = (n2, d * k) if part == "числитель" else (n * k, d2)
            return _value(x, nn, dd) if dd else None

        est = _line(_value(x, n, d), with_other(other["effect_pct"]), x, sig)
        lo, hi = with_other(other["low_pct"]), with_other(other["high_pct"])
        if est["after"] is not None and lo is not None and hi is not None:
            est.update(after_low=r(min(lo, hi)), after_high=r(max(lo, hi)), level_pct=LEVEL_PCT, based_on=other["metric"])
            line["with_estimate"] = est
    return line


# ---------- оценка по истории ----------


def _clean_pairs(t: Table, driver: Metric, target: Metric):
    """Ряды драйвера и цели по одним и тем же корзинам; неполные и пустые корзины убираются."""
    sd, st = S.series(t, driver, bucket=series_bucket(t)), S.series(t, target, bucket=series_bucket(t))
    if not sd["series"] or not st["series"] or sd["labels"] != st["labels"]:
        return None
    keep = [i for i, (a, b, pa, pb) in enumerate(zip(sd["series"][0]["values"], st["series"][0]["values"], sd["partial"], st["partial"]))
            if a is not None and b is not None and not pa and not pb]
    xs = np.array([sd["series"][0]["values"][i] for i in keep], dtype=float)
    ys = np.array([st["series"][0]["values"][i] for i in keep], dtype=float)
    return keep, xs, ys


def history_effect(t: Table, driver: Metric, target: Metric, win: Window, x0_pct: float | None) -> tuple[dict | None, dict | None]:
    """Оценка эффекта на target, если driver в целом изменился на x0_pct процентов: регрессия процентных изменений по
    прошлым периодам. Возвращает (строка эффекта, None) либо (None, причина отказа). Это корреляция, а не причинность."""
    def refuse(reason: str, **numbers) -> tuple[None, dict]:
        return None, {"metric": target.name, "reason": reason, **numbers}

    if t.grain is None:
        return refuse("В данных нет дат: по истории изменений оценить нечем.")
    if x0_pct is None:
        return refuse("Изменение драйвера в процентах не определено (в окне он равен нулю).")
    pairs = _clean_pairs(t, driver, target)
    if pairs is None:
        return refuse("Ряды показателей не удалось сопоставить по периодам.")
    keep, xs, ys = pairs
    gx, gy = [], []
    for i in range(1, len(keep)):
        if keep[i] - keep[i - 1] != 1 or xs[i - 1] == 0 or ys[i - 1] == 0:  # только соседние корзины
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
    line = _line(before, before * (1 + effect / 100), target, t.spec.significant_pct)
    line.update(
        metric=target.name, unit=target.unit, relation="оценка по истории",
        estimate={"effect_pct": r(effect, 1), "effect_low_pct": r(slope * x0_pct - half, 1), "effect_high_pct": r(slope * x0_pct + half, 1),
                  "level_pct": LEVEL_PCT, "sensitivity": r(slope), "correlation": r(corr), "r2": r(corr ** 2), "n_periods": n,
                  "beyond_history": bool(abs(x0_pct) > 2 * float(np.abs(gx).max()))},
        after_low=r(before * (1 + (effect - half) / 100)), after_high=r(before * (1 + (effect + half) / 100)),
        caveat="Оценка по прошлым изменениям (корреляция), а не причинно-следственная связь: она верна, только если связь "
               "между показателями останется такой же. Это ориентир, а не обещание результата.")
    return line, None


def _sums(t: Table, m: Metric, win: Window) -> tuple[float, float | None]:
    F = _window_rows(t, m, win)
    ratio = m.kind in ("mean", "ratio")
    return float(F[m.N].sum()), (float(F[m.D].sum()) if ratio else None)


# ---------- прогноз ----------


def _forecast_input(t: Table, m: Metric, where=()):
    """Ряд для прогноза: корзины времени без неполных по краям (недосчёт сумм выглядел бы как падение)."""
    s = S.series(t, m, where=where, bucket=series_bucket(t))
    if not s["series"]:
        return None
    labels, values, partial, dates = s["labels"], s["series"][0]["values"], s["partial"], s["dates"]
    lo, hi = 0, len(values)
    while hi - lo > 2 and partial[hi - 1]:
        hi -= 1
    while hi - lo > 2 and partial[lo]:
        lo += 1
    return {"bucket": s["bucket"], "labels": labels[lo:hi], "dates": dates[lo:hi], "values": values[lo:hi]}


def _next_periods(inp: dict, horizon: int) -> tuple[list[str], list[str]]:
    """Подписи и даты начала следующих корзин после последней корзины ряда."""
    last = pd.Timestamp(inp["dates"][-1])
    step = pd.tseries.frequencies.to_offset(S.PANDAS_FREQ[inp["bucket"]])
    stamps = [last + step * k for k in range(1, horizon + 1)]
    return [S._label(ts, inp["bucket"]) for ts in stamps], [str(ts.date()) for ts in stamps]


def forecast_block(t: Table, items: list[tuple[Metric, float | None]], horizon: int | None, where=()) -> dict:
    """Базовый прогноз («если ничего не менять») и прогноз с эффектом сценария для каждого показателя из items.
    items — пары (показатель, относительный эффект в процентах или None, если сценария нет)."""
    from tools.forecast import forecast_series

    if t.grain is None:
        return {"note": "В данных нет дат: прогноз по времени не строится."}
    first = _forecast_input(t, items[0][0], where)
    if first is None:
        return {"note": "В данных нет ряда для прогноза."}
    bucket = first["bucket"]
    horizon = horizon or DEFAULT_HORIZON[bucket]
    labels, dates = _next_periods(first, horizon)
    out = {"bucket": bucket, "unit_word": BUCKET_WORD[bucket], "horizon": horizon, "level_pct": LEVEL_PCT, "labels": labels, "dates": dates,
           "metrics": [], "not_forecast": []}
    for m, effect_pct in items:
        inp = _forecast_input(t, m, where)
        if inp is None:
            out["not_forecast"].append({"metric": m.name, "reason": "Нет ряда по периодам."})
            continue
        known = [v for v in inp["values"] if v is not None]
        fc = forecast_series(inp["values"], horizon, SEASON[bucket], nonnegative=bool(known) and min(known) >= 0)
        if not fc["ok"]:
            out["not_forecast"].append({"metric": m.name, "reason": fc["reason"]})
            continue
        base = fc["point"]
        factor = 1 + (effect_pct or 0) / 100
        scen = [None if b is None else b * factor for b in base]
        tail = slice(-HISTORY_POINTS, None)
        item = {"metric": m.name, "unit": m.unit, "last_actual": r(known[-1]) if known else None,
                "history": {"labels": inp["labels"][tail], "dates": inp["dates"][tail], "values": inp["values"][tail]},
                "baseline": base, "low": fc["low"], "high": fc["high"],
                "method": fc["method_label"], "trend": fc["trend"], "reliability": fc["reliability"], "backtest": fc["backtest"],
                "history_n": fc["history_n"]}
        if effect_pct is not None:
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
    out = {
        "change": {"type": change_type, "value": r(change_value), "selection": ({d: v for d, v in where} if where else None)},
        "selection_value": {"before": r(res["selection_before"]), "after": r(res["selection_after"])},
        "total": {**_line(res["total_before"], res["total_after"], m, sig)},
    }
    total_pct = out["total"]["delta_pct"]
    x0 = None if res["total_before"] in (None, 0) else (res["total_after"] - res["total_before"]) / abs(res["total_before"]) * 100

    linked = linked_ratios(t, m)
    linked_names = {x.name for x, _ in linked}
    asked, refused = [], []
    for name in effect_on or []:
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
    for x, part in linked[:MAX_EFFECTS]:
        other_name = _other_part_metric(t, x, part)
        est = history.get(other_name) if other_name else None
        other = None if est is None else {"metric": other_name, "effect_pct": est["estimate"]["effect_pct"],
                                           "low_pct": est["estimate"]["effect_low_pct"], "high_pct": est["estimate"]["effect_high_pct"]}
        line = formula_effect(t, x, part, win, res["dn"], other)
        if line is None:
            refused.append({"metric": x.name, "reason": "Знаменатель после изменения равен нулю."})
            continue
        effects.append({"metric": x.name, "unit": x.unit, "relation": f"формула: {m.name} — {part} показателя", **line})
        relative.append((x, (line.get("with_estimate") or line)["delta_pct"]))
    for name, line in history.items():
        effects.append(line)
        relative.append((t.metric(name), line["estimate"]["effect_pct"]))
    if effects:
        asked_first = {str(n).strip().casefold(): i for i, n in enumerate(effect_on or [])}
        effects.sort(key=lambda e: asked_first.get(e["metric"].casefold(), len(asked_first)))  # названные в вопросе — первыми
        out["effects"] = effects
    if refused:
        out["not_estimated"] = refused
    out["assumptions"] = _assumptions(m, change_type, bool(where), bool(linked), bool(asked))
    if with_forecast:
        items = [(m, total_pct if total_pct is not None else 0.0)] + [(x, e) for x, e in relative]
        out["forecast"] = forecast_block(t, items, horizon)
    return out


def _other_part_metric(t: Table, x: Metric, part: str) -> str | None:
    """Показатель-сумма, чей столбец — вторая часть формулы отношения x (знаменатель, если m — числитель, и наоборот)."""
    col = ((x.den if part == "числитель" else x.num) or "").strip()
    return next((y.name for y in t.spec.metrics if y.kind == "sum" and (y.column or "").strip() == col), None)


def _assumptions(m: Metric, change_type: str, sliced: bool, linked: bool, historical: bool) -> list[str]:
    notes = ["Остальные значения и периоды остаются как в окне анализа: меняется только названное."]
    if m.kind in ("mean", "ratio"):
        notes.append("Меняется ставка (значение показателя) в срезе; объём, на который она приходится, остаётся прежним.")
    if sliced:
        notes.append("Изменение применено только к названному срезу; итог пересчитан по всему отчёту.")
    if linked:
        notes.append("Показатели-отношения пересчитаны по формуле из тех же сумм. Основной расчёт механический: вторая часть "
                     "формулы (например, объём результата) принята неизменной, хотя в жизни она тоже реагирует. Реакцию можно "
                     "оценить по истории: назовите её показатель в effect_on, тогда появится вариант with_estimate.")
    if historical:
        notes.append("Эффект на показатели без формулы связи — оценка по истории, с интервалом и оговоркой.")
    return notes


def build_forecast(t: Table, m: Metric, horizon: int | None, where=()) -> dict:
    """Прогноз показателя «если ничего не менять» (для forecast_metric)."""
    return forecast_block(t, [(m, None)], horizon, where)
