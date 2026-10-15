"""Инструменты над активным отчётом. Все числа считает код (tools/engine.py); модель лишь выбирает вызовы.

Каждый инструмент возвращает компактный JSON с call_id; целые таблицы не отдаются.
"""
import functools
import uuid
from typing import Annotated, Callable

from pydantic import Field

from tools import datastore
from tools import engine as E
from tools import whatif as W
from tools.engine import ToolError, r
from tools.text import scores

MIN_MATCH_SCORE = 0.5
MAX_VALUES = 8  # значений измерения в ответе break_down
MAX_STRUCTURE_DIMENSIONS = 3  # измерений в ответе get_structure, если не названо конкретное
BETTER_WORD = {"up": "больше", "down": "меньше", "none": "нейтрально"}

MetricArg = Annotated[str, Field(description="Название показателя (из describe_dataset или find_metrics)")]
FromArg = Annotated[str | None, Field(description="Начало периода: 2026-09-15, 2026-09 (месяц) или 2026-Q3. "
                                                   "Не указано — окно по умолчанию (см. describe_dataset)")]
ToArg = Annotated[str | None, Field(description="Конец периода в том же формате. Не указан — конец данных")]
LastMonthsArg = Annotated[int | None, Field(description="Только последние N месяцев до конца данных "
                                                         "(для «за последние 3 месяца»)", ge=1, le=120)]
LastDaysArg = Annotated[int | None, Field(description="Только последние N дней (для «за последние две недели»: 14)",
                                           ge=1, le=3650)]
BaselineArg = Annotated[str | None, Field(description="С чем сравнивать: plan (план), prev (предыдущий период той же "
                                                       "длины), yoy (тот же период год назад). Не указано — "
                                                       "план, если он есть, иначе предыдущий период")]
WhereDimArg = Annotated[str | None, Field(description="Измерение для фильтра, например «Канал» (вместе с where_value)")]
WhereValueArg = Annotated[str | None, Field(description="Значение измерения для фильтра, например «Поиск»")]
ByArg = Annotated[str | None, Field(description="Измерение, по которому раскладывать (например «Канал»). "
                                                "Не указано — выбирается самое объясняющее изменение")]


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


# ---------- общая подготовка ----------


def _prepare(metric, period_from, period_to, last_months, last_days, baseline, where_dimension, where_value):
    t = datastore.get_table()
    m = E.find_metric(t, metric)
    win, note = E.resolve_window(t, period_from, period_to, last_months, last_days)
    where = E.resolve_where(t, where_dimension, where_value)
    base = E.resolve_baseline(t, m, win, baseline)
    return t, m, win, base, where, note


def _period(t, win) -> dict:
    if t.grain is None:
        return {"note": "в данных нет дат"}
    return {"from": t.label(win.a), "to": t.label(win.b), "periods": win.n}


def _baseline(t, base) -> dict:
    info = {"type": base.kind, "label": base.label}
    if base.win is not None:
        info.update({"from": t.label(base.win.a), "to": t.label(base.win.b)})
    return info


def _deviation(cmp) -> dict:
    return {"fact": r(cmp.v1), "base": r(cmp.v0), "deviation_abs": r(cmp.diff), "deviation_pct": r(cmp.pct, 1),
            "unfavorable": bool(cmp.unfavorable), "significant": bool(cmp.significant), "assessment": cmp.assessment}


def _head(t, m, win, base, where, note) -> dict:
    out = {"metric": m.name, "unit": m.unit, "better": BETTER_WORD[m.better], "definition": m.definition(),
           "period": _period(t, win), "baseline": _baseline(t, base)}
    if where:
        out["filter"] = {d: v for d, v in where}
    if note:
        out["note"] = note
    return out


def _available(t, win, kind) -> bool:
    shift = win.n if kind == E.PREV else E.YOY_SHIFT[t.grain]
    return bool(E._coverage(t, E.Window(win.a - shift, win.b - shift)) >= E.MIN_COVERAGE)


# ---------- инструменты ----------


@_tool
def describe_dataset() -> dict:
    """Что в отчёте: период и детализация, показатели (как считаются, что лучше), измерения и их значения, доступные
    базы сравнения. Вызывайте, если нужно понять структуру данных или вопрос про сам отчёт."""
    t = datastore.get_table()
    info = t.summary()
    out = {"source": t.public_name, "rows": info["rows"], "grain": t.grain, "periods": info["periods"],
           "metrics_count": len(t.spec.metrics)}
    if t.grain is not None:
        win, _ = E.resolve_window(t)
        out["period"] = {"from": info["first_period"], "to": info["last_period"]}
        out["default_window"] = {"from": t.label(win.a), "to": t.label(win.b), "periods": win.n}
        out["baselines_available"] = {"plan": any(m.has_plan for m in t.spec.metrics),
                                      "prev": _available(t, win, E.PREV), "yoy": _available(t, win, E.YOY)}
    if t.rows_excluded:
        # фильтр по столбцу «Не отправлять в модель» работает локально: ни его название, ни значения не раскрываются, остаётся счётчик
        shown = [{"column": f["column"], "kept_values": f["values"]} for f in t.spec.filters if f["column"] not in t.spec.excluded]
        if shown:
            out["filters"] = shown
        out["rows_excluded_by_filters"] = t.rows_excluded
    out["metrics"] = [{"name": m.name, "unit": m.unit, "better": BETTER_WORD[m.better], "definition": m.definition(),
                       "has_plan": m.has_plan, **({"goal": m.goal} if m.goal else {})} for m in t.spec.metrics[:40]]
    out["dimensions"] = [{"name": d, "values_count": len(t.dim_values[d]), "top_values": t.dim_values[d][:8]}
                         for d in t.dims[:8]]
    if t.goals:
        out["goals"] = list(t.goals)
    return out


@_tool
def find_metrics(
    query: Annotated[str, Field(description="Название показателя, цели или значения измерения своими словами, "
                                            "например «конверсия» или «канал Поиск»")],
) -> dict:
    """Найти показатели (и цели, если они есть) по названию, а также значения измерений, упомянутые в запросе.
    Вызывайте первым, если в вопросе названа цель, показатель или конкретное значение (канал, отдел, регион)."""
    t = datastore.get_table()
    metric_hits = [(n, s) for n, s in scores(query, {m.name: m.name for m in t.spec.metrics}) if s >= MIN_MATCH_SCORE][:4]
    out = {"query": query, "metric_matches": [], "goal_matches": [], "value_matches": []}
    for name, score in metric_hits:
        m = t.metric(name)
        out["metric_matches"].append({"metric": name, "unit": m.unit, "match_score": score, "definition": m.definition(),
                                      **({"goal": m.goal} if m.goal else {})})
    if t.goals:
        goal_hits = [(g, s) for g, s in scores(query, {g: g for g in t.goals}) if s >= MIN_MATCH_SCORE][:3]
        for goal, score in goal_hits:
            out["goal_matches"].append({
                "goal": goal, "match_score": score, "parent_goal": t.goals[goal]["parent"],
                "metrics": [{"metric": m.name, "unit": m.unit, "better": BETTER_WORD[m.better]}
                            for m in t.spec.metrics if m.goal == goal]})
    for dim in t.dims:
        values = t.dim_values[dim]
        if len(values) > 2000:
            continue
        for value, score in scores(query, {v: v for v in values})[:2]:
            if score >= 0.75:
                out["value_matches"].append({"dimension": dim, "value": value})
    if not (out["metric_matches"] or out["goal_matches"] or out["value_matches"]):
        out["note"] = "Совпадений нет. Уточните название; список показателей приведён в available_metrics."
        out["available_metrics"] = [m.name for m in t.spec.metrics[:40]]
    return out


@_tool
def get_change(metric: MetricArg, period_from: FromArg = None, period_to: ToArg = None, last_months: LastMonthsArg = None,
               last_days: LastDaysArg = None, baseline: BaselineArg = None, where_dimension: WhereDimArg = None,
               where_value: WhereValueArg = None) -> dict:
    """Изменение показателя за период относительно базы (план, предыдущий период или год назад): факт, база, разница,
    процент, лучше или хуже. Можно сузить фильтром по одному значению измерения."""
    t, m, win, base, where, note = _prepare(metric, period_from, period_to, last_months, last_days, baseline,
                                            where_dimension, where_value)
    cmp = E.compare(t, m, win, base, where=where)
    return {**_head(t, m, win, base, where, note), **_deviation(cmp)}


def _pick_dimension(t, by: str | None, m, win, base, where) -> str:
    if by:
        dims = {d.casefold(): d for d in t.dims}
        if by.strip().casefold() not in dims:
            raise ToolError(f"Измерения «{by}» нет. Доступны: {', '.join(t.dims) or 'нет измерений'}.", dimensions=list(t.dims))
        return dims[by.strip().casefold()]
    if not t.dims:
        raise ToolError("В данных нет измерений (разрезов): раскладывать нечего.")
    scan = E.scan_dimensions(t, m, win, base, where)
    if not scan:
        raise ToolError("Ни по одному измерению нельзя разложить изменение: значений меньше двух.")
    return scan[0]["dimension"]


@_tool
def break_down(metric: MetricArg, by: ByArg = None, period_from: FromArg = None, period_to: ToArg = None,
               last_months: LastMonthsArg = None, last_days: LastDaysArg = None, baseline: BaselineArg = None,
               where_dimension: WhereDimArg = None, where_value: WhereValueArg = None) -> dict:
    """Разложить изменение показателя по значениям измерения: факт, база и вклад каждого значения в процентах."""
    t, m, win, base, where, note = _prepare(metric, period_from, period_to, last_months, last_days, baseline,
                                            where_dimension, where_value)
    dim = _pick_dimension(t, by, m, win, base, where)
    cmp = E.compare(t, m, win, base, by=dim, where=where)
    g = cmp.groups
    out = {**_head(t, m, win, base, where, note), "dimension": dim, "total": _deviation(cmp),
           "values_count": int(len(g)), "values": E.group_rows(cmp, MAX_VALUES),
           "top_value": str(g.index[0]) if len(g) and abs(cmp.diff) > 1e-12 else None,
           "top_value_share_pct": r(g["share"].iloc[0], 1) if len(g) else None}
    if m.kind in ("mean", "ratio"):
        out["aggregation_note"] = "Вклад значения в отношение — разложение «ставка и структура»: учитывает его собственную динамику и сдвиг доли в объёме."
    if not cmp.significant:
        out["interpretation_note"] = "Общее изменение незначимо: вклады значений трактовать осторожно."
    return out


@_tool
def get_related_metrics(metric: MetricArg, period_from: FromArg = None, period_to: ToArg = None,
                        last_months: LastMonthsArg = None, last_days: LastDaysArg = None, baseline: BaselineArg = None,
                        where_dimension: WhereDimArg = None, where_value: WhereValueArg = None) -> dict:
    """Связанные показатели и их изменение за тот же период: числитель и знаменатель отношения, показатели из таблицы
    связей, показатели, изменившиеся в то же время (связь у них не установлена). Для каждого — какое значение
    измерения дало основной вклад."""
    t, m, win, base, where, note = _prepare(metric, period_from, period_to, last_months, last_days, baseline,
                                            where_dimension, where_value)
    main = E.compare(t, m, win, base, where=where)
    scan = E.scan_dimensions(t, m, win, base, where) if t.dims else []
    dim = scan[0]["dimension"] if scan else None
    top_value = str(scan[0]["_cmp"].groups.index[0]) if scan else None
    items = E.related(t, m, main, where, dim, primary_value=top_value)
    out = {**_head(t, m, win, base, where, note), "primary_dimension": dim, "related": items}
    if not items:
        out["related_note"] = "Связанных показателей со значимым изменением не найдено."
    return out


@_tool
def get_goal_info(name: Annotated[str, Field(description="Название цели или показателя")]) -> dict:
    """Владелец цели, родительская цель (и её владелец), дочерние цели и показатели цели. Есть только в отчётах
    стратегического формата (листы goals, facts, links); в обычной таблице цели и владельцев нет."""
    t = datastore.get_table()
    if not t.goals:
        return {"available": False, "note": "В этом отчёте нет информации о целях и владельцах."}
    goal_scores = scores(name, {g: g for g in t.goals})
    metric_scores = scores(name, {m.name: m.name for m in t.spec.metrics})
    if metric_scores and metric_scores[0][1] >= MIN_MATCH_SCORE and metric_scores[0][1] > (goal_scores[0][1] if goal_scores else 0):
        goal = t.metric(metric_scores[0][0]).goal
    elif goal_scores and goal_scores[0][1] >= MIN_MATCH_SCORE:
        goal = goal_scores[0][0]
    else:
        raise ToolError(f"Цель или показатель «{name}» не найдены.", available_goals=list(t.goals))
    parent = t.goals[goal]["parent"]
    return {"available": True, "goal": goal, "owner": t.goals[goal]["owner"], "parent_goal": parent,
            "parent_owner": t.goals[parent]["owner"] if parent else None,
            "child_goals": [{"goal": g, "owner": info["owner"]} for g, info in t.goals.items() if info["parent"] == goal],
            "metrics": [{"metric": m.name, "unit": m.unit} for m in t.spec.metrics if m.goal == goal]}


@_tool
def diagnose_change(metric: MetricArg, period_from: FromArg = None, period_to: ToArg = None,
                    last_months: LastMonthsArg = None, last_days: LastDaysArg = None, baseline: BaselineArg = None,
                    where_dimension: WhereDimArg = None, where_value: WhereValueArg = None) -> dict:
    """Главный инструмент для вопросов «почему изменилось / отстаём»: за один вызов даёт изменение показателя, с какого
    периода оно держится, вклад значений по каждому измерению (самое объясняющее — первым), углубление внутрь главного
    значения, связанные показатели и арифметику сценариев. Все числа для записки берите из его результата."""
    t, m, win, base, where, note = _prepare(metric, period_from, period_to, last_months, last_days, baseline,
                                            where_dimension, where_value)
    total = E.compare(t, m, win, base, where=where)
    out = _head(t, m, win, base, where, note)
    if m.goal:
        out.update(goal=m.goal, owner=m.owner, parent_goal=t.goals.get(m.goal, {}).get("parent"))
    out["deviation"] = _deviation(total)
    since, run, unit_label = E.trailing_run(t, m, win, base, total, where)
    out.update(unfavorable_periods_in_row=run, unfavorable_since=since, interval=unit_label)
    scan = E.scan_dimensions(t, m, win, base, where) if t.dims else []
    primary = scan[0] if scan and scan[0]["top_share_pct"] is not None else None
    out["primary_dimension"] = primary["dimension"] if primary else None
    out["dimensions"] = [{k: v for k, v in d.items() if k != "_cmp"} for d in scan]
    out["drill_down"] = E.drill_down(t, m, win, base, where, primary) if primary else []
    local = E.local_deviations(m, total, scan)
    if local:
        out["local_deviations"] = local
        out["local_note"] = ("Итог в пределах допуска или лучше базы, но внутри есть заметные ухудшения: это повод проверить "
                             "указанные значения, а не объяснение изменения итога.")
    top_value = str(primary["_cmp"].groups.index[0]) if primary else None
    out["related_metrics"] = E.related(t, m, total, where, out["primary_dimension"], primary_value=top_value)
    scenarios = E.scenarios(primary["_cmp"], m.unit) if primary else []
    out["recovery_scenarios"] = scenarios
    if not scenarios:
        out["scenarios_note"] = ("Изменение незначимо или в лучшую сторону либо нет измерений: сценарии возврата к базе "
                                 "не строились.")
    if m.kind in ("mean", "ratio"):
        out["aggregation_note"] = "Для отношения вклад значения — разложение «ставка и структура»: его динамика и сдвиг доли в объёме."
    return out


@_tool
def get_structure(metric: MetricArg, by: ByArg = None, period_from: FromArg = None, period_to: ToArg = None,
                  last_months: LastMonthsArg = None, last_days: LastDaysArg = None, where_dimension: WhereDimArg = None,
                  where_value: WhereValueArg = None) -> dict:
    """Значение показателя за период и из чего оно состоит: значение и доля каждого значения измерения. Работает у любого
    отчёта, в том числе без дат и без плана, где сравнивать не с чем: для вопросов «какое значение», «из чего состоит»,
    «какой канал даёт основную долю». Вопросы про изменение и причины решает diagnose_change."""
    t = datastore.get_table()
    m = E.find_metric(t, metric)
    win, note = E.resolve_window(t, period_from, period_to, last_months, last_days)
    where = E.resolve_where(t, where_dimension, where_value)
    out = {"metric": m.name, "unit": m.unit, "better": BETTER_WORD[m.better], "definition": m.definition(),
           "period": _period(t, win), "value": r(E.value_only(t, m, win, where))}
    if where:
        out["filter"] = {d: v for d, v in where}
    if note:
        out["note"] = note
    free = [d for d in t.dims if all(d != w for w, _ in where)]
    if by:
        dims = {d.casefold(): d for d in free}
        if by.strip().casefold() not in dims:
            raise ToolError(f"Измерения «{by}» нет или оно занято фильтром. Доступны: {', '.join(free) or 'нет измерений'}.",
                            dimensions=free)
        free = [dims[by.strip().casefold()]]
    out["structure"] = [{"dimension": d, "values_count": len(t.dim_values[d]), "values": E.structure(t, m, win, d, where, MAX_VALUES)}
                        for d in free[:MAX_STRUCTURE_DIMENSIONS]]
    if not out["structure"]:
        out["structure_note"] = "В данных нет измерений: значение показателя назвать можно, а из чего оно состоит — нет."
    return out


ChangeTypeArg = Annotated[str, Field(description="Как меняем значение: pct — на N процентов (−20 — снизить на 20%), "
                                                  "abs — прибавить N единиц показателя (для процентов — п.п.), "
                                                  "set — задать значение N")]
ChangeValueArg = Annotated[float, Field(description="Величина изменения: для pct — проценты со знаком, для abs — единицы "
                                                     "показателя со знаком, для set — само новое значение")]
EffectOnArg = Annotated[list[str] | None, Field(description="Показатели, на которые нужно оценить влияние. Не указано — "
                                                             "считаются сам показатель и показатели, связанные с ним формулой. "
                                                             "Назовите здесь показатель без формулы связи (например заказы при "
                                                             "изменении расходов): будет оценка по истории с интервалом и оговоркой")]
HorizonArg = Annotated[int | None, Field(description="На сколько периодов вперёд строить прогноз, целое от 1 до 24. Период — шаг "
                                                      "ряда: день, неделя или месяц (ответ называет его в unit_word; у данных с "
                                                      "выходными ряд берётся по неделям). Не указано — подбирается по "
                                                      "детализации (7 дней, 4 недели, 3 месяца, 2 квартала)", ge=1, le=24)]


@_tool
def what_if(metric: MetricArg, change_type: ChangeTypeArg, change_value: ChangeValueArg, where_dimension: WhereDimArg = None,
            where_value: WhereValueArg = None, effect_on: EffectOnArg = None, period_from: FromArg = None, period_to: ToArg = None,
            last_months: LastMonthsArg = None, last_days: LastDaysArg = None, horizon: HorizonArg = None) -> dict:
    """Вопросы «что если…»: что будет с итогом и связанными показателями, если значение показателя изменится (в целом или
    в одном срезе: канале, сегменте, отделе). Даёт арифметику по окну (точно), эффект на показатели с формулой связи
    (точно), оценку по истории для показателей без формулы (приблизительно, с интервалом) и прогноз на ближайшие периоды
    «если ничего не менять» и «со сценарием». Для вопроса «что будет дальше» без изменений вызывайте forecast_metric."""
    t = datastore.get_table()
    m = E.find_metric(t, metric)
    try:
        value = float(change_value)
    except (TypeError, ValueError):
        raise ToolError("change_value должно быть числом.") from None
    ctype = W.normalize_change_type(change_type)
    horizon = W.normalize_horizon(horizon)
    win, note = E.resolve_window(t, period_from, period_to, last_months, last_days)
    where = E.resolve_where(t, where_dimension, where_value)
    out = {"metric": m.name, "unit": m.unit, "better": BETTER_WORD[m.better], "definition": m.definition(), "period": _period(t, win)}
    if note:
        out["note"] = note
    out.update(W.build_scenario(t, m, win, where, ctype, value, effect_on, horizon))
    return out


@_tool
def forecast_metric(metric: MetricArg, horizon: HorizonArg = None, where_dimension: WhereDimArg = None,
                    where_value: WhereValueArg = None) -> dict:
    """Прогноз показателя «если ничего не менять»: значения на ближайшие периоды с интервалом 80%, метод, тренд и
    надёжность (по длине истории и ошибке на прошлых периодах). Для вопросов «что будет с показателем», «каким будет
    значение через 3 месяца». Для вопросов «что если изменить…» вызывайте what_if."""
    t = datastore.get_table()
    m = E.find_metric(t, metric)
    horizon = W.normalize_horizon(horizon)
    where = E.resolve_where(t, where_dimension, where_value)
    out = {"metric": m.name, "unit": m.unit, "better": BETTER_WORD[m.better], "definition": m.definition()}
    if where:
        out["filter"] = {d: v for d, v in where}
    out["forecast"] = W.build_forecast(t, m, horizon, where)
    return out


TOOLS: dict[str, Callable[..., dict]] = {
    fn.__name__: fn for fn in (describe_dataset, find_metrics, get_change, break_down, get_related_metrics,
                               get_goal_info, diagnose_change, get_structure, what_if, forecast_metric)
}
