"""Дашборд по любому отчёту: сводка для модели, разбор её плана и сборка данных блоков. Все числа считает код.

Порядок работы:
  build_digest(table)        — сводка отчёта для GigaChat: показатели, итоги к базе, ряды, вклады значений измерений,
                               скрытые проблемы. Построчных данных в ней нет;
  assemble(table, plan, ...) — план модели (какие блоки показать) сверяется с отчётом, блоки строятся кодом;
                               то, что не удалось построить, убирается с пояснением, недостающее добирается по умолчанию;
  fallback_plan(digest)      — план «по умолчанию» из сводки: им дополняется ответ модели, и по нему же работает заглушка.

Отчёт может быть любым: без дат (нет графиков динамики), без измерений (только плитки и динамика), без плана и без
прошлых периодов (значения без сравнения и структура), плоская таблица или стратегическая витрина.
"""
import json
import time
from collections import Counter

from data.model import Table
from data.parsing import parse_date_value
from tools import engine as E
from tools import fmt
from tools import series as S
from tools.engine import ToolError, r

MAX_DIGEST_METRICS = 24  # показателей с подробностями в сводке; остальные — одним списком названий
MAX_DIGEST_CHARS = 30_000  # сводка не больше этого размера (примерно 12 тысяч токенов)
SERIES_POINTS = 10  # точек ряда на показатель в сводке
SCAN_METRICS = 16  # у скольких показателей искать вклады по измерениям
SCAN_BUDGET_S = 25.0  # потолок времени на перебор измерений: на очень больших файлах часть показателей идёт без него
MIN_WIDGETS = 3
LIMITS = {"kpi": 8, "trend": 4, "breakdown": 4}
BETTER_WORD = {"up": "больше", "down": "меньше", "none": "нейтрально"}
SEVERITY_ORDER = {"problem": 0, "risk": 1, "opportunity": 2, "info": 3}


# ======================================================================================================================
# Сводка отчёта для модели
# ======================================================================================================================


def _window(t: Table):
    try:
        return E.resolve_window(t)[0]
    except ToolError:
        return E.Window(0, 0)


def _analyse(t: Table, m, win, where=(), requested: str | None = None):
    """(Comparison, Baseline) показателя или (None, None), если сравнивать не с чем."""
    try:
        base = E.resolve_baseline(t, m, win, requested)
        return E.compare(t, m, win, base, where=where), base
    except ToolError:
        if requested not in (None, "auto"):
            return _analyse(t, m, win, where, None)
        return None, None


def _priority(cmp) -> tuple:
    """Что показать первым: значимое ухудшение, затем значимое улучшение, затем остальное, затем без сравнения."""
    if cmp is None:
        return (3, 0.0)
    pct = abs(cmp.pct) if cmp.pct is not None else 0.0
    if cmp.significant and cmp.unfavorable:
        return (0, -pct)
    if cmp.significant:
        return (1, -pct)
    return (2, -pct)


def _top_rows(rows: list[dict], n: int = 3) -> list[dict]:
    return [{k: row[k] for k in ("value", "contribution_pct", "change_pct") if k in row} for row in rows[:n]]


def _metric_entry(t: Table, m, win, cmp, base, scan: bool, points: int) -> dict:
    item = {"name": m.name, "unit": m.unit, "better": BETTER_WORD[m.better], "how": m.definition()}
    if m.goal:
        item["goal"] = m.goal
        if m.owner:
            item["owner"] = m.owner
    if cmp is not None:
        item.update(fact=r(cmp.v1), base=r(cmp.v0), deviation_abs=r(cmp.diff), deviation_pct=r(cmp.pct, 1),
                    assessment=cmp.assessment, significant=bool(cmp.significant), unfavorable=bool(cmp.unfavorable),
                    baseline=base.label)
    else:
        try:
            item["fact"] = r(E.value_only(t, m, win))
        except ToolError:
            item["fact"] = None
        item["baseline"] = None  # сравнивать не с чем: нет плана и нет прошлых периодов
    if t.grain is not None:
        sp = S.series(t, m, window=None)
        if sp["series"]:
            vals = sp["series"][0]["values"]
            item["series"] = {"bucket": sp["bucket"], "points": [[a, b] for a, b in zip(sp["labels"][-points:], vals[-points:])]}
    if not scan or not t.dims:
        return item
    if cmp is not None:
        found = E.scan_dimensions(t, m, win, base)
        if found:
            item["dimensions"] = [{"dimension": d["dimension"], "values_count": d["values_count"], "top": _top_rows(d["top"])}
                                  for d in found[:2]]
            local = E.local_deviations(m, cmp, found, limit=2)
            if local:
                item["local_deviations"] = [{k: x[k] for k in ("dimension", "value", "change_pct", "effect_on_total_pct")}
                                            for x in local]
        if cmp.unfavorable and cmp.significant:
            since, run, unit_label = E.trailing_run(t, m, win, base, cmp)
            if since:
                item.update(unfavorable_since=since, unfavorable_periods=run, interval=unit_label)
    else:
        parts = []
        for dim in t.dims[:2]:
            try:
                rows = E.structure(t, m, win, dim, top=3)
            except ToolError:
                continue
            parts.append({"dimension": dim, "values_count": len(t.dim_values[dim]),
                          "top": [{"value": x["value"], "share_pct": x["share_pct"]} for x in rows if not x["value"].startswith("прочие")]})
        if parts:
            item["structure"] = parts
    return item


def build_digest(t: Table, *, max_metrics: int = MAX_DIGEST_METRICS, points: int = SERIES_POINTS,
                 budget_s: float = SCAN_BUDGET_S) -> dict:
    """Сводка отчёта для модели. Построчных данных нет: итоги, ряды, вклады, названия. Все числа посчитаны кодом."""
    deadline = time.monotonic() + budget_s
    win = _window(t)
    info = t.summary()
    scored = []
    for m in t.spec.metrics:
        cmp, base = _analyse(t, m, win)
        scored.append((m, cmp, base))
    order = sorted(range(len(scored)), key=lambda i: (_priority(scored[i][1]), i))
    entries = []
    for rank, i in enumerate(order[:max_metrics]):
        m, cmp, base = scored[i]
        scan = rank < SCAN_METRICS and time.monotonic() < deadline
        entries.append(_metric_entry(t, m, win, cmp, base, scan, points))
    bases = Counter(b.label for _, _, b in scored if b is not None)
    report = {
        "source": t.source, "kind": "стратегическая витрина с целями" if t.goals else "таблица", "rows": info["rows"],
        "metrics_count": len(t.spec.metrics), "grain": t.grain, "has_plan": info["has_plan"],
        "period": ({"from": info["first_period"], "to": info["last_period"], "periods": info["periods"]}
                   if t.grain is not None else None),
        "analysis_window": ({"from": t.label(win.a), "to": t.label(win.b), "periods": win.n} if t.grain is not None else None),
        "baseline": bases.most_common(1)[0][0] if bases else None,
    }
    if t.rows_excluded:
        report["rows_excluded_by_filters"] = t.rows_excluded
    digest = {
        "report": report,
        "metrics": entries,
        "dimensions": [{"name": d, "values_count": len(t.dim_values[d]), "top_values": t.dim_values[d][:8]} for d in t.dims[:8]],
    }
    rest = [{"name": scored[i][0].name, "unit": scored[i][0].unit} for i in order[max_metrics:]]
    if rest:
        digest["other_metrics"] = rest
    problems = [e["name"] for e in entries if e.get("significant") and e.get("unfavorable")]
    gains = [e["name"] for e in entries if e.get("significant") and not e.get("unfavorable")]
    hidden = [e["name"] for e in entries if e.get("local_deviations")]
    digest["highlights"] = {"significant_deterioration": problems, "significant_improvement": gains,
                            "hidden_problems_inside_flat_total": hidden}
    if t.goals:
        digest["goals"] = [{"goal": g, "owner": info_g["owner"], "parent": info_g["parent"],
                            "metrics": [m.name for m in t.spec.metrics if m.goal == g]} for g, info_g in list(t.goals.items())[:20]]
    if t.links:
        digest["links"] = [{"cause": x["cause"], "effect": x["effect"], "direction": x["direction"]} for x in t.links[:30]]
    if t.warnings:
        digest["data_quality"] = [w[:240] for w in t.warnings[:5]]
    return _fit(digest, points)


def _fit(digest: dict, points: int) -> dict:
    """Сводка должна уложиться в MAX_DIGEST_CHARS: сначала укорачиваем ряды, потом убираем второстепенное."""
    def size() -> int:
        return len(digest_text(digest))

    if size() > MAX_DIGEST_CHARS:
        for m in digest["metrics"]:
            if "series" in m:
                m["series"]["points"] = m["series"]["points"][-max(points // 2, 4):]
    if size() > MAX_DIGEST_CHARS:
        digest.pop("other_metrics", None)
        digest.pop("links", None)
    while size() > MAX_DIGEST_CHARS and len(digest["metrics"]) > 8:
        digest["metrics"].pop()
        digest["report"]["truncated_metrics"] = True
    return digest


def digest_text(digest: dict) -> str:
    return json.dumps(digest, ensure_ascii=False, separators=(",", ":"))


# ======================================================================================================================
# План по умолчанию: из сводки
# ======================================================================================================================


def _cluster(m: dict) -> tuple | None:
    """Где сосредоточено изменение показателя: (измерение, главное значение). Показатели с общей причиной — один кластер."""
    d = (m.get("dimensions") or [None])[0]
    return (d["dimension"], d["top"][0]["value"]) if d and d.get("top") else None


def _diverse(items: list[dict], per_cluster: int = 2) -> list[dict]:
    """Порядок показателей, при котором на экране не оказываются только следствия одной и той же причины:
    сначала не больше per_cluster показателей на кластер, остальные — после."""
    head, tail, counts = [], [], Counter()
    for m in items:
        c = _cluster(m)
        if c is None or counts[c] < per_cluster:
            head.append(m)
            counts[c] += 1
        else:
            tail.append(m)
    return head + tail


def _first_dim(m: dict) -> str | None:
    for key in ("dimensions", "structure"):
        if m.get(key):
            return m[key][0]["dimension"]
    return None


def fallback_plan(digest: dict) -> dict:
    """План дашборда по одной лишь сводке: плитки по самым заметным показателям (не только следствия одной причины),
    динамика и разбивка там, где изменение сосредоточено, и инсайты из чисел сводки (дословно). Им дополняется ответ
    модели, и по нему же работает заглушка."""
    metrics = digest["metrics"]
    rep = digest["report"]
    grain = rep.get("grain")
    problems = _diverse([m for m in metrics if m.get("significant") and m.get("unfavorable")])
    gains = [m for m in metrics if m.get("significant") and not m.get("unfavorable")]
    hidden = [m for m in metrics if m.get("local_deviations") and m not in problems]
    others = [m for m in metrics if m not in problems and m not in gains and m not in hidden]

    widgets: list[dict] = []
    ids: dict[str, list[str]] = {}

    def add(kind: str, m: dict, title: str, note: str = "", dimension: str | None = None) -> None:
        wid = f"w{len(widgets) + 1}"
        widgets.append({"id": wid, "type": kind, "metric": m["name"], "title": title, "note": note, "dimension": dimension,
                        "baseline": "auto"})
        ids.setdefault(m["name"], []).append(wid)

    picks: list[dict] = []
    for m in problems[:3] + hidden[:2] + gains[:1] + problems[3:] + gains[1:] + hidden[2:] + others:
        if m not in picks and len(picks) < 7:
            picks.append(m)
    for m in picks:
        add("kpi", m, m["name"], "Итог за период к базе сравнения" if m.get("baseline") else "Значение за период")
    focus = [m for m in dict.fromkeys(id(x) for x in problems[:1] + hidden[:1] + problems[1:2] + gains[:1] + others[:1])]
    by_id = {id(x): x for x in problems + hidden + gains + others}
    several_periods = (rep.get("period") or {}).get("periods", 0) >= 2
    for key in focus[:2] if grain and several_periods else []:
        m = by_id[key]
        dim = _first_dim(m)
        add("trend", m, f"Динамика: {m['name']}" + (f" (по «{dim}»)" if dim else ""),
            "Линия показателя по периодам" + (f" в разрезе «{dim}»" if dim else ""), dim)
    for m in problems[:2] + hidden[:1] + gains[:1] + others:
        if len([w for w in widgets if w["type"] == "breakdown"]) >= 2:
            break
        dim = _first_dim(m)
        if dim and not any(w["type"] == "breakdown" and w["metric"] == m["name"] for w in widgets):
            word = "вклад в изменение" if m.get("baseline") else "структура"
            add("breakdown", m, f"{m['name']}: {word} по «{dim}»",
                "Какие значения дают основное изменение" if m.get("baseline") else "Из чего состоит показатель", dim)

    insights: list[dict] = []
    for m in problems[:3]:
        top = (m.get("dimensions") or [{}])[0]
        text = _fact_sentence(m, rep)
        if top.get("top"):
            lead = top["top"][0]
            text += (f" Основной вклад — {fmt.dim_word(top['dimension'])} «{lead['value']}»: {fmt.num(lead['contribution_pct'])}% изменения"
                     f" (собственное изменение {fmt.num(lead['change_pct'], True)}%).")
        if m.get("unfavorable_since"):
            text += f" Ухудшение держится {m['unfavorable_periods']} {m['interval']} подряд, с {m['unfavorable_since']}."
        insights.append({"title": f"{m['name']}: {m['assessment']}", "text": text, "severity": "problem",
                         "widgets": ids.get(m["name"], [])})
    for m in hidden[:2]:
        parts = "; ".join(f"{fmt.dim_word(x['dimension'])} «{x['value']}»: {fmt.num(x['change_pct'], True)}% к базе "
                          f"(влияние на итог {fmt.num(x['effect_on_total_pct'], True)}%)" for x in m["local_deviations"])
        insights.append({"title": f"{m['name']}: итог в норме, но внутри есть ухудшения",
                         "text": f"{_fact_sentence(m, rep)} Внутри итога заметно просели: {parts}.", "severity": "risk",
                         "widgets": ids.get(m["name"], [])})
    for m in gains[:1]:
        insights.append({"title": f"{m['name']}: {m['assessment']}", "text": _fact_sentence(m, rep), "severity": "opportunity",
                         "widgets": ids.get(m["name"], [])})
    if not insights:
        if any(m.get("baseline") for m in metrics):
            text = "Ни один показатель не отклонился от базы сравнения значимо: изменения в пределах допуска."
        else:
            text = "В отчёте нет плана и прошлых периодов, поэтому дашборд показывает значения и структуру без сравнения."
            for m in metrics[:3]:
                if m.get("structure"):
                    st = m["structure"][0]
                    text += (f" Например, у показателя «{m['name']}» больше всего даёт {fmt.dim_word(st['dimension'])} "
                             f"«{st['top'][0]['value']}»: {fmt.num(st['top'][0]['share_pct'])}%.")
                    break
        insights.append({"title": "Существенных изменений нет", "text": text, "severity": "info", "widgets": []})
    if digest.get("data_quality"):
        insights.append({"title": "Качество данных", "text": " ".join(digest["data_quality"][:2]), "severity": "info", "widgets": []})

    next_steps: list[str] = []
    for m in problems[:2]:
        top = (m.get("dimensions") or [{}])[0]
        where = f" ({top['dimension']} «{top['top'][0]['value']}»)" if top.get("top") else ""
        since = m.get("unfavorable_since") or (rep["analysis_window"]["from"] if rep.get("analysis_window") else "начала периода")
        who = f" Владелец цели: {m['owner']}." if m.get("owner") else ""
        next_steps.append(f"Выяснить, что изменилось по показателю «{m['name']}»{where} с {since}: условия, каналы, поставщики, "
                          f"качество данных.{who}")
    for m in hidden[:1]:
        x = m["local_deviations"][0]
        next_steps.append(f"Проверить {fmt.dim_word(x['dimension'])} «{x['value']}»: показатель «{m['name']}» просел там при ровном итоге.")
    next_steps.append("Сверить цифры дашборда с первичным источником и договориться, какую базу сравнения считать основной.")
    if not problems and not hidden:
        next_steps.insert(0, "Выбрать показатели, за которыми нужно следить регулярно, и задать для них плановые значения.")

    questions: list[str] = []
    for m in (problems + hidden + gains + others)[:2]:
        questions.append(f"Почему изменился показатель «{m['name']}» и где именно?")
    lead_metric = (problems + hidden)[:1]
    if lead_metric:
        m = lead_metric[0]
        dim = _first_dim(m)
        lead = (((m.get("dimensions") or [{}])[0].get("top") or [{}])[0].get("value")
                or (m.get("local_deviations") or [{}])[0].get("value"))
        if lead and dim:
            questions.append(f"Что произойдёт, если вернуть {fmt.dim_word(dim)} «{lead}» к базе сравнения?")
            questions.append(f"Какие риски, если ничего не менять по {fmt.dim_word(dim)} «{lead}»?")
    questions.append("Что делать в первую очередь по итогам дашборда и в каком порядке?")
    questions.append("Каких данных не хватает, чтобы уверенно объяснить причины?")

    head = f"Отчёт «{rep['source']}»: {fmt.num(rep['rows'])} строк, {rep['metrics_count']} показателей"
    if rep.get("period"):
        head += f", период {rep['period']['from']} – {rep['period']['to']}"
    parts = [head]
    if problems:
        parts.append("Значимое ухудшение: " + ", ".join(f"«{m['name']}»" for m in problems[:4]))
    elif hidden:
        parts.append("Итоги в норме, но внутри есть ухудшения: " + ", ".join(f"«{m['name']}»" for m in hidden[:3]))
    else:
        parts.append("Значимых ухудшений нет")
    return {"title": f"Дашборд: {rep['source']}", "summary": ". ".join(parts) + ".", "widgets": widgets, "insights": insights,
            "next_steps": next_steps[:5], "questions": questions[:6]}


def _fact_sentence(m: dict, rep: dict) -> str:
    u = fmt.unit(m["unit"])
    window = rep.get("analysis_window")
    span = f" за {window['from']} – {window['to']}" if window else ""
    if m.get("baseline") is None:
        return f"{m['name']}{span}: {fmt.num(m['fact'])}{u}."
    return (f"{m['name']}{span}: факт {fmt.num(m['fact'])}{u}, база ({m['baseline']}) {fmt.num(m['base'])}{u}, "
            f"изменение {fmt.num(m['deviation_pct'], True)}% — {m['assessment']}.")


# ======================================================================================================================
# Сборка блоков
# ======================================================================================================================


def _period(t: Table, win) -> dict | None:
    return {"from": t.label(win.a), "to": t.label(win.b), "periods": win.n} if t.grain is not None else None


def _baseline_info(t: Table, base) -> dict:
    info = {"type": base.kind, "label": base.label}
    if base.win is not None:
        info.update({"from": t.label(base.win.a), "to": t.label(base.win.b)})
    return info


def _iso(label: str) -> str | None:
    ts = parse_date_value(label)
    return str(ts.date()) if ts is not None else None


def _build_widget(t: Table, raw: dict) -> tuple[dict, dict]:
    """(спецификация блока с каноническими названиями, данные блока). Бросает ToolError с русским пояснением."""
    m = E.find_metric(t, raw["metric"])
    kind = raw["type"]
    where = E.resolve_where(t, raw.get("where_dimension"), raw.get("where_value")) if (raw.get("where_dimension") or raw.get("where_value")) else []
    dims = {d.casefold(): d for d in t.dims}
    dimension = dims.get(str(raw.get("dimension") or "").strip().casefold())
    notes: list[str] = []
    if raw.get("dimension") and dimension is None:
        notes.append(f"измерения «{raw['dimension']}» в отчёте нет")
    if dimension is not None and any(d == dimension for d, _ in where):
        dimension = None
        notes.append("разрез совпал с фильтром блока")
    win = _window(t)
    if raw.get("last_days") or raw.get("last_months"):
        try:
            win = E.resolve_window(t, None, None, raw.get("last_months"), raw.get("last_days"))[0]
        except ToolError as exc:
            notes.append(str(exc))
    spec = {"type": kind, "metric": m.name, "title": raw["title"], "note": raw.get("note", ""), "dimension": dimension,
            "baseline": raw.get("baseline") or "auto", "where": {d: v for d, v in where} or None,
            "last_days": raw.get("last_days"), "last_months": raw.get("last_months")}
    data = {"type": kind, "metric": m.name, "unit": m.unit, "better": m.better, "definition": m.definition(),
            "title": raw["title"], "note": raw.get("note", ""), "filter": spec["where"], "period": _period(t, win)}
    if m.goal:
        data.update(goal=m.goal, owner=m.owner)
    if kind == "kpi":
        data.update(_kpi_data(t, m, win, where, spec["baseline"]))
    elif kind == "trend":
        data.update(_trend_data(t, m, win, where, dimension, bool(raw.get("last_days") or raw.get("last_months")), spec["baseline"]))
        spec["dimension"] = dimension
    elif kind == "breakdown":
        if dimension is None:
            if not t.dims:
                raise ToolError("в отчёте нет измерений: разбивку по значениям построить нельзя")
            fallback = next((d for d in t.dims if all(d != w for w, _ in where)), None)
            if fallback is None:
                raise ToolError("все измерения заняты фильтром блока")
            dimension = fallback
            notes.append(f"разрез подобран автоматически: «{dimension}»")
        spec["dimension"] = dimension
        data.update(_breakdown_data(t, m, win, where, dimension, spec["baseline"]))
    data["adjustments"] = notes
    return spec, data


def _kpi_data(t, m, win, where, requested) -> dict:
    cmp, base = _analyse(t, m, win, where, requested)
    if cmp is not None:
        out = {"fact": r(cmp.v1), "base": r(cmp.v0), "deviation_abs": r(cmp.diff), "deviation_pct": r(cmp.pct, 1),
               "assessment": cmp.assessment, "significant": bool(cmp.significant), "unfavorable": bool(cmp.unfavorable),
               "baseline": _baseline_info(t, base)}
    else:
        out = {"fact": r(E.value_only(t, m, _window(t), where)), "baseline": None}
    out["spark"] = S.sparkline(t, m, where=where) if t.grain is not None else {"labels": [], "values": []}
    return out


def _trend_data(t, m, win, where, split, restricted: bool, requested) -> dict:
    if t.grain is None:
        raise ToolError("в отчёте нет дат: график динамики построить нельзя")
    s = S.series(t, m, where=where, by=split, window=win if restricted else None)
    if len(s["labels"]) < 2:
        raise ToolError("в данных меньше двух периодов: график динамики не нужен")
    out = {"bucket": s["bucket"], "labels": s["labels"], "dates": s["dates"], "partial": s["partial"], "series": s["series"],
           "plan": s["plan"] if split is None else None, "others": s["others"], "split_by": split}
    cmp, base = _analyse(t, m, _window(t), where, requested)
    if cmp is not None and cmp.significant and cmp.unfavorable:
        since, run, _ = E.trailing_run(t, m, _window(t), base, cmp, where)
        marker = _iso(since) if since else None
        if marker:
            out["marker"] = {"date": marker, "label": f"ухудшение с {since}"}
    return out


def _breakdown_data(t, m, win, where, dim, requested) -> dict:
    cmp, base = _analyse(t, m, win, where, requested)
    if cmp is not None:
        full = E.compare(t, m, win, base, by=dim, where=where)
        return {"mode": "change", "dimension": dim, "values_count": int(len(full.groups)), "baseline": _baseline_info(t, base),
                "total": {"fact": r(cmp.v1), "base": r(cmp.v0), "deviation_abs": r(cmp.diff), "deviation_pct": r(cmp.pct, 1),
                          "assessment": cmp.assessment, "significant": bool(cmp.significant), "unfavorable": bool(cmp.unfavorable)},
                "rows": E.group_rows(full, 8)}
    rows = E.structure(t, m, win, dim, where, 8)
    return {"mode": "structure", "dimension": dim, "values_count": len(t.dim_values[dim]), "baseline": None, "rows": rows}


def assemble(t: Table, plan: dict, digest: dict | None = None) -> dict:
    """Проверяет план модели по отчёту и строит блоки. Возвращает {"plan", "widgets", "notes"}:
    plan — очищенный план (канонические названия, сквозные id, ссылки инсайтов на существующие блоки);
    widgets — данные блоков, посчитанные кодом; notes — что пришлось убрать, исправить или добрать."""
    notes: list[str] = []
    specs, built, seen = [], [], set()
    counts: Counter = Counter()
    id_map: dict[str, str] = {}

    def take(raw: dict, label: str, original_id: str | None, quiet: bool = False) -> None:
        def note(text: str) -> None:
            if not quiet:  # блоки по умолчанию добираются молча: пользователю важно, что убрано из ответа модели
                notes.append(text)

        try:
            spec, data = _build_widget(t, raw)
        except ToolError as exc:
            note(f"Блок «{label}» не построен: {exc}.")
            return
        key = (spec["type"], spec["metric"], spec["dimension"], json.dumps(spec["where"], ensure_ascii=False, sort_keys=True),
               spec["last_days"], spec["last_months"])
        if key in seen:
            note(f"Блок «{label}» убран: такой уже есть.")
            return
        if counts[spec["type"]] >= LIMITS[spec["type"]]:
            note(f"Блок «{label}» убран: блоков этого вида не больше {LIMITS[spec['type']]}.")
            return
        seen.add(key)
        counts[spec["type"]] += 1
        wid = f"w{len(built) + 1}"
        spec["id"], data["id"] = wid, wid
        if original_id:
            id_map[original_id] = wid
        for adj in data["adjustments"]:
            note(f"Блок «{label}»: {adj}.")
        specs.append(spec)
        built.append(data)

    for idx, raw in enumerate(plan["widgets"], 1):
        take(raw, raw.get("title") or f"№{idx}", raw.get("id") or f"w{idx}")
    if len(built) < MIN_WIDGETS and digest is not None:
        added = 0
        for raw in fallback_plan(digest)["widgets"]:
            if len(built) >= MIN_WIDGETS + 2:
                break
            before = len(built)
            take(raw, raw["title"], None, quiet=True)
            added += len(built) - before
        if added:
            notes.append(f"Предложенных блоков оказалось слишком мало, дашборд дополнен блоками по умолчанию: {added}.")
    insights = []
    for ins in plan["insights"]:
        refs = [id_map[w] for w in ins.get("widgets", []) if w in id_map]
        if ins.get("widgets") and not refs:
            notes.append(f"У инсайта «{ins['title']}» убрана ссылка на блок, которого нет на дашборде.")
        insights.append({**ins, "widgets": list(dict.fromkeys(refs))})
    insights.sort(key=lambda x: SEVERITY_ORDER.get(x.get("severity", "info"), 3))
    clean = {k: plan[k] for k in ("title", "summary", "next_steps", "questions")}
    clean.update(widgets=specs, insights=insights)
    return {"plan": clean, "widgets": built, "notes": notes}
