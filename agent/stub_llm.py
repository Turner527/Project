"""Детерминированная заглушка вместо GigaChat: для тестов, evals без сети и демо без ключа.

Это не «умная» модель, а сценарий: find_metrics -> diagnose_change -> записка из его результата.
Числа берутся дословно из результатов инструментов (ничего не считается), поэтому записка проходит
верификатор. Качество настоящей модели заглушкой не проверяется — только конвейер вокруг неё.
"""
import json
import re

from agent.llm import TEMPERATURE, FunctionCall, LLMResponse, Usage
from agent.prompts import CONTEXT_HEADER, DIGEST_CLOSE, DIGEST_OPEN
from tools import dashboard, fmt
from tools.text import stems

_WORD_NUMBERS = {"два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8,
                 "девять": 9, "десять": 10, "двенадцать": 12}
_NUM = r"(\d+|" + "|".join(_WORD_NUMBERS) + r")"


_fmt, _u, _dim = fmt.num, fmt.unit, fmt.dim_word


def _n(word: str) -> int:
    return int(word) if word.isdigit() else _WORD_NUMBERS[word]


def _period_args(question: str) -> dict:
    """Период из вопроса: «последние 3 месяца», «за две недели», «последние 14 дней», «квартал», «полгода»."""
    q = question.lower()
    if m := re.search(r"последн\w+\s+" + _NUM + r"\s+(дн|недел|месяц)", q) or re.search(r"за\s+" + _NUM + r"\s+(дн|недел|месяц)", q):
        n, unit = _n(m.group(1)), m.group(2)
        return {"last_months": n} if unit == "месяц" else {"last_days": n * (7 if unit == "недел" else 1)}
    if "квартал" in q:
        return {"last_months": 3}
    if "полгода" in q:
        return {"last_months": 6}
    if re.search(r"последн\w+\s+недел", q):
        return {"last_days": 7}
    if re.search(r"последн\w+\s+месяц", q):
        return {"last_months": 1}
    return {}


_STRATEGIC = re.compile(r"в первую очередь|приоритет|стратег|риск|как распредел|стоит ли|не хватает|каких данных|"
                        r"в каком порядке|что делать дальше|рекомендуе", re.I)


def _is_strategic(question: str) -> bool:
    return bool(_STRATEGIC.search(question))


_RECOVERY = re.compile(r"к базе|к плану|к уровню базы|к прошлому уровню", re.I)  # «вернуть к базе» — арифметика возврата, не прогноз
_WHATIF = re.compile(r"что\s+(?:будет|получится|станет)[^?]*\bесли|\bа\s+если|что\s+если|\bесли\s+(?:мы\s+)?(?:\S+\s+){0,4}"
                     r"(?:снизим|снизит|снизить|уменьш\w+|сократ\w+|увелич\w+|вырастет|вырастут|повысим|повысит|повысить|поднимем|"
                     r"упадёт|упадет|упадут|нарастим)", re.I)
_FORECAST = re.compile(r"прогноз|каким\s+будет|какой\s+будет|какими\s+будут|что\s+будет\s+(?:с|со|через|дальше)|"
                       r"через\s+(?:\d+|\w+)\s+(?:месяц|недел|дн|квартал)|если\s+ничего\s+не\s+(?:менять|делать|изменится)", re.I)
_DOWN = re.compile(r"снизим|снизит|снизить|уменьш|сократ|упад|ниже|минус|−|падени|урезать|урежем", re.I)
_UP = re.compile(r"вырастет|вырастут|увелич|повыс|подним|нарастим|выше|плюс|рост|добавим", re.I)
_AMOUNT = re.compile(r"(\d+(?:[.,]\d+)?)\s*(%|процент\w*|п\.\s?п\.?|пп\b|пункт\w*)", re.I)


def _is_whatif(question: str) -> bool:
    return bool(_WHATIF.search(question)) and not _RECOVERY.search(question) and _AMOUNT.search(question) is not None


def _is_forecast(question: str) -> bool:
    return bool(_FORECAST.search(question)) and not _is_whatif(question)


def _change_args(question: str) -> dict | None:
    """Величина изменения из вопроса: «на 20%», «на 2 п.п.»; знак по словам «снизим / вырастет». Нет величины — None."""
    m = _AMOUNT.search(question)
    if not m:
        return None
    value = float(m.group(1).replace(",", "."))
    if _DOWN.search(question) and not _UP.search(question):
        value = -value
    elif _DOWN.search(question) and _UP.search(question):  # «вырастет, а потом снизим»: берём слово, ближайшее к числу слева
        left = question[: m.start()]
        down = max((x.end() for x in _DOWN.finditer(left)), default=-1)
        up = max((x.end() for x in _UP.finditer(left)), default=-1)
        value = -value if down > up else value
    kind = "abs" if m.group(2).lower().startswith(("п", "пп")) else "pct"
    return {"change_type": kind, "change_value": value}


def _dashboard_context(messages: list) -> dict | None:
    """Выводы дашборда из системного промпта: инсайты (важность, заголовок, текст) и рекомендованные шаги."""
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    if CONTEXT_HEADER not in system:
        return None
    block = system.split(CONTEXT_HEADER, 1)[1]
    insights = [{"severity": a, "title": b, "text": c} for a, b, c in re.findall(r"^- \[([^\]]+)\] (.*?) \| (.*)$", block, re.M)]
    steps_line = next((ln for ln in block.splitlines() if ln.startswith("Рекомендованные шаги:")), "")
    steps = [x.strip() for x in re.split(r"\s*\d+\)\s+", steps_line.split(":", 1)[1])[1:]] if steps_line else []
    return {"insights": insights, "steps": steps} if insights else None


def _scoped(question: str, value: str) -> bool:
    """Значение названо как область анализа («в канале Поиск», «по Казани»), а не как гипотеза («если вернуть …»)."""
    pos = question.lower().find(value.lower())
    if pos < 0:
        return False
    return bool(re.search(r"(^|\s)(в|во|по|для|у|среди|внутри)\s+(\S+\s+){0,2}[«\"]?$", question[:pos].lower()))


class ScriptedLLM:
    name = "stub"

    def chat(self, messages, functions=None, *, function_call="auto", temperature=TEMPERATURE, max_tokens=None):
        question = next(m["content"] for m in messages if m["role"] == "user")
        if DIGEST_OPEN in question:  # запрос плана дашборда: сводка отчёта между метками
            return self._dashboard(messages, question)
        results = [(m["name"], json.loads(m["content"])) for m in messages if m["role"] == "function"]
        calls = [m["function_call"] for m in messages if m["role"] == "assistant" and m.get("function_call")]
        context = _dashboard_context(messages)
        strategic = context is not None and _is_strategic(question)
        if function_call == "none":
            memo = self._strategic_memo(question, context) if strategic else self._memo(results)
            content = json.dumps(memo, ensure_ascii=False)
            return LLMResponse(content=content, usage=self._usage(messages, content), finish_reason="stop")
        if strategic:  # стратегический вопрос после дашборда отвечается по его выводам, инструменты не нужны
            return LLMResponse(content="Данных достаточно.", usage=self._usage(messages, ""), finish_reason="stop")
        call = self._next_call(question, results, context, calls)
        usage = self._usage(messages, json.dumps(call.__dict__, ensure_ascii=False) if call else "")
        if call is None:
            return LLMResponse(content="Данных достаточно.", usage=usage, finish_reason="stop")
        return LLMResponse(function_call=call, usage=usage, finish_reason="function_call")

    # --- выбор вызова ---
    def _dashboard(self, messages: list, question: str) -> LLMResponse:
        raw = question.split(DIGEST_OPEN, 1)[1].rsplit(DIGEST_CLOSE, 1)[0]
        content = json.dumps(dashboard.fallback_plan(json.loads(raw)), ensure_ascii=False)
        return LLMResponse(content=content, usage=self._usage(messages, content), finish_reason="stop")

    def _next_call(self, question: str, results: list, context: dict | None = None, calls: list | None = None) -> FunctionCall | None:
        if not results:
            return FunctionCall("find_metrics", {"query": question})
        name, data = results[-1]
        if name in ("what_if", "forecast_metric"):
            return None  # прогноз и сценарий посчитаны, данных достаточно
        if name == "diagnose_change" and "Не с чем сравнивать" in str(data.get("error", "")) \
                and not any(n == "get_structure" for n, _ in results):
            # у отчёта нет базы сравнения: как велит промпт, вместо изменения берём значение и состав показателя
            last = next((c["arguments"] for c in reversed(calls or []) if c["name"] == "diagnose_change"), {})
            keep = ("metric", "where_dimension", "where_value", "period_from", "period_to", "last_months", "last_days")
            return FunctionCall("get_structure", {k: v for k, v in last.items() if k in keep})
        if name != "find_metrics":
            return None
        metric_hits, goal_hits = data.get("metric_matches", []), data.get("goal_matches", [])
        metric = None
        if metric_hits and (not goal_hits or metric_hits[0]["match_score"] >= 0.9):
            metric = metric_hits[0]["metric"]
        elif goal_hits and goal_hits[0]["metrics"]:
            metric = goal_hits[0]["metrics"][0]["metric"]  # главный показатель цели — первый в списке
        elif metric_hits:
            metric = metric_hits[0]["metric"]
        if metric is None and context:  # вопрос про «этот канал» без названия показателя — берём главную проблему дашборда
            metric = next((i["title"].split(":")[0] for i in context["insights"]
                           if i["severity"] in ("проблема", "риск") and ": " in i["title"]), None)  # «Показатель: оценка»
        if _is_whatif(question) or _is_forecast(question):
            return self._whatif_call(question, data, metric_hits)
        if metric is None:
            return None
        args = {"metric": metric, **_period_args(question)}
        hit = next((h for h in data.get("value_matches", []) if _scoped(question, h["value"])), None)
        if hit:
            args.update(where_dimension=hit["dimension"], where_value=hit["value"])
        return FunctionCall("diagnose_change", args)

    @staticmethod
    def _whatif_call(question: str, data: dict, hits: list) -> FunctionCall | None:
        """«Что если…» -> what_if, «что будет дальше» -> forecast_metric. Меняется показатель, названный после «если»;
        остальные названные показатели просим оценить как следствия. Значение измерения из вопроса — срез изменения."""
        if not hits:
            return None
        head, _, tail = question.lower().partition("если")
        pick = lambda text: [h["metric"] for h in hits if set(stems(h["metric"], stop=False)) & set(stems(text))]  # noqa: E731
        changed = (pick(tail) or [hits[0]["metric"]])[0] if tail else hits[0]["metric"]
        value_hit = next(iter(data.get("value_matches", [])), None)
        if _is_forecast(question):
            args = {"metric": changed}
            if value_hit:
                args.update(where_dimension=value_hit["dimension"], where_value=value_hit["value"])
            return FunctionCall("forecast_metric", args)
        change = _change_args(question)
        if change is None:
            return None
        args = {"metric": changed, **change, **_period_args(question)}
        if value_hit:
            args.update(where_dimension=value_hit["dimension"], where_value=value_hit["value"])
        others = [m for m in pick(head) if m != changed][:3]
        if others:
            args["effect_on"] = others
        return FunctionCall("what_if", args)

    # --- сборка записки ---
    def _strategic_memo(self, question: str, context: dict) -> dict:
        """Ответ на стратегический вопрос по выводам дашборда: только его формулировки и числа, без прогнозов."""
        insights, steps = context["insights"], context["steps"]
        quoted = re.findall(r"«([^»]+)»", question)
        related = [i for i in insights if any(q in i["title"] or q in i["text"] for q in quoted)] or insights[:1]
        if re.search(r"не хватает|каких данных", question, re.I):
            summary = ("По данным дашборда видно, где и насколько изменились показатели, но не почему: причин вне отчёта "
                       "(решения, акции, поведение клиентов, сбои) в данных нет.")
            checks = ["Собрать сведения о решениях, акциях и изменениях условий за период ухудшения.",
                      "Уточнить у владельцев показателей, что менялось в процессах."]
            root = "Причины вне отчёта в данных отсутствуют"
        elif re.search(r"риск", question, re.I):
            summary = ("Если ничего не менять, ухудшение сохранится на достигнутом уровне; прогноз не строился, это допущение. "
                       + " ".join(f"{i['title']}: {i['text']}" for i in related[:2]))
            checks = steps[:2] or ["Проверить причины ухудшения с владельцами показателей."]
            root = f"Риск: {related[0]['title']}"
        else:
            order = [i for i in insights if i["severity"] in ("проблема", "риск")] or insights
            summary = ("Приоритеты по выводам дашборда: " + "; ".join(f"{n}) {i['title']}" for n, i in enumerate(order[:3], 1))
                       + ". " + order[0]["text"])
            checks = steps[:3] or ["Проверить причины ухудшения с владельцами показателей."]
            root = f"Первый приоритет: {order[0]['title']}"
        scenarios = [{"name": f"Шаг {n}", "description": step,
                      "effect_estimate": "Количественная оценка недоступна: эффект зависит от причин вне данных.",
                      "assumptions": ["Шаг следует из выводов дашборда; причину нужно подтвердить владельцем показателя."]}
                     for n, step in enumerate(steps[:3], 1)]
        return {"diagnosis": {"summary": summary, "root_cause": root, "key_segments": []}, "scenarios": scenarios,
                "manual_checks": checks + ["Проверить гипотезы о причинах вне отчёта: это предположения, данных для них нет."],
                "used_tool_calls": []}

    def _memo(self, results: list) -> dict:
        ids = [d["call_id"] for _, d in results if "call_id" in d]
        scenario = next((d for n, d in reversed(results) if n == "what_if" and "error" not in d), None)
        forecast = next((d for n, d in reversed(results) if n == "forecast_metric" and "error" not in d), None)
        if scenario is not None:
            return self._memo_from_whatif(scenario, [scenario["call_id"]])
        if forecast is not None:
            return self._memo_from_forecast(forecast, [forecast["call_id"]])
        failed = next((d for n, d in reversed(results) if n in ("what_if", "forecast_metric") and "error" in d), None)
        if failed is not None:
            return {"diagnosis": {"summary": f"Сценарий не посчитан: {failed['error']}", "root_cause": "Сценарий не посчитан",
                                  "key_segments": []}, "scenarios": [],
                    "manual_checks": ["Уточнить показатель, значение среза или величину изменения и повторить вопрос."],
                    "used_tool_calls": [failed["call_id"]]}
        diag = next((d for n, d in reversed(results) if n == "diagnose_change" and "error" not in d), None)
        struct = next((d for n, d in reversed(results) if n == "get_structure" and "error" not in d), None)
        if diag is None and struct is not None:
            return self._memo_from_structure(struct, [struct["call_id"]])  # числа — из этого вызова, не из неудавшегося diagnose_change
        if diag is None:
            return {
                "diagnosis": {
                    "summary": "Не удалось сопоставить вопрос с показателем из отчёта, поэтому данных для диагноза нет.",
                    "root_cause": "Причина не установлена: данных недостаточно",
                    "key_segments": [],
                },
                "scenarios": [],
                "manual_checks": ["Уточнить название показателя и повторить вопрос."],
                "used_tool_calls": ids,
            }
        return self._memo_from_diagnosis(diag, ids)

    # --- «что если» и прогноз: записка целиком из результата инструмента ---
    @staticmethod
    def _ln(name: str, unit: str, d: dict) -> str:
        pct = "н/д" if d.get("delta_pct") is None else f"{_fmt(d['delta_pct'], True)}%"
        return f"{name}: {_fmt(d['before'])}{_u(unit)} → {_fmt(d['after'])}{_u(unit)} ({pct}, {d['verdict']})"

    @staticmethod
    def _forecast_text(fc: dict, item: dict, with_scenario: bool) -> str:
        u, level = _u(item["unit"]), item["reliability"]["level"]
        if "horizon_total_baseline" in item:  # сумма за горизонт: интервал тоже для суммы, а не для последнего периода
            text = f"{item['metric']}: за {fc['horizon']} {fc['unit_word']} без изменений {_fmt(item['horizon_total_baseline'])}{u}"
            if with_scenario and "horizon_total_scenario" in item:
                text += f", со сценарием {_fmt(item['horizon_total_scenario'])}{u} (разница {_fmt(item['horizon_total_delta'], True)}{u})"
            low, high = item["horizon_total_low"], item["horizon_total_high"]
            tail = f"интервал суммы для базового прогноза {_fmt(low)}–{_fmt(high)}{u} (границы периодов сложены, интервал консервативный)"
        else:
            text = f"{item['metric']}: к {fc['labels'][-1]} без изменений {_fmt(item['baseline'][-1])}{u}"
            if with_scenario and "scenario" in item:
                text += f", со сценарием {_fmt(item['scenario'][-1])}{u}"
            tail = f"интервал {fc['level_pct']}% для базового прогноза {_fmt(item['low'][-1])}–{_fmt(item['high'][-1])}{u}"
        return f"{text}; {tail}, надёжность {level}"

    def _memo_from_whatif(self, w: dict, ids: list[str]) -> dict:
        unit, m = w["unit"], w["metric"]
        c = w["change"]
        scope = "".join(f" ({k} = «{v}»)" for k, v in (c.get("selection") or {}).items())
        step = " п.п." if unit == "%" else _u(unit)
        what = {"pct": f"{_fmt(c['value'], True)}%", "abs": f"{_fmt(c['value'], True)}{step}", "set": f"до {_fmt(c['value'])}{_u(unit)}"}[c["type"]]
        period = w["period"]
        span = f" за {period['from']}–{period['to']}" if period.get("from") else ""
        sel = w["selection_value"]
        sentences = [f"Сценарий: {m}{scope} {what}{span}. Значение в срезе {_fmt(sel['before'])}{_u(unit)} → {_fmt(sel['after'])}{_u(unit)}.",
                     self._ln(f"Итог по отчёту, {m}", unit, w["total"]) + "."]
        effects = w.get("effects", [])
        lines = []
        for e in effects:
            main = e.get("with_estimate") or e
            tag = "оценка по истории" if e["relation"] == "оценка по истории" else "по формуле"
            if e.get("with_estimate"):
                est = e["with_estimate"]
                tag = f"по формуле с учётом реакции «{est['based_on']}» по истории: от {_fmt(est['after_low'])} до {_fmt(est['after_high'])}{_u(e['unit'])}"
            elif e["relation"] == "оценка по истории":
                est = e["estimate"]
                tag = f"оценка по истории, интервал {est['level_pct']}% от {_fmt(e['after_low'])} до {_fmt(e['after_high'])}{_u(e['unit'])}"
            lines.append(f"{self._ln(e['metric'], e['unit'], main)} [{tag}]")
        if lines:
            sentences.append("Связанные показатели: " + "; ".join(lines[:3]) + ".")
        for r_ in w.get("not_estimated", [])[:2]:
            sentences.append(f"«{r_['metric']}» по истории не оценён: {r_['reason']}")
        fc = w.get("forecast") or {}
        fmetrics = fc.get("metrics", [])
        if fmetrics:
            sentences.append("Прогноз: " + self._forecast_text(fc, fmetrics[0], True) + ".")
        elif fc.get("note"):
            sentences.append(f"Прогноз не построен: {fc['note']}")
        scenarios = [{"name": f"Если {m}{scope} {what}", "description": "Арифметика по окну анализа: значение в срезе меняется, итог пересчитывается из тех же сумм.",
                      "effect_estimate": self._ln(f"Итог, {m}", unit, w["total"]) + ".", "assumptions": w["assumptions"][:2]}]
        if lines:
            scenarios.append({"name": "Что будет со связанными показателями", "description": "Пересчёт по формуле и оценка по истории там, где формулы нет.",
                              "effect_estimate": "; ".join(lines[:3]) + ".",
                              "assumptions": [e["caveat"] for e in effects if e.get("caveat")][:1] or w["assumptions"][2:3]})
        if fmetrics:
            scenarios.append({"name": "Прогноз на ближайшие периоды", "description": "Базовый прогноз продолжает прошлую динамику, сценарий добавляет эффект изменения.",
                              "effect_estimate": self._forecast_text(fc, fmetrics[0], True) + ".",
                              "assumptions": [fc["assumption"]] if fc.get("assumption") else ["Прогноз продолжает прошлую динамику и не обещает результат."]})
        checks = ["Проверить допущения сценария с владельцем показателя: изменение может сработать иначе, чем в арифметике.",
                  "Сверить прогноз с планами, акциями и сезонностью, которых в данных нет."]
        if effects and any(e["relation"] == "оценка по истории" for e in effects):
            checks.append("Оценки по истории — корреляция: подтвердить на пилоте или у владельцев показателей.")
        return {"diagnosis": {"summary": " ".join(sentences), "root_cause": "Сценарий «что если»: расчёт, а не диагноз",
                              "key_segments": list((c.get("selection") or {}).values())},
                "scenarios": scenarios[:3], "manual_checks": checks, "used_tool_calls": ids}

    def _memo_from_forecast(self, f: dict, ids: list[str]) -> dict:
        fc, m, unit = f["forecast"], f["metric"], f["unit"]
        scope = "".join(f" ({k} = «{v}»)" for k, v in (f.get("filter") or {}).items())
        items = fc.get("metrics", [])
        if not items:
            note = fc.get("note") or "прогноз не построен"
            return {"diagnosis": {"summary": f"{m}{scope}: прогноз не построен. {note}", "root_cause": "Прогноз не построен: данных недостаточно",
                                  "key_segments": []}, "scenarios": [],
                    "manual_checks": ["Загрузить отчёт с более длинной историей по периодам, чтобы строить прогноз."], "used_tool_calls": ids}
        item = items[0]
        trend = item["trend"]
        text = (f"Прогноз {m}{scope} «если ничего не менять»: {self._forecast_text(fc, item, False)}. Тренд: {trend['direction']}. "
                f"Метод: {item['method']}. Это продолжение прошлой динамики, а не обещание.")
        scenario = {"name": f"Базовый прогноз: {m}", "description": "Продолжение прошлой динамики без изменений в работе.",
                    "effect_estimate": self._forecast_text(fc, item, False) + ".",
                    "assumptions": ["Условия остаются как в прошлых периодах: акции, сезонность и решения не учтены."]}
        return {"diagnosis": {"summary": text, "root_cause": "Прогноз, а не диагноз: причины изменений не искались",
                              "key_segments": list((f.get("filter") or {}).values())},
                "scenarios": [scenario],
                "manual_checks": ["Сверить прогноз с планами, акциями и сезонностью, которых в данных нет.",
                                  "Для оценки результата решений задайте вопрос «что если…» с величиной изменения."],
                "used_tool_calls": ids}

    @staticmethod
    def _memo_from_structure(s: dict, ids: list[str]) -> dict:
        """Ответ там, где сравнивать не с чем: значение показателя и из чего оно состоит, без слов об изменении и причинах."""
        u = _u(s["unit"])
        summary = f"{s['metric']}: {_fmt(s['value'])}{u}" + (f" за {s['period']['from']} – {s['period']['to']}" if s["period"].get("from") else "") + "."
        part = s["structure"][0] if s["structure"] else None
        top = part["values"][0] if part and part["values"] else None
        if top:
            summary += f" Больше всего даёт {_dim(part['dimension'])} «{top['value']}»: {_fmt(top['share_pct'])}%"
            summary += f" ({_fmt(top['fact'])}{u})." if top["fact"] is not None else "."
            if len(part["values"]) > 1 and part["values"][1]["fact"] is not None:
                second = part["values"][1]
                summary += f" Следом — {_dim(part['dimension'])} «{second['value']}»: {_fmt(second['share_pct'])}%."
        summary += " Сравнивать не с чем: нет плана и подходящего прошлого периода, поэтому об изменении и его причинах сказать нельзя."
        return {"diagnosis": {"summary": summary, "root_cause": "Причины изменения не определены: нет базы сравнения",
                              "key_segments": [top["value"]] if top else []},
                "scenarios": [],
                "manual_checks": ["Загрузить отчёт с планом или с прошлыми периодами, чтобы сравнивать значения и искать причины изменений."],
                "used_tool_calls": ids}

    @staticmethod
    def _driver(d: dict, dim_value: str | None):
        """Что объясняет изменение: причина > составляющая формулы > последствие > совпадение по времени —
        только если оно в том же значении измерения."""
        order = [("cause", lambda x: x["relation"].startswith("влияет") and x["consistent_with_link"] is True),
                 ("component", lambda x: re.match(r"(числитель|знаменатель) этого показателя", x["relation"]) is not None),
                 ("consequence", lambda x: (x["relation"].startswith("зависит") and x["consistent_with_link"] is True)
                  or x["relation"].startswith("этот показатель — "))]
        pool = [x for x in d["related_metrics"] if x["significant"] and (dim_value is None or x["top_value"] == dim_value)]
        for kind, test in order:
            found = sorted((x for x in pool if test(x)), key=lambda x: -abs(x["deviation_pct"] or 0))
            if found:
                return kind, found[0]
        return None, None

    def _memo_from_diagnosis(self, d: dict, ids: list[str]) -> dict:
        dev, unit, metric = d["deviation"], d["unit"], d["metric"]
        period, base = d["period"], d["baseline"]
        scope = "".join(f" ({k} = «{v}»)" for k, v in (d.get("filter") or {}).items())
        span = f" за {period['from']}–{period['to']}" if period.get("from") else ""
        head = (f"{metric}{scope}{span}: факт {_fmt(dev['fact'])}{_u(unit)}; база ({base['label']}) {_fmt(dev['base'])}{_u(unit)}; "
                f"изменение {_fmt(dev['deviation_abs'], True)}{_u(unit)} ({_fmt(dev['deviation_pct'], True)}%) — "
                f"{dev['assessment']}.")
        local = d.get("local_deviations") or []
        local_text = ""
        if local:  # итог в норме или лучше базы, но внутри есть заметные ухудшения
            items = "; ".join(f"{_dim(x['dimension'])} «{x['value']}»: {_fmt(x['change_pct'], True)}% к базе "
                              f"(влияние на итог {_fmt(x['effect_on_total_pct'], True)}%)" for x in local)
            local_text = f" Внутри есть заметные ухудшения: {items}."
        if not dev["significant"]:
            checks = ["Убедиться, что база сравнения (план или период) выбрана верно."]
            if local:
                names = ", ".join(f"«{x['value']}»" for x in local)
                checks.insert(0, f"Проверить, что происходит со значениями {names}: условия, качество данных, сбои.")
                checks.append("Решить, нужно ли разбирать эти значения отдельно: итог их ухудшение пока не показывает.")
            return {
                "diagnosis": {"summary": f"{head} Значимого изменения нет, искать причины итога не нужно.{local_text}",
                              "root_cause": "Итог в норме, но есть локальные ухудшения" if local else "Значимого изменения нет",
                              "key_segments": list(dict.fromkeys(x["value"] for x in local))},
                "scenarios": [],
                "manual_checks": checks,
                "used_tool_calls": ids,
            }

        sentences = [head + local_text]
        key_values: list[str] = []
        top_value = dim_name = None
        primary = next((x for x in d["dimensions"] if x["dimension"] == d["primary_dimension"]), None)
        if primary:
            dim_name, top = _dim(primary["dimension"]), primary["top"][0]
            top_value = top["value"]
            sentences.append(f"Основной вклад — {dim_name} «{top_value}»: {_fmt(top['contribution_pct'])}% изменения, "
                             f"собственное изменение {_fmt(top['change_pct'], True)}%.")
            key_values = [top_value] + [x["value"] for x in primary["top"][1:2]
                                        if (x["contribution_pct"] or 0) >= 10 and not x["value"].startswith("прочие")]
            if d["drill_down"]:
                inner = d["drill_down"][0]
                first = inner["top"][0]
                sentences.append(f"Внутри «{top_value}» сильнее всего {_dim(inner['dimension'])} «{first['value']}» "
                                 f"({_fmt(first['contribution_pct'])}% изменения внутри).")
        if d.get("unfavorable_since"):
            sentences.append(f"Неблагоприятное изменение держится {d['unfavorable_periods_in_row']} {d['interval']} "
                             f"подряд, с {d['unfavorable_since']}.")

        kind, driver = self._driver(d, top_value)
        root = f"Изменение сосредоточено: {dim_name} «{top_value}»" if top_value else "Причина в данных не найдена"
        if driver:
            where = (f"; сильнее всего — {dim_name} «{driver['top_value']}» ({_fmt(driver['top_value_share_pct'])}% изменения)"
                     if driver["top_value"] else "")
            lead, verb = {"cause": ("Связанный показатель", "изменился"), "component": ("Составляющая формулы", "изменилась"),
                          "consequence": ("Следствие — показатель", "изменился")}[kind]
            sentences.append(f"{lead} «{driver['metric']}» ({driver['relation']}) {verb} на "
                             f"{_fmt(driver['deviation_pct'], True)}%{where}.")
            if kind != "consequence":  # следствие причиной не называем
                root = (f"{driver['metric']}: изменение сосредоточено — {dim_name} «{driver['top_value']}»"
                        if driver["top_value"] else f"{driver['metric']}: изменение связанного показателя")

        scenarios = []
        for s in d["recovery_scenarios"]:
            names = " и ".join(f"«{x}»" for x in s["values"])
            title = (f"Вернуть к базе: {names}" if s["close_share_pct"] == 100 else f"Вернуть половину изменения: {names}")
            scenarios.append({
                "name": title,
                "description": "Адресные меры по этому значению до возврата к уровню базы; конкретный набор мер "
                               "определяет владелец показателя.",
                "effect_estimate": f"Эффект {_fmt(s['effect_abs'])}{_u(s['unit'])}: закрывает {_fmt(s['gap_closed_pct'])}% "
                                   f"изменения, остаток {_fmt(s['remaining_gap_abs'])}{_u(s['unit'])}.",
                "assumptions": ["Эффект — арифметика возврата значения к базе, а не прогноз результата мер.",
                                "Причина изменения подтверждена владельцем показателя."],
            })
        target = f"«{top_value}» ({dim_name})" if top_value else f"показателем «{metric}»"
        checks = [f"Проверить, что изменилось с {target} с {d.get('unfavorable_since') or period.get('from', 'начала периода')}: "
                  "условия, тарифы, каналы, поставщики, качество данных.",
                  f"Сверить данные по {target} с первичным источником."]
        if d.get("owner"):
            checks.insert(0, f"Обсудить с владельцем цели ({d['owner']}), что изменилось в {target}.")
        if kind not in ("cause", "component"):
            checks.append("Связанные показатели не объясняют изменение: причину стоит искать вне отчёта.")
        return {
            "diagnosis": {"summary": " ".join(sentences),
                          "root_cause": "Изменение в лучшую сторону" if dev["assessment"].startswith("лучше") else root,
                          "key_segments": key_values},
            "scenarios": scenarios,
            "manual_checks": checks,
            "used_tool_calls": ids,
        }

    @staticmethod
    def _usage(messages, output: str) -> Usage:
        prompt = sum(len(str(m.get("content", ""))) for m in messages) // 4
        completion = len(output) // 4
        return Usage(prompt, completion, prompt + completion)
