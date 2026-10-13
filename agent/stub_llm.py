"""Детерминированная заглушка вместо GigaChat: для тестов, evals без сети и демо без ключа.

Это не «умная» модель, а сценарий: find_metrics -> diagnose_change -> записка из его результата.
Числа берутся дословно из результатов инструментов (ничего не считается), поэтому записка проходит
верификатор. Качество настоящей модели заглушкой не проверяется — только конвейер вокруг неё.
"""
import json
import re

from agent.llm import TEMPERATURE, FunctionCall, LLMResponse, Usage

_WORD_NUMBERS = {"два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8,
                 "девять": 9, "десять": 10, "двенадцать": 12}
_NUM = r"(\d+|" + "|".join(_WORD_NUMBERS) + r")"
_MINUS = "−"


def _fmt(x, signed: bool = False) -> str:
    """Число в русской записи с теми же цифрами, что в результате инструмента (знак — для наглядности)."""
    if x is None:
        return "н/д"
    text = str(abs(x)).replace(".", ",")
    if x < 0:
        return f"{_MINUS}{text}"
    return f"+{text}" if signed and x > 0 else text


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


def _dim(name: str) -> str:
    """Название измерения в тексте: «Канал» -> «канал», но «CRM» и «ИП» остаются как есть."""
    return name.lower() if name[1:].islower() else name


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
        results = [(m["name"], json.loads(m["content"])) for m in messages if m["role"] == "function"]
        if function_call == "none":
            content = json.dumps(self._memo(results), ensure_ascii=False)
            return LLMResponse(content=content, usage=self._usage(messages, content), finish_reason="stop")
        call = self._next_call(question, results)
        usage = self._usage(messages, json.dumps(call.__dict__, ensure_ascii=False) if call else "")
        if call is None:
            return LLMResponse(content="Данных достаточно.", usage=usage, finish_reason="stop")
        return LLMResponse(function_call=call, usage=usage, finish_reason="function_call")

    # --- выбор вызова ---
    def _next_call(self, question: str, results: list) -> FunctionCall | None:
        if not results:
            return FunctionCall("find_metrics", {"query": question})
        name, data = results[-1]
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
        if metric is None:
            return None
        args = {"metric": metric, **_period_args(question)}
        hit = next((h for h in data.get("value_matches", []) if _scoped(question, h["value"])), None)
        if hit:
            args.update(where_dimension=hit["dimension"], where_value=hit["value"])
        return FunctionCall("diagnose_change", args)

    # --- сборка записки ---
    def _memo(self, results: list) -> dict:
        ids = [d["call_id"] for _, d in results if "call_id" in d]
        diag = next((d for n, d in reversed(results) if n == "diagnose_change" and "error" not in d), None)
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
        head = (f"{metric}{scope}{span}: факт {_fmt(dev['fact'])} {unit}; база ({base['label']}) {_fmt(dev['base'])} {unit}; "
                f"изменение {_fmt(dev['deviation_abs'], True)} {unit} ({_fmt(dev['deviation_pct'], True)}%) — "
                f"{dev['assessment']}.")
        if not dev["significant"]:
            return {
                "diagnosis": {"summary": f"{head} Значимого изменения нет, искать причины не нужно.",
                              "root_cause": "Значимого изменения нет", "key_segments": []},
                "scenarios": [],
                "manual_checks": ["Убедиться, что база сравнения (план или период) выбрана верно."],
                "used_tool_calls": ids,
            }

        sentences = [head]
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
                "effect_estimate": f"Эффект {_fmt(s['effect_abs'])} {s['unit']}: закрывает {_fmt(s['gap_closed_pct'])}% "
                                   f"изменения, остаток {_fmt(s['remaining_gap_abs'])} {s['unit']}.",
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
