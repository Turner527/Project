"""Детерминированная заглушка вместо GigaChat: для тестов, evals без сети и демо без ключа.

Это не «умная» модель, а сценарий: найти KPI -> diagnose_deviation -> собрать записку из его результата.
Числа берутся дословно из результатов инструментов (ничего не считается), поэтому записка проходит
верификатор. Качество настоящей модели заглушкой не проверяется — только конвейер вокруг неё.
"""
import json
import re

from agent.llm import TEMPERATURE, FunctionCall, LLMResponse, Usage

_WORD_NUMBERS = {"два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8,
                 "девять": 9, "десять": 10, "двенадцать": 12}
_DASH = "−"


def _fmt(x) -> str:
    """Число в русской записи с дословно теми же знаками, что и в результате инструмента."""
    if x is None:
        return "н/д"
    text = str(abs(x)).replace(".", ",")
    return f"{_DASH}{text}" if x < 0 else text


def _last_months(question: str) -> int | None:
    q = question.lower()
    if "квартал" in q:
        return 3
    if "полгода" in q:
        return 6
    m = re.search(r"последн\w+\s+(\d+|" + "|".join(_WORD_NUMBERS) + r")\s+месяц", q)
    if m:
        return int(m.group(1)) if m.group(1).isdigit() else _WORD_NUMBERS[m.group(1)]
    m = re.search(r"за\s+(\d+|" + "|".join(_WORD_NUMBERS) + r")\s+месяц", q)
    if m:
        return int(m.group(1)) if m.group(1).isdigit() else _WORD_NUMBERS[m.group(1)]
    return None


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
            return FunctionCall("find_kpis_by_goal", {"query": question})
        name, data = results[-1]
        if name != "find_kpis_by_goal":
            return None
        kpi = None
        kpi_hits, goal_hits = data.get("kpi_matches", []), data.get("goal_matches", [])
        if kpi_hits and (not goal_hits or kpi_hits[0]["match_score"] >= 0.9):
            kpi = kpi_hits[0]["kpi"]
        elif goal_hits:
            kpi = goal_hits[0]["kpis"][0]["kpi"]  # главный KPI цели — первый в списке
        if kpi is None:
            return None
        args = {"kpi": kpi}
        if (n := _last_months(question)) is not None:
            args["last_months"] = n
        return FunctionCall("diagnose_deviation", args)

    # --- сборка записки ---
    def _memo(self, results: list) -> dict:
        all_ids = [d["call_id"] for _, d in results if "call_id" in d]
        diag = next((d for n, d in reversed(results) if n == "diagnose_deviation" and "error" not in d), None)
        if diag is None:
            return {
                "diagnosis": {
                    "summary": "Не удалось сопоставить вопрос с целью или KPI из витрины, поэтому данных для "
                    "диагноза нет.",
                    "root_cause": "Причина не установлена: данных недостаточно",
                    "key_segments": [],
                },
                "scenarios": [],
                "manual_checks": ["Уточнить название цели или KPI и повторить вопрос."],
                "used_tool_calls": all_ids,
            }
        return self._memo_from_diagnosis(diag, all_ids)

    def _memo_from_diagnosis(self, d: dict, ids: list[str]) -> dict:
        dev, unit, kpi = d["deviation"], d["unit"], d["kpi"]
        period = d["period"]
        head = (
            f"{kpi} за {period['from']}–{period['to']}: факт {_fmt(dev['fact'])} {unit} при плане "
            f"{_fmt(dev['plan'])} {unit}, отклонение {_fmt(dev['deviation_abs'])} {unit} "
            f"({_fmt(dev['deviation_pct'])}%) — {dev['assessment']}."
        )
        if not dev["significant"]:
            return {
                "diagnosis": {
                    "summary": f"{head} Значимого отклонения нет, искать причины не нужно.",
                    "root_cause": "Значимого отклонения нет",
                    "key_segments": [],
                },
                "scenarios": [],
                "manual_checks": ["Убедиться, что плановое значение показателя актуально."],
                "used_tool_calls": ids,
            }

        top = d["breakdown"]["segments"][0]
        seg = top["segment"]
        extra_segs = [s["segment"] for s in d["breakdown"]["segments"][1:2] if (s["contribution_pct"] or 0) >= 10]
        sentences = [
            head,
            f"Основной вклад даёт сегмент «{seg}»: {_fmt(top['contribution_pct'])}% отклонения, отклонение "
            f"самого сегмента {_fmt(top['deviation_pct'])}%.",
        ]
        if d.get("off_plan_since"):
            sentences.append(f"Отклонение держится {d['months_off_plan_in_row']} мес. подряд, с {d['off_plan_since']}.")

        confirming = [r for r in d["related_kpis"] if r["consistent_with_link"] is True]
        confirming.sort(key=lambda r: (r["most_deviating_segment"] != seg, -abs(r["deviation_pct"] or 0)))
        driver = confirming[0] if confirming else None
        if driver:
            tail = ""
            if driver["most_deviating_segment"]:
                tail = (f"; сильнее всего в сегменте «{driver['most_deviating_segment']}» "
                        f"({_fmt(driver['most_deviating_segment_pct'])}%)")
            sentences.append(
                f"Связанный показатель «{driver['kpi']}» ({driver['link_direction']} связан с «{kpi}») "
                f"отклоняется на {_fmt(driver['deviation_pct'])}%{tail}."
            )
            root = f"{driver['kpi']}: отклонение в сегменте «{driver['most_deviating_segment'] or seg}»"
        else:
            root = f"Отклонение сосредоточено в сегменте «{seg}»"

        favorable = not dev["unfavorable"]
        scenarios = []
        for s in d["recovery_scenarios"]:
            names = " и ".join(f"«{x}»" for x in s["segments"])
            title = (f"Вернуть к плану сегмент{'ы' if len(s['segments']) > 1 else ''} {names}"
                     if s["close_share_pct"] == 100
                     else f"Вернуть половину отставания в сегменте {names}")
            scenarios.append(
                {
                    "name": title,
                    "description": "Адресные меры в сегменте до возврата к плану; конкретный набор мер "
                    "определяет владелец цели.",
                    "effect_estimate": f"Эффект {_fmt(s['effect_abs'])} {s['unit']}: закрывает "
                    f"{_fmt(s['gap_closed_pct'])}% отклонения, остаток {_fmt(s['remaining_gap_abs'])} {s['unit']}.",
                    "assumptions": [
                        "Эффект — арифметика возврата сегмента к плану, а не прогноз результата мер.",
                        "Причина отклонения в сегменте подтверждена владельцем цели.",
                    ],
                }
            )
        checks = [
            f"Проверить вместе с владельцем цели ({d['owner']}), что изменилось в сегменте «{seg}»: условия "
            "продуктов, скоринг, тарифы, каналы продаж.",
            f"Сверить данные по сегменту «{seg}» с первичным источником.",
        ]
        if not driver:
            checks.append("Связанные KPI не подтверждают причину: искать её вне витрины.")
        return {
            "diagnosis": {
                "summary": " ".join(sentences),
                "root_cause": "Отклонение в лучшую сторону" if favorable else root,
                "key_segments": [seg] + extra_segs,
            },
            "scenarios": scenarios,
            "manual_checks": checks,
            "used_tool_calls": ids,
        }

    @staticmethod
    def _usage(messages, output: str) -> Usage:
        prompt = sum(len(str(m.get("content", ""))) for m in messages) // 4
        completion = len(output) // 4
        return Usage(prompt, completion, prompt + completion)
