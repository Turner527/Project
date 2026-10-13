"""Оценка одного ответа агента по эталону и сводка по прогону. Чистые функции, без ввода-вывода.

Три метрики прогона:
  cause_accuracy    — верно ли названа причина (нужный сегмент, а где требуется — и KPI-драйвер);
  numbers_accuracy  — совпадают ли числа ответа с эталонными числами витрины (с допуском на округление записи);
  unverified_share  — доля ответов, где верификатор нашёл числа, которых нет в результатах инструментов.
"""
import math
from collections import defaultdict

from agent.verifier import STATUS_OK, extract_numbers, numbers_match
from tools.core import _stems  # та же нормализация названий, что при поиске KPI


def mentioned(name: str, text: str) -> bool:
    """Название KPI упомянуто в тексте: найдено не меньше 2/3 его основ (допускает «одобрение заявок»)."""
    needed = math.ceil(len(_stems(name)) * 2 / 3)
    return len(_stems(name) & _stems(text)) >= needed


def score_cause(memo, expect: dict) -> bool:
    if memo is None:
        return False
    segment = expect["key_segment"]
    if segment is None:  # контрольный KPI: правильный ответ — «проблем нет», сегменты не называются
        return not memo.diagnosis.key_segments
    named = {s.casefold() for s in memo.diagnosis.key_segments}
    segment_ok = segment.casefold() in named or segment.casefold() in memo.diagnosis.root_cause.casefold()
    drivers = expect["driver_kpis_any"]
    cause_text = f"{memo.diagnosis.root_cause} {memo.diagnosis.summary}"
    return segment_ok and (not drivers or any(mentioned(d, cause_text) for d in drivers))


def score_numbers(memo, expect: dict) -> tuple[bool | None, list[str]]:
    """(все ли эталонные числа найдены в ответе, имена ненайденных). None — если эталонных чисел нет."""
    expected = expect["numbers"]
    if not expected:
        return None, []
    if memo is None:
        return False, [e["name"] for e in expected]
    written = [n for text in memo.texts().values() for n in extract_numbers(text, skip_trivial=False)]
    def found(e: dict) -> bool:
        candidates = e["any_of"] if "any_of" in e else [e["value"]]
        return any(numbers_match(n, v) for n in written for v in candidates)

    missing = [e["name"] for e in expected if not found(e)]
    return not missing, missing


def evaluate(result, question: dict) -> dict:
    memo = result.memo
    expect = question["expect"]
    numbers_ok, missing = score_numbers(memo, expect)
    verification = result.verification
    return {
        "id": question["id"],
        "anomaly": question["anomaly"],
        "type": question["type"],
        "question": question["question"],
        "cause_ok": score_cause(memo, expect),
        "numbers_ok": numbers_ok,
        "missing_numbers": missing,
        "status": verification.status if verification else "ошибка",
        "unverified": [u.value for u in verification.unverified] if verification else [],
        "error": result.error,
        "tools": [c.name for c in result.tool_calls],
        "latency_ms": result.latency_ms,
        "tokens": result.usage.total_tokens,
        "summary": (memo.diagnosis.summary[:300] if memo else (result.raw_text or "")[:300]),
    }


def _share(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 3) if flags else None


def aggregate(items: list[dict]) -> dict:
    answered = [i for i in items if i["status"] != "ошибка"]
    with_numbers = [i for i in items if i["numbers_ok"] is not None]
    by_group: dict[str, list[dict]] = defaultdict(list)
    for i in items:
        by_group[i["anomaly"]].append(i)
    return {
        "n": len(items),
        "errors": len(items) - len(answered),
        "cause_accuracy": _share([i["cause_ok"] for i in items]),
        "numbers_accuracy": _share([i["numbers_ok"] for i in with_numbers]),
        "unverified_share": _share([i["status"] != STATUS_OK for i in answered]),
        "avg_latency_ms": round(sum(i["latency_ms"] for i in items) / len(items)) if items else None,
        "avg_tokens": round(sum(i["tokens"] for i in items) / len(items)) if items else None,
        "by_group": {
            g: {
                "n": len(rows),
                "cause_accuracy": _share([r["cause_ok"] for r in rows]),
                "numbers_accuracy": _share([r["numbers_ok"] for r in rows if r["numbers_ok"] is not None]),
            }
            for g, rows in sorted(by_group.items())
        },
    }
