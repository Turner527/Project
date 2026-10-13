"""Строит evals/questions.json из data/anomalies.json: вопросы разной формулировки с эталонами.

Запуск:  python -m evals.build_questions
Эталон вопроса: сегмент и KPI-драйверы (для проверки причины) и числа из эталонного расчёта генератора
(для проверки чисел). Для контрольных KPI без аномалии эталон — «значимого отклонения нет».
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANOMALIES = ROOT / "data" / "anomalies.json"
OUT = Path(__file__).parent / "questions.json"


def _months(checkpoint: dict) -> int:
    (y1, m1), (y2, m2) = (map(int, checkpoint[k].split("-")) for k in ("from", "to"))
    return (y2 - y1) * 12 + (m2 - m1) + 1


def _num(name: str, value: float | list[float]) -> dict:
    """Одно эталонное число или список допустимых (достаточно любого из них)."""
    return {"name": name, "any_of": value} if isinstance(value, list) else {"name": name, "value": value}


# тип: (шаблон вопроса, окно эталона, нужен ли KPI-драйвер в причине, числа из эталона)
TEMPLATES = {
    "why_goal": ("Почему мы отстаём по цели «{goal}» и что делать?", "last_6m", True,
                 ["deviation_pct", "key_segment_contribution_pct"]),
    "explain_kpi": ("Объясни отклонение показателя «{kpi}» от плана за последние {n} месяцев.", "full", True,
                    ["deviation_pct", "key_segment_contribution_pct"]),
    "which_segment": ("Какой сегмент сильнее всего влияет на отклонение показателя «{kpi}» и каков его вклад?",
                      "last_6m", False, ["key_segment_contribution_pct"]),
    "related": ("Какие связанные показатели объясняют ситуацию с показателем «{kpi}»?", "last_6m", True,
                ["driver_deviation_pct"]),
    "board_memo": ("Подготовь записку для правления: что случилось с показателем «{kpi}» и какие есть варианты "
                   "действий?", "last_6m", True, ["deviation_pct", "s1_effect_abs"]),
    "recovery": ("На сколько улучшится показатель «{kpi}», если вернуть сегмент «{segment}» к плану?", "last_6m",
                 False, ["s1_effect_abs"]),
    "last_3m": ("Что происходит с показателем «{kpi}» за последние три месяца и в чём причина?", "last_3m", True,
                ["deviation_pct", "key_segment_contribution_pct"]),
    "colloquial": ("Шеф спрашивает, что не так с показателем «{kpi}». Разберись и предложи меры.", "last_6m", True,
                   ["deviation_pct"]),
}
CONTROL_QUESTION = "Есть ли проблемы с показателем «{kpi}»? Нужно ли что-то менять?"


def build(anomalies: dict) -> list[dict]:
    questions = []
    for a in anomalies["anomalies"]:
        for qtype, (template, window, need_driver, number_names) in TEMPLATES.items():
            cp = a["checkpoints"][window]
            expected = dict(cp["expected"])
            expected["driver_deviation_pct"] = [cp["drivers"][d]["deviation_pct"] for d in a["driver_kpis_any"]]
            questions.append(
                {
                    "id": f"{a['id']}-{qtype}",
                    "anomaly": a["id"],
                    "type": qtype,
                    "question": template.format(
                        goal=a["goal"], kpi=a["primary_kpi"], segment=a["key_segment"], n=_months(cp)
                    ),
                    "expect": {
                        "key_segment": a["key_segment"],
                        "driver_kpis_any": a["driver_kpis_any"] if need_driver else [],
                        "cause_label": a["cause_label"],
                        "numbers": [_num(n, expected[n]) for n in number_names],
                        "window": {"from": cp["from"], "to": cp["to"]},
                    },
                }
            )
    for c in anomalies["controls"]:
        cp = c["checkpoints"]["last_6m"]
        questions.append(
            {
                "id": f"{c['id']}-control",
                "anomaly": c["id"],
                "type": "control",
                "question": CONTROL_QUESTION.format(kpi=c["kpi"]),
                "expect": {
                    "key_segment": None,
                    "driver_kpis_any": [],
                    "cause_label": "Значимого отклонения нет",
                    "numbers": [_num("deviation_pct", cp["expected"]["deviation_pct"])],
                    "window": {"from": cp["from"], "to": cp["to"]},
                },
            }
        )
    return questions


def main() -> None:
    anomalies = json.loads(ANOMALIES.read_text("utf-8"))
    questions = build(anomalies)
    payload = {"source": "data/anomalies.json", "count": len(questions), "questions": questions}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"Записано {len(questions)} вопросов в {OUT}")


if __name__ == "__main__":
    main()
