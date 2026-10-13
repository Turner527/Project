"""Строит evals/questions_flat.json: вопросы по плоским отчётам (data/samples) с эталонами, посчитанными на pandas.

Запуск:  python -m evals.build_flat_questions

Эталон считается здесь, независимо от инструментов проекта: сырые таблицы берутся из генератора образцов
(data/generate_samples.py), суммы и отношения сумм считает pandas, вклад значения измерения в отношение — по формуле
«ставка и структура» (README, §13). В образцы заложены аномалии (SAMPLE_SIGNALS), поэтому для каждого вопроса известен
и нужный разрез, и нужное значение измерения.
"""
import json
from pathlib import Path

import pandas as pd

from data import generate_samples as gs

OUT = Path(__file__).parent / "questions_flat.json"
DAILY_WINDOW = 30  # окно по умолчанию для дневных данных: последние 30 дней против предыдущих 30
MONTHLY_WINDOW = 4  # в «branches» девять закрытых месяцев: окно по умолчанию — четыре, база — четыре до них


# показатель эталонного вызова для каждого вопроса: по нему тесты сверяют инструменты с эталоном pandas
REFERENCE_METRIC = {
    "M1-why_spend": "Расходы", "M1-conversion": "Конверсия в лиды", "M1-cost_lead": "Стоимость лида", "M1-roas": "ROAS",
    "M1-two_weeks": "Конверсия в лиды", "M1-board_memo": "Выручка", "M2-control_impressions": "Показы",
    "M2-control_clicks": "Клики", "S1-sla": "Выполнение SLA", "S1-csat": "CSAT",
    "S1-handle_time": "Среднее время обработки", "S1-two_weeks": "Среднее время обработки", "S1-colloquial": "CSAT",
    "S2-control_new": "Новые обращения", "B1-conversion": "Конверсия сделок", "B1-hidden_problem": "Конверсия сделок",
    "B1-expenses": "Расходы",
    "B2-control_income": "Доход",
}


def _r(x: float, nd: int = 2) -> float:
    """Округление как в ответах инструментов: два знака, а для малых величин четыре."""
    return round(float(x), 4 if nd == 2 and 0 < abs(x) < 0.1 else nd)


def _frames():
    marketing, support, branches = gs.marketing(), gs.support(), gs.branches()
    marketing["Дата"], support["Дата"] = pd.to_datetime(marketing["Дата"]), pd.to_datetime(support["Дата"])
    branches["Период"] = pd.to_datetime(branches["Период"])
    return marketing, support, branches[branches["Период закрыт"] == "Да"].copy()  # незакрытый месяц не анализируется


def _window(df: pd.DataFrame, date: str, length: int, unit: str):
    """(факт, база): последние length дней или месяцев и столько же перед ними."""
    end = df[date].max()
    if unit == "D":
        step = pd.Timedelta(days=length)
        f0, f1 = end - step + pd.Timedelta(days=1), end
        return (f0, f1), (f0 - step, f1 - step)
    period = end.to_period("M")
    return ((period - length + 1).start_time, end), ((period - 2 * length + 1).start_time, (period - length).start_time)


def _slice(df, date, win):
    return df[(df[date] >= win[0]) & (df[date] <= win[1])]


def reference(df: pd.DataFrame, date: str, dim: str, metric: tuple, fact_win, base_win, plan: str | None = None) -> dict:
    """Изменение показателя, главное значение измерения и его вклад.

    metric: ("sum", столбец) или ("ratio", числитель, знаменатель, множитель). plan — столбец плана вместо прошлого периода.
    """
    F = _slice(df, date, fact_win)
    B = F if plan else _slice(df, date, base_win)
    if metric[0] == "sum":
        col = metric[1]
        base_col = plan or col
        v1, v0 = F[col].sum(), B[base_col].sum()
        g1, g0 = F.groupby(dim)[col].sum(), B.groupby(dim)[base_col].sum()
        idx = g1.index.union(g0.index)
        contrib = g1.reindex(idx, fill_value=0) - g0.reindex(idx, fill_value=0)
    else:
        _, num, den, scale = metric
        n1, d1, n0, d0 = F[num].sum(), F[den].sum(), B[num].sum(), B[den].sum()
        v1, v0 = scale * n1 / d1, scale * n0 / d0
        r0 = n0 / d0
        g1, g0 = F.groupby(dim)[[num, den]].sum(), B.groupby(dim)[[num, den]].sum()
        idx = g1.index.union(g0.index)
        g1, g0 = g1.reindex(idx, fill_value=0), g0.reindex(idx, fill_value=0)
        contrib = scale * (g1[num] / d1 - g0[num] / d0 - r0 * (g1[den] / d1 - g0[den] / d0))
    diff = v1 - v0
    top = (contrib * (1 if diff > 0 else -1)).idxmax()
    return {"deviation_pct": _r(diff / abs(v0) * 100, 1), "top_value": top,
            "key_segment_contribution_pct": _r(contrib[top] / diff * 100, 1), "s1_effect_abs": _r(abs(contrib[top])),
            "fact": v1, "base": v0}


def own_change(df, date: str, dim: str, value: str, metric: tuple, fact_win, base_win) -> float:
    """Собственное изменение ставки одного значения измерения, % к базе (для «скрытых» проблем при ровном итоге)."""
    part = df[df[dim] == value]
    F, B = _slice(part, date, fact_win), _slice(part, date, base_win)
    _, num, den, scale = metric
    v1, v0 = scale * F[num].sum() / F[den].sum(), scale * B[num].sum() / B[den].sum()
    return _r((v1 / v0 - 1) * 100, 1)


def _question(qid, group, qtype, text, expect_numbers, key_segment, drivers=(), cause="", window=None, dataset="",
              control=False):
    """window — окно эталонного расчёта (вместе с REFERENCE_METRIC это вызов diagnose_change для сверки инструментов)."""
    return {
        "id": qid, "anomaly": group, "type": qtype, "dataset": dataset, "question": text,
        "expect": {"key_segment": None if control else key_segment, "driver_kpis_any": list(drivers),
                   "cause_label": "Значимого отклонения нет" if control else cause,
                   "numbers": expect_numbers, "metric": "", "window": window},
    }


def _numbers(ref: dict, names: tuple) -> list[dict]:
    return [{"name": n, "value": ref[n]} for n in names]


def _win_text(fact_win) -> dict:
    return {"from": str(fact_win[0].date()), "to": str(fact_win[1].date())}


def build() -> list[dict]:
    marketing, support, branches = _frames()
    sig = gs.SAMPLE_SIGNALS
    questions: list[dict] = []
    DEV, SHARE, EFFECT = ("deviation_pct",), ("deviation_pct", "key_segment_contribution_pct"), ("deviation_pct", "s1_effect_abs")

    # ---------- маркетинг: с середины сентября в канале «Поиск» растут расходы и падает конверсия ----------
    channel = sig["marketing"]["channel"]
    f30, b30 = _window(marketing, "Дата", DAILY_WINDOW, "D")
    f14, b14 = _window(marketing, "Дата", 14, "D")
    ref = lambda metric, f=f30, b=b30: reference(marketing, "Дата", "Канал", metric, f, b)  # noqa: E731
    spend, conv = ref(("sum", "Расходы, руб")), ref(("ratio", "Лиды", "Клики", 100))
    cpl, roas = ref(("ratio", "Расходы, руб", "Лиды", 1)), ref(("ratio", "Выручка, руб", "Расходы, руб", 1))
    revenue, shows, clicks = ref(("sum", "Выручка, руб")), ref(("sum", "Показы")), ref(("sum", "Клики"))
    conv14 = ref(("ratio", "Лиды", "Клики", 100), f14, b14)
    mk = dict(dataset="marketing", key_segment=channel)
    questions += [
        _question("M1-why_spend", "marketing", "why_spend", "Почему выросли расходы на рекламу и в каком канале?",
                  _numbers(spend, SHARE), cause="Рост расходов в канале «Поиск»", window=_win_text(f30), **mk),
        _question("M1-conversion", "marketing", "conversion", "Почему упала конверсия из кликов в лиды?",
                  _numbers(conv, SHARE), drivers=["Лиды"], cause="Падение лидов в канале «Поиск»", window=_win_text(f30), **mk),
        _question("M1-cost_lead", "marketing", "cost_lead", "Почему выросла стоимость лида и где именно?",
                  _numbers(cpl, SHARE), drivers=["Расходы", "Лиды"], cause="Рост расходов при падении лидов в «Поиске»",
                  window=_win_text(f30), **mk),
        _question("M1-roas", "marketing", "roas", "Что случилось с ROAS, окупаемостью рекламы?", _numbers(roas, SHARE),
                  drivers=["Выручка", "Расходы"], cause="Расходы выросли, выручка упала в «Поиске»", window=_win_text(f30), **mk),
        _question("M1-two_weeks", "marketing", "two_weeks", "Что происходит с конверсией в лиды за последние две недели?",
                  _numbers(conv14, SHARE), drivers=["Лиды"], cause="Падение лидов в канале «Поиск»", window=_win_text(f14), **mk),
        _question("M1-board_memo", "marketing", "board_memo", "Подготовь записку: что случилось с выручкой и какие есть варианты "
                  "действий?", _numbers(revenue, EFFECT), cause="Падение выручки в канале «Поиск»", window=_win_text(f30), **mk),
        _question("M2-control_impressions", "marketing", "control", "Есть ли проблемы с показами? Нужно ли что-то менять?",
                  _numbers(shows, DEV), None, cause="", window=_win_text(f30), dataset="marketing", control=True),
        _question("M2-control_clicks", "marketing", "control", "Всё ли в порядке с кликами? Нужно ли что-то менять?",
                  _numbers(clicks, DEV), None, cause="", window=_win_text(f30), dataset="marketing", control=True),
    ]

    # ---------- поддержка: с 10 сентября в очереди «Техническая поддержка» падают SLA и CSAT, растёт время ----------
    queue = sig["support"]["queue"]
    sf30, sb30 = _window(support, "Дата", DAILY_WINDOW, "D")
    sf14, sb14 = _window(support, "Дата", 14, "D")
    sref = lambda metric, f=sf30, b=sb30: reference(support, "Дата", "Очередь", metric, f, b)  # noqa: E731
    sla, csat = sref(("ratio", "SLA выполнено", "Решено", 100)), sref(("ratio", "Положительные оценки", "Всего оценок", 100))
    handle = sref(("ratio", "Минуты обработки", "Решено", 1))
    handle14 = sref(("ratio", "Минуты обработки", "Решено", 1), sf14, sb14)
    new_tickets = sref(("sum", "Новые обращения"))
    sp = dict(dataset="support", key_segment=queue)
    questions += [
        _question("S1-sla", "support", "sla", "Почему упало выполнение SLA и в какой очереди?", _numbers(sla, SHARE),
                  drivers=["SLA выполнено"], cause="Падение SLA в очереди «Техническая поддержка»", window=_win_text(sf30), **sp),
        _question("S1-csat", "support", "csat", "Что случилось с CSAT и почему?", _numbers(csat, SHARE),
                  drivers=["Положительные оценки"], cause="Падение оценок в очереди «Техническая поддержка»",
                  window=_win_text(sf30), **sp),
        _question("S1-handle_time", "support", "handle_time", "Почему выросло среднее время обработки обращений?",
                  _numbers(handle, SHARE), drivers=["Минуты обработки", "Решено"],
                  cause="Рост времени обработки в очереди «Техническая поддержка»", window=_win_text(sf30), **sp),
        _question("S1-two_weeks", "support", "two_weeks", "Что со средним временем обработки за последние 14 дней?",
                  _numbers(handle14, SHARE), drivers=["Минуты обработки", "Решено"],
                  cause="Рост времени обработки в очереди «Техническая поддержка»", window=_win_text(sf14), **sp),
        _question("S1-colloquial", "support", "colloquial", "Шеф спрашивает, что не так с CSAT. Разберись и предложи меры.",
                  _numbers(csat, DEV), cause="Падение оценок в очереди «Техническая поддержка»", window=_win_text(sf30), **sp),
        _question("S2-control_new", "support", "control", "Есть ли проблемы с количеством новых обращений?",
                  _numbers(new_tickets, DEV), None, window=_win_text(sf30), dataset="support", control=True),
    ]

    # ---------- сеть отделений: конверсия падает в двух отделениях, расходы растут в Новосибирске; октябрь не закрыт ----------
    bf4, bb4 = _window(branches, "Период", MONTHLY_WINDOW, "M")
    bf2, bb2 = _window(branches, "Период", 2, "M")
    bref = lambda metric, f=bf4, b=bb4, plan=None: reference(branches, "Период", "Отделение", metric, f, b, plan)  # noqa: E731
    deals2 = bref(("ratio", "Сделки", "Обращения", 100), bf2, bb2)
    conv4 = bref(("ratio", "Сделки", "Обращения", 100))
    expenses = bref(("sum", "Расходы"))
    income = bref(("sum", "Доход"), plan="План дохода")
    office = sig["branches"]["conversion_offices"][0]
    questions += [
        _question("B1-conversion", "branches", "conversion", "Почему упала конверсия сделок за последние два месяца и в каких "
                  "отделениях?", _numbers(deals2, SHARE), office, cause="Падение конверсии в отделении «Москва Север»",
                  window=_win_text(bf2), dataset="branches"),
        _question("B1-hidden_problem", "branches", "hidden", "Что происходит с конверсией сделок?",
                  [{"name": "deviation_pct", "value": conv4["deviation_pct"]},
                   {"name": "local_change_pct", "value": own_change(branches, "Период", "Отделение", office,
                                                                    ("ratio", "Сделки", "Обращения", 100), bf4, bb4)}],
                  office, cause="Итог в норме, но конверсия упала в отделении «Москва Север»", window=_win_text(bf4),
                  dataset="branches"),
        _question("B1-expenses", "branches", "expenses", "Почему растут расходы отделений?", _numbers(expenses, SHARE),
                  sig["branches"]["expense_office"], cause="Рост расходов в отделении «Новосибирск»",
                  window=_win_text(bf4), dataset="branches"),
        _question("B2-control_income", "branches", "control", "Выполняется ли план по доходу?", _numbers(income, DEV), None,
                  window=_win_text(bf4), dataset="branches", control=True),
    ]
    for q in questions:
        q["expect"]["metric"] = REFERENCE_METRIC[q["id"]]
    return questions


def main() -> None:
    questions = build()
    payload = {"source": "data/generate_samples.py", "count": len(questions), "questions": questions}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", "utf-8")
    print(f"Записано {len(questions)} вопросов в {OUT}")


if __name__ == "__main__":
    main()
