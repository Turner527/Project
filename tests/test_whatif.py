"""Сценарии «что если» и прогноз: расчёты сверяются с независимым пересчётом по исходной таблице (pandas), а не с кодом движка."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from data import generate_samples
from tests.helpers import activate_flat
from tools import core, datastore
from tools import whatif as W


def rel(a, b, tol=1e-9):
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def close(value, expected):
    """Инструмент округляет до двух знаков (малые числа — до четырёх): сравниваем с допуском округления."""
    return abs(value - expected) <= (0.0051 if abs(expected) >= 0.1 else 0.00051)


@pytest.fixture()
def marketing(tmp_path, monkeypatch):
    activate_flat(tmp_path, monkeypatch, "marketing")
    path = tmp_path / "raw.xlsx"
    generate_samples.write_sample("marketing", path)
    raw = pd.read_excel(path)
    return raw[raw["Дата"] >= "2026-09-01"]  # окно по умолчанию для дневных данных — последние 30 дней


def activate_custom(tmp_path, monkeypatch, df: pd.DataFrame, spec: dict):
    df.to_excel(tmp_path / "data.xlsx", index=False)
    (tmp_path / "spec.json").write_text(json.dumps({"data_file": "data.xlsx", "filename": "t.xlsx", "spec": spec}, ensure_ascii=False), "utf-8")
    monkeypatch.setenv("KPI_DATA_PATH", str(tmp_path / "spec.json"))
    datastore._cache.clear()


def metric(name, column, kind="sum", better="up", **extra):
    return {"name": name, "kind": kind, "column": column, "unit": "", "better": better, "scale": 1, **extra}


def months(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="MS")


# ---------- арифметика по окну: сверка с pandas ----------


def test_pct_change_in_a_slice_matches_independent_recount(marketing):
    raw = marketing
    res = core.what_if("Расходы", "pct", -20, where_dimension="Канал", where_value="Поиск", horizon=None)
    assert "error" not in res
    spend = raw["Расходы, руб"].sum()
    search = raw.loc[raw["Канал"] == "Поиск", "Расходы, руб"].sum()
    assert close(res["selection_value"]["before"], search) and close(res["selection_value"]["after"], search * 0.8)
    total = res["total"]
    assert close(total["before"], spend) and close(total["after"], spend - 0.2 * search)
    assert close(total["delta_abs"], -0.2 * search) and close(total["delta_pct"], -0.2 * search / spend * 100) or abs(total["delta_pct"] - (-0.2 * search / spend * 100)) < 0.06
    assert total["verdict"] == "лучше"  # расходы: меньше — лучше

    effects = {e["metric"]: e for e in res["effects"]}
    orders, revenue, clicks = raw["Заказы"].sum(), raw["Выручка, руб"].sum(), raw["Клики"].sum()
    new_spend = spend - 0.2 * search
    assert close(effects["ROAS"]["before"], revenue / spend) and close(effects["ROAS"]["after"], revenue / new_spend)
    assert close(effects["Стоимость заказа"]["before"], spend / orders) and close(effects["Стоимость заказа"]["after"], new_spend / orders)
    assert close(effects["Стоимость клика"]["after"], new_spend / clicks)
    assert effects["ROAS"]["relation"].startswith("формула") and "знаменатель" in effects["ROAS"]["relation"]
    assert effects["ROAS"]["verdict"] == "лучше" and effects["Стоимость заказа"]["verdict"] == "лучше"


def test_abs_change_of_a_ratio_changes_the_rate_and_keeps_volume(marketing):
    raw = marketing
    res = core.what_if("Конверсия в лиды", "abs", 2, where_dimension="Канал", where_value="Email", horizon=None)
    email = raw[raw["Канал"] == "Email"]
    rate_email = email["Лиды"].sum() / email["Клики"].sum() * 100
    assert close(res["selection_value"]["before"], rate_email) and close(res["selection_value"]["after"], rate_email + 2)
    leads_after = raw["Лиды"].sum() - email["Лиды"].sum() + (rate_email + 2) / 100 * email["Клики"].sum()
    assert close(res["total"]["before"], raw["Лиды"].sum() / raw["Клики"].sum() * 100)
    assert close(res["total"]["after"], leads_after / raw["Клики"].sum() * 100)
    assert "effects" not in res  # ставка изменена, объём (клики) тот же: формульных следствий нет
    assert any("ставка" in a for a in res["assumptions"])


def test_set_value_and_percent_of_the_whole_report(marketing):
    raw = marketing
    res = core.what_if("Расходы", "set", 0, where_dimension="Канал", where_value="Email", horizon=None)
    assert close(res["selection_value"]["after"], 0)
    assert close(res["total"]["after"], raw["Расходы, руб"].sum() - raw.loc[raw["Канал"] == "Email", "Расходы, руб"].sum())
    whole = core.what_if("Заказы", "pct", 10, horizon=None)
    assert close(whole["total"]["after"], raw["Заказы"].sum() * 1.1) and whole["change"]["selection"] is None
    assert whole["total"]["verdict"] == "лучше"


def test_last_value_metric_changes_only_the_closing_period():
    res = core.what_if("Розничный кредитный портфель", "pct", 5, horizon=None)  # демо-витрина: остаток — значение на конец периода
    assert "error" not in res
    t = datastore.get_table()
    m = t.metric("Розничный кредитный портфель")
    assert m.kind == "last"
    last = t.df[t.df["_p"] == t.df.loc[(t.df["_p"] <= t.p_max), "_p"].max()]
    assert close(res["selection_value"]["before"], m.scale * last[m.N].sum())
    assert close(res["total"]["after"], m.scale * last[m.N].sum() * 1.05)


@pytest.mark.parametrize("kwargs, text", [
    ({"metric": "Расходы", "change_type": "pct", "change_value": -100}, "100"),
    ({"metric": "Расходы", "change_type": "pct", "change_value": -150}, "100"),
    ({"metric": "Расходы", "change_type": "умножить", "change_value": 1}, "change_type"),
    ({"metric": "Нет такого", "change_type": "pct", "change_value": 5}, "не найден"),
    ({"metric": "Расходы", "change_type": "pct", "change_value": 5, "where_dimension": "Канал", "where_value": "Телепорт"}, "нет в измерении"),
    ({"metric": "Расходы", "change_type": "pct", "change_value": 5, "where_dimension": "Канал"}, "оба параметра"),
    ({"metric": "Расходы", "change_type": "pct", "change_value": "много"}, "числом"),
])
def test_invalid_requests_return_clear_errors(marketing, kwargs, text):
    res = core.what_if(**kwargs, horizon=None)
    assert "error" in res and text in res["error"]


def test_change_type_synonyms_and_json_result(marketing):
    a = core.what_if("Расходы", "percent", -10, horizon=None)
    b = core.what_if("Расходы", "pct", -10, horizon=None)
    assert a["total"] == b["total"] and a["change"]["type"] == "pct"
    json.dumps(a, ensure_ascii=False, allow_nan=False)  # без NaN и бесконечностей


def test_verdict_words_follow_better_and_tolerance():
    assert W.verdict(-5, -5.0, "down", 3) == "лучше" and W.verdict(-5, -5.0, "up", 3) == "хуже"
    assert W.verdict(1, 1.0, "up", 3) == "в пределах допуска" and W.verdict(0, 0.0, "up", 3) == "без изменений"
    assert W.verdict(5, 5.0, "none", 3) == "выше" and W.verdict(-5, -5.0, "none", 3) == "ниже"


# ---------- оценка по истории: известная зависимость, шум, короткая история ----------


@pytest.fixture()
def history(tmp_path, monkeypatch):
    """24 месяца, 2 региона. Заказы реагируют на расходы ровно вдвое слабее (изменения в процентах), шум не связан ни с чем."""
    rng = np.random.RandomState(3)
    n = 24
    growth = 0.12 * np.sin(np.arange(n) * 1.3) + rng.normal(0, 0.03, n)
    spend = 1000 * np.cumprod(1 + growth)
    orders = np.empty(n)
    orders[0] = 500.0
    for i in range(1, n):
        orders[i] = orders[i - 1] * (1 + 0.5 * (spend[i] - spend[i - 1]) / spend[i - 1])
    noise = 300 * (1 + rng.normal(0, 0.2, n))
    rows = []
    for i, d in enumerate(months(n)):
        for region, share in (("Север", 0.6), ("Юг", 0.4)):
            rows.append({"Месяц": d, "Регион": region, "Spend": spend[i] * share, "Orders": orders[i] * share, "Noise": noise[i] * share})
    df = pd.DataFrame(rows)
    ratio = {"name": "Цена заказа", "kind": "ratio", "num": "Spend", "den": "Orders", "unit": "руб", "better": "down", "scale": 1}
    spec = {"date": "Месяц", "dimensions": ["Регион"], "metrics": [metric("Расход", "Spend", better="down"), metric("Заказы", "Orders"),
                                                                  metric("Шум", "Noise"), ratio], "filters": [], "significant_pct": 3.0}
    activate_custom(tmp_path, monkeypatch, df, spec)
    return df, spend, orders, noise


def test_history_estimate_recovers_a_known_sensitivity(history):
    df, spend, orders, noise = history
    res = core.what_if("Расход", "pct", 10, effect_on=["Заказы"], last_months=1, horizon=None)
    assert "error" not in res and res["total"]["delta_pct"] == 10.0
    est = next(e for e in res["effects"] if e["metric"] == "Заказы")
    assert est["relation"] == "оценка по истории" and est["caveat"]
    e = est["estimate"]
    assert close(e["sensitivity"], 0.5) and e["correlation"] >= 0.999 and e["n_periods"] == 23
    assert abs(e["effect_pct"] - 5.0) < 0.06 and e["effect_low_pct"] <= e["effect_pct"] <= e["effect_high_pct"]
    assert abs((e["effect_high_pct"] - e["effect_low_pct"])) < 0.2  # зависимость точная: интервал узкий
    assert close(est["before"], orders[-1]) and abs(est["after"] - orders[-1] * 1.05) < 0.02 * orders[-1] / 100 + 0.01
    assert est["verdict"] == "лучше"


def test_ratio_with_estimate_uses_the_reaction_of_the_other_part(history):
    df, spend, orders, noise = history
    res = core.what_if("Расход", "pct", 10, effect_on=["Заказы"], last_months=1, horizon=None)
    price = next(e for e in res["effects"] if e["metric"] == "Цена заказа")
    assert close(price["before"], spend[-1] / orders[-1]) and close(price["after"], spend[-1] * 1.1 / orders[-1])  # механика: заказы не меняются
    est = price["with_estimate"]
    assert est["based_on"] == "Заказы" and close(est["after"], spend[-1] * 1.1 / (orders[-1] * 1.05))
    assert est["after_low"] <= est["after"] <= est["after_high"] and est["level_pct"] == 80
    assert est["delta_pct"] > price["delta_pct"] * 0 and abs(est["delta_pct"]) < abs(price["delta_pct"])  # учёт реакции смягчает эффект
    # без effect_on варианта with_estimate нет
    plain = core.what_if("Расход", "pct", 10, last_months=1, horizon=None)
    assert "with_estimate" not in next(e for e in plain["effects"] if e["metric"] == "Цена заказа")


def test_uncorrelated_metric_is_refused_not_invented(history):
    df, spend, orders, noise = history
    g = lambda x: (x[1:] - x[:-1]) / x[:-1]  # noqa: E731
    assert abs(np.corrcoef(g(spend), g(noise))[0, 1]) < W.MIN_CORR  # исходные данные действительно не связаны
    res = core.what_if("Расход", "pct", 10, effect_on=["Шум"], last_months=1, horizon=None)
    assert all(e["metric"] != "Шум" for e in res.get("effects", []))
    refused = next(x for x in res["not_estimated"] if x["metric"] == "Шум")
    assert "слаб" in refused["reason"] and refused["n_periods"] == 23 and abs(refused["correlation"]) < 0.5


def test_short_history_is_refused(tmp_path, monkeypatch):
    n = 7
    df = pd.DataFrame({"Месяц": months(n), "Spend": np.linspace(100, 160, n), "Orders": np.linspace(50, 80, n)})
    spec = {"date": "Месяц", "dimensions": [], "metrics": [metric("Расход", "Spend", better="down"), metric("Заказы", "Orders")], "filters": [], "significant_pct": 3.0}
    activate_custom(tmp_path, monkeypatch, df, spec)
    res = core.what_if("Расход", "pct", 10, effect_on=["Заказы"], last_months=1, horizon=None)
    refused = res["not_estimated"][0]
    assert refused["metric"] == "Заказы" and refused["n_periods"] < W.MIN_HISTORY_DIFFS and "мало" in refused["reason"]


def test_history_estimate_is_flagged_when_the_scenario_is_far_beyond_the_past(history):
    res = core.what_if("Расход", "pct", 900, effect_on=["Заказы"], last_months=1, horizon=None)
    assert next(e for e in res["effects"] if e["metric"] == "Заказы")["estimate"]["beyond_history"] is True


def test_unknown_effect_metric_is_reported_not_raised(history):
    res = core.what_if("Расход", "pct", 10, effect_on=["Нет такого показателя"], last_months=1, horizon=None)
    assert "error" not in res and res["not_estimated"][0]["metric"] == "Нет такого показателя"


def test_report_without_dates_gives_arithmetic_but_no_history_or_forecast(tmp_path, monkeypatch):
    df = pd.DataFrame({"Канал": ["А", "Б", "В"], "Spend": [100.0, 200.0, 300.0], "Orders": [10.0, 30.0, 60.0]})
    spec = {"date": None, "dimensions": ["Канал"], "metrics": [metric("Расход", "Spend", better="down"), metric("Заказы", "Orders"),
                                                               {"name": "Цена", "kind": "ratio", "num": "Spend", "den": "Orders", "unit": "руб", "better": "down", "scale": 1}],
            "filters": [], "significant_pct": 3.0}
    activate_custom(tmp_path, monkeypatch, df, spec)
    res = core.what_if("Расход", "pct", -50, where_dimension="Канал", where_value="В", effect_on=["Заказы"])
    assert close(res["total"]["after"], 450) and close(next(e for e in res["effects"] if e["metric"] == "Цена")["after"], 450 / 100)
    assert "по истории" not in json.dumps(res["effects"], ensure_ascii=False) or all(e["relation"] != "оценка по истории" for e in res["effects"])
    assert "нет дат" in res["not_estimated"][0]["reason"]
    assert "нет дат" in res["forecast"]["note"]


def test_forecast_metric_needs_dates(tmp_path, monkeypatch):
    df = pd.DataFrame({"Канал": ["А", "Б", "В"], "Spend": [100.0, 200.0, 300.0]})
    spec = {"date": None, "dimensions": ["Канал"], "metrics": [metric("Расход", "Spend")], "filters": [], "significant_pct": 3.0}
    activate_custom(tmp_path, monkeypatch, df, spec)
    res = core.forecast_metric("Расход")
    assert "нет дат" in res["forecast"]["note"]


# ---------- вопрос из чата целиком: заглушка → инструменты → записка → верификатор ----------


def ask(question):
    from agent.loop import Agent
    from agent.stub_llm import ScriptedLLM
    from agent.tool_adapter import make_adapter

    adapter = make_adapter("direct")
    try:
        return Agent(ScriptedLLM(), adapter).run(question)
    finally:
        adapter.close()


def test_what_if_question_from_chat_calls_the_tool_and_every_number_is_verified(marketing):
    result = ask("Что будет с ROAS, если расходы в канале Поиск снизить на 20%?")
    assert [c.name for c in result.tool_calls] == ["find_metrics", "what_if"]
    args = result.tool_calls[1].arguments
    assert args["metric"] == "Расходы" and args["change_type"] == "pct" and args["change_value"] == -20
    assert args["where_dimension"] == "Канал" and args["where_value"] == "Поиск"
    assert result.error is None and result.verification.status == "проверен", result.verification.unverified
    memo = result.memo
    assert memo.diagnosis.root_cause.startswith("Сценарий") and memo.diagnosis.key_segments == ["Поиск"]
    assert 1 <= len(memo.scenarios) <= 3 and "ROAS" in memo.diagnosis.summary
    assert result.tool_calls[1].call_id in memo.used_tool_calls


def test_what_if_with_other_metrics_asks_for_history_estimates(marketing):
    result = ask("Что будет с заказами, если расходы на Поиск вырастут на 10%?")
    call = result.tool_calls[1]
    assert call.name == "what_if" and call.arguments["metric"] == "Расходы" and call.arguments["change_value"] == 10
    assert call.arguments.get("effect_on") == ["Заказы"]
    assert result.verification.status == "проверен", result.verification.unverified


def test_points_change_for_a_ratio_is_absolute(marketing):
    result = ask("Что если конверсия в лиды в канале Email вырастет на 2 п.п.?")
    call = result.tool_calls[1]
    assert call.arguments["metric"] == "Конверсия в лиды" and call.arguments["change_type"] == "abs" and call.arguments["change_value"] == 2
    assert result.verification.status == "проверен", result.verification.unverified


def test_forecast_question_calls_forecast_metric(marketing):
    result = ask("Какой будет выручка в ближайшие недели? Нужен прогноз.")
    assert [c.name for c in result.tool_calls] == ["find_metrics", "forecast_metric"]
    assert result.tool_calls[1].arguments["metric"] == "Выручка"
    assert result.verification.status == "проверен", result.verification.unverified
    assert result.memo.diagnosis.root_cause.startswith("Прогноз")


def test_recovery_question_keeps_the_old_path(marketing):
    result = ask("Что произойдёт, если вернуть канал «Поиск» к базе сравнения?")
    assert "what_if" not in [c.name for c in result.tool_calls]


def test_what_if_without_a_named_amount_does_not_invent_one(marketing):
    result = ask("Что если расходы вырастут?")
    assert "what_if" not in [c.name for c in result.tool_calls]


# ---------- прогноз: ряд, подписи будущих периодов, эффект сценария ----------


@pytest.fixture()
def linear_months(tmp_path, monkeypatch):
    n = 24
    df = pd.DataFrame({"Месяц": months(n), "Spend": 100.0 + 10.0 * np.arange(n), "Orders": 50.0 + 4.0 * np.arange(n)})
    spec = {"date": "Месяц", "dimensions": [], "metrics": [metric("Расход", "Spend", better="down"), metric("Заказы", "Orders"),
                                                           {"name": "Цена заказа", "kind": "ratio", "num": "Spend", "den": "Orders", "unit": "руб", "better": "down", "scale": 1}],
            "filters": [], "significant_pct": 3.0}
    activate_custom(tmp_path, monkeypatch, df, spec)
    return df


def test_forecast_metric_continues_a_linear_series_and_names_future_months(linear_months):
    res = core.forecast_metric("Расход", horizon=3)
    fc = res["forecast"]
    assert fc["bucket"] == "month" and fc["horizon"] == 3 and fc["unit_word"] == "мес."
    assert fc["labels"] == ["2026-01", "2026-02", "2026-03"] and fc["dates"] == ["2026-01-01", "2026-02-01", "2026-03-01"]
    item = fc["metrics"][0]
    assert item["baseline"] == pytest.approx([100 + 10 * k for k in (24, 25, 26)], abs=0.011)
    assert item["low"][0] <= item["baseline"][0] <= item["high"][0] and item["trend"]["direction"] == "растёт"
    assert item["reliability"]["level"] == "высокая" and item["history_n"] == 24 and item["method"]
    assert len(item["history"]["values"]) == W.HISTORY_POINTS and item["history"]["labels"][-1] == "2025-12"
    assert item["last_actual"] == 330.0 and item["horizon_total_baseline"] == pytest.approx(sum(100 + 10 * k for k in (24, 25, 26)), abs=0.02)
    assert "scenario" not in item  # без изменений сценария нет
    json.dumps(res, ensure_ascii=False, allow_nan=False)


def test_default_horizon_follows_the_grain(linear_months):
    assert core.forecast_metric("Расход")["forecast"]["horizon"] == 3  # месяцы — три месяца


def test_what_if_scenario_forecast_is_baseline_times_relative_effect(linear_months):
    res = core.what_if("Расход", "pct", 10, horizon=3)
    fc = res["forecast"]
    spend = next(i for i in fc["metrics"] if i["metric"] == "Расход")
    assert res["total"]["delta_pct"] == 10.0
    assert spend["scenario"] == pytest.approx([b * 1.1 for b in spend["baseline"]], abs=0.011)
    assert spend["horizon_total_scenario"] == pytest.approx(spend["horizon_total_baseline"] * 1.1, abs=0.03)
    assert spend["horizon_total_delta"] == pytest.approx(spend["horizon_total_baseline"] * 0.1, abs=0.03)
    price = next(i for i in fc["metrics"] if i["metric"] == "Цена заказа")  # отношение: относительный эффект из окна
    effect = next(e for e in res["effects"] if e["metric"] == "Цена заказа")
    assert price["scenario"] == pytest.approx([b * (1 + effect["delta_pct"] / 100) for b in price["baseline"]], abs=0.011)
    assert "horizon_total_baseline" not in price  # отношения за горизонт не складываются
    assert "относительному эффекту" in fc["assumption"]


def test_partial_edge_weeks_are_not_used_for_the_forecast(tmp_path, monkeypatch):
    days = pd.date_range("2026-01-07", periods=150, freq="D")  # среда ... пятница: недели по краям неполные
    df = pd.DataFrame({"Дата": days, "Spend": 100.0 + np.arange(150) % 7 * 5})
    spec = {"date": "Дата", "dimensions": [], "metrics": [metric("Расход", "Spend")], "filters": [], "significant_pct": 3.0}
    activate_custom(tmp_path, monkeypatch, df, spec)
    fc = core.forecast_metric("Расход", horizon=2)["forecast"]
    item = fc["metrics"][0]
    assert fc["bucket"] == "week" and item["history"]["labels"][0] >= "2026-01-12" and item["history"]["labels"][-1] == "2026-05-25"
    assert fc["labels"][0] == "2026-06-01"  # следующая корзина после последней полной недели
    # неполная последняя неделя (3 дня из 7) не занижает прогноз: недельная сумма близка к полной неделе
    assert item["baseline"][0] > 0.8 * item["last_actual"]


def test_short_history_gives_a_note_instead_of_a_forecast(tmp_path, monkeypatch):
    df = pd.DataFrame({"Месяц": months(4), "Spend": [10.0, 12.0, 11.0, 13.0]})
    spec = {"date": "Месяц", "dimensions": [], "metrics": [metric("Расход", "Spend")], "filters": [], "significant_pct": 3.0}
    activate_custom(tmp_path, monkeypatch, df, spec)
    fc = core.forecast_metric("Расход")["forecast"]
    assert fc["metrics"] == [] and "не меньше" in fc["note"]
    what = core.what_if("Расход", "pct", 10)  # арифметика по окну всё равно есть
    assert what["total"]["delta_pct"] == 10.0 and "не меньше" in what["forecast"]["note"]


def test_forecast_for_a_slice_uses_only_that_slice(history):
    df, spend, orders, noise = history
    north = core.forecast_metric("Расход", where_dimension="Регион", where_value="Север")
    assert north["filter"] == {"Регион": "Север"}
    assert north["forecast"]["metrics"][0]["last_actual"] == pytest.approx(spend[-1] * 0.6, abs=0.011)


def test_horizon_total_interval_is_for_the_sum_not_for_the_last_period(linear_months):
    item = core.forecast_metric("Расход", horizon=3)["forecast"]["metrics"][0]
    assert item["horizon_total_low"] == pytest.approx(sum(item["low"]), abs=0.02)
    assert item["horizon_total_high"] == pytest.approx(sum(item["high"]), abs=0.02)
    assert item["horizon_total_low"] <= item["horizon_total_baseline"] <= item["horizon_total_high"]
    assert item["horizon_total_low"] > item["low"][-1]  # интервал суммы не путается с интервалом последнего периода
