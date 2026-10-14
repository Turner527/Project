"""Прогноз ряда (tools/forecast.py): методы, выбор по проверке на истории, интервал, надёжность, отказы."""
import json
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from tools import forecast as F
from tools.forecast import LEVEL, MIN_POINTS, forecast_series

Z = NormalDist().inv_cdf(0.5 + LEVEL / 2) * F.SIGMA_INFLATION  # полуширина = z · поправка · sigma


def noisy(n, seed, base=100.0, sd=5.0):
    return list(base + np.random.RandomState(seed).normal(0, sd, n))


def line(n, a=10.0, b=2.0):
    return [a + b * t for t in range(n)]


def check_sane(res: dict, horizon: int):
    """Общие свойства любого успешного ответа: длины, порядок границ, конечные числа, сериализуемость."""
    assert res["ok"] is True
    assert res["horizon"] == horizon
    for key in ("point", "low", "high"):
        assert len(res[key]) == horizon
        assert all(isinstance(v, float) and math.isfinite(v) for v in res[key])
    for lo, p, hi in zip(res["low"], res["point"], res["high"]):
        assert lo <= p <= hi
    assert res["level_pct"] == 80
    assert res["trend"]["direction"] in ("растёт", "падает", "стабильно")
    assert res["reliability"]["level"] in ("низкая", "средняя", "высокая")
    assert res["reliability"]["reasons"]
    assert res["method"] in F.METHODS
    assert res["method_label"]
    json.dumps(res, allow_nan=False)  # ни NaN, ни inf, ни numpy-типов


# ---------- независимый «оракул»: скалярные циклы вместо векторных формул модуля ----------


def o_naive(y, h):
    return y[-1]


def o_mean3(y, h):
    return sum(y[-3:]) / 3


def o_linear(y, h):
    w = y[-12:]
    k = len(w)
    if k < 2:
        return w[-1]
    slope, intercept = np.polyfit(range(k), w, 1)
    return intercept + slope * (k - 1 + h)


def o_holt(y, h):
    phi, best = 0.9, None
    k = min(3, len(y) - 1)
    for alpha in (0.1, 0.3, 0.5, 0.7, 0.9):
        for beta in (0.05, 0.1, 0.2, 0.4):
            level, trend, sse = y[0], (y[k] - y[0]) / k, 0.0
            for v in y[1:]:
                pred = level + phi * trend
                sse += (v - pred) ** 2
                new = alpha * v + (1 - alpha) * pred
                trend = beta * (new - level) + (1 - beta) * phi * trend
                level = new
            if best is None or sse < best[0]:
                best = (sse, level, trend)
    _, level, trend = best
    return level + trend * sum(phi ** i for i in range(1, h + 1))


def o_seasonal(y, h, season):
    return y[len(y) - season + (h - 1) % season]


def oracle(y, horizon, season=None):
    """Проверка на истории по описанию из docstring модуля: {метод: {h: [ошибки]}}, {h: [факты]}, число проверок."""
    n = len(y)
    seasonal = bool(season) and n >= 2 * season
    min_train = max(5, season) if seasonal else 5
    folds = min(6, n - min_train)
    fns = {"naive": o_naive, "mean3": o_mean3, "linear": o_linear, "holt": o_holt}
    if seasonal:
        fns["seasonal_naive"] = lambda yy, h: o_seasonal(yy, h, season)
    errs = {m: {h: [] for h in range(1, horizon + 1)} for m in fns}
    actual = {h: [] for h in range(1, horizon + 1)}
    for length in range(n - folds, n):
        for h in range(1, min(horizon, n - length) + 1):
            actual[h].append(y[length + h - 1])
            for m, fn in fns.items():
                errs[m][h].append(y[length + h - 1] - fn(y[:length], h))
    return folds, errs, actual


def oracle_choice(y, horizon, season=None):
    folds, errs, actual = oracle(y, horizon, season)
    hs = range(1, min(horizon, 3) + 1)
    maes = {m: np.mean([abs(e) for h in hs for e in errs[m][h]]) for m in errs}
    order = ["naive", "mean3", "linear", "holt", "seasonal_naive"]
    best = min((m for m in order if m in maes), key=lambda m: (maes[m], order.index(m)))
    return best, maes, errs, actual, folds


# ---------- точные ряды ----------


def test_точная_прямая_продолжается_без_неопределённости():
    res = forecast_series(line(15), 4)
    check_sane(res, 4)
    assert res["method"] == "linear"
    assert res["method_label"] == "линейный тренд по последним 12 периодам"
    assert res["point"] == pytest.approx(line(19)[15:], abs=1e-6)
    assert np.array(res["high"]) - np.array(res["low"]) == pytest.approx(0, abs=1e-6)
    assert res["backtest"]["mae"] == pytest.approx(0, abs=1e-6)
    assert res["trend"] == {"slope_per_period": 2.0, "direction": "растёт"}


def test_точная_падающая_прямая():
    res = forecast_series(line(10, 100, -3), 3)
    assert res["method"] == "linear"
    assert res["point"] == pytest.approx([100 - 3 * t for t in (10, 11, 12)], abs=1e-6)
    assert res["trend"] == {"slope_per_period": -3.0, "direction": "падает"}


def test_метка_линейного_тренда_называет_окно():
    assert forecast_series(line(8), 2)["method_label"] == "линейный тренд по последним 8 периодам"
    assert forecast_series(line(30), 2)["method_label"] == "линейный тренд по последним 12 периодам"


def test_линейный_прогноз_берёт_только_последние_12_точек():
    # до 18-й точки ряд падает, дальше растёт: прогноз обязан продолжить рост (окно 12), а не среднюю линию
    y = [100 - 5 * t for t in range(18)] + [10 + 4 * t for t in range(20)]
    res = forecast_series(y, 3)
    assert res["method"] == "linear"
    assert res["point"] == pytest.approx([10 + 4 * t for t in (20, 21, 22)], abs=1e-6)


def test_постоянный_ряд():
    res = forecast_series([7.5] * 10, 5)
    check_sane(res, 5)
    assert res["method"] == "naive"
    assert res["point"] == [7.5] * 5 and res["low"] == [7.5] * 5 and res["high"] == [7.5] * 5
    assert res["trend"] == {"slope_per_period": 0.0, "direction": "стабильно"}
    assert res["backtest"]["mape_pct"] == 0.0


def test_нулевой_ряд_допустим_а_mape_не_определён():
    res = forecast_series([0] * 9, 3)
    check_sane(res, 3)
    assert res["point"] == [0.0] * 3
    assert res["backtest"]["mape_pct"] is None
    assert res["trend"]["direction"] == "стабильно"


# ---------- сезонность ----------


def seasonal_series(n=36, seed=0):
    t = np.arange(n)
    return list(100 + 0.1 * t + 15 * np.sin(2 * np.pi * t / 12) + np.random.RandomState(seed).normal(0, 1, n))


def test_сезонный_ряд_выбирает_сезонный_повтор_и_точнее_линейного_на_отложенном_годе():
    y = seasonal_series()
    train, held = y[:24], np.array(y[24:])  # ровно два сезона истории, год отложен
    res = forecast_series(train, 12, season=12)
    check_sane(res, 12)
    assert res["method"] == "seasonal_naive"
    assert "12" in res["method_label"]
    mae = np.abs(np.array(res["point"]) - held).mean()
    slope, intercept = np.polyfit(range(12), train[-12:], 1)  # линейный тренд по последним 12 точкам
    linear = intercept + slope * (11 + np.arange(1, 13))
    assert mae < 0.5 * np.abs(linear - held).mean()
    assert mae < 3.0


def test_сезонный_повтор_на_полной_истории_и_длинный_горизонт_идёт_по_кругу():
    y = seasonal_series()
    res = forecast_series(y, 18, season=12)
    assert res["method"] == "seasonal_naive"
    assert res["point"] == [round(v, 2) for v in y[-12:] + y[-12:][:6]]


def test_без_season_сезонный_метод_не_используется():
    res = forecast_series(seasonal_series(), 6)
    assert res["method"] != "seasonal_naive"
    assert not any("сезонность" in s for s in res["reliability"]["reasons"])


def test_сезон_длиннее_половины_истории_игнорируется_с_пояснением():
    res = forecast_series(seasonal_series(20), 3, season=12)
    assert res["method"] != "seasonal_naive"
    assert any("сезонность (длина сезона 12) не учитывалась" in s and "24" in s for s in res["reliability"]["reasons"])


def test_сезон_нет_если_он_меньше_двух():
    assert forecast_series(noisy(20, 1), 3, season=1) == forecast_series(noisy(20, 1), 3)
    assert forecast_series(noisy(20, 1), 3, season=0) == forecast_series(noisy(20, 1), 3)
    with pytest.raises(ValueError):
        forecast_series(noisy(20, 1), 3, season=-4)


@pytest.mark.parametrize("season", [3, 4, 7, 12])
def test_история_ровно_два_сезона(season):
    t = np.arange(2 * season)
    y = list(50 + 10 * np.sin(2 * np.pi * t / season) + np.random.RandomState(season).normal(0, 0.3, len(t)))
    res = forecast_series(y, season, season=season)
    check_sane(res, season)
    assert res["history_n"] == 2 * season


# ---------- выбор метода: сверка с независимой реализацией ----------


@pytest.mark.parametrize("seed", range(24))
def test_выбор_метода_и_ошибки_совпадают_с_независимым_расчётом(seed):
    rng = np.random.RandomState(100 + seed)
    n = int(rng.randint(8, 40))
    kind = seed % 3
    t = np.arange(n)
    if kind == 0:
        y = np.cumsum(rng.normal(0, 1, n)) + 50
    elif kind == 1:
        y = 20 + rng.uniform(-1, 1) * t + rng.normal(0, 2, n)
    else:
        y = 100 + 8 * np.sin(2 * np.pi * t / 4) + rng.normal(0, 1, n)
    y = [float(v) for v in y]
    season = 4 if kind == 2 else None
    horizon = int(rng.randint(1, 9))

    best, maes, errs, actual, folds = oracle_choice(y, horizon, season)
    res = forecast_series(y, horizon, season)
    assert res["method"] == best
    assert res["backtest"]["folds"] == folds
    assert res["backtest"]["mae"] == pytest.approx(maes[best], abs=0.006)

    hs = range(1, min(horizon, 3) + 1)
    pooled = [e for h in hs for e in errs[best][h]]
    assert res["backtest"]["rmse"] == pytest.approx(math.sqrt(np.mean(np.square(pooled))), abs=0.006)

    # интервал: sigma_h — RMSE ошибок на горизонте h (при трёх и более), иначе sigma_1 * sqrt(h); не убывает
    sigma1 = math.sqrt(np.mean(np.square(errs[best][1])))
    prev = 0.0
    for h in range(1, horizon + 1):
        e = errs[best][h]
        sigma = math.sqrt(np.mean(np.square(e))) if len(e) >= 3 else sigma1 * math.sqrt(h)
        prev = max(prev, sigma)
        assert (res["high"][h - 1] - res["low"][h - 1]) / 2 == pytest.approx(Z * prev, abs=0.011)


def test_запасной_путь_на_шести_точках_сверен_вручную():
    # n = 6: проверок по правилу меньше двух -> mean3; ошибки считаются по обучающим частям от 3 точек
    # origin 3: среднее [1,2,3]=2, факты 4,5,7 -> ошибки 2,3,5; origin 4: среднее [2,3,4]=3, факты 5,7 -> 2,4;
    # origin 5: среднее [3,4,5]=4, факт 7 -> 3. На горизонтах 1..3 вместе MAE = 19/6
    res = forecast_series([1, 2, 3, 4, 5, 7], 3)
    assert res["method"] == "mean3"
    assert res["backtest"]["folds"] == 3
    assert res["backtest"]["mae"] == pytest.approx(19 / 6, abs=0.006)
    sigma1 = math.sqrt((2 ** 2 + 2 ** 2 + 3 ** 2) / 3)  # ошибки на горизонте 1: 2, 2, 3
    assert res["point"] == pytest.approx([16 / 3] * 3, abs=0.006)
    for h in (1, 2, 3):  # на горизонтах 2 и 3 ошибок меньше трёх: sigma_1 * sqrt(h)
        assert res["high"][h - 1] - res["point"][h - 1] == pytest.approx(Z * sigma1 * math.sqrt(h), abs=0.011)
    assert res["reliability"]["level"] == "низкая"
    assert any("запасной" in s for s in res["reliability"]["reasons"])


def test_равные_ошибки_решает_простота_метода():
    # константа: у всех методов ошибка 0 (с точностью до погрешности float) -> naive, самый простой
    assert forecast_series([3.3] * 12, 3)["method"] == "naive"
    assert forecast_series([0.1 + 0.2] * 9, 3)["method"] == "naive"


def test_затухающий_тренд_холта():
    # рост с насыщением: прогноз замедляется, а не уходит по прямой вверх
    t = np.arange(24)
    y = list(1000 * (100 * (1 - np.exp(-t / 5)) + np.random.RandomState(3).normal(0, 0.3, 24)))
    res = forecast_series(y, 8)
    assert res["method"] == "holt"
    p = np.array(res["point"])
    steps = np.diff(p)
    assert (steps > 0).all() and (np.diff(steps) < 0).all()  # растёт всё медленнее
    assert steps[1:] / steps[:-1] == pytest.approx([0.9] * 6, abs=0.02)  # затухание phi = 0,9
    slope, intercept = np.polyfit(range(12), y[-12:], 1)
    assert p[-1] < intercept + slope * (11 + 8)  # ниже продолжения прямой


def test_приращения_прогноза_холта_образуют_геометрию_со_знаменателем_phi():
    # f(h) - f(h-1) = phi**h * тренд: проверка формулы затухания без округления ответа
    f = F._holt(np.array(line(10, 0, 1.0)), 6, None)
    steps = np.diff(f)
    assert steps[1:] / steps[:-1] == pytest.approx([F.HOLT_PHI] * 4, abs=1e-9)


# ---------- интервал ----------


def test_интервал_случайного_блуждания_покрывает_около_80_процентов():
    rng = np.random.RandomState(1)
    n, runs, hits = 36, 200, 0
    for _ in range(runs):
        x = np.cumsum(rng.normal(0, 1, n)) + rng.normal(0, 0.5, n)
        res = forecast_series(list(x[:-1]), 1)
        hits += res["low"][0] <= x[-1] <= res["high"][0]
    assert 0.72 <= hits / runs <= 0.92


@pytest.mark.parametrize("kind", ["walk", "trend", "season"])
def test_интервал_80_процентов_откалиброван_на_трёх_видах_рядов(kind):
    """Без поправки SIGMA_INFLATION охват был 65–75%: интервал, названный 80%-ным, обещал бы больше, чем даёт."""
    rng = np.random.RandomState(11)
    n, runs, hits, total = 36, 200, 0, 0
    for _ in range(runs):
        t = np.arange(n + 3)
        if kind == "walk":
            y = 100 + np.cumsum(rng.normal(0, 3, n + 3))
        elif kind == "trend":
            y = 100 + 2 * t + rng.normal(0, 5, n + 3)
        else:
            y = 100 + 20 * np.sin(t * 2 * np.pi / 12) + rng.normal(0, 4, n + 3)
        res = forecast_series(list(y[:n]), 3, 12 if kind == "season" else None)
        for k in range(3):
            total += 1
            hits += res["low"][k] <= y[n + k] <= res["high"][k]
    assert 0.74 <= hits / total <= 0.92


def test_интервал_не_сужается_с_горизонтом_и_шире_нуля_на_шуме():
    res = forecast_series(noisy(30, 5), 12)
    half = (np.array(res["high"]) - np.array(res["low"])) / 2
    assert (half > 0).all()
    assert (np.diff(half) >= -0.011).all()
    # дальние горизонты (ошибок на горизонте меньше трёх) не уже sigma_1 * sqrt(h)
    assert (half[4:] >= half[0] * np.sqrt(np.arange(5, 13)) - 0.02).all()


def test_середина_интервала_точечный_прогноз_а_границы_симметричны():
    res = forecast_series(noisy(24, 9), 6)
    mid = (np.array(res["high"]) + np.array(res["low"])) / 2
    assert mid == pytest.approx(res["point"], abs=0.011)


def test_nonnegative_обрезает_прогноз_и_границы_нулём():
    y = [float(max(v, 0)) for v in noisy(20, 7, base=2.0, sd=3.0)]
    raw = forecast_series(y, 6)
    clipped = forecast_series(y, 6, nonnegative=True)
    assert min(raw["low"]) < 0  # без обрезки нижняя граница уходит в минус
    for key in ("point", "low", "high"):
        assert clipped[key] == pytest.approx([max(v, 0.0) for v in raw[key]], abs=1e-9)
        assert min(clipped[key]) >= 0
    assert clipped["method"] == raw["method"]


def test_nonnegative_прямая_уходит_ниже_нуля():
    y = [28 - 4 * t for t in range(7)]  # 28 ... 4, дальше 0, -4, -8
    assert forecast_series(y, 3)["point"] == pytest.approx([0, -4, -8], abs=1e-6)
    assert forecast_series(y, 3, nonnegative=True)["point"] == [0.0, 0.0, 0.0]


# ---------- подготовка ряда ----------


def test_пропуски_по_краям_отбрасываются_а_внутри_заполняются_линейно():
    res = forecast_series([None, float("nan"), 1, 2, None, 4, 5, 6, 7, 8, None, None], 3)
    clean = forecast_series([1, 2, 3, 4, 5, 6, 7, 8], 3)
    assert res["history_n"] == 8 and res["filled"] == 1
    assert {**res, "filled": 0} == clean  # пропуск между 2 и 4 стал 3


def test_несколько_подряд_идущих_пропусков_считаются_по_одному():
    res = forecast_series([0, 2, None, None, None, 10, 12, 14], 2)
    assert res["filled"] == 3 and res["history_n"] == 8
    assert res["method"] == "linear"
    assert res["point"] == pytest.approx([16, 18], abs=1e-6)  # 0,2,4,6,8,10,12,14 — прямая


def test_бесконечность_и_нечисло_считаются_пропуском():
    res = forecast_series([1, 2, float("inf"), 4, 5, "x", 7, 8, "9"], 2)
    assert res["filled"] == 2 and res["history_n"] == 9
    check_sane(res, 2)


def test_слишком_много_восстановленных_точек_отказ():
    res = forecast_series([1, None, None, None, None, 6, 7], 3)  # 4 из 7 придуманы интерполяцией
    assert res["ok"] is False and "пропущено" in res["reason"] and "4 из 7" in res["reason"]


def test_вход_не_меняется_и_принимает_разные_контейнеры():
    src = [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    keep = list(src)
    base = forecast_series(src, 3)
    assert src == keep
    assert forecast_series(tuple(src), 3) == base
    assert forecast_series(np.array([np.nan if v is None else v for v in src]), 3) == base
    assert forecast_series(pd.Series(src, dtype="float64"), 3) == base


# ---------- отказы и проверка аргументов ----------


@pytest.mark.parametrize("n", [0, 1, 4, 5])
def test_короткий_ряд_отказ(n):
    res = forecast_series(list(range(1, n + 1)), 3)
    assert res["ok"] is False
    if n:
        assert res["reason"] == f"Для прогноза нужно не меньше 6 периодов, в данных — {n}."


def test_строка_вместо_ряда_ошибка():
    for bad in ("123456789", b"123456789", 5):
        with pytest.raises(ValueError):
            forecast_series(bad, 3)


def test_отказ_на_пустом_и_без_чисел():
    for values in ([], None, [None, None, None], [float("nan")] * 10):
        res = forecast_series(values, 3)
        assert res["ok"] is False and "ни одного числового значения" in res["reason"]


def test_отказ_считает_периоды_после_обрезки_краёв():
    res = forecast_series([None, None, 1, 2, 3, 4, 5, None], 3)
    assert res["ok"] is False and "в данных — 5." in res["reason"]


def test_ровно_минимум_точек_допустим():
    assert MIN_POINTS == 6
    res = forecast_series([3, 5, 4, 6, 5, 7], 3)
    check_sane(res, 3)
    assert res["history_n"] == 6


@pytest.mark.parametrize("horizon", [0, -1, 25, 100])
def test_горизонт_вне_диапазона_ошибка(horizon):
    with pytest.raises(ValueError):
        forecast_series(line(10), horizon)


@pytest.mark.parametrize("horizon", [2.5, "3", None])
def test_горизонт_не_целое_ошибка(horizon):
    with pytest.raises(ValueError):
        forecast_series(line(10), horizon)


@pytest.mark.parametrize("horizon", [1, 24])
def test_границы_горизонта_допустимы(horizon):
    res = forecast_series(noisy(10, 2), horizon)
    check_sane(res, horizon)


def test_горизонт_длиннее_истории_допустим_с_пометкой():
    res = forecast_series(noisy(8, 4), 24)
    check_sane(res, 24)
    assert any("горизонт (24) длиннее истории (8)" in s for s in res["reliability"]["reasons"])


def test_детерминизм():
    y = noisy(30, 11)
    assert forecast_series(y, 6, 12) == forecast_series(y, 6, 12)
    assert forecast_series(y, 6, season=4, nonnegative=True) == forecast_series(y, 6, season=4, nonnegative=True)


# ---------- устойчивость: ни исключений, ни NaN/inf ----------

EDGE_SERIES = {
    "все равные": [4.2] * 15,
    "все нули": [0.0] * 15,
    "один огромный выброс": [10, 11, 9, 10, 1e9, 10, 11, 9, 10, 12, 11, 10],
    "огромный выброс в конце": [10, 11, 9, 10, 11, 9, 10, 12, 11, 1e9],
    "отрицательные": [-5, -9, -3, -12, -7, -15, -6, -11, -4, -10, -8, -13],
    "через ноль": [-3, 4, -2, 5, -1, 6, 0, 7, 1, 8],
    "ровно минимум точек": [1, 2, 3, 4, 5, 6],
    "крошечные значения": [1e-6 * v for v in (3, 5, 4, 6, 5, 7, 6, 8)],
    "огромные значения": [1e15 * v for v in (3, 5, 4, 6, 5, 7, 6, 8)],
    "экстремально большие": [1e300 * v for v in (3, 5, 4, 6, 5, 7, 6, 8)],
    "с нулями среди фактов": [0, 5, 0, 5, 0, 5, 0, 5, 0, 5],
}


@pytest.mark.parametrize("name", list(EDGE_SERIES))
@pytest.mark.parametrize("horizon, season", [(1, None), (3, 2), (24, None), (24, 3), (5, 40)])
@pytest.mark.parametrize("nonnegative", [False, True])
def test_крайние_ряды_без_исключений_и_без_nan(name, horizon, season, nonnegative):
    res = forecast_series(EDGE_SERIES[name], horizon, season, nonnegative=nonnegative)
    if res["ok"]:
        check_sane(res, horizon)
        if nonnegative:
            assert min(res["low"]) >= 0
    else:
        assert res["reason"]


def test_огромные_числа_дают_отказ_а_не_бесконечность():
    res = forecast_series(EDGE_SERIES["экстремально большие"], 3)
    assert res["ok"] is False and "слишком велики" in res["reason"]


def test_выброс_расширяет_интервал_а_не_прячется():
    calm = forecast_series([10, 11, 9, 10, 11, 9, 10, 12, 11, 10, 9, 10], 3)
    wild = forecast_series([10, 11, 9, 10, 11, 9, 10, 12, 11, 100, 9, 10], 3)
    assert wild["high"][0] - wild["low"][0] > 5 * (calm["high"][0] - calm["low"][0])
    assert wild["reliability"]["level"] == "низкая"


# ---------- направление тренда ----------


def test_направление_стабильно_если_изменение_за_окно_меньше_5_процентов():
    flat = [100 + 0.2 * t + (0.5 if t % 2 else -0.5) for t in range(12)]  # за окно ~2,4%
    assert forecast_series(flat, 3)["trend"]["direction"] == "стабильно"
    assert forecast_series([100 + 1.0 * t for t in range(12)], 3)["trend"]["direction"] == "растёт"
    assert forecast_series([100 - 1.0 * t for t in range(12)], 3)["trend"]["direction"] == "падает"


def test_наклон_считается_по_последним_12_точкам():
    y = [50 - 10 * t for t in range(10)] + [0 + 3 * t for t in range(12)]
    trend = forecast_series(y, 3)["trend"]
    assert trend["slope_per_period"] == pytest.approx(3.0)
    assert trend["direction"] == "растёт"


# ---------- надёжность ----------


def level(res):
    return res["reliability"]["level"]


def reasons(res):
    return " | ".join(res["reliability"]["reasons"])


@pytest.mark.parametrize("n, text", [(1, "1 период"), (2, "2 периода"), (5, "5 периодов"), (11, "11 периодов"),
                                     (12, "12 периодов"), (21, "21 период"), (23, "23 периода"), (112, "112 периодов")])
def test_склонение_периодов_в_причинах(n, text):
    assert F._periods(n) == text


def test_надёжность_низкая_на_короткой_истории():
    res = forecast_series(line(10), 3)  # точная прямая, но 10 < 12 периодов
    assert level(res) == "низкая"
    assert "10 периодов (меньше 12)" in reasons(res)


def test_надёжность_низкая_при_малом_числе_проверок():
    res = forecast_series(noisy(7, 3), 3)  # n = 7: проверок две
    assert res["backtest"]["folds"] == 2 and level(res) == "низкая"
    assert "проверок на истории: 2 (меньше 3)" in reasons(res)


def test_надёжность_низкая_если_метод_запасной():
    res = forecast_series(line(6), 3)
    assert level(res) == "низкая"
    assert "запасной" in reasons(res) and "проверок 1" in reasons(res)


def test_надёжность_низкая_при_большой_ошибке_на_длинной_истории():
    res = forecast_series([20, 180] * 20, 3)  # пилообразный ряд: ни один метод его не угадывает
    assert res["history_n"] >= 24 and res["backtest"]["folds"] >= 4
    assert res["backtest"]["mape_pct"] > 30
    assert level(res) == "низкая"
    assert "относительная ошибка" in reasons(res) and "больше 30%" in reasons(res)


def test_надёжность_относительная_ошибка_без_mape_считается_от_среднего_факта():
    # факты содержат ноль -> MAPE не определён, берётся MAE / среднее |факт|; пилообразный ряд даёт большую ошибку
    res = forecast_series([0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8], 3)
    assert res["backtest"]["mape_pct"] is None
    assert level(res) == "низкая"
    assert "относительная ошибка" in reasons(res)


def test_надёжность_высокая_на_длинном_точном_ряде():
    res = forecast_series(seasonal_series(36), 6, season=12)
    assert res["history_n"] == 36 and res["backtest"]["folds"] == 6 and res["backtest"]["mape_pct"] <= 15
    assert level(res) == "высокая"
    assert "история 36 периодов (не меньше 24)" in reasons(res)


def test_граница_истории_24_периода_для_высокой():
    y = [100 + 0.5 * t + (1 if t % 2 else -1) * 0.2 for t in range(24)]
    assert level(forecast_series(y, 3)) == "высокая"
    short = forecast_series(y[1:], 3)  # 23 периода
    assert level(short) == "средняя"
    assert "история 23 периода: для высокой надёжности нужно не меньше 24" in reasons(short)


def test_надёжность_средняя_между_порогами():
    res = forecast_series(noisy(30, 4, base=100, sd=14), 3)  # MAPE между 15% и 30%, история 30
    assert 15 < res["backtest"]["mape_pct"] <= 30
    assert level(res) == "средняя"
    assert "относительная ошибка на истории" in reasons(res) and "не больше 15%" in reasons(res)


def test_все_сработавшие_правила_перечислены():
    res = forecast_series(noisy(7, 8, base=10, sd=9), 3)  # короткая, две проверки, большая ошибка
    assert level(res) == "низкая"
    text = reasons(res)
    assert "меньше 12" in text and "меньше 3" in text and "больше 30%" in text


@pytest.mark.parametrize("seed", range(60))
def test_уровень_надёжности_подчиняется_правилам(seed):
    rng = np.random.RandomState(500 + seed)
    n = int(rng.randint(6, 45))
    y = list(np.abs(60 + np.cumsum(rng.normal(0, 3, n)) + rng.normal(0, rng.uniform(1, 25), n)) + 1)
    res = forecast_series(y, int(rng.randint(1, 13)))
    assert res["ok"]
    b, mape = res["backtest"], res["backtest"]["mape_pct"]
    fell_back = n == 6  # единственный случай, когда проверок меньше двух
    if n < 12 or b["folds"] < 3 or mape > 30 or fell_back:
        expected = "низкая"
    elif n >= 24 and b["folds"] >= 4 and mape <= 15:
        expected = "высокая"
    else:
        expected = "средняя"
    assert level(res) == expected
    assert res["reliability"]["reasons"]


# ---------- публичные константы и вид ответа ----------


def test_состав_ответа():
    res = forecast_series(noisy(14, 6), 3)
    assert set(res) == {"ok", "method", "method_label", "horizon", "point", "low", "high", "level_pct", "history_n",
                        "filled", "backtest", "trend", "reliability"}
    assert set(res["backtest"]) == {"folds", "mae", "rmse", "mape_pct"}
    assert set(res["trend"]) == {"slope_per_period", "direction"}
    assert set(res["reliability"]) == {"level", "reasons"}
    assert res["history_n"] == 14 and res["filled"] == 0
    assert F.LEVEL == 0.8 and F.MIN_POINTS == 6


def test_без_отрицательного_нуля():
    # прямая пересекает ноль с погрешностью float (-4e-16): после округления не должно остаться -0.0
    res = forecast_series([0.7 * (7 - t) for t in range(7)], 3)
    assert res["method"] == "linear" and res["point"][0] == 0.0
    for key in ("point", "low", "high"):
        assert all(not (v == 0 and math.copysign(1, v) < 0) for v in res[key])
