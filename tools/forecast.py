"""Прогноз ряда на несколько периодов вперёд: детерминированный, с интервалом и оценкой надёжности.

Все числа считает код (только numpy), модель языка их не придумывает. Прогноз честен в неопределённости: он
отказывается строиться на коротком ряду, рисует доверительный интервал, выведенный из ошибок самого метода на
истории, и сам оценивает, насколько ему можно верить.

Подготовка ряда: ведущие и замыкающие пропуски отбрасываются, внутренние заполняются линейной интерполяцией
(их число — в "filled"). Если интерполяцией восстановлено больше половины периодов, прогноз не строится.

Методы-кандидаты:
  * naive — последнее значение;
  * mean3 — среднее трёх последних значений;
  * linear — линейная регрессия по последним min(n, 12) точкам, продлённая вперёд;
  * holt — экспоненциальное сглаживание с аддитивным затухающим трендом (phi = 0,9); alpha и beta берутся из
    фиксированной сетки по минимуму суммы квадратов одношаговых ошибок на обучающей части;
  * seasonal_naive — значение того же периода прошлого сезона; только если задан season и n >= 2 * season.

Выбор метода — скользящая проверка на истории (rolling origin). Точки отсчёта — последние
min(6, n - MIN_TRAIN) длин обучающей части, MIN_TRAIN = 5 (а при доступной сезонности — max(5, season), чтобы в
обучении был хотя бы один сезон). Для каждой точки метод прогнозирует вперёд, пока есть факт, и ошибка факта
считается для горизонтов 1..min(horizon, 3), собранных вместе. Побеждает метод с наименьшей средней абсолютной
ошибкой (MAE); при равенстве — более простой в порядке naive, mean3, linear, holt, seasonal_naive. Если проверок
меньше двух, метод не выбирается: берётся linear при n >= 8, иначе mean3, надёжность «низкая», а ошибки для
интервала считаются по более коротким обучающим частям (от 3 точек).

Интервал: sigma_h — RMSE ошибок выбранного метода на горизонте h; если ошибок на этом горизонте меньше трёх,
sigma_h = sigma_1 * sqrt(h); sigma_h не убывает с ростом h. Полуширина = z * sigma_h, z — квантиль нормального
распределения для LEVEL (80%). С nonnegative значения и границы обрезаются снизу нулём.

Тренд (описывает историю, а не прогноз): наклон МНК по последним min(n, 12) точкам; «стабильно», если
|наклон| * длина окна < 5% среднего |значения| окна, иначе «растёт» или «падает» по знаку наклона.

Надёжность (относительная ошибка — MAPE на проверке, а если среди фактов есть ноль, то MAE / среднее |факт|):
  * «низкая» — если выполнено хотя бы одно: история < 12 периодов; проверок < 3; относительная ошибка > 30%;
    метод выбран запасным путём (проверок меньше двух);
  * «высокая» — только если история >= 24 периодов, проверок >= 4 и относительная ошибка <= 15%;
  * иначе «средняя».
В reasons перечислены все сработавшие правила с числами; для «средней» — не выполненные условия «высокой»,
для «высокой» — выполненные. В конец добавляются справочные замечания, не влияющие на уровень (сезонность не
использована; горизонт длиннее истории).

Функция чистая: тот же вход даёт тот же результат, случайности и глобального состояния нет.
"""
import math
import operator
from statistics import NormalDist

import numpy as np

from tools.engine import r
from tools.fmt import num

MIN_POINTS = 6  # меньше точек прогноз не строится
LEVEL = 0.8  # доверительный интервал 80%
SIGMA_INFLATION = 1.3  # ошибок для оценки sigma мало (до 6), а метод выбран по ним же: без поправки интервал «80%» покрывал 65–75% фактов
# (400 смоделированных рядов: блуждание, тренд, сезонность); с 1,3 покрытие 77–85%
MAX_HORIZON = 24
MAX_FOLDS = 6  # число точек отсчёта скользящей проверки
MIN_TRAIN = 5  # минимальная длина обучающей части при проверке
FALLBACK_MIN_TRAIN = 3  # то же для оценки ошибок запасного метода
SCORE_HORIZON = 3  # метод выбирается по ошибкам на горизонтах 1..min(horizon, 3)
FIT_WINDOW = 12  # окно линейного тренда и оценки направления
LAST_MEAN = 3  # окно mean3
HOLT_PHI = 0.9
HOLT_ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9)
HOLT_BETAS = (0.05, 0.1, 0.2, 0.4)
STABLE_SHARE = 0.05  # |наклон| * окно < 5% среднего |значения| -> «стабильно»
MAX_FILLED_SHARE = 0.5  # доля восстановленных интерполяцией периодов, выше которой прогноз не строится
LOW_HISTORY, HIGH_HISTORY = 12, 24
LOW_FOLDS, HIGH_FOLDS = 3, 4
LOW_ERROR_PCT, HIGH_ERROR_PCT = 30, 15
ORDER = ("naive", "mean3", "linear", "holt", "seasonal_naive")  # от простого к сложному: порядок при равных ошибках
TIE_EPS = 1e-9  # относительная точность, в пределах которой ошибки методов считаются равными


# ---------- методы: (обучающая часть, число шагов, сезон) -> прогноз на 1..H шагов ----------


def _naive(y: np.ndarray, steps: int, season: int | None) -> np.ndarray:
    return np.full(steps, y[-1])


def _mean3(y: np.ndarray, steps: int, season: int | None) -> np.ndarray:
    return np.full(steps, y[-LAST_MEAN:].mean())


def _ols(w: np.ndarray) -> tuple[float, float]:
    """Среднее окна и наклон МНК по номерам точек (наклон 0, если точка одна)."""
    k = len(w)
    if k < 2:
        return float(w.mean()), 0.0
    t = np.arange(k) - (k - 1) / 2
    return float(w.mean()), float((t * w).sum() / (t * t).sum())


def _linear(y: np.ndarray, steps: int, season: int | None) -> np.ndarray:
    w = y[-FIT_WINDOW:]
    mean, slope = _ols(w)
    return mean + slope * ((len(w) - 1) / 2 + np.arange(1, steps + 1))


def _holt(y: np.ndarray, steps: int, season: int | None) -> np.ndarray:
    """Затухающий аддитивный тренд: сетка alpha x beta считается разом, выбирается минимум SSE одношаговых ошибок."""
    if len(y) < 3:
        return _naive(y, steps, season)
    alpha = np.array([a for a in HOLT_ALPHAS for _ in HOLT_BETAS])
    beta = np.array([b for _ in HOLT_ALPHAS for b in HOLT_BETAS])
    k = min(3, len(y) - 1)
    level = np.full(len(alpha), y[0])
    trend = np.full(len(alpha), (y[k] - y[0]) / k)
    sse = np.zeros(len(alpha))
    for v in y[1:]:
        pred = level + HOLT_PHI * trend
        err = v - pred
        sse += err * err
        new = pred + alpha * err
        trend = beta * (new - level) + (1 - beta) * HOLT_PHI * trend
        level = new
    best = int(np.argmin(sse))  # при равенстве — первая пара сетки
    damp = np.cumsum(HOLT_PHI ** np.arange(1, steps + 1))
    return level[best] + trend[best] * damp


def _seasonal_naive(y: np.ndarray, steps: int, season: int | None) -> np.ndarray:
    return y[len(y) - season + np.arange(steps) % season]


METHODS = {"naive": _naive, "mean3": _mean3, "linear": _linear, "holt": _holt, "seasonal_naive": _seasonal_naive}


def _label(method: str, n: int, season: int | None) -> str:
    if method == "linear":
        return f"линейный тренд по последним {min(n, FIT_WINDOW)} периодам"
    if method == "mean3":
        return f"среднее по последним {min(n, LAST_MEAN)} периодам"
    if method == "holt":
        return "экспоненциальное сглаживание с затухающим трендом (Хольт)"
    if method == "seasonal_naive":
        return f"значение того же периода прошлого сезона (длина сезона — {season})"
    return "последнее значение ряда"


# ---------- скользящая проверка ----------


def _backtest(y: np.ndarray, methods, horizon: int, min_train: int, season: int | None):
    """Ошибки методов на истории: (число проверок, {метод: {горизонт: [ошибка]}}, {горизонт: [факт]}).
    Ошибка = факт - прогноз. Из точки отсчёта L прогнозируются шаги 1..min(horizon, n - L)."""
    n = len(y)
    folds = max(0, min(MAX_FOLDS, n - min_train))
    errs = {m: {h: [] for h in range(1, horizon + 1)} for m in methods}
    actual = {h: [] for h in range(1, horizon + 1)}
    for length in range(n - folds, n):
        steps = min(horizon, n - length)
        for h in range(1, steps + 1):
            actual[h].append(float(y[length + h - 1]))
        for m in methods:
            pred = METHODS[m](y[:length], steps, season)
            for h in range(1, steps + 1):
                errs[m][h].append(float(y[length + h - 1] - pred[h - 1]))
    return folds, errs, actual


def _pooled(errs_m: dict, actual: dict, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Ошибки и факты на горизонтах 1..min(horizon, 3), собранные вместе."""
    hs = range(1, min(horizon, SCORE_HORIZON) + 1)
    return (np.array([e for h in hs for e in errs_m[h]], dtype=float),
            np.array([a for h in hs for a in actual[h]], dtype=float))


def _choose(methods, errs, actual, horizon: int, scale: float) -> str:
    """Метод с наименьшим MAE; побеждает следующий в порядке ORDER, только если лучше строго (с допуском TIE_EPS)."""
    best, best_mae = None, math.inf
    for m in methods:
        e, _ = _pooled(errs[m], actual, horizon)
        mae = float(np.abs(e).mean())
        if best is None or mae < best_mae - TIE_EPS * scale:
            best, best_mae = m, mae
    return best


def _sigmas(errs_m: dict, horizon: int) -> np.ndarray:
    """sigma_h: RMSE ошибок на горизонте h, а при числе ошибок меньше трёх — sigma_1 * sqrt(h); не убывает с h."""
    first = np.array(errs_m[1], dtype=float)
    sigma1 = float(np.sqrt((first ** 2).mean())) if len(first) else 0.0
    out, prev = [], 0.0
    for h in range(1, horizon + 1):
        e = np.array(errs_m[h], dtype=float)
        s = float(np.sqrt((e ** 2).mean())) if len(e) >= 3 else sigma1 * math.sqrt(h)
        prev = max(prev, s)
        out.append(prev)
    return np.array(out)


# ---------- вспомогательное ----------


def _pct(x: float) -> str:
    if not math.isfinite(x):
        return "не определена"
    v = round(x, 1)
    return num(int(v) if v.is_integer() else v) + "%"


def _periods(n: int) -> str:
    """«1 период», «3 периода», «5 периодов»."""
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} период"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} периода"
    return f"{n} периодов"


def _clean(values) -> tuple[np.ndarray, int]:
    """Числовой ряд без пропусков по краям и с интерполяцией внутри: (ряд, число заполненных точек).
    Пропуском считается всё, что не число: None, NaN, бесконечности, строки, не похожие на число."""
    if isinstance(values, (str, bytes)):
        raise ValueError("values должен быть последовательностью чисел, а не строкой")
    try:
        seq = [] if values is None else list(values)
    except TypeError:
        raise ValueError("values должен быть последовательностью чисел") from None
    arr = np.full(len(seq), np.nan)
    for i, v in enumerate(seq):
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            arr[i] = x
    ok = np.flatnonzero(~np.isnan(arr))
    if len(ok) == 0:
        return arr[:0], 0
    arr = arr[ok[0]: ok[-1] + 1]
    gaps = np.isnan(arr)
    if gaps.any():
        idx = np.arange(len(arr))
        arr[gaps] = np.interp(idx[gaps], idx[~gaps], arr[~gaps])
    return arr, int(gaps.sum())


def _int_arg(value, name: str, lo: int, hi: int | None = None) -> int:
    try:
        v = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} должен быть целым числом") from None
    if v < lo or (hi is not None and v > hi):
        raise ValueError(f"{name} должен быть от {lo}" + (f" до {hi}" if hi is not None else "") + f", получено {v}")
    return v


def _reliability(n: int, folds: int, rel_pct: float, fell_back: bool, folds_regular: int) -> tuple[str, list[str]]:
    low = []
    if n < LOW_HISTORY:
        low.append(f"короткая история: {_periods(n)} (меньше {LOW_HISTORY})")
    if folds < LOW_FOLDS:
        low.append(f"мало проверок на истории: {folds} (меньше {LOW_FOLDS})")
    if rel_pct > LOW_ERROR_PCT:
        low.append(f"большая относительная ошибка на истории: {_pct(rel_pct)} (больше {num(LOW_ERROR_PCT)}%)")
    if fell_back:
        low.append(f"метод не выбран проверкой на истории (проверок {folds_regular}, нужно не меньше 2), "
                   "использован запасной")
    if low:
        return "низкая", low
    if n >= HIGH_HISTORY and folds >= HIGH_FOLDS and rel_pct <= HIGH_ERROR_PCT:
        return "высокая", [f"история {_periods(n)} (не меньше {HIGH_HISTORY})",
                           f"проверок на истории {folds} (не меньше {HIGH_FOLDS})",
                           f"относительная ошибка на истории {_pct(rel_pct)} (не больше {num(HIGH_ERROR_PCT)}%)"]
    reasons = []
    if n < HIGH_HISTORY:
        reasons.append(f"история {_periods(n)}: для высокой надёжности нужно не меньше {HIGH_HISTORY}")
    if folds < HIGH_FOLDS:
        reasons.append(f"проверок на истории {folds}: для высокой надёжности нужно не меньше {HIGH_FOLDS}")
    if rel_pct > HIGH_ERROR_PCT:
        reasons.append(f"относительная ошибка на истории {_pct(rel_pct)}: для высокой надёжности нужна не больше "
                       f"{num(HIGH_ERROR_PCT)}%")
    return "средняя", reasons


def _refuse(reason: str) -> dict:
    return {"ok": False, "reason": reason}


def forecast_series(values, horizon: int, season: int | None = None, *, nonnegative: bool = False) -> dict:
    """Прогноз ряда на horizon периодов вперёд (1..24; иначе ValueError).

    values — числа, None или NaN (старые периоды первыми, периоды равноотстоящие); season — длина сезона в периодах
    (12 для месяцев, 4 для кварталов, 7 для дней, 52 для недель) или None.

    Успех: {"ok": True, "method", "method_label", "horizon", "point", "low", "high", "level_pct", "history_n",
    "filled", "backtest": {"folds", "mae", "rmse", "mape_pct"}, "trend": {"slope_per_period", "direction"},
    "reliability": {"level", "reasons"}}. Отказ: {"ok": False, "reason": "..."}. Правила — в описании модуля.
    """
    horizon = _int_arg(horizon, "horizon", 1, MAX_HORIZON)
    if season is not None:
        season = _int_arg(season, "season", 0)
        season = season if season >= 2 else None  # сезон из одного периода или 0 — сезонности нет
    y, filled = _clean(values)
    n = len(y)
    if n == 0:
        return _refuse("В данных нет ни одного числового значения.")
    if n < MIN_POINTS:
        return _refuse(f"Для прогноза нужно не меньше {MIN_POINTS} периодов, в данных — {n}.")
    if filled > MAX_FILLED_SHARE * n:
        return _refuse(f"Более половины периодов пропущено ({filled} из {n}): прогноз по такому ряду ненадёжен.")

    with np.errstate(all="ignore"):  # переполнение на огромных значениях ловится ниже проверкой на конечность
        return _forecast(y, filled, horizon, season, nonnegative)


def _forecast(y: np.ndarray, filled: int, horizon: int, season: int | None, nonnegative: bool) -> dict:
    n = len(y)
    season_ok = season is not None and n >= 2 * season
    methods = [m for m in ORDER if m != "seasonal_naive" or season_ok]
    min_train = max(MIN_TRAIN, season) if season_ok else MIN_TRAIN
    scale = float(np.abs(y).mean())

    folds, errs, actual = _backtest(y, methods, horizon, min_train, season)
    fell_back = folds < 2
    folds_regular = folds
    if fell_back:
        method = "linear" if n >= 8 else "mean3"
        folds, errs, actual = _backtest(y, [method], horizon, FALLBACK_MIN_TRAIN, season)
    else:
        method = _choose(methods, errs, actual, horizon, scale)

    e, a = _pooled(errs[method], actual, horizon)
    mae = float(np.abs(e).mean())
    rmse = float(np.sqrt((e ** 2).mean()))
    mape = None if (a == 0).any() else float((np.abs(e) / np.abs(a)).mean() * 100)
    if mape is not None:
        rel_pct = mape
    else:
        mean_abs = float(np.abs(a).mean())
        rel_pct = mae / mean_abs * 100 if mean_abs > 0 else (0.0 if mae == 0 else math.inf)

    z = NormalDist().inv_cdf(0.5 + LEVEL / 2)
    point = METHODS[method](y, horizon, season)
    half = z * SIGMA_INFLATION * _sigmas(errs[method], horizon)
    low, high = point - half, point + half
    if nonnegative:
        point, low, high = (np.maximum(x, 0.0) for x in (point, low, high))

    w = y[-FIT_WINDOW:]
    slope = _ols(w)[1]
    mean_abs_w = float(np.abs(w).mean())
    if mean_abs_w == 0 or abs(slope) * len(w) < STABLE_SHARE * mean_abs_w:
        direction = "стабильно"
    else:
        direction = "растёт" if slope > 0 else "падает"

    level, reasons = _reliability(n, folds, rel_pct, fell_back, folds_regular)
    if season is not None and not season_ok:
        reasons.append(f"сезонность (длина сезона {season}) не учитывалась: для неё нужна история не меньше "
                       f"двух сезонов ({2 * season}), в данных — {n}")
    if horizon > n:
        reasons.append(f"горизонт ({horizon}) длиннее истории ({n}): дальние значения очень неопределённы")

    numbers = [point, low, high, [mae, rmse, slope]]
    if not all(np.all(np.isfinite(x)) for x in numbers):
        return _refuse("Не удалось рассчитать прогноз: значения ряда слишком велики для расчёта.")

    def clean(x) -> list[float]:
        return [r(v) + 0.0 for v in x]  # + 0.0 превращает -0.0 в 0.0

    return {"ok": True, "method": method, "method_label": _label(method, n, season), "horizon": horizon,
            "point": clean(point), "low": clean(low), "high": clean(high), "level_pct": round(LEVEL * 100),
            "history_n": n, "filled": filled,
            "backtest": {"folds": folds, "mae": r(mae), "rmse": r(rmse), "mape_pct": r(mape)},
            "trend": {"slope_per_period": r(slope), "direction": direction},
            "reliability": {"level": level, "reasons": reasons}}
