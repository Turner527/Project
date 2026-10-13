"""Генератор синтетической банковской витрины (реальные данные не используются).

Запуск:  python -m data.generate
Создаёт data/kpi.xlsx (листы goals, facts, links) и data/anomalies.json — описание
четырёх заложенных аномалий и эталонных чисел для evals. Результат детерминирован (SEED).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.schema import (
    BETTER_DOWN,
    BETTER_UP,
    DIR_NEGATIVE,
    DIR_POSITIVE,
    FACTS_COLUMNS,
    GOALS_COLUMNS,
    KIND_FLOW,
    KIND_RATIO,
    KIND_STOCK,
    LINKS_COLUMNS,
    SHEET_FACTS,
    SHEET_GOALS,
    SHEET_LINKS,
)

SEED = 20260101
FIRST_MONTH, LAST_MONTH = "2025-04", "2026-09"  # 18 месяцев
MONTHS = pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M")
SEGMENTS = ["Массовый", "Зарплатные", "Премиум", "Молодёжь", "Самозанятые и ИП"]
SIGNIFICANT_PCT = 3.0  # порог значимого отклонения; тот же, что в tools.core
AMORTIZATION = 0.035  # месячное погашение розничного портфеля

OUT_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Kpi:
    name: str
    goal: str
    unit: str
    kind: str
    better: str
    base: tuple[float, ...]  # уровень первого месяца по сегментам (порядок SEGMENTS)
    trend: float = 0.0  # месячный относительный рост
    season: str = "none"
    noise: float = 0.02  # стандартное отклонение относительного шума


# сезонные профили по календарным месяцам (янв..дек), нормируются к среднему 1
SEASON = {
    "none": [1] * 12,
    "lending": [0.86, 0.92, 1.00, 1.02, 0.98, 1.02, 1.05, 1.04, 1.03, 1.05, 1.03, 1.20],
    "spending": [0.85, 0.88, 0.97, 0.96, 0.98, 1.00, 1.03, 1.04, 1.00, 1.02, 1.10, 1.30],
    "profit": [0.94, 0.96, 1.00, 1.00, 0.99, 1.01, 1.02, 1.01, 1.00, 1.02, 1.01, 1.08],
    "deposit": [1.02, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.01, 1.03],
}

G_PROFIT = "Рост прибыли и капитала"
G_LENDING = "Рост розничного кредитования"
G_QUALITY = "Качество кредитного портфеля"
G_NII = "Рост чистого процентного дохода"
G_FEES = "Рост комиссионных доходов"
G_CLIENTS = "Привлечение и удержание клиентов"
G_FUNDING = "Рост ресурсной базы"
G_DIGITAL = "Цифровизация продаж и обслуживания"

GOALS = [  # цель, родительская цель, владелец
    (G_PROFIT, None, "Финансовый директор"),
    (G_LENDING, G_PROFIT, "Директор департамента розничного бизнеса"),
    (G_QUALITY, G_PROFIT, "Директор по рискам"),
    (G_NII, G_PROFIT, "Казначей"),
    (G_FEES, G_PROFIT, "Директор по транзакционному бизнесу"),
    (G_CLIENTS, G_PROFIT, "Директор по клиентскому опыту"),
    (G_FUNDING, G_NII, "Директор по привлечению средств населения"),
    (G_DIGITAL, G_CLIENTS, "Директор по цифровым каналам"),
]

# базовые уровни по сегментам: Массовый, Зарплатные, Премиум, Молодёжь, Самозанятые и ИП
KPIS = [
    Kpi("Чистая прибыль", G_PROFIT, "млн ₽", KIND_FLOW, BETTER_UP, (3300, 2200, 1900, 500, 1100), 0.006, "profit", 0.035),
    Kpi("Cost/Income", G_PROFIT, "%", KIND_RATIO, BETTER_DOWN, (45, 38, 36, 60, 48), -0.001, "none", 0.012),
    Kpi("Выдачи розничных кредитов", G_LENDING, "млрд ₽", KIND_FLOW, BETTER_UP, (16.2, 11.7, 5.4, 8.1, 3.6), 0.010, "lending", 0.03),
    Kpi("Розничный кредитный портфель", G_LENDING, "млрд ₽", KIND_STOCK, BETTER_UP, (0,) * 5),  # считается из выдач
    Kpi("Доля одобрения заявок", G_LENDING, "%", KIND_RATIO, BETTER_UP, (52, 58, 62, 41, 38), 0.0, "none", 0.02),
    Kpi("Доля просрочки 30+ в рознице", G_QUALITY, "%", KIND_RATIO, BETTER_DOWN, (4.0, 2.2, 1.0, 6.5, 7.5), 0.0, "none", 0.03),
    Kpi("Стоимость риска", G_QUALITY, "%", KIND_RATIO, BETTER_DOWN, (2.4, 1.4, 0.8, 3.8, 4.5), 0.0, "none", 0.03),
    Kpi("Чистая процентная маржа", G_NII, "%", KIND_RATIO, BETTER_UP, (5.6, 4.8, 3.9, 6.8, 6.2), 0.0, "none", 0.015),
    Kpi("Чистый процентный доход", G_NII, "млн ₽", KIND_FLOW, BETTER_UP, (14000, 9500, 5500, 3000, 6000), 0.006, "profit", 0.03),
    Kpi("Стоимость фондирования", G_NII, "%", KIND_RATIO, BETTER_DOWN, (8.4, 6.2, 8.9, 8.0, 8.6), 0.0, "none", 0.012),
    Kpi("Комиссионный доход", G_FEES, "млн ₽", KIND_FLOW, BETTER_UP, (3400, 2300, 3200, 600, 1900), 0.008, "spending", 0.03),
    Kpi("Оборот по картам", G_FEES, "млрд ₽", KIND_FLOW, BETTER_UP, (110, 82, 90, 36, 32), 0.010, "spending", 0.03),
    Kpi("Активные клиенты", G_CLIENTS, "тыс. клиентов", KIND_STOCK, BETTER_UP, (1900, 1500, 250, 700, 450), 0.003, "none", 0.004),
    Kpi("Отток клиентов", G_CLIENTS, "% в месяц", KIND_RATIO, BETTER_DOWN, (1.2, 0.7, 0.6, 1.8, 1.4), 0.0, "none", 0.05),
    Kpi("NPS", G_CLIENTS, "пункты", KIND_RATIO, BETTER_UP, (38, 45, 58, 36, 34), 0.0, "none", 0.03),
    Kpi("Остатки на счетах и вкладах физлиц", G_FUNDING, "млрд ₽", KIND_STOCK, BETTER_UP, (780, 520, 560, 90, 150), 0.007, "deposit", 0.006),
    Kpi("Доля цифровых продаж", G_DIGITAL, "%", KIND_RATIO, BETTER_UP, (66, 71, 58, 85, 60), 0.004, "none", 0.015),
    Kpi("Время решения по кредиту", G_DIGITAL, "мин", KIND_RATIO, BETTER_DOWN, (15, 12, 25, 6, 30), 0.0, "none", 0.04),
]
KPI_BY_NAME = {k.name: k for k in KPIS}
PORTFOLIO = "Розничный кредитный портфель"
ISSUANCE = "Выдачи розничных кредитов"

LINKS = [  # причина, следствие, направление влияния
    ("Доля одобрения заявок", ISSUANCE, DIR_POSITIVE),
    ("Время решения по кредиту", ISSUANCE, DIR_NEGATIVE),
    ("Время решения по кредиту", "Доля просрочки 30+ в рознице", DIR_NEGATIVE),
    ("Доля цифровых продаж", ISSUANCE, DIR_POSITIVE),
    (ISSUANCE, PORTFOLIO, DIR_POSITIVE),
    (PORTFOLIO, "Чистый процентный доход", DIR_POSITIVE),
    ("Доля одобрения заявок", "Доля просрочки 30+ в рознице", DIR_POSITIVE),
    ("Доля просрочки 30+ в рознице", "Стоимость риска", DIR_POSITIVE),
    ("Стоимость риска", "Чистая прибыль", DIR_NEGATIVE),
    ("Чистый процентный доход", "Чистая прибыль", DIR_POSITIVE),
    ("Комиссионный доход", "Чистая прибыль", DIR_POSITIVE),
    ("Оборот по картам", "Комиссионный доход", DIR_POSITIVE),
    ("NPS", "Отток клиентов", DIR_NEGATIVE),
    ("Отток клиентов", "Активные клиенты", DIR_NEGATIVE),
    ("Активные клиенты", "Оборот по картам", DIR_POSITIVE),
    ("Отток клиентов", "Остатки на счетах и вкладах физлиц", DIR_NEGATIVE),
    ("Активные клиенты", "Остатки на счетах и вкладах физлиц", DIR_POSITIVE),
    ("Остатки на счетах и вкладах физлиц", "Стоимость фондирования", DIR_NEGATIVE),
    ("Стоимость фондирования", "Чистая процентная маржа", DIR_NEGATIVE),
    ("Чистая процентная маржа", "Чистый процентный доход", DIR_POSITIVE),
    ("Cost/Income", "Чистая прибыль", DIR_NEGATIVE),
]


@dataclass(frozen=True)
class Effect:
    """Мультипликативное влияние аномалии на факт KPI в одном сегменте."""

    kpi: str
    segment: str
    start: str  # 'YYYY-MM'
    amplitude: float
    shape: str = "ramp"  # ramp: выход на amplitude за ramp_months; linear: amplitude за каждый месяц
    ramp_months: int = 1

    def factor(self, month_idx: int) -> float:
        k = month_idx - MONTHS.get_loc(pd.Period(self.start, "M"))
        if k < 0:
            return 1.0
        if self.shape == "linear":
            return 1.0 + self.amplitude * (k + 1)
        return 1.0 + self.amplitude * min(1.0, (k + 1) / self.ramp_months)


ANOMALIES = [
    {
        "id": "A1",
        "title": "Отставание по выдачам розничных кредитов из-за сегмента «Молодёжь»",
        "goal": G_LENDING,
        "primary_kpi": ISSUANCE,
        "key_segment": "Молодёжь",
        "cause_label": "Падение одобрения заявок в сегменте «Молодёжь»",
        "hidden_story": "С апреля 2026 ужесточён скоринг для молодёжных кредитов: доля одобрения заявок упала, "
        "выдачи сегмента ушли ниже плана почти вдвое. Остальные сегменты в норме.",
        "driver_kpis_any": ["Доля одобрения заявок"],
        "period_start": "2026-04",
        "effects": [
            Effect("Доля одобрения заявок", "Молодёжь", "2026-04", -0.28, "ramp", 2),
            Effect(ISSUANCE, "Молодёжь", "2026-04", -0.50, "ramp", 2),
        ],
    },
    {
        "id": "A2",
        "title": "Рост просрочки 30+ в сегменте «Самозанятые и ИП» после упрощения проверки заёмщиков",
        "goal": G_QUALITY,
        "primary_kpi": "Доля просрочки 30+ в рознице",
        "key_segment": "Самозанятые и ИП",
        "cause_label": "Сокращение времени решения по кредиту (упрощённая проверка) в сегменте «Самозанятые и ИП» "
        "и последующий рост просрочки",
        "hidden_story": "С февраля 2026 для самозанятых и ИП упрощена проверка заёмщиков: время решения "
        "сократилось на 40%. С лагом в три месяца выросли просрочка 30+ и стоимость риска — побочный эффект "
        "ускорения выдач.",
        "driver_kpis_any": ["Время решения по кредиту"],
        "period_start": "2026-05",
        "effects": [
            Effect("Время решения по кредиту", "Самозанятые и ИП", "2026-02", -0.40, "ramp", 2),
            Effect("Доля просрочки 30+ в рознице", "Самозанятые и ИП", "2026-05", 0.90, "ramp", 3),
            Effect("Стоимость риска", "Самозанятые и ИП", "2026-05", 0.60, "ramp", 3),
        ],
    },
    {
        "id": "A3",
        "title": "Недобор комиссионного дохода из-за падения оборота по картам в сегменте «Премиум»",
        "goal": G_FEES,
        "primary_kpi": "Комиссионный доход",
        "key_segment": "Премиум",
        "cause_label": "Падение оборота по картам в сегменте «Премиум»",
        "hidden_story": "С мая 2026 изменены условия кешбэка по премиальным картам: клиенты Премиум перенесли "
        "траты на карты других банков, оборот по картам и комиссии сегмента просели, NPS снизился.",
        "driver_kpis_any": ["Оборот по картам"],
        "period_start": "2026-05",
        "effects": [
            Effect("Оборот по картам", "Премиум", "2026-05", -0.38, "ramp", 1),
            Effect("Комиссионный доход", "Премиум", "2026-05", -0.32, "ramp", 1),
            Effect("NPS", "Премиум", "2026-05", -0.12, "ramp", 2),
        ],
    },
    {
        "id": "A4",
        "title": "Рост оттока клиентов в сегменте «Зарплатные»",
        "goal": G_CLIENTS,
        "primary_kpi": "Отток клиентов",
        "key_segment": "Зарплатные",
        "cause_label": "Рост оттока в сегменте «Зарплатные» с падением числа активных клиентов и остатков",
        "hidden_story": "С февраля 2026 несколько крупных работодателей перевели зарплатные проекты в другой банк: "
        "отток в сегменте вырос, число активных клиентов и остатки снижаются, фондирование дорожает.",
        "driver_kpis_any": ["Активные клиенты", "Остатки на счетах и вкладах физлиц"],
        "period_start": "2026-02",
        "effects": [
            Effect("Отток клиентов", "Зарплатные", "2026-02", 1.20, "ramp", 2),
            Effect("Активные клиенты", "Зарплатные", "2026-02", -0.012, "linear"),
            Effect("Остатки на счетах и вкладах физлиц", "Зарплатные", "2026-02", -0.020, "linear"),
            Effect("Стоимость фондирования", "Зарплатные", "2026-02", 0.020, "linear"),
        ],
    },
]
CONTROL_KPIS = ["Cost/Income", "Доля цифровых продаж", "Чистая процентная маржа", "Чистый процентный доход"]


def _season_vector(profile: str) -> np.ndarray:
    raw = np.array(SEASON[profile], dtype=float)
    return (raw / raw.mean())[MONTHS.month - 1]


def _ar1_noise(rng: np.random.Generator, sd: float, n: int, phi: float = 0.4) -> np.ndarray:
    eps = np.zeros(n)
    shocks = rng.normal(0.0, sd * np.sqrt(1 - phi**2), n)
    eps[0] = rng.normal(0.0, sd)
    for t in range(1, n):
        eps[t] = phi * eps[t - 1] + shocks[t]
    return eps


def _effects_factor(effects: list[Effect], kpi: str, segment: str) -> np.ndarray:
    factor = np.ones(len(MONTHS))
    for e in effects:
        if e.kpi == kpi and e.segment == segment:
            factor *= np.array([e.factor(i) for i in range(len(MONTHS))])
    return factor


def _generate_series(seed: int = SEED) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """(kpi, segment) -> (план, факт) по месяцам."""
    rng = np.random.default_rng(seed)
    effects = [e for a in ANOMALIES for e in a["effects"]]
    t = np.arange(len(MONTHS))
    series: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for kpi in KPIS:
        if kpi.name == PORTFOLIO:
            continue
        bias = 0.01 if kpi.better == BETTER_UP else -0.01  # план чуть амбициознее факта
        for i, seg in enumerate(SEGMENTS):
            base = kpi.base[i] * (1 + kpi.trend) ** t * _season_vector(kpi.season)
            eps = _ar1_noise(rng, kpi.noise, len(MONTHS))
            plan = base * (1 + bias)
            fact = base * (1 + eps) * _effects_factor(effects, kpi.name, seg)
            series[(kpi.name, seg)] = (plan, fact)

    # портфель: P[t] = P[t-1]*(1-a) + выдачи[t]; план и факт считаются из своих выдач
    for seg in SEGMENTS:
        v_plan, v_fact = series[(ISSUANCE, seg)]
        p_plan, p_fact = np.zeros(len(MONTHS)), np.zeros(len(MONTHS))
        p_plan[0] = v_plan[0] / AMORTIZATION
        p_fact[0] = p_plan[0] * (1 + rng.normal(0, 0.003))
        for k in range(1, len(MONTHS)):
            p_plan[k] = p_plan[k - 1] * (1 - AMORTIZATION) + v_plan[k]
            p_fact[k] = p_fact[k - 1] * (1 - AMORTIZATION) + v_fact[k]
        series[(PORTFOLIO, seg)] = (p_plan, p_fact)
    return series


def build_frames(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Три листа в русских названиях колонок, как в xlsx."""
    series = _generate_series(seed)
    goal_info = {g: (parent, owner) for g, parent, owner in GOALS}
    goals = pd.DataFrame(
        [
            {
                "цель": k.goal,
                "родительская цель": goal_info[k.goal][0],
                "владелец": goal_info[k.goal][1],
                "KPI": k.name,
                "единица": k.unit,
                "тип": k.kind,
                "лучше": k.better,
            }
            for k in KPIS
        ]
    )
    rows = []
    for kpi in KPIS:
        for seg in SEGMENTS:
            plan, fact = series[(kpi.name, seg)]
            for m, p, f in zip(MONTHS, plan, fact):
                rows.append((kpi.name, m.to_timestamp(), seg, round(float(p), 2), round(float(f), 2)))
    facts = pd.DataFrame(rows, columns=list(FACTS_COLUMNS))
    links = pd.DataFrame(LINKS, columns=list(LINKS_COLUMNS))
    assert list(goals.columns) == list(GOALS_COLUMNS)
    return goals, facts, links


# ---------- эталон для evals: независимый расчёт (не использует пакет tools) ----------


def _segment_values(sub: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == KIND_FLOW:
        return sub.groupby("segment")[["plan", "fact"]].sum()
    if kind == KIND_STOCK:
        return sub[sub["month"] == sub["month"].max()].set_index("segment")[["plan", "fact"]]
    return sub.groupby("segment")[["plan", "fact"]].mean()


def reference_stats(facts: pd.DataFrame, kpi: str, start: str, end: str, key_segment: str | None = None) -> dict:
    """Эталонные отклонение и вклады сегментов за [start, end]; округление как в инструментах."""
    meta = KPI_BY_NAME[kpi]
    sub = facts[(facts["kpi"] == kpi) & (facts["month"] >= start) & (facts["month"] <= end)]
    seg = _segment_values(sub, meta.kind)
    n = len(seg)
    agg = (lambda s: s.sum()) if meta.kind != KIND_RATIO else (lambda s: s.mean())
    plan, fact = agg(seg["plan"]), agg(seg["fact"])
    dev = fact - plan
    out = {
        "plan": round(float(plan), 2),
        "fact": round(float(fact), 2),
        "deviation_abs": round(float(dev), 2),
        "deviation_pct": round(float(dev / plan * 100), 1),
    }
    seg_dev = seg["fact"] - seg["plan"]
    out["segment_deviation_pct"] = {s: round(float(seg_dev[s] / seg["plan"][s] * 100), 1) for s in seg.index}
    out["segment_contribution_pct"] = {s: round(float(seg_dev[s] / seg_dev.sum() * 100), 1) for s in seg.index}
    share = n if meta.kind == KIND_RATIO else 1
    out["segment_contribution_abs"] = {s: round(float(seg_dev[s] / share), 2) for s in seg.index}
    if key_segment:
        out["key_segment_contribution_pct"] = out["segment_contribution_pct"][key_segment]
        out["key_segment_deviation_pct"] = out["segment_deviation_pct"][key_segment]
    return out


def _window_end(offset: int) -> str:
    return str(MONTHS[-1 - offset])


def _checkpoint(facts: pd.DataFrame, anomaly: dict, start: str, end: str) -> dict:
    seg = anomaly["key_segment"]
    primary = reference_stats(facts, anomaly["primary_kpi"], start, end, seg)
    # главный сегмент в данных должен совпасть с заложенным — иначе аномалия неудачно подобрана
    top = max(primary["segment_contribution_pct"], key=primary["segment_contribution_pct"].get)
    assert top == seg, f"{anomaly['id']} {start}..{end}: главный сегмент {top}, ожидался {seg}"
    expected = {
        "deviation_abs": primary["deviation_abs"],
        "deviation_pct": primary["deviation_pct"],
        "key_segment_contribution_pct": primary["key_segment_contribution_pct"],
        "key_segment_deviation_pct": primary["key_segment_deviation_pct"],
        "s1_effect_abs": abs(primary["segment_contribution_abs"][seg]),
    }
    drivers = {}
    for name in anomaly["driver_kpis_any"]:
        d = reference_stats(facts, name, start, end, seg)
        drivers[name] = {
            "deviation_pct": d["deviation_pct"],
            "key_segment_deviation_pct": d["key_segment_deviation_pct"],
        }
    return {"from": start, "to": end, "plan": primary["plan"], "fact": primary["fact"], "expected": expected,
            "drivers": drivers}


def build_anomalies(facts_ts: pd.DataFrame) -> dict:
    facts = facts_ts.copy()
    facts.columns = list(FACTS_COLUMNS.values())
    facts["month"] = facts["month"].dt.strftime("%Y-%m")
    owners = {g: o for g, _, o in GOALS}
    out = []
    for a in ANOMALIES:
        windows = {
            "full": (a["period_start"], _window_end(0)),
            "last_6m": (_window_end(5), _window_end(0)),
            "last_3m": (_window_end(2), _window_end(0)),
        }
        out.append(
            {
                **{k: v for k, v in a.items() if k != "effects"},
                "owner": owners[a["goal"]],
                "effects": [
                    {"kpi": e.kpi, "segment": e.segment, "start": e.start, "amplitude": e.amplitude,
                     "shape": e.shape, "ramp_months": e.ramp_months}
                    for e in a["effects"]
                ],
                "checkpoints": {name: _checkpoint(facts, a, *w) for name, w in windows.items()},
            }
        )
    controls = []
    goal_of = {k.name: k.goal for k in KPIS}
    for i, name in enumerate(CONTROL_KPIS, 1):
        ref = reference_stats(facts, name, _window_end(5), _window_end(0))
        assert abs(ref["deviation_pct"]) < SIGNIFICANT_PCT, f"контрольный KPI {name} отклоняется: {ref['deviation_pct']}"
        controls.append({"id": f"C{i}", "kpi": name, "goal": goal_of[name],
                         "checkpoints": {"last_6m": {"from": _window_end(5), "to": _window_end(0),
                                                     "expected": {"deviation_pct": ref["deviation_pct"],
                                                                  "deviation_abs": ref["deviation_abs"]}}}})
    return {
        "meta": {"seed": SEED, "first_month": FIRST_MONTH, "last_month": LAST_MONTH,
                 "significant_deviation_pct": SIGNIFICANT_PCT, "default_window_months": 6},
        "anomalies": out,
        "controls": controls,
    }


def write_workbook(path: Path, goals: pd.DataFrame, facts: pd.DataFrame, links: pd.DataFrame) -> None:
    with pd.ExcelWriter(path, engine="openpyxl", datetime_format="yyyy-mm") as xw:
        for sheet, df in ((SHEET_GOALS, goals), (SHEET_FACTS, facts), (SHEET_LINKS, links)):
            df.to_excel(xw, sheet_name=sheet, index=False)
            ws = xw.sheets[sheet]
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 60)
            ws.freeze_panes = "A2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Сгенерировать kpi.xlsx и anomalies.json")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="каталог для файлов (по умолчанию data/)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    goals, facts, links = build_frames()
    write_workbook(args.out / "kpi.xlsx", goals, facts, links)
    anomalies = build_anomalies(facts)
    (args.out / "anomalies.json").write_text(json.dumps(anomalies, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"Записано: {args.out / 'kpi.xlsx'} ({len(facts)} строк facts, {len(goals)} KPI), "
          f"{args.out / 'anomalies.json'} ({len(anomalies['anomalies'])} аномалии, "
          f"{len(anomalies['controls'])} контрольных KPI)")


if __name__ == "__main__":
    main()
