"""Синтетические плоские отчёты для демонстрации и тестов универсального режима (реальные данные не используются).

Запуск:  python -m data.generate_samples   ->  data/samples/{marketing,support,branches}.xlsx

В каждый отчёт заложена аномалия (описана в SAMPLE_SIGNALS), которую должен найти агент:
  * marketing — с 15 сентября в канале «Поиск» растут расходы и падает конверсия клик -> лид;
  * support   — с 10 сентября в очереди «Техническая поддержка» падают SLA и CSAT, растёт время обработки;
  * branches  — в августе-сентябре падает конверсия в отделениях «Москва Север» и «Екатеринбург», с августа растут
                расходы в «Новосибирске»; октябрь неполный (помечен «Период закрыт = Нет»).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent / "samples"
SEED = 20260930

SAMPLE_SIGNALS = {
    "marketing": {"since": "2026-09-15", "channel": "Поиск", "spend_up": 1.45, "lead_conversion_down": 0.62},
    "support": {"since": "2026-09-10", "queue": "Техническая поддержка", "sla_down": 0.80, "csat_down": 0.85,
                "time_up": 1.5},
    "branches": {"conversion_offices": ["Москва Север", "Екатеринбург"], "conversion_down": 0.70,
                 "expense_office": "Новосибирск", "expense_up": 1.30, "since_month": "2026-08"},
}
OFFICES = ["Москва Центр", "Москва Север", "Санкт-Петербург", "Казань", "Екатеринбург", "Новосибирск", "Краснодар",
           "Самара", "Нижний Новгород", "Ростов-на-Дону", "Уфа", "Воронеж"]


def _noise(rng: np.random.Generator, n: int, sd: float) -> np.ndarray:
    return 1 + rng.normal(0, sd, n)


def marketing(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2026-08-01", "2026-09-30", freq="D")
    channels = {"Поиск": (20000, 0.028, 0.10, 0.25), "Социальные сети": (13000, 0.021, 0.08, 0.20),
                "Ретаргетинг": (9000, 0.034, 0.12, 0.30), "Email": (4000, 0.040, 0.09, 0.22)}
    campaigns = {"Новые клиенты": 1.0, "Повторные покупки": 0.9}
    rows = []
    for day in days:
        weekend = 0.9 if day.dayofweek >= 5 else 1.0
        for channel, (spend0, ctr, cvr_lead, cvr_order) in channels.items():
            for campaign, factor in campaigns.items():
                spend = spend0 * factor * weekend * _noise(rng, 1, 0.07)[0]
                impressions = spend * 1.35 * _noise(rng, 1, 0.05)[0]
                clicks = impressions * ctr * _noise(rng, 1, 0.05)[0]
                lead_rate, spend_mult = cvr_lead, 1.0
                if channel == SAMPLE_SIGNALS["marketing"]["channel"] and day >= pd.Timestamp(SAMPLE_SIGNALS["marketing"]["since"]):
                    spend_mult = SAMPLE_SIGNALS["marketing"]["spend_up"]
                    lead_rate *= SAMPLE_SIGNALS["marketing"]["lead_conversion_down"]
                leads = clicks * lead_rate * _noise(rng, 1, 0.06)[0]
                orders = leads * cvr_order * _noise(rng, 1, 0.08)[0]
                rows.append([day, channel, campaign, round(spend * spend_mult), round(impressions), round(clicks),
                             round(leads), round(orders), round(orders * 4200 * _noise(rng, 1, 0.06)[0]), "RUB"])
    return pd.DataFrame(rows, columns=["Дата", "Канал", "Кампания", "Расходы, руб", "Показы", "Клики", "Лиды", "Заказы",
                                       "Выручка, руб", "Валюта"])


def support(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    days = pd.date_range("2026-08-01", "2026-09-30", freq="D")
    queues = {"Общие вопросы": (60, 9.0), "Техническая поддержка": (70, 17.0), "Оплата": (45, 8.0)}
    channels = {"Чат": 1.0, "Телефон": 0.8}
    rows = []
    for day in days:
        for queue, (base, minutes) in queues.items():
            for channel, share in channels.items():
                new = base * share * _noise(rng, 1, 0.08)[0]
                resolved = new * _noise(rng, 1, 0.05)[0]
                sla, csat, handle = 0.95, 0.93, minutes
                if queue == SAMPLE_SIGNALS["support"]["queue"] and day >= pd.Timestamp(SAMPLE_SIGNALS["support"]["since"]):
                    sla *= SAMPLE_SIGNALS["support"]["sla_down"]
                    csat *= SAMPLE_SIGNALS["support"]["csat_down"]
                    handle *= SAMPLE_SIGNALS["support"]["time_up"]
                ratings = resolved * 0.5 * _noise(rng, 1, 0.08)[0]
                rows.append([day, queue, channel, round(new), round(resolved), round(resolved * sla * _noise(rng, 1, 0.02)[0]),
                             round(ratings), round(ratings * csat * _noise(rng, 1, 0.02)[0]),
                             round(resolved * handle * _noise(rng, 1, 0.05)[0])])
    return pd.DataFrame(rows, columns=["Дата", "Очередь", "Канал", "Новые обращения", "Решено", "SLA выполнено",
                                       "Всего оценок", "Положительные оценки", "Минуты обработки"])


def branches(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)
    months = pd.date_range("2026-01-01", "2026-10-01", freq="MS")
    products = {"Карты": 1.0, "Кредиты": 2.4, "Сбережения": 2.0}
    sig = SAMPLE_SIGNALS["branches"]
    rows = []
    for i, month in enumerate(months):
        closed = month < pd.Timestamp("2026-10-01")
        for k, office in enumerate(OFFICES):
            size = 0.7 + 0.12 * ((k * 7) % 9)
            for product, weight in products.items():
                plan = 1_400_000 * size * weight * (1 + 0.01 * i)
                income = plan * _noise(rng, 1, 0.04)[0] * (1.0 if closed else 0.33)
                expenses = income * 0.55 * _noise(rng, 1, 0.03)[0] if closed else plan * 0.18
                inquiries = 1500 * size * _noise(rng, 1, 0.05)[0]
                deals = inquiries * 0.27 * _noise(rng, 1, 0.04)[0]
                after = month >= pd.Timestamp(sig["since_month"] + "-01")
                if after and office in sig["conversion_offices"] and closed:
                    deals *= sig["conversion_down"]
                if after and office == sig["expense_office"] and closed:
                    expenses *= sig["expense_up"]
                rows.append([month, office, product, round(income), round(expenses), round(plan), round(inquiries),
                             round(deals), "RUB", "Да" if closed else "Нет"])
    return pd.DataFrame(rows, columns=["Период", "Отделение", "Продукт", "Доход", "Расходы", "План дохода", "Обращения",
                                       "Сделки", "Валюта", "Период закрыт"])


SAMPLES = {"marketing": (marketing, "Маркетинговые кампании", {"Расходы, руб": '#,##0" ₽"', "Выручка, руб": '#,##0" ₽"'}),
           "support": (support, "Работа службы поддержки", {}),
           "branches": (branches, "Сеть отделений", {"Доход": "#,##0", "Расходы": "#,##0", "План дохода": "#,##0"})}


def write_sample(name: str, path: Path) -> int:
    build, title, formats = SAMPLES[name]
    df = build()
    notes = pd.DataFrame({"Описание": [title, "Синтетические данные для демонстрации, не реальные результаты",
                                       "Лист «Данные»: одна строка = один день или месяц × разрезы отчёта"]})
    with pd.ExcelWriter(path, engine="openpyxl", datetime_format="yyyy-mm-dd") as xw:
        df.to_excel(xw, sheet_name="Данные", index=False)
        notes.to_excel(xw, sheet_name="О файле", index=False)
        ws = xw.sheets["Данные"]
        ws.freeze_panes = "A2"
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(max(len(str(col[0].value)) + 4, 12), 28)
            if col[0].value in formats:
                for cell in col[1:]:
                    cell.number_format = formats[col[0].value]
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Сгенерировать демо-отчёты для универсального режима")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in SAMPLES:
        rows = write_sample(name, args.out / f"{name}.xlsx")
        print(f"Записано: {args.out / (name + '.xlsx')} ({rows} строк)")


if __name__ == "__main__":
    main()
