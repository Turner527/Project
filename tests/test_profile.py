import json

import pandas as pd
import pytest

from data import generate_samples
from data.model import Spec, build_table
from data.profile import better_by_name, profile, split_unit
from data.readers import read_file
from tests.helpers import xlsx_book


def profiled(header, rows, formats=None, sheet="Данные"):
    raw = read_file(xlsx_book({sheet: [header, *rows]}, formats), "t.xlsx")
    return profile(raw), raw


def roles(report):
    return {c["name"]: c["role"] for c in report["columns"]}


def metrics(report):
    return {m["name"]: m for m in report["spec"]["metrics"]}


def sales_rows(days=6):
    regions = ["Север", "Юг", "Запад", "Восток"]
    return [[f"2026-03-{d:02d}", region, 100 + 10 * i + d, 5 + i] for d in range(1, days + 1) for i, region in enumerate(regions)]


# ---------- роли столбцов ----------


def test_typical_table_gets_date_dimension_and_metrics():
    report, _ = profiled(["Дата", "Регион", "Выручка", "Сделки"], sales_rows())
    assert roles(report) == {"Дата": "date", "Регион": "dimension", "Выручка": "metric", "Сделки": "metric"}
    spec = report["spec"]
    assert spec["date"] == "Дата" and spec["dimensions"] == ["Регион"] and spec["sheet"] == "Данные"
    assert report["grain"] == "day" and report["period"] == {"from": "2026-03-01", "to": "2026-03-06", "periods": 6}
    assert report["rows"] == 24 and report["header_row"] == 1 and report["sheets"] == ["Данные"]


def test_text_dates_are_recognised():
    rows = [[f"{d:02d}.03.2026", "Север", d] for d in range(1, 8)]
    report, _ = profiled(["Дата", "Регион", "Продажи"], rows)
    assert roles(report)["Дата"] == "date" and report["grain"] == "day"


def test_month_names_are_recognised_as_monthly_dates():
    months = ["январь 2026", "февраль 2026", "март 2026", "апрель 2026"]
    report, _ = profiled(["Месяц", "Отдел", "Продажи"], [[m, o, 10 + i] for i, m in enumerate(months) for o in ("А", "Б")])
    assert roles(report)["Месяц"] == "date" and report["grain"] == "month"


def test_second_date_column_is_ignored_with_a_reason():
    rows = [[f"2026-03-{d:02d}", f"2026-04-{d:02d}", "Север", d] for d in range(1, 8)]
    report, _ = profiled(["Дата заказа", "Дата оплаты", "Регион", "Сумма"], rows)
    assert roles(report)["Дата заказа"] == "date" and roles(report)["Дата оплаты"] == "ignore"
    ignored = next(c for c in report["columns"] if c["name"] == "Дата оплаты")
    assert "ещё один столбец с датами" in ignored["reason"] and "Дата заказа" in ignored["reason"]


def test_identifiers_constants_and_calendar_columns_are_ignored():
    rows = [[i, f"2026-03-{(i % 6) + 1:02d}", "Север" if i % 2 else "Юг", "RUB", 2026, 1000 + 7 * i * i % 13 + i]
            for i in range(1, 31)]
    report, _ = profiled(["ID", "Дата", "Регион", "Валюта", "Год", "Выручка"], rows)
    r = roles(report)
    assert r["ID"] == r["Валюта"] == r["Год"] == "ignore" and r["Выручка"] == "metric"
    reasons = {c["name"]: c["reason"] for c in report["columns"]}
    assert "идентификатор" in reasons["ID"] and "одно и то же значение" in reasons["Валюта"]
    assert "календарный" in reasons["Год"]


def test_sequential_integers_are_ids_but_unique_amounts_are_not():
    rows = [[i, 1000 + 37 * i] for i in range(1, 31)]  # первый столбец идёт подряд, второй — уникальные суммы
    report, _ = profiled(["Строка", "Сумма"], rows)
    assert roles(report) == {"Строка": "ignore", "Сумма": "metric"}


def test_long_unique_text_is_not_a_dimension():
    rows = [[f"2026-03-{(i % 5) + 1:02d}", f"Комментарий номер {i}", 10 + i] for i in range(60)]
    report, _ = profiled(["Дата", "Комментарий", "Сумма"], rows)
    assert roles(report)["Комментарий"] == "ignore" and "идентификатор или комментарий" in next(
        c["reason"] for c in report["columns"] if c["name"] == "Комментарий")


def test_currency_symbol_goes_only_to_money_columns():
    rows = [[f"2026-03-{d:02d}", "Север", "RUB", 1000 + d, 3 + d] for d in range(1, 8)]
    report, _ = profiled(["Дата", "Регион", "Валюта", "Выручка", "Сделки"], rows)
    m = metrics(report)
    assert m["Выручка"]["unit"] == "₽" and m["Сделки"]["unit"] == ""


def test_units_from_column_names_and_excel_formats():
    rows = [["2026-01", "А", 1000, 50, 0.25], ["2026-02", "А", 1100, 55, 0.3], ["2026-03", "А", 900, 60, 0.2]]
    report, _ = profiled(["Месяц", "Отдел", "Оборот, тыс. руб", "Время (мин)", "Доля брака"], rows,
                         {("Данные", "E", 2): "0.0%"})
    m = metrics(report)
    assert m["Оборот"]["unit"] == "тыс. руб" and m["Оборот"]["column"] == "Оборот, тыс. руб"
    assert m["Время"]["unit"] == "мин"
    assert m["Доля брака"]["unit"] == "%" and m["Доля брака"]["kind"] == "mean" and m["Доля брака"]["scale"] == 100.0


def test_percentages_are_averaged_not_summed():
    rows = [[f"2026-03-{d:02d}", "Север", 10 + d, 0.2 + d / 100] for d in range(1, 8)]
    report, _ = profiled(["Дата", "Регион", "Продажи", "Конверсия"], rows)
    conv = metrics(report)["Конверсия"]
    assert conv["kind"] == "mean" and conv["unit"] == "%" and conv["scale"] == 100.0
    assert metrics(report)["Продажи"]["kind"] == "sum"
    assert any("доля или процент" in w for w in report["warnings"])
    rows = [[f"2026-03-{d:02d}", "Север", 10 + d, 12.5 + d] for d in range(1, 8)]
    report, _ = profiled(["Дата", "Регион", "Продажи", "Конверсия, %"], rows)
    conv = metrics(report)["Конверсия"]
    assert conv["kind"] == "mean" and conv["unit"] == "%" and conv["scale"] == 1.0


@pytest.mark.parametrize("name, expected", [("Расходы", "down"), ("Стоимость лида", "down"), ("Время ответа", "down"),
                                            ("Отток клиентов", "down"), ("Выручка", "up"), ("Конверсия", "up"),
                                            ("Обращения", "none"), ("Остаток на счетах", "none")])
def test_direction_is_guessed_from_the_name(name, expected):
    assert better_by_name(name) == expected


@pytest.mark.parametrize("column, expected", [("Расходы, руб", ("Расходы", "руб")), ("Выручка (тыс. руб.)", ("Выручка", "тыс. руб.")),
                                              ("Среднее время, мин", ("Среднее время", "мин")),
                                              ("Лиды", ("Лиды", "")), ("Регион, город", ("Регион, город", ""))])
def test_split_unit(column, expected):
    assert split_unit(column) == expected


# ---------- план и фильтры ----------


def plan_rows():
    return [[f"2026-{m:02d}", o, 100 + m + i, 110 + m, 20 + i, 25 + m] for m in range(1, 7) for i, o in enumerate(("А", "Б"))]


def test_plan_columns_are_paired_with_their_metric():
    report, _ = profiled(["Месяц", "Отдел", "Доход", "План дохода", "Заявки", "План продаж"], plan_rows())
    r = roles(report)
    assert r["План дохода"] == "plan" and next(c for c in report["columns"] if c["name"] == "План дохода")["plan_for"] == "Доход"
    assert metrics(report)["Доход"]["plan_column"] == "План дохода" and "План дохода" not in metrics(report)
    # «План продаж» не относится к «Заявки»: остаётся обычным показателем, пользователь может связать вручную
    assert r["План продаж"] == "metric" and metrics(report)["Заявки"]["plan_column"] is None


def test_a_constant_plan_is_kept_when_it_has_a_metric_but_a_lonely_constant_is_ignored():
    rows = [[f"2026-{m:02d}", o, 40 + m + i, 50, 7] for m in range(1, 7) for i, o in enumerate(("А", "Б"))]
    report, _ = profiled(["Месяц", "Отдел", "NPS", "План NPS", "План прочего"], rows)
    r = roles(report)
    assert r["План NPS"] == "plan" and metrics(report)["NPS"]["plan_column"] == "План NPS"
    assert r["План прочего"] == "ignore"


def test_completeness_flag_becomes_a_row_filter():
    rows = [[f"2026-{m:02d}", o, 100 + m, "Да" if m < 6 else "Нет"] for m in range(1, 7) for o in ("А", "Б")]
    report, raw = profiled(["Месяц", "Отдел", "Доход", "Период закрыт"], rows)
    col = next(c for c in report["columns"] if c["name"] == "Период закрыт")
    assert col["role"] == "filter" and col["keep"] == ["Да"] and col["values"] == ["Да", "Нет"]
    assert report["spec"]["filters"] == [{"column": "Период закрыт", "values": ["Да"]}]
    table = build_table(raw.frames["Данные"], Spec.from_dict(report["spec"]), "t")
    assert table.rows_excluded == 2 and len(table.df) == 10 and table.label(table.p_max) == "2026-05"
    # период в профиле показан уже после фильтра: неполный месяц в анализ не попадёт
    assert report["rows_excluded"] == 2 and report["period"] == {"from": "2026-01", "to": "2026-05", "periods": 5}


def test_an_ordinary_yes_no_dimension_is_not_a_filter():
    rows = [[f"2026-03-{d:02d}", "Да" if d % 2 else "Нет", d] for d in range(1, 9)]
    report, _ = profiled(["Дата", "Новый клиент", "Продажи"], rows)
    assert roles(report)["Новый клиент"] == "dimension" and report["spec"]["filters"] == []


# ---------- отношения ----------


def test_ratio_suggestions_for_the_marketing_sample(tmp_path):
    p = tmp_path / "m.xlsx"
    generate_samples.write_sample("marketing", p)
    report = profile(read_file(p.read_bytes(), "m.xlsx"))
    suggested = {r["name"]: r for r in report["ratios"]}
    assert set(suggested) == {"CTR", "Конверсия в лиды", "Конверсия лидов в заказы", "Стоимость клика", "Стоимость лида",
                              "Стоимость заказа", "ROAS", "Средний чек"}
    assert (suggested["CTR"]["num"], suggested["CTR"]["den"], suggested["CTR"]["scale"]) == ("Клики", "Показы", 100)
    assert suggested["Стоимость лида"]["num"] == "Расходы, руб" and suggested["Стоимость лида"]["better"] == "down"
    in_spec = {m["name"]: m for m in report["spec"]["metrics"]}
    assert all(name in in_spec and in_spec[name]["kind"] == "ratio" for name in suggested)


def test_ratio_suggestions_for_support_and_branches(tmp_path):
    names = {}
    for sample in ("support", "branches"):
        p = tmp_path / f"{sample}.xlsx"
        generate_samples.write_sample(sample, p)
        names[sample] = {r["name"]: r for r in profile(read_file(p.read_bytes(), p.name))["ratios"]}
    assert set(names["support"]) == {"Выполнение SLA", "CSAT", "Среднее время обработки", "Доля решённых обращений"}
    assert {"Конверсия сделок", "Прибыль", "Маржа"} <= set(names["branches"])
    assert names["branches"]["Прибыль"]["column"] == "Доход - Расходы"
    assert names["branches"]["Маржа"]["num"] == "Доход - Расходы" and names["branches"]["Маржа"]["den"] == "Доход"


def test_only_ratios_with_both_columns_present_are_suggested():
    rows = [[f"2026-03-{d:02d}", "А", 1000 + 10 * d, 20 + d] for d in range(1, 8)]
    report, _ = profiled(["Дата", "Канал", "Показы", "Клики"], rows)
    assert [r["name"] for r in report["ratios"]] == ["CTR"]
    report, _ = profiled(["Дата", "Канал", "Показы", "Вызовы"], [[*r[:3], r[3]] for r in rows])
    assert report["ratios"] == []


# ---------- предупреждения о качестве ----------


def test_warnings_about_the_data():
    rows = sales_rows()
    rows += [rows[0][:], [rows[1][0], rows[1][1], None, -5]]  # дубль ключа, пустое значение, отрицательное
    report, _ = profiled(["Дата", "Регион", "Выручка", "Сделки"], rows)
    text = " | ".join(report["warnings"])
    assert "повторяют сочетание" in text and "пустые значения" in text and "отрицательные значения" in text


def test_warning_when_there_is_no_date():
    report, _ = profiled(["Отдел", "Продажи"], [["А", 1], ["Б", 2], ["В", 3], ["Г", 4]])
    assert report["grain"] is None and report["period"] is None and report["spec"]["date"] is None
    assert any("Нет столбца с датой" in w for w in report["warnings"])
    assert any("сравнивать не с чем" in w for w in report["warnings"])


def test_warning_when_plan_and_fact_are_in_one_column():
    rows = [[f"2026-03-{d:02d}", kind, 100 + d] for d in range(1, 6) for kind in ("План", "Факт")]
    report, _ = profiled(["Дата", "Тип", "Сумма"], rows)
    assert any("«план» и «факт»" in w for w in report["warnings"])


def test_warning_about_gaps_in_the_date_series():
    days = ("2026-03-01", "2026-03-02", "2026-03-03", "2026-03-10", "2026-03-11", "2026-03-12")
    report, _ = profiled(["Дата", "Канал", "Сумма"], [[d, "А" if i % 2 else "Б", 10 + i] for i, d in enumerate(days)])
    assert any("пропуски" in w and "6 из 12" in w for w in report["warnings"])


def test_several_currencies_are_flagged():
    rows = [[f"2026-03-{d:02d}", "RUB" if d % 2 else "USD", 100 + d] for d in range(1, 9)]
    report, _ = profiled(["Дата", "Валюта", "Выручка"], rows)
    assert any("несколько валют" in w for w in report["warnings"])


def test_non_numeric_cells_in_a_metric_are_reported_not_hidden():
    rows = sales_rows()
    rows[3][2] = "н/д"
    report, _ = profiled(["Дата", "Регион", "Выручка", "Сделки"], rows)
    assert any("нечисловые значения" in w and "н/д" in w for w in report["warnings"])


def test_a_bad_date_is_reported_with_the_excel_row_number():
    rows = sales_rows()
    rows[2][0] = "вчера"
    report, _ = profiled(["Дата", "Регион", "Выручка", "Сделки"], rows)
    assert any("не удалось распознать значения в строках 4" in w and "вчера" in w for w in report["warnings"])


# ---------- общие свойства ----------


def test_duplicate_metric_names_fall_back_to_column_names():
    rows = [[f"2026-03-{d:02d}", "Север", 100 + d, 5 + d] for d in range(1, 8)]
    report, _ = profiled(["Дата", "Регион", "Затраты, руб", "Затраты, шт"], rows)
    assert set(metrics(report)) == {"Затраты, руб", "Затраты, шт"}


def test_other_sheet_can_be_profiled():
    raw = read_file(xlsx_book({"Первый": [["Дата", "Канал", "Клики"], *[[f"2026-03-0{d}", "А", d] for d in range(1, 5)]],
                               "Второй": [["Месяц", "Отдел", "Выручка", "Расходы"],
                                          *[[f"2026-0{m}", o, 10 * m, 5 * m] for m in range(1, 5) for o in ("Х", "Ю")]]}), "t.xlsx")
    assert raw.sheet == "Второй"  # больше строк × столбцов
    assert profile(raw)["sheet"] == "Второй"
    first = profile(raw, "Первый")
    assert first["sheet"] == "Первый" and first["spec"]["date"] == "Дата" and set(metrics(first)) == {"Клики"}
    assert profile(raw, "Нет такого")["sheet"] == "Второй"  # неизвестный лист — лист по умолчанию


@pytest.mark.parametrize("sample, grain, rows", [("marketing", "day", 488), ("support", "day", 366), ("branches", "month", 360)])
def test_profile_is_json_safe_and_its_spec_builds_a_table(tmp_path, sample, grain, rows):
    path = tmp_path / "s.xlsx"
    generate_samples.write_sample(sample, path)
    raw = read_file(path.read_bytes(), "s.xlsx")
    report = profile(raw)
    json.dumps(report, ensure_ascii=False, allow_nan=False)  # numpy-типы и NaN в профиль попадать не должны
    assert report["grain"] == grain and report["rows"] == rows
    assert {c["role"] for c in report["columns"]} <= {"date", "dimension", "metric", "plan", "filter", "ignore"}
    table = build_table(raw.frames[raw.sheet], Spec.from_dict(report["spec"]), "s.xlsx")
    assert table.grain == grain and table.rows_total == rows
    assert [m.name for m in table.spec.metrics] == [m["name"] for m in report["spec"]["metrics"]]
    assert report["notes"] == ["Описание", generate_samples.SAMPLES[sample][1],
                               "Синтетические данные для демонстрации, не реальные результаты",
                               "Лист «Данные»: одна строка = один день или месяц × разрезы отчёта"]


def test_profile_does_not_modify_the_data():
    report, raw = profiled(["Дата", "Регион", "Выручка"], [[f"2026-03-0{d}", "Север", d] for d in range(1, 6)])
    before = raw.frames["Данные"].copy()
    profile(raw)
    pd.testing.assert_frame_equal(raw.frames["Данные"], before)
