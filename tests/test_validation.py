import pandas as pd
import pytest

from data import generate
from data.validation import WorkbookError, load_dataset


@pytest.fixture
def frames():
    goals, facts, links = generate.build_frames()
    return goals.copy(), facts.copy(), links.copy()


def write(tmp_path, goals, facts, links, name="x.xlsx", sheets=("goals", "facts", "links")):
    path = tmp_path / name
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for sheet, df in zip(("goals", "facts", "links"), (goals, facts, links)):
            if sheet in sheets:
                df.to_excel(xw, sheet_name=sheet, index=False)
    return path


def errors_of(path) -> str:
    with pytest.raises(WorkbookError) as exc:
        load_dataset(path)
    return "\n".join(exc.value.errors)


def test_valid_file_loads(tmp_path, frames):
    ds = load_dataset(write(tmp_path, *frames))
    assert ds.months[0] == "2025-04" and ds.months[-1] == "2026-09"


def test_not_an_excel_file(tmp_path):
    bad = tmp_path / "broken.xlsx"
    bad.write_bytes("это не xlsx".encode("utf-8"))
    assert "Не удалось открыть файл как Excel" in errors_of(bad)


def test_missing_sheet(tmp_path, frames):
    msg = errors_of(write(tmp_path, *frames, sheets=("goals", "facts")))
    assert "Нет листа «links»" in msg and "«goals»" in msg  # перечислены найденные листы


def test_missing_column(tmp_path, frames):
    goals, facts, links = frames
    msg = errors_of(write(tmp_path, goals, facts.drop(columns=["план"]), links))
    assert "Лист «facts»" in msg and "нет колонок «план»" in msg and "Найденные колонки" in msg


def test_blank_cells_report_rows(tmp_path, frames):
    goals, facts, links = frames
    facts["факт"] = facts["факт"].astype(object)
    facts.loc[[3, 7, 11], "факт"] = None
    msg = errors_of(write(tmp_path, goals, facts, links))
    assert "колонка «факт»: пустые ячейки в строках 5, 9, 13 (всего 3)" in msg


def test_text_instead_of_number(tmp_path, frames):
    goals, facts, links = frames
    facts["план"] = facts["план"].astype(object)
    facts.loc[2, "план"] = "12,5"
    msg = errors_of(write(tmp_path, goals, facts, links))
    assert "нужно число" in msg and "«12,5»" in msg and "строках 4" in msg


def test_unparseable_month(tmp_path, frames):
    goals, facts, links = frames
    facts["месяц"] = facts["месяц"].astype(object)
    facts.loc[0, "месяц"] = "апрель"
    assert "не удалось распознать месяц" in errors_of(write(tmp_path, goals, facts, links))


def test_text_months_are_accepted(tmp_path, frames):
    goals, facts, links = frames
    facts["месяц"] = facts["месяц"].dt.strftime("%Y-%m")
    assert load_dataset(write(tmp_path, goals, facts, links)).months[0] == "2025-04"


def test_unknown_kpi_in_facts_suggests_close_name(tmp_path, frames):
    goals, facts, links = frames
    facts["KPI"] = facts["KPI"].replace("Оборот по картам", "Оборот по карточкам")
    msg = errors_of(write(tmp_path, goals, facts, links))
    assert "«Оборот по карточкам»" in msg and "не описан на листе «goals»" in msg
    assert "Возможно, имелось в виду «Оборот по картам»" in msg


def test_duplicate_fact_rows(tmp_path, frames):
    goals, facts, links = frames
    facts = pd.concat([facts, facts.iloc[[0]]], ignore_index=True)
    assert "повторяется комбинация" in errors_of(write(tmp_path, goals, facts, links))


def test_goals_semantic_errors(tmp_path, frames):
    goals, facts, links = frames
    goals["тип"] = goals["тип"].replace({"поток": "оборот"})
    msg = errors_of(write(tmp_path, goals, facts, links))
    assert "колонка «тип»" in msg and "Допустимо: поток, остаток, доля" in msg

    goals, facts, links = frames
    goals.loc[goals["цель"] == "Рост розничного кредитования", "родительская цель"] = "Несуществующая цель"
    assert "родительская цель «Несуществующая цель»" in errors_of(write(tmp_path, goals, facts, links))


def test_goal_hierarchy_cycle(tmp_path, frames):
    goals, facts, links = frames
    goals.loc[goals["родительская цель"].isna(), "родительская цель"] = "Рост розничного кредитования"
    msg = errors_of(write(tmp_path, goals, facts, links))
    assert "зациклена" in msg


def test_links_errors(tmp_path, frames):
    goals, facts, links = frames
    links.loc[0, "влияние"] = "сильно"
    links.loc[1, "KPI-следствие"] = "Чудо-показатель"
    msg = errors_of(write(tmp_path, goals, facts, links))
    assert "недопустимое значение «сильно»" in msg and "KPI «Чудо-показатель»" in msg


def test_all_problems_reported_together(tmp_path, frames):
    goals, facts, links = frames
    facts["факт"] = facts["факт"].astype(object)
    facts.loc[0, "факт"] = None
    goals["лучше"] = goals["лучше"].replace({"больше": "много"})
    msg = errors_of(write(tmp_path, goals, facts, links))
    assert "колонка «факт»" in msg and "колонка «лучше»" in msg


def test_bytes_source_is_supported(frames, tmp_path):
    path = write(tmp_path, *frames)
    assert load_dataset(path.read_bytes()).months


def test_zip_bomb_is_rejected_before_parsing(tmp_path, monkeypatch):
    import zipfile

    from data import validation

    monkeypatch.setattr(validation, "MAX_UNPACKED_BYTES", 1024 * 1024)
    bomb = tmp_path / "bomb.xlsx"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/worksheets/sheet1.xml", b"0" * (3 * 1024 * 1024))  # 3 МБ нулей сжимаются в килобайты
    assert bomb.stat().st_size < 100_000
    msg = errors_of(bomb)
    assert "слишком большой после распаковки" in msg
