import pandas as pd
import pytest

from data.readers import NOTE_SHEET_RE, pick_sheet, read_file
from data.validation import WorkbookError
from tests.helpers import xlsx_book as book


DATA = [["Дата", "Канал", "Расходы"], ["2026-01-01", "Поиск", 100], ["2026-01-02", "Поиск", 120], ["2026-01-03", "Email", 90]]


def test_plain_xlsx_is_read_with_header_in_first_row():
    raw = read_file(book({"Данные": DATA}), "report.xlsx")
    assert raw.kind == "xlsx" and raw.filename == "report.xlsx" and raw.sheet == "Данные"
    df = raw.frames["Данные"]
    assert list(df.columns) == ["Дата", "Канал", "Расходы"] and len(df) == 3
    assert raw.header_rows["Данные"] == 1 and df["Расходы"].tolist() == [100, 120, 90]


def test_title_rows_above_the_header_are_skipped_and_kept_as_notes():
    rows = [["Отчёт по расходам за январь"], ["Подготовил: финансовый отдел"], [None], *DATA]
    raw = read_file(book({"Лист1": rows}), "r.xlsx")
    assert raw.header_rows["Лист1"] == 4  # номер строки в файле, нужен для сообщений об ошибках
    assert list(raw.frames["Лист1"].columns) == ["Дата", "Канал", "Расходы"] and len(raw.frames["Лист1"]) == 3
    assert "Отчёт по расходам за январь" in raw.notes and "Подготовил: финансовый отдел" in raw.notes


def test_notes_sheet_is_collected_for_the_user_and_never_chosen_as_data():
    notes = [["Описание"], ["Расходы указаны в рублях без НДС"], ["Источник: рекламный кабинет"], ["Обновляется раз в неделю"]]
    raw = read_file(book({"О файле": notes, "Данные": DATA}), "r.xlsx")
    assert raw.sheet == "Данные"
    assert "Расходы указаны в рублях без НДС" in raw.notes and "Источник: рекламный кабинет" in raw.notes
    # заметки не должны попадать в таблицы данных
    assert all("рублях без НДС" not in str(v) for v in raw.frames["Данные"].to_numpy().ravel())


def test_note_like_sheet_names():
    for name in ("О файле", "Описание", "Справка", "README", "Notes", "Легенда", "Инструкция"):
        assert NOTE_SHEET_RE.search(name), name
    for name in ("Данные", "Продажи", "Лист1", "Факты"):
        assert not NOTE_SHEET_RE.search(name), name


def test_biggest_sheet_wins_and_all_sheets_are_listed():
    big = [["День", "Регион", "Продажи"]] + [[f"2026-01-{d:02d}", "Север", d] for d in range(1, 21)]
    raw = read_file(book({"Малый": DATA, "Большой": big}), "r.xlsx")
    assert raw.sheet == "Большой" and set(raw.sheets) == {"Малый", "Большой"}
    assert pick_sheet(raw) == "Большой"


def test_number_formats_of_the_first_data_row_are_exposed():
    rows = [["Месяц", "Доля", "Выручка"], ["2026-01", 0.25, 1000], ["2026-02", 0.3, 1100], ["2026-03", 0.2, 900]]
    raw = read_file(book({"Данные": rows}, {("Данные", "B", 2): "0.0%", ("Данные", "C", 2): '#,##0" ₽"'}), "r.xlsx")
    fmts = raw.formats["Данные"]
    assert fmts["Доля"] == "0.0%" and "₽" in fmts["Выручка"]


def test_duplicate_and_empty_headers_get_unique_names():
    rows = [["Дата", "Сумма", "Сумма", None], ["2026-01-01", 1, 2, 3], ["2026-01-02", 4, 5, 6], ["2026-01-03", 7, 8, 9]]
    df = read_file(book({"Данные": rows}), "r.xlsx").frames["Данные"]
    assert list(df.columns) == ["Дата", "Сумма", "Сумма (2)", "Столбец 4"]


def test_blank_rows_and_columns_are_dropped():
    rows = [["Дата", "Канал", None, "Расходы"], ["2026-01-01", "Поиск", None, 1], [None, None, None, None],
            ["2026-01-02", "Email", None, 2], ["2026-01-03", "Email", None, 3]]
    df = read_file(book({"Данные": rows}), "r.xlsx").frames["Данные"]
    assert list(df.columns) == ["Дата", "Канал", "Расходы"] and len(df) == 3


# ---------- csv ----------


def test_utf8_csv_with_commas():
    raw = read_file("Дата,Канал,Расходы\n2026-01-01,Поиск,100\n2026-01-02,Email,120\n2026-01-03,Email,90\n".encode("utf-8-sig"),
                    "data.csv")
    assert raw.kind == "csv" and raw.sheet == "data"
    assert list(raw.frames["data"].columns) == ["Дата", "Канал", "Расходы"] and len(raw.frames["data"]) == 3


def test_cp1251_csv_with_semicolons_and_decimal_commas():
    text = "Дата;Канал;Расходы\n01.01.2026;Поиск;100,5\n02.01.2026;Email;120,25\n03.01.2026;Email;90\n"
    raw = read_file(text.encode("cp1251"), "Выгрузка.csv")
    df = raw.frames["Выгрузка"]
    assert list(df.columns) == ["Дата", "Канал", "Расходы"]
    assert df["Канал"].tolist() == ["Поиск", "Email", "Email"] and df["Расходы"].tolist() == ["100,5", "120,25", "90"]


def test_csv_title_rows_above_the_header():
    text = "Выгрузка за январь\n\nДата;Канал;Расходы\n2026-01-01;Поиск;1\n2026-01-02;Email;2\n2026-01-03;Email;3\n"
    raw = read_file(text.encode("utf-8"), "r.csv")
    assert raw.header_rows["r"] == 3 and len(raw.frames["r"]) == 3


# ---------- ошибки ----------


def test_unsupported_and_broken_files_get_russian_messages():
    with pytest.raises(WorkbookError, match="устарел"):
        read_file(b"x", "old.xls")
    with pytest.raises(WorkbookError, match="не в поддерживаемом формате"):
        read_file(b"x", "data.txt")
    with pytest.raises(WorkbookError, match="Не удалось открыть файл как Excel"):
        read_file(b"PK\x03\x04" + b"\x00" * 50, "broken.xlsx")
    with pytest.raises(WorkbookError, match="слишком мало данных"):
        read_file(b"a,b\n1,2\n", "tiny.csv")
    with pytest.raises(WorkbookError, match="не найдено ни одной таблицы"):
        read_file(book({"Пусто": [[None, None]], "Одна": [["заголовок", "x"], [1, 2]]}), "empty.xlsx")


def test_a_book_of_one_column_has_no_table():
    with pytest.raises(WorkbookError, match="не найдено ни одной таблицы"):
        read_file(book({"Список": [["Имя"], ["Анна"], ["Борис"], ["Вера"]]}), "list.xlsx")


def test_frames_are_plain_dataframes_with_object_cells():
    raw = read_file(book({"Данные": DATA}), "r.xlsx")
    assert isinstance(raw.frames["Данные"], pd.DataFrame) and raw.frames["Данные"].index.tolist() == [0, 1, 2]
