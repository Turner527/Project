import json
import os
import time

import pytest

from data import generate_samples
from data.validation import WorkbookError
from tests.conftest import DEMO_XLSX
from tests.helpers import xlsx_book
from tools import datastore


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    """Хранилище во временной папке; KPI_DATA_PATH снят, чтобы работал обычный порядок выбора источника."""
    monkeypatch.delenv("KPI_DATA_PATH", raising=False)
    monkeypatch.setattr(datastore, "UPLOAD_PATH", tmp_path / "uploads" / "active.xlsx")
    monkeypatch.setattr(datastore, "DEFAULT_PATH", DEMO_XLSX)
    datastore._cache.clear()
    return tmp_path / "uploads"


def sample(tmp_path, name="marketing"):
    path = tmp_path / f"{name}.xlsx"
    generate_samples.write_sample(name, path)
    return path.read_bytes()


def spec_for(content, name="marketing.xlsx"):
    from data.profile import profile
    from data.readers import read_file

    return profile(read_file(content, name))["spec"]


def test_default_source_is_the_demo_workbook(store_dir):
    assert datastore.active_source() == ("strategic", DEMO_XLSX)
    info = datastore.describe()
    assert info["is_upload"] is False and info["kind"] == "strategic" and info["kpis"] == 18


def test_priority_of_sources(store_dir, tmp_path, monkeypatch):
    content = sample(tmp_path)
    upload_id = datastore.save_pending(content, "marketing.xlsx", "xlsx")
    datastore.save_flat(upload_id, spec_for(content))
    assert datastore.active_source() == ("flat", store_dir / "active.spec.json")
    monkeypatch.setenv("KPI_DATA_PATH", str(DEMO_XLSX))  # переменная окружения сильнее загруженного
    assert datastore.active_source() == ("strategic", DEMO_XLSX)
    monkeypatch.setenv("KPI_DATA_PATH", str(store_dir / "active.spec.json"))
    assert datastore.active_source() == ("flat", store_dir / "active.spec.json")  # .json — плоский отчёт


def test_flat_report_survives_between_calls_and_reloads_on_change(store_dir, tmp_path):
    content = sample(tmp_path, "support")
    datastore.save_flat(datastore.save_pending(content, "support.xlsx", "xlsx"), spec_for(content, "support.xlsx"))
    first = datastore.get_table()
    assert datastore.get_table() is first  # кэш
    assert first.source == "support.xlsx" and first.grain == "day" and first.dims == ["Очередь", "Канал"]
    other = sample(tmp_path, "branches")
    datastore.save_flat(datastore.save_pending(other, "branches.xlsx", "xlsx"), spec_for(other, "branches.xlsx"))
    second = datastore.get_table()
    assert second is not first and second.source == "branches.xlsx" and second.grain == "month"
    assert second.rows_excluded == 36
    # метаданные лежат на диске в читаемом виде
    meta = json.loads((store_dir / "active.spec.json").read_text("utf-8"))
    assert meta["filename"] == "branches.xlsx" and meta["data_file"] == "active.data.xlsx" and meta["spec"]["date"] == "Период"


def test_cache_follows_file_changes_made_by_another_process(store_dir, tmp_path):
    content = sample(tmp_path)
    datastore.save_flat(datastore.save_pending(content, "m.xlsx", "xlsx"), spec_for(content, "m.xlsx"))
    before = datastore.get_table()
    meta_path = store_dir / "active.spec.json"
    meta = json.loads(meta_path.read_text("utf-8"))
    meta["spec"]["metrics"] = meta["spec"]["metrics"][:2]
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), "utf-8")
    later = time.time() + 5
    os.utime(meta_path, (later, later))  # как будто файл изменил соседний процесс (MCP-сервер видит то же самое)
    after = datastore.get_table()
    assert after is not before and len(after.spec.metrics) == 2


def test_reset_removes_everything_that_was_uploaded(store_dir, tmp_path):
    content = sample(tmp_path)
    datastore.save_flat(datastore.save_pending(content, "m.xlsx", "xlsx"), spec_for(content, "m.xlsx"))
    assert sorted(p.name for p in store_dir.glob("active.*")) == ["active.data.xlsx", "active.spec.json"]
    info = datastore.reset()
    assert info["is_upload"] is False and not list(store_dir.glob("active.*"))


def test_csv_and_xlsx_data_files_do_not_pile_up(store_dir, tmp_path):
    xlsx = sample(tmp_path)
    datastore.save_flat(datastore.save_pending(xlsx, "m.xlsx", "xlsx"), spec_for(xlsx, "m.xlsx"))
    csv = "Дата;Канал;Лиды\n" + "\n".join(f"{d:02d}.03.2026;{c};{d + i}" for d in range(1, 10) for i, c in enumerate(("А", "Б")))
    csv_bytes = csv.encode("cp1251")
    datastore.save_flat(datastore.save_pending(csv_bytes, "m.csv", "csv"), spec_for(csv_bytes, "m.csv"))
    assert sorted(p.name for p in store_dir.glob("active.data.*")) == ["active.data.csv"]
    assert datastore.get_table().source == "m.csv"


def test_failed_confirm_changes_nothing(store_dir, tmp_path):
    content = sample(tmp_path)
    upload_id = datastore.save_pending(content, "m.xlsx", "xlsx")
    with pytest.raises(WorkbookError):
        datastore.save_flat(upload_id, {"date": "нет такого", "metrics": []})
    assert not list(store_dir.glob("active.*")) and datastore.active_source()[1] == DEMO_XLSX


def test_pending_ids_are_validated_and_old_uploads_are_cleaned(store_dir, monkeypatch):
    for bad in ("", "../x", "A" * 32, "g" * 32, "0" * 31):
        with pytest.raises(WorkbookError, match="Неверный идентификатор"):
            datastore.load_pending(bad)
    with pytest.raises(WorkbookError, match="не найден или устарел"):
        datastore.load_pending("0" * 32)
    content = xlsx_book({"Данные": [["Дата", "Канал", "Клики"], ["2026-03-01", "А", 1], ["2026-03-02", "Б", 2], ["2026-03-03", "А", 3]]})
    ids = [datastore.save_pending(content, f"f{i}.xlsx", "xlsx") for i in range(14)]
    left = sorted(p.stem for p in (store_dir / "pending").glob("*.json"))
    assert len(left) <= datastore.PENDING_KEEP and set(left) <= set(ids) and ids[-1] in left
    assert datastore.load_pending(ids[-1]) == (content, "f13.xlsx", "xlsx")
    assert len(list((store_dir / "pending").glob("*.xlsx"))) == len(left)  # данные чистятся вместе с метаданными


def test_strategic_upload_replaces_a_flat_report(store_dir, tmp_path):
    from data import generate

    content = sample(tmp_path)
    datastore.save_flat(datastore.save_pending(content, "m.xlsx", "xlsx"), spec_for(content, "m.xlsx"))
    goals, facts, links = generate.build_frames()
    path = tmp_path / "strategic.xlsx"
    generate.write_workbook(path, goals, facts, links)
    info = datastore.save_upload(path.read_bytes())
    assert info["kind"] == "strategic" and info["is_upload"] is True
    assert not (store_dir / "active.spec.json").exists() and not list(store_dir.glob("active.data.*"))
