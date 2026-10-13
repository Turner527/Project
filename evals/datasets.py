"""Наборы данных для evals: демо-витрина и плоские образцы (профиль строится так же, как при загрузке в интерфейсе)."""
import json
from pathlib import Path

from data import generate_samples
from data.profile import profile
from data.readers import read_file
from tools import datastore

DEMO = "demo"


def available() -> list[str]:
    return [DEMO, *generate_samples.SAMPLES]


def activate(name: str, workdir: Path) -> Path:
    """Готовит файлы набора в workdir и возвращает путь для KPI_DATA_PATH (xlsx демо или spec.json плоского отчёта)."""
    if name == DEMO:
        return datastore.DEFAULT_PATH
    if name not in generate_samples.SAMPLES:
        raise ValueError(f"Неизвестный набор данных «{name}». Доступны: {', '.join(available())}.")
    data_path = workdir / f"{name}.xlsx"
    generate_samples.write_sample(name, data_path)
    raw = read_file(data_path.read_bytes(), data_path.name)
    spec_path = workdir / f"{name}.spec.json"
    spec_path.write_text(json.dumps({"data_file": data_path.name, "filename": data_path.name,
                                     "spec": profile(raw)["spec"]}, ensure_ascii=False), "utf-8")
    return spec_path
