import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DEMO_XLSX = ROOT / "data" / "kpi.xlsx"
ANOMALIES_JSON = ROOT / "data" / "anomalies.json"


@pytest.fixture(autouse=True)
def demo_dataset(monkeypatch):
    """Тесты всегда работают на демо-витрине, даже если в data/uploads лежит чужой файл."""
    monkeypatch.setenv("KPI_DATA_PATH", str(DEMO_XLSX))


@pytest.fixture(scope="session")
def anomalies() -> dict:
    return json.loads(ANOMALIES_JSON.read_text("utf-8"))
