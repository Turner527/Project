import json

import pandas as pd

from data import generate
from data.schema import FACTS_COLUMNS
from data.validation import load_dataset
from tests.conftest import ANOMALIES_JSON, DEMO_XLSX


def test_demo_workbook_is_valid_and_has_expected_shape():
    ds = load_dataset(DEMO_XLSX)
    assert 15 <= ds.goals["kpi"].nunique() <= 20
    assert len(ds.months) == 18
    assert 4 <= len(ds.segments) <= 5
    assert set(ds.links["direction"]) == {"положительно", "отрицательно"}
    assert len(ds.facts) == ds.goals["kpi"].nunique() * len(ds.months) * len(ds.segments)


def test_generator_is_deterministic_and_matches_committed_files():
    goals1, facts1, _ = generate.build_frames()
    _, facts2, _ = generate.build_frames()
    pd.testing.assert_frame_equal(facts1, facts2)

    committed = load_dataset(DEMO_XLSX).facts.sort_values(["kpi", "month", "segment"]).reset_index(drop=True)
    fresh = facts1.copy()
    fresh.columns = list(FACTS_COLUMNS.values())
    fresh["month"] = fresh["month"].dt.strftime("%Y-%m")
    fresh = fresh.sort_values(["kpi", "month", "segment"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(committed[["kpi", "month", "segment", "plan", "fact"]], fresh)

    # anomalies.json не отстал от генератора
    expected = generate.build_anomalies(facts1)
    assert json.loads(ANOMALIES_JSON.read_text("utf-8")) == json.loads(json.dumps(expected, ensure_ascii=False))


def test_four_anomalies_with_expected_structure(anomalies):
    assert len(anomalies["anomalies"]) == 4
    for a in anomalies["anomalies"]:
        assert {"id", "goal", "primary_kpi", "key_segment", "cause_label", "driver_kpis_any", "owner"} <= set(a)
        for window in ("full", "last_6m", "last_3m"):
            exp = a["checkpoints"][window]["expected"]
            assert {"deviation_pct", "key_segment_contribution_pct", "s1_effect_abs"} <= set(exp)
            assert abs(exp["deviation_pct"]) >= 3.0  # аномалия заметна на любом окне
    assert len(anomalies["controls"]) >= 3


def test_data_has_seasonality_and_noise():
    ds = load_dataset(DEMO_XLSX)
    f = ds.facts[ds.facts["kpi"] == "Выдачи розничных кредитов"]
    monthly = f.groupby("month")["plan"].sum()
    assert monthly["2025-12"] > monthly["2026-01"] * 1.1  # декабрьский пик, январский провал
    # шум: факт не совпадает с планом ни в одном сегменте-месяце целиком
    healthy = ds.facts[(ds.facts["kpi"] == "Cost/Income") & (ds.facts["month"] < "2026-02")]
    ratio = (healthy["fact"] / healthy["plan"]).round(4)
    assert ratio.nunique() > 20 and ratio.between(0.9, 1.1).all()
