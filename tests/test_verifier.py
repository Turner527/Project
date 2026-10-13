from types import SimpleNamespace

import pytest

from agent.verifier import STATUS_OK, STATUS_UNVERIFIED, extract_numbers, numbers_match, verify


def rec(call_id, result):
    return SimpleNamespace(call_id=call_id, result=result)


def values(text, **kw):
    return [(n.value, n.decimals) for n in extract_numbers(text, **kw)]


# ---------- извлечение чисел ----------


def test_extracts_russian_and_english_decimals_percent_and_signs():
    assert values("отклонение −9,5% при вкладе 86.7%") == [(9.5, 1), (86.7, 1)]
    assert values("факт 288,0 млрд ₽ против 317,19") == [(288.0, 1), (317.19, 2)]


def test_thousands_separators():
    assert values("прибыль 12 345,6 млн, оборот 10 000 млн ₽") == [(12345.6, 1), (10000.0, 0)]
    assert values("остаток 1 234 млрд") == [(1234.0, 0)]


def test_dates_years_ids_labels_and_trivial_integers_are_not_numbers():
    text = (
        "С 2026-04 по 2026-09 (2026 год), сценарий S1 и S2, просрочка 30+, 3 сценария, 2 сегмента, "
        "источник c_ab12cd34, 1) пункт, квартал Q3, дата 01.10.2026"
    )
    assert values(text) == []


def test_small_integers_with_units_are_still_checked():
    assert values("рост на 5% и 8 млрд ₽") == [(5.0, 0), (8.0, 0)]
    assert values("держится 6 мес.") == []  # «6 мес.» — счёт, а не метрика
    assert values("потеряем 4 п.п.") == [(4.0, 0)]


def test_question_numbers_are_kept_when_not_skipping():
    assert values("за последние 3 месяца", skip_trivial=False) == [(3.0, 0)]


# ---------- допуск на округление ----------


@pytest.mark.parametrize(
    "written, source, ok",
    [
        ("86,7", 86.7, True),
        ("87", 86.7, True),      # целое допускает отклонение до 0,5
        ("86,8", 86.7, False),   # записано с десятыми — должно совпасть с точностью до 0,05
        ("90", 86.7, False),
        ("−9,5", -9.5, True),    # знак не важен
        ("9,5", -9.5, True),
        ("26,09", 26.09, True),
        ("26,1", 26.09, True),
        ("26,2", 26.09, False),
        ("1 234", 1233.8, True),
    ],
)
def test_rounding_tolerance(written, source, ok):
    num = extract_numbers(f"значение {written} млрд")[0]
    assert numbers_match(num, source) is ok


# ---------- проверка записки ----------


def test_all_numbers_found_gives_verified_status_and_sources():
    calls = [rec("c_1", {"deviation_pct": -9.5, "plan": 317.19}), rec("c_2", {"contribution_pct": 86.7})]
    res = verify({"diagnosis.summary": "Отклонение −9,5% от плана 317,19 млрд ₽, вклад сегмента 87%."}, calls)
    assert res.status == STATUS_OK and res.ok and res.checked == 3 and not res.unverified
    assert res.sources["9,5"] == ["c_1"] and res.sources["87"] == ["c_2"]


def test_fabricated_number_is_flagged_with_field_and_context():
    calls = [rec("c_1", {"deviation_pct": -9.5})]
    texts = {"diagnosis.summary": "Отклонение −9,5%.", "scenarios[1].effect_estimate": "Эффект составит 123,45 млрд ₽ в год."}
    res = verify(texts, calls)
    assert res.status == STATUS_UNVERIFIED
    assert [(u.value, u.field) for u in res.unverified] == [("123,45", "scenarios[1].effect_estimate")]
    assert "Эффект составит" in res.unverified[0].context


def test_number_of_a_different_value_does_not_match_by_accident():
    res = verify({"x": "Вклад 90% отклонения"}, [rec("c_1", {"contribution_pct": 86.7})])
    assert res.status == STATUS_UNVERIFIED


def test_numbers_from_the_question_are_allowed():
    res = verify({"x": "За последние 18 месяцев показатель стабилен."}, [rec("c_1", {})], question="Что за 18 месяцев?")
    assert res.ok


def test_answer_with_numbers_but_no_tool_calls_is_unverified():
    res = verify({"x": "Выдачи упали на 15%."}, [])
    assert res.status == STATUS_UNVERIFIED


def test_answer_without_numbers_is_verified():
    assert verify({"x": "Данных недостаточно, уточните вопрос."}, []).ok


def test_unknown_cited_call_ids_are_reported():
    res = verify({"x": "ok"}, [rec("c_1", {})], cited_call_ids=["c_1", "c_zzz", "c_zzz"])
    assert res.unknown_call_ids == ["c_zzz"]


def test_nested_json_numbers_and_bools():
    calls = [rec("c_1", {"a": {"b": [{"c": 42.5}]}, "flag": True, "count": 1})]
    assert verify({"x": "значение 42,5 млн"}, calls).ok
    assert not verify({"x": "значение 100 млн"}, calls).ok  # True не превращается в 1, а 100 не найдено
