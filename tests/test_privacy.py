"""Приватность: «Не отправлять в модель», маскирование имени файла, подсказка о персональных данных.

Главная проверка — инвариант утечки: у отчёта есть столбцы с пометкой «Не отправлять» и файл с «секретным» именем, и ни
название столбца, ни его значения, ни имя файла не находятся ни в сводке, ни в каталоге, ни в результатах инструментов,
ни в сообщениях, которые получает модель (при построении дашборда и в чате). Метки содержат слово «СЕКРЕТ», поэтому
проверка ловит любой след; контрольный отчёт без пометки доказывает, что проверка вообще способна что-то поймать.
"""
import copy
import inspect
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.dashboard import DashboardAgent
from agent.loop import Agent
from agent.prompts import catalog_text
from agent.stub_llm import ScriptedLLM
from agent.tool_adapter import make_adapter
from app.main import create_app
from data import generate_samples
from data.model import Metric, Spec, Table, build_table
from data.profile import profile
from data.readers import read_file
from data.validation import WorkbookError
from logs.store import LogStore
from tests.conftest import DEMO_XLSX
from tests.helpers import FakeLLM, call, memo_json, plan_json, text, xlsx_book
from tools import dashboard as D
from tools import datastore
from tools import privacy
from tools.core import TOOLS, describe_dataset

SECRET_NAME = "СЕКРЕТНОЕ_ИМЯ"
FILE_NAME = SECRET_NAME + ".xlsx"
CLIENT_COL = "ФИО-СЕКРЕТ-7731"  # столбец-«разрез» с чужими именами
STATUS_COL = "Статус-СЕКРЕТ-9912"  # служебный столбец, по которому чистятся строки
SALARY_COL = "Оклад-СЕКРЕТ-5566"  # числовой столбец, который тоже не должен попасть в модель
CLIENTS = ["СЕКРЕТ-КЛИЕНТ-7731-А", "СЕКРЕТ-КЛИЕНТ-7731-Б", "СЕКРЕТ-КЛИЕНТ-7731-В"]
KEEP_VALUE, DROP_VALUE = "СЕКРЕТ-ФИЛЬТР-9912", "СЕКРЕТ-ЧУЖОЕ-9913"
EXCLUDED = [CLIENT_COL, STATUS_COL, SALARY_COL]
COLUMN_MARKERS = [CLIENT_COL, STATUS_COL, SALARY_COL, KEEP_VALUE, DROP_VALUE, *CLIENTS]
ALL_MARKERS = [SECRET_NAME, *COLUMN_MARKERS]
QUESTION = "Почему выросли расходы в канале Поиск и что делать?"


def report_rows() -> list[list]:
    rows = []
    for month in range(1, 7):
        for ch, channel in enumerate(["Поиск", "Соцсети", "Почта"]):
            for cl, client in enumerate(CLIENTS):
                n = len(rows)
                spend = 100 + (25 * month if channel == "Поиск" else 0) + 3 * cl
                rows.append([f"2026-{month:02d}-01", channel, client, DROP_VALUE if n % 7 == 3 else KEEP_VALUE, spend,
                             10 + month + ch + cl, 1000 + 10 * month, 50_000 + n])
    return rows


HEADER = ["Дата", "Канал", CLIENT_COL, STATUS_COL, "Расходы", "Клики", "Показы", SALARY_COL]


def make_report(tmp_path, monkeypatch, *, excluded: bool, name: str = FILE_NAME) -> Table:
    """Плоский отчёт с «секретными» столбцами становится активным. excluded=False — контроль: те же данные без пометки."""
    content = xlsx_book({"Данные": [HEADER, *report_rows()]})
    spec = profile(read_file(content, name))["spec"]
    spec["dimensions"] = ["Канал"] + ([] if excluded else [CLIENT_COL])
    spec["filters"] = [{"column": STATUS_COL, "values": [KEEP_VALUE]}]
    if excluded:
        spec["excluded"] = list(EXCLUDED)
        banned = set(EXCLUDED)
        spec["metrics"] = [m for m in spec["metrics"] if not banned & {m.get("column"), m.get("num"), m.get("den"), m.get("plan_column")}]
    (tmp_path / "data.xlsx").write_bytes(content)
    (tmp_path / "spec.json").write_text(json.dumps({"data_file": "data.xlsx", "filename": name, "spec": spec}, ensure_ascii=False), "utf-8")
    monkeypatch.setenv("KPI_DATA_PATH", str(tmp_path / "spec.json"))
    datastore._cache.clear()
    return datastore.get_table()


@pytest.fixture
def secret(tmp_path, monkeypatch) -> Table:
    return make_report(tmp_path, monkeypatch, excluded=True)


@pytest.fixture
def store(tmp_path):
    return LogStore(tmp_path / "pilot.db")


def dumped(obj) -> str:
    """Любой объект как текст без экранирования юникода: иначе русские метки в проверке были бы невидимы."""
    return obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)


def assert_clean(label: str, obj, markers=ALL_MARKERS) -> None:
    body = dumped(obj)
    for marker in markers:
        assert marker not in body, f"{label}: в тексте для модели найдено «{marker}»"
    assert "СЕКРЕТ" not in body.upper(), f"{label}: остался след секретной метки"


class RecordingLLM:
    """Заглушка, которая запоминает все сообщения и описания функций, полученные «моделью»."""

    name = "recording"

    def __init__(self):
        self.inner = ScriptedLLM()
        self.sent: list = []

    def chat(self, messages, functions=None, **kwargs):
        self.sent.append({"messages": copy.deepcopy(messages), "functions": copy.deepcopy(functions)})
        return self.inner.chat(messages, functions, **kwargs)


# ======================================================================================================================
# Инвариант утечки
# ======================================================================================================================


def test_control_without_the_mark_the_markers_do_leak(tmp_path, monkeypatch):
    """Проверка чувствительна: без пометки те же метки действительно попадают в сводку, каталог и результаты инструментов."""
    table = make_report(tmp_path, monkeypatch, excluded=False)
    digest_text = D.digest_text(D.build_digest(table))
    assert all(client in digest_text for client in CLIENTS) and CLIENT_COL in digest_text
    assert CLIENT_COL in catalog_text(table) and STATUS_COL in dumped(describe_dataset())
    assert KEEP_VALUE in dumped(describe_dataset())  # открытый фильтр раскрывает и столбец, и значения
    # а имя файла скрыто всегда: маскирование не зависит от пометок
    assert SECRET_NAME not in digest_text and SECRET_NAME not in catalog_text(table) and SECRET_NAME not in dumped(describe_dataset())


def test_digest_catalog_and_describe_dataset_hide_excluded_columns_and_file_name(secret):
    digest = D.build_digest(secret)
    assert_clean("сводка", D.digest_text(digest))
    assert_clean("каталог", catalog_text(secret))
    info = describe_dataset()
    assert_clean("describe_dataset", info)
    # фильтр по скрытому столбцу не раскрывается, остаётся только число отброшенных строк
    assert "filters" not in info and info["rows_excluded_by_filters"] > 0
    assert digest["report"]["rows_excluded_by_filters"] == info["rows_excluded_by_filters"]
    assert info["source"] == "отчёт.xlsx" and digest["report"]["source"] == "отчёт.xlsx"
    assert [d["name"] for d in digest["dimensions"]] == ["Канал"]


def test_a_filter_on_a_visible_column_is_still_described(tmp_path, monkeypatch):
    """Скрывается только фильтр по помеченному столбцу; обычный фильтр по-прежнему виден модели."""
    table = make_report(tmp_path, monkeypatch, excluded=False)
    info = describe_dataset()
    assert info["filters"] == [{"column": STATUS_COL, "kept_values": [KEEP_VALUE]}] and table.rows_excluded > 0


def test_excluded_columns_never_enter_the_table(secret):
    """Защитная проверка: столбцы с пометкой не попадают в Table.df, значения разрезов и итоговую сводку таблицы."""
    assert secret.spec.excluded == EXCLUDED
    assert set(secret.df.columns) <= {"_p", "Канал"} | {f"{p}::{m.name}" for m in secret.spec.metrics for p in "NDP"}
    assert_clean("Table.df", secret.df.astype(str).to_dict("list"), COLUMN_MARKERS)
    assert_clean("Table.dim_values", secret.dim_values, COLUMN_MARKERS)
    assert_clean("Table.summary", {**secret.summary(), "source": secret.public_name}, COLUMN_MARKERS)
    assert secret.rows_excluded > 0  # фильтр по скрытому столбцу работает локально: строки отброшены


TOOL_CALLS = {
    "describe_dataset": [{}],
    "find_metrics": [{"query": "расходы"}, {"query": SECRET_NAME}, {"query": CLIENT_COL}, {"query": CLIENTS[0]},
                     {"query": KEEP_VALUE}, {"query": STATUS_COL}, {"query": "ФИО клиент"}],
    "get_change": [{"metric": "Расходы"}, {"metric": "CTR"}, {"metric": SALARY_COL},
                   {"metric": "Расходы", "where_dimension": CLIENT_COL, "where_value": CLIENTS[0]},
                   {"metric": "Расходы", "where_dimension": STATUS_COL, "where_value": KEEP_VALUE}],
    "break_down": [{"metric": "Расходы"}, {"metric": "Расходы", "by": "Канал"}, {"metric": "Расходы", "by": CLIENT_COL},
                   {"metric": "Расходы", "by": STATUS_COL}],
    "get_related_metrics": [{"metric": "Расходы"}, {"metric": "Клики"}],
    "get_goal_info": [{"name": "Расходы"}, {"name": CLIENTS[0]}, {"name": SECRET_NAME}],
    "diagnose_change": [{"metric": "Расходы"}, {"metric": "Расходы", "where_dimension": "Канал", "where_value": "Поиск"},
                        {"metric": "Клики", "baseline": "yoy"}, {"metric": "Расходы", "where_dimension": CLIENT_COL, "where_value": CLIENTS[1]}],
    "get_structure": [{"metric": "Расходы"}, {"metric": "Расходы", "by": "Канал"}, {"metric": "Расходы", "by": CLIENT_COL}],
}


def generic_arguments(fn) -> dict:
    """Правдоподобные аргументы для инструмента, которого нет в TOOL_CALLS (например, добавленного позже)."""
    values = {"metric": "Расходы", "by": "Канал", "query": SECRET_NAME, "name": CLIENTS[0], "where_dimension": CLIENT_COL,
              "where_value": CLIENTS[0]}
    return {n: values.get(n, "Поиск") for n, p in inspect.signature(fn).parameters.items()
            if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}


def without_echo(result: dict, arguments: dict) -> str:
    """Результат без повтора собственных аргументов вызова: модель сама их передала, так что эхо ничего не раскрывает."""
    body = dumped(result)
    for value in arguments.values():
        body = body.replace(str(value), "")
    return body


def assert_messages_clean(messages: list[dict]) -> None:
    """Сообщения модели: промпт и вопрос чисты целиком, ответы инструментов — за вычетом аргументов, которые модель передала
    сама (её собственные слова, эхо ничего не раскрывает)."""
    arguments: dict = {}
    for m in messages:
        if m["role"] == "assistant":
            arguments = (m.get("function_call") or {}).get("arguments") or {}
        elif m["role"] == "function":
            assert_clean(f"ответ инструмента {m['name']}", without_echo(json.loads(m["content"]), arguments))
        else:
            assert_clean(f"сообщение {m['role']}", m["content"])


def test_every_tool_hides_excluded_columns_and_file_name(secret):
    calls = 0
    for name, fn in TOOLS.items():
        for arguments in TOOL_CALLS.get(name) or [generic_arguments(fn)]:
            result = fn(**arguments)
            assert result["tool"] == name
            assert_clean(f"{name}({arguments})", without_echo(result, arguments))
            calls += 1
    assert calls >= len(TOOLS)
    # результат с ошибкой («измерения нет») тоже не должен называть скрытые столбцы среди доступных
    err = TOOLS["break_down"](metric="Расходы", by=CLIENT_COL)
    assert "error" in err and "Канал" in err["error"]


def test_dashboard_agent_sends_and_returns_no_trace_of_excluded_columns(secret):
    llm = RecordingLLM()
    result = DashboardAgent(llm).run(secret)
    assert result.plan_source == "model" and result.widgets
    sent = result.to_dict()["sent_to_model"]
    assert_clean("sent_to_model", sent)
    assert sent["chars"] == len(D.digest_text(sent["digest"]))
    assert_clean("сообщения при построении дашборда", llm.sent)
    # локально пользователь видит настоящее имя; скрытых столбцов нет нигде в результате
    local = result.to_dict()
    assert local["source"] == FILE_NAME and FILE_NAME in dumped(local["plan"])
    assert_clean("результат дашборда", dumped(local).replace(FILE_NAME, ""))  # без настоящего имени файла следов нет


def test_chat_sends_no_trace_of_excluded_columns(secret):
    llm = RecordingLLM()
    result = Agent(llm, make_adapter("direct")).run(QUESTION)
    assert result.memo is not None and result.error is None and [c.name for c in result.tool_calls][:2] == ["find_metrics", "diagnose_change"]
    assert_clean("сообщения чата", llm.sent)
    assert_clean("следы вызовов", [c.to_dict() for c in result.tool_calls])


def test_chat_with_probing_tool_calls_sends_no_trace(secret):
    """Модель сама пробует спросить про скрытые столбцы и значения: ответы инструментов их не раскрывают."""
    llm = FakeLLM([call("describe_dataset"), call("find_metrics", query=CLIENTS[0]), call("break_down", metric="Расходы", by=CLIENT_COL),
                   call("diagnose_change", metric="Расходы", where_dimension=STATUS_COL, where_value=KEEP_VALUE),
                   call("get_structure", metric="Расходы"), text("данных достаточно"), text(memo_json("Расходы выросли."))])
    result = Agent(llm, make_adapter("direct")).run("Что в отчёте и есть ли клиенты?")
    assert result.memo is not None and len(result.tool_calls) == 5
    for sent in llm.calls:
        assert_messages_clean(sent["messages"])
    assert_clean("описания функций", [c["functions"] for c in llm.calls])


def test_api_dashboard_then_chat_sends_no_trace(secret, store):
    """Через API: дашборд, затем вопрос с выводами дашборда в промпте. В сохранённом плане стоит настоящее имя файла, и
    в промпт чата оно не должно попасть."""
    llm = RecordingLLM()
    with TestClient(create_app(llm=llm, adapter=make_adapter("direct"), store=store, load_env_file=False)) as client:
        dash = client.post("/api/dashboard").json()
        assert FILE_NAME in dash["plan"]["title"] and dash["source"] == FILE_NAME
        assert_clean("sent_to_model", dash["sent_to_model"])
        answer = client.post("/api/ask", json={"question": QUESTION}).json()
        assert answer["memo"] and answer["status"] == "проверен"
    assert len(llm.sent) >= 3  # план дашборда и два вызова чата
    assert_clean("сообщения дашборда и чата", llm.sent)
    system = next(m["content"] for m in llm.sent[-1]["messages"] if m["role"] == "system")
    assert "Контекст дашборда" in system and "Дашборд: отчёт.xlsx" in system  # выводы дашборда в промпте, с подписью вместо имени
    assert_clean("ответ чата", answer["tool_calls"])


# ======================================================================================================================
# Спецификация: исключённые столбцы и роли
# ======================================================================================================================

FRAME = pd.DataFrame({
    "Дата": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"], "Канал": ["Поиск", "Поиск", "Почта", "Почта"],
    "Клиент": ["А", "Б", "А", "Б"], "Расходы": [100, 120, 90, 80], "Клики": [10, 12, 9, 8], "План": [110, 110, 90, 90],
    "Статус": ["Да", "Да", "Да", "Нет"],
}, dtype=object)
BASE = dict(date="Дата", dimensions=["Канал"], metrics=[Metric("Расходы", "sum", "Расходы"), Metric("Клики", "sum", "Клики")])


def spec_of(**changes) -> Spec:
    return Spec(**{**copy.deepcopy(BASE), **changes})


def errors_of(spec: Spec) -> list[str]:
    with pytest.raises(WorkbookError) as exc:
        build_table(FRAME, spec, "t.xlsx")
    return exc.value.errors


def test_spec_round_trips_exclusions():
    spec = spec_of(excluded=["Клиент", "Статус"])
    assert spec.to_dict()["excluded"] == ["Клиент", "Статус"]
    assert Spec.from_dict(spec.to_dict()).excluded == ["Клиент", "Статус"]
    assert Spec.from_dict({"excluded": ["Клиент", "Клиент", "Статус"]}).excluded == ["Клиент", "Статус"]  # без повторов
    assert Spec.from_dict({}).excluded == [] and Spec().to_dict()["excluded"] == []  # старые файлы настроек без поля читаются


def test_excluded_column_cannot_be_the_date():
    errors = errors_of(spec_of(excluded=["Дата"]))
    assert errors == ["Столбец «Дата» помечен «Не отправлять в модель», поэтому не может быть датой."]


def test_excluded_column_cannot_be_a_dimension():
    errors = errors_of(spec_of(dimensions=["Канал", "Клиент"], excluded=["Клиент"]))
    assert errors == ["Столбец «Клиент» помечен «Не отправлять в модель», поэтому не может быть разрезом (измерением)."]


def test_excluded_column_cannot_be_a_metric_column():
    spec = spec_of(excluded=["Расходы"])
    assert errors_of(spec) == ["Показатель «Расходы»: название совпадает со столбцом «Расходы», который помечен «Не отправлять в модель».",
                               "Показатель «Расходы»: столбец «Расходы» помечен «Не отправлять в модель», поэтому не может быть столбцом показателя."]
    spec = spec_of(metrics=[Metric("Затраты", "sum", "Расходы")], excluded=["Расходы"])
    assert errors_of(spec) == ["Показатель «Затраты»: столбец «Расходы» помечен «Не отправлять в модель», поэтому не может быть столбцом показателя."]


def test_excluded_column_cannot_be_a_ratio_part_or_a_plan():
    ratio = Metric("Цена клика", "ratio", num="Расходы", den="Клики")
    assert errors_of(spec_of(metrics=[ratio], excluded=["Расходы"])) == [
        "Показатель «Цена клика»: столбец «Расходы» помечен «Не отправлять в модель», поэтому не может быть числителем."]
    assert errors_of(spec_of(metrics=[ratio], excluded=["Клики"])) == [
        "Показатель «Цена клика»: столбец «Клики» помечен «Не отправлять в модель», поэтому не может быть знаменателем."]
    planned = Metric("Расходы", "sum", "Расходы", plan_column="План")
    assert errors_of(spec_of(metrics=[planned], excluded=["План"])) == [
        "Показатель «Расходы»: столбец «План» помечен «Не отправлять в модель», поэтому не может быть столбцом плана."]


def test_excluded_column_inside_an_expression_is_found():
    profit = Metric("Разница", "sum", "Расходы - Клики")
    assert errors_of(spec_of(metrics=[profit], excluded=["Клики"])) == [
        "Показатель «Разница»: столбец «Клики» помечен «Не отправлять в модель», поэтому не может быть столбцом показателя."]
    ratio = Metric("Доля", "ratio", num="Расходы - Клики", den="Расходы")
    assert any("Клики" in e and "числителем" in e for e in errors_of(spec_of(metrics=[ratio], excluded=["Клики"])))


def test_metric_named_like_an_excluded_column_is_rejected():
    spec = spec_of(metrics=[Metric("Клиент", "sum", "Расходы")], excluded=["Клиент"])
    assert errors_of(spec) == ["Показатель «Клиент»: название совпадает со столбцом «Клиент», который помечен «Не отправлять в модель»."]


def test_unknown_excluded_column_is_an_error_and_errors_are_listed_together():
    assert errors_of(spec_of(excluded=["Нет такого"])) == ["Не отправлять: столбца «Нет такого» нет в таблице."]
    metrics = [Metric("Расходы", "sum", "Расходы"), Metric("Отклики", "sum", "Клики")]
    errors = errors_of(spec_of(dimensions=["Канал", "Клиент"], metrics=metrics, excluded=["Дата", "Клиент", "Клики"]))
    assert len(errors) == 3 and "датой" in errors[0] and "разрезом" in errors[1] and "Клики" in errors[2] and "«Отклики»" in errors[2]


def test_excluded_column_may_still_filter_rows_locally():
    spec = spec_of(excluded=["Клиент", "Статус"], filters=[{"column": "Статус", "values": ["Да"]}])
    table = build_table(FRAME, spec, "t.xlsx")
    assert table.rows_excluded == 1 and len(table.df) == 3
    assert not {"Клиент", "Статус"} & set(table.df.columns)


def test_confirm_endpoint_reports_the_forbidden_use_in_russian(tmp_path, monkeypatch, store):
    monkeypatch.delenv("KPI_DATA_PATH", raising=False)
    monkeypatch.setattr(datastore, "DEFAULT_PATH", DEMO_XLSX)
    datastore._cache.clear()
    path = tmp_path / "marketing.xlsx"
    generate_samples.write_sample("marketing", path)
    with TestClient(create_app(llm=ScriptedLLM(), adapter=make_adapter("direct"), store=store, load_env_file=False)) as client:
        body = client.post("/api/upload", files={"file": ("marketing.xlsx", path.read_bytes(), "application/vnd.ms-excel")}).json()
        spec = body["profile"]["spec"]
        assert spec["excluded"] == []  # профиль никогда не ставит пометку сам
        channel = spec["dimensions"][0]
        r = client.post("/api/datasets/confirm", json={"upload_id": body["upload_id"], "spec": {**spec, "excluded": [channel]}})
        assert r.status_code == 422
        assert any(channel in e and "помечен «Не отправлять в модель»" in e and "разрезом" in e for e in r.json()["errors"])
        # без разреза с этим столбцом отчёт принимается, а пометка сохраняется в настройках
        ok = {**spec, "dimensions": [d for d in spec["dimensions"] if d != channel], "excluded": [channel]}
        r = client.post("/api/datasets/confirm", json={"upload_id": body["upload_id"], "spec": ok})
        assert r.status_code == 200, r.text
        saved = json.loads((datastore._dir() / "active.spec.json").read_text("utf-8"))
        assert saved["spec"]["excluded"] == [channel] and datastore.get_table().spec.excluded == [channel]


def test_exclusions_are_part_of_the_report_fingerprint(tmp_path, monkeypatch):
    monkeypatch.delenv("KPI_DATA_PATH", raising=False)
    monkeypatch.setattr(datastore, "DEFAULT_PATH", DEMO_XLSX)
    datastore._cache.clear()
    content = xlsx_book({"Данные": [HEADER, *report_rows()]})
    spec = profile(read_file(content, "r.xlsx"))["spec"]
    spec["dimensions"], spec["filters"] = ["Канал"], []
    spec["metrics"] = [m for m in spec["metrics"] if m.get("column") in ("Расходы", "Клики")]
    upload_id = datastore.save_pending(content, "r.xlsx", "xlsx")
    datastore.save_flat(upload_id, spec)
    plain = datastore.fingerprint()
    datastore.save_flat(upload_id, {**spec, "excluded": [CLIENT_COL]})
    assert datastore.fingerprint() != plain
    datastore.save_flat(upload_id, spec)
    assert datastore.fingerprint() == plain  # те же роли — тот же отпечаток


# ======================================================================================================================
# Маскирование имени файла и обратная подстановка
# ======================================================================================================================


def test_public_name_keeps_only_the_extension():
    assert privacy.public_name("Продажи Q3 (клиенты).xlsx") == "отчёт.xlsx"
    assert privacy.public_name("ДАННЫЕ.CSV") == "отчёт.csv"
    assert privacy.public_name("report") == "отчёт" and privacy.public_name("") == "отчёт" and privacy.public_name("a.txt") == "отчёт"
    assert privacy.public_name("kpi.xlsx") == privacy.public_name("другое.xlsx")


def test_mask_and_unmask_are_inverse_and_walk_nested_data():
    real = "Секрет.xlsx"
    plan = {"title": "Дашборд: отчёт.xlsx", "items": [{"text": "Из файла отчёт.xlsx", "n": 3}, "без имени"], "flag": True}
    shown = privacy.unmask(plan, real)
    assert shown == {"title": "Дашборд: Секрет.xlsx", "items": [{"text": "Из файла Секрет.xlsx", "n": 3}, "без имени"], "flag": True}
    assert privacy.mask(shown, real) == plan
    assert plan["title"] == "Дашборд: отчёт.xlsx"  # исходный объект не меняется
    assert privacy.unmask("отчёт.xlsx", "") == "отчёт.xlsx" and privacy.mask("текст", "") == "текст"  # пустое имя ничего не стирает
    assert privacy.unmask("отчёт.csv", "x.csv") == "x.csv" and privacy.unmask("отчёт.csv", "отчёт.csv") == "отчёт.csv"


def test_table_public_name_is_used_for_the_model_but_source_stays_real(secret):
    assert secret.source == FILE_NAME and secret.public_name == "отчёт.xlsx"
    assert secret.summary()["source"] == FILE_NAME  # для интерфейса


def test_model_plan_text_gets_the_real_file_name_back(tmp_path, monkeypatch):
    table = make_report(tmp_path, monkeypatch, excluded=True)
    plan = json.loads(plan_json(title="Расходы в отчёт.xlsx", summary="Сводка по файлу отчёт.xlsx: расходы выросли.",
                                next_steps=["Сверить отчёт.xlsx с источником."]))
    for w in plan["widgets"]:
        w["metric"] = "Расходы"
        w["dimension"] = "Канал" if w["type"] != "kpi" else None
    plan["widgets"][0]["title"] = "Расходы из отчёт.xlsx"
    llm = FakeLLM([text(json.dumps(plan, ensure_ascii=False))])
    result = DashboardAgent(llm).run(table)
    assert result.plan_source == "model", result.error
    assert result.plan["title"] == f"Расходы в {FILE_NAME}" and FILE_NAME in result.plan["summary"]
    assert result.plan["next_steps"] == [f"Сверить {FILE_NAME} с источником."]
    assert result.plan["widgets"][0]["title"] == f"Расходы из {FILE_NAME}" and result.widgets[0]["title"] == f"Расходы из {FILE_NAME}"
    assert "отчёт.xlsx" not in dumped(result.plan) and result.source == FILE_NAME
    assert "отчёт.xlsx" in llm.calls[0]["messages"][1]["content"]  # а модель видела только подпись
    assert_clean("сводка", llm.calls[0]["messages"])


def test_default_plan_title_shows_the_real_file_name_too(secret):
    result = DashboardAgent(RecordingLLM()).run(secret)
    assert result.plan["title"] == f"Дашборд: {FILE_NAME}" and FILE_NAME in result.plan["summary"]


def test_chat_answer_gets_the_real_file_name_back_while_the_trace_shows_what_the_model_saw(secret):
    memo = memo_json(summary="Данные взяты из файла отчёт.xlsx.", checks=["Сверить отчёт.xlsx с источником."])
    llm = FakeLLM([call("describe_dataset"), text("ok"), text(memo)])
    result = Agent(llm, make_adapter("direct")).run("Что за отчёт загружен?")
    assert result.memo.diagnosis.summary == f"Данные взяты из файла {FILE_NAME}."
    assert result.memo.manual_checks == [f"Сверить {FILE_NAME} с источником."]
    assert result.tool_calls[0].result["source"] == "отчёт.xlsx"  # след вызова — то, что получила модель
    assert_clean("сообщения", llm.calls)


def test_unverified_number_context_is_unmasked_too(secret):
    memo = memo_json(summary="В отчёт.xlsx расходы выросли на 987,6%.")
    result = Agent(FakeLLM([text("ok"), text(memo)]), make_adapter("direct")).run("Что с расходами?")
    context = [u.context for u in result.verification.unverified]
    assert context and all("отчёт.xlsx" not in c for c in context) and any(FILE_NAME in c for c in context)


def test_api_answers_and_header_use_the_real_name(secret, store):
    memo = memo_json(summary="Источник: отчёт.xlsx.")
    llm = FakeLLM([call("describe_dataset"), text("ok"), text(memo)])
    with TestClient(create_app(llm=llm, adapter=make_adapter("direct"), store=store, load_env_file=False)) as client:
        answer = client.post("/api/ask", json={"question": "Что за отчёт загружен?"}).json()
        assert answer["memo"]["diagnosis"]["summary"] == f"Источник: {FILE_NAME}."
        assert client.get("/api/status").json()["dataset"]["source"] == FILE_NAME


def test_dashboard_header_reads_the_real_file_name_from_the_result():
    html = (DEMO_XLSX.parent.parent / "app" / "static" / "index.html").read_text("utf-8")
    assert "`Источник: ${d.source}`" in html and "Источник: ${rep.source}" not in html


# ======================================================================================================================
# Подсказка «возможно, персональные данные»
# ======================================================================================================================


def profiled(header, rows):
    return profile(read_file(xlsx_book({"Данные": [header, *rows]}), "t.xlsx"))


def table_rows(extra, n=12):
    return [[f"2026-03-{d:02d}", "Север" if d % 2 else "Юг", 100 + d, extra(d)] for d in range(1, n + 1)]


def hints(report) -> dict:
    return {c["name"]: c["sensitive_hint"] for c in report["columns"]}


@pytest.mark.parametrize("name", ["Клиент", "ФИО", "Фамилия", "Имя", "Сотрудник", "Контрагент", "Телефон", "Мобильный телефон", "E-mail",
                                  "Почта", "Паспорт", "ИНН", "СНИЛС", "Адрес доставки", "Номер договора", "Договор", "Номер карты",
                                  "Счёт", "Дата рождения", "Менеджер"])
def test_sensitive_hint_by_text_column_name(name):
    report = profiled(["Дата", "Регион", "Расходы", name], table_rows(lambda d: f"значение {d % 5}"))
    column = next(c for c in report["columns"] if c["name"] == name)
    assert column["sensitive_hint"] is True and column["sensitive_reason"]
    assert any(name in w and "персональные данные" in w for w in report["warnings"])


@pytest.mark.parametrize("name", ["Канал", "Регион", "Продукт", "Наименование товара", "Сегмент клиентов", "Клиентская группа", "Время обработки"])
def test_no_sensitive_hint_for_ordinary_names(name):
    report = profiled(["Дата", "Регион", "Расходы", name], table_rows(lambda d: f"значение {d % 5}"))
    column = next(c for c in report["columns"] if c["name"] == name)
    assert column["sensitive_hint"] is False and column["sensitive_reason"] == ""
    assert not any("персональные данные" in w for w in report["warnings"])


def test_numeric_counts_named_like_people_are_not_flagged_but_phone_numbers_are():
    rows = [[f"2026-03-{d:02d}", "Север" if d % 2 else "Юг", 100 + d, 10 + d, 79_990_000_000 + d] for d in range(1, 13)]
    report = profiled(["Дата", "Регион", "Расходы", "Клиентов, шт", "Телефон"], rows)
    assert hints(report)["Клиентов, шт"] is False and hints(report)["Телефон"] is True


def test_sensitive_hint_by_values_under_a_neutral_name():
    emails = profiled(["Дата", "Регион", "Расходы", "Контакт"], table_rows(lambda d: f"user{d}@example.com"))
    assert hints(emails)["Контакт"] and "почты" in next(c for c in emails["columns"] if c["name"] == "Контакт")["sensitive_reason"]
    phones = profiled(["Дата", "Регион", "Расходы", "Связь"], table_rows(lambda d: f"+7 (999) 123-45-{d:02d}"))
    assert hints(phones)["Связь"] and "телефон" in next(c for c in phones["columns"] if c["name"] == "Связь")["sensitive_reason"]
    cards = profiled(["Дата", "Регион", "Расходы", "Реквизит"], table_rows(lambda d: ["4111 1111 1111 1111", "5555 5555 5555 4444"][d % 2]))
    assert hints(cards)["Реквизит"] and "карт" in next(c for c in cards["columns"] if c["name"] == "Реквизит")["sensitive_reason"]


def test_sensitive_hint_needs_most_values_to_look_alike():
    mixed = profiled(["Дата", "Регион", "Расходы", "Заметка"], table_rows(lambda d: f"user{d}@example.com" if d <= 4 else f"текст {d}"))
    assert hints(mixed)["Заметка"] is False  # 4 из 12: меньше большинства
    codes = profiled(["Дата", "Регион", "Расходы", "Код"], table_rows(lambda d: f"{1234567800 + d}"))  # 10 цифр подряд — это код, не телефон
    assert hints(codes)["Код"] is False
    wrong_luhn = profiled(["Дата", "Регион", "Расходы", "Реквизит"], table_rows(lambda d: "4111 1111 1111 1112"))
    assert hints(wrong_luhn)["Реквизит"] is False


def test_the_hint_never_changes_roles_or_marks_anything():
    report = profiled(["Дата", "Клиент", "Расходы", "Телефон"], [[f"2026-03-{d:02d}", f"Клиент {d % 4}", 100 + d, f"+7 999 000-00-{d:02d}"] for d in range(1, 13)])
    assert hints(report)["Клиент"] and hints(report)["Телефон"]
    assert report["spec"]["excluded"] == []
    roles = {c["name"]: c["role"] for c in report["columns"]}
    assert roles["Клиент"] == "dimension"  # решение за пользователем: подсказка роль не меняет
    assert all("sensitive_hint" in c and "sensitive_reason" in c for c in report["columns"])


def test_sample_reports_have_no_false_alarms(tmp_path):
    for sample in ("marketing", "support", "branches"):
        path = tmp_path / f"{sample}.xlsx"
        generate_samples.write_sample(sample, path)
        report = profile(read_file(path.read_bytes(), path.name))
        assert not [c["name"] for c in report["columns"] if c["sensitive_hint"]], sample


# ======================================================================================================================
# Интерфейс (статические проверки; поведение проверено в браузере)
# ======================================================================================================================


def test_roles_dialog_has_the_exclusion_controls():
    html = (DEMO_XLSX.parent.parent / "app" / "static" / "index.html").read_text("utf-8")
    for fragment in ("Не отправлять в модель", "не отправляется", "возможно, персональные данные", "Дату исключить нельзя",
                     "excluded: cols.filter((c) => c.excluded)", "ratioBlockers", "aria-label\": `Не отправлять в модель: столбец"):
        assert fragment in html, fragment
    assert "prefers-color-scheme" not in html
