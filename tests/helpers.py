import copy
import io
import json

from agent.llm import FunctionCall, LLMResponse, Usage


class FakeLLM:
    """Отдаёт заранее заданные ответы по очереди и запоминает, что ему присылали."""

    name = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, functions=None, *, function_call="auto", temperature=0.0, max_tokens=None):
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "functions": functions,
                "function_call": function_call,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def call(name, **arguments) -> LLMResponse:
    return LLMResponse(function_call=FunctionCall(name, arguments), usage=Usage(10, 5, 15), finish_reason="function_call")


def text(content) -> LLMResponse:
    return LLMResponse(content=content, usage=Usage(20, 10, 30), finish_reason="stop")


def memo_json(summary="Отклонение небольшое.", scenarios=None, checks=None, used=None) -> str:
    return json.dumps(
        {
            "diagnosis": {"summary": summary, "root_cause": "Причина", "key_segments": ["Молодёжь"]},
            "scenarios": scenarios or [],
            "manual_checks": checks or ["Проверить вручную."],
            "used_tool_calls": used or [],
        },
        ensure_ascii=False,
    )


# ---------- плоские отчёты для универсального режима ----------


def activate_flat(tmp_path, monkeypatch, sample: str):
    """Записывает образец (marketing / support / branches) во временную папку, строит спецификацию профилировщиком
    и делает отчёт активным. Возвращает (профиль, DataFrame-источник)."""
    import json

    from data import generate_samples
    from data.profile import profile
    from data.readers import read_file
    from tools import datastore

    data_path = tmp_path / "data.xlsx"
    generate_samples.write_sample(sample, data_path)
    raw = read_file(data_path.read_bytes(), f"{sample}.xlsx")
    report = profile(raw)
    (tmp_path / "spec.json").write_text(json.dumps(
        {"data_file": "data.xlsx", "filename": raw.filename, "spec": report["spec"]}, ensure_ascii=False), "utf-8")
    monkeypatch.setenv("KPI_DATA_PATH", str(tmp_path / "spec.json"))
    datastore._cache.clear()
    return report, raw.frames[raw.sheet]


def table_from(rows, columns, spec_kwargs):
    """Table из списка строк: rows — значения, columns — названия столбцов, spec_kwargs — аргументы Spec."""
    import pandas as pd

    from data.model import Metric, Spec, build_table

    metrics = [m if isinstance(m, Metric) else Metric(**m) for m in spec_kwargs.pop("metrics")]
    return build_table(pd.DataFrame(rows, columns=columns), Spec(metrics=metrics, **spec_kwargs), "test")


def xlsx_book(sheets: dict[str, list[list]], formats: dict | None = None) -> bytes:
    """xlsx из словаря «лист -> строки»; formats: (лист, буква столбца, номер строки) -> формат числа ячейки."""
    import openpyxl

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for title, rows in sheets.items():
        ws = wb.create_sheet(title)
        for row in rows:
            ws.append(row)
    for (title, letter, row), fmt in (formats or {}).items():
        wb[title][f"{letter}{row}"].number_format = fmt
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ---------- план дашборда, как его мог бы вернуть GigaChat (под образец marketing) ----------

MODEL_PLAN = {
    "title": "Маркетинг: расходы и конверсия",
    "summary": "Расходы выросли, конверсия в лиды упала; изменение сосредоточено в канале «Поиск».",
    "widgets": [
        {"id": "w1", "type": "kpi", "metric": "Расходы", "title": "Расходы", "note": "К предыдущему периоду", "baseline": "auto"},
        {"id": "w2", "type": "trend", "metric": "Расходы", "title": "Динамика расходов", "note": "Линии по каналам",
         "dimension": "Канал"},
        {"id": "w3", "type": "breakdown", "metric": "Конверсия в лиды", "dimension": "Канал", "title": "Конверсия по каналам",
         "note": "Вклад каналов"},
    ],
    "insights": [
        {"title": "Поиск тянет расходы вверх", "text": "Расходы выросли на 10,9%: основной вклад даёт канал «Поиск».",
         "severity": "problem", "widgets": ["w1", "w2"]},
        {"title": "Конверсия просела", "text": "Конверсия в лиды упала в канале «Поиск».", "severity": "risk", "widgets": ["w3"]},
    ],
    "next_steps": ["Проверить ставки в канале «Поиск»."],
    "questions": ["Что с конверсией в «Поиске»?"],
}


def plan_json(**changes):
    return json.dumps({**MODEL_PLAN, **changes}, ensure_ascii=False)


def activate_csv(tmp_path, monkeypatch, content: str, name: str = "report.csv"):
    """Делает активным плоский отчёт из текста csv: роли столбцов определяет профилировщик, как при загрузке в интерфейсе."""
    from data.profile import profile
    from data.readers import read_file
    from tools import datastore

    data = content.encode("utf-8")
    raw = read_file(data, name)
    (tmp_path / "data.csv").write_bytes(data)
    (tmp_path / "spec.json").write_text(json.dumps({"data_file": "data.csv", "filename": name, "spec": profile(raw)["spec"]},
                                                   ensure_ascii=False), "utf-8")
    monkeypatch.setenv("KPI_DATA_PATH", str(tmp_path / "spec.json"))
    datastore._cache.clear()
    return datastore.get_table()
