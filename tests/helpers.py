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
