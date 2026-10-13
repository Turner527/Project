"""Прогон агента по вопросам с эталонами.

Запуск:  python -m evals.run_evals [--backend stub|gigachat] [--tools mcp|direct] [--limit N] [--note "..."]

Результат: строка в evals/results.csv (дата, версия промпта, три метрики) и подробности по каждому вопросу в
evals/details/<время>.json. В конце печатается сравнение с предыдущим прогоном той же модели — так видно
эффект правки промпта. Всегда используется демо-витрина data/kpi.xlsx, загруженные пользователем файлы не влияют.
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from agent.llm import ConfigError, load_env, make_llm
from agent.loop import Agent
from agent.prompts import PROMPT_VERSION
from agent.tool_adapter import make_adapter
from evals.scoring import aggregate, evaluate
from tools import datastore

HERE = Path(__file__).parent
QUESTIONS = HERE / "questions.json"
RESULTS = HERE / "results.csv"
DETAILS_DIR = HERE / "details"
COLUMNS = ["date", "prompt_version", "llm", "tools_backend", "dataset", "n", "errors", "cause_accuracy",
           "numbers_accuracy", "unverified_share", "avg_latency_ms", "avg_tokens", "note", "details"]
METRICS = (("cause_accuracy", "причина названа верно"), ("numbers_accuracy", "числа совпадают с витриной"),
           ("unverified_share", "ответы с непроверенными числами"))


def load_questions(path: Path = QUESTIONS, ids: list[str] | None = None, limit: int | None = None) -> list[dict]:
    questions = json.loads(path.read_text("utf-8"))["questions"]
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    return questions[:limit] if limit else questions


def run_all(agent: Agent, questions: list[dict], progress=print) -> list[dict]:
    items = []
    for i, q in enumerate(questions, 1):
        result = agent.run(q["question"])
        item = evaluate(result, q)
        items.append(item)
        mark = lambda ok: "—" if ok is None else ("✓" if ok else "✗")  # noqa: E731
        progress(f"[{i:>2}/{len(questions)}] {q['id']:<20} причина {mark(item['cause_ok'])}  "
                 f"числа {mark(item['numbers_ok'])}  {item['status']:<25} {item['latency_ms'] / 1000:.1f} с"
                 + (f"  ОШИБКА: {item['error']}" if item["error"] else ""))
    return items


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def append_row(path: Path, row: dict) -> None:
    exists = path.exists() and path.stat().st_size > 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in COLUMNS})


def _pct(x) -> str:
    return "н/д" if x in (None, "") else f"{float(x) * 100:.1f}%"


def format_report(summary: dict, row: dict, previous: dict | None) -> str:
    lines = ["", f"Прогон {row['date']} · промпт {row['prompt_version']} · {row['llm']} · инструменты: "
                 f"{row['tools_backend']} · вопросов: {summary['n']} (ошибок: {summary['errors']})", ""]
    for key, title in METRICS:
        line = f"  {title:<34} {_pct(summary[key]):>7}"
        if previous and previous.get(key) not in (None, "") and summary[key] is not None:
            delta = (summary[key] - float(previous[key])) * 100
            line += f"   Δ {delta:+.1f} п.п. к прогону {previous['date']} (промпт {previous['prompt_version']})"
        lines.append(line)
    lines += ["", "  По группам (причина / числа):"]
    for group, g in summary["by_group"].items():
        lines.append(f"    {group:<4} n={g['n']:<2} {_pct(g['cause_accuracy']):>7} / {_pct(g['numbers_accuracy']):>7}")
    if previous is None:
        lines.append("\n  Предыдущих прогонов этой модели нет — сравнивать пока не с чем.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Прогнать агента по вопросам с эталонами")
    parser.add_argument("--backend", default=None, help="gigachat | stub | auto (по умолчанию по окружению)")
    parser.add_argument("--tools", default=None, help="mcp | direct (по умолчанию как в приложении: mcp)")
    parser.add_argument("--limit", type=int, default=None, help="взять только первые N вопросов")
    parser.add_argument("--ids", default=None, help="id вопросов через запятую")
    parser.add_argument("--note", default="", help="пометка к прогону, например «правка про сценарии»")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--details-dir", type=Path, default=DETAILS_DIR)
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # старые кодировки консоли Windows не знают ✓ и Δ
    load_env()
    os.environ["KPI_DATA_PATH"] = str(datastore.DEFAULT_PATH)  # эталоны посчитаны по демо-витрине
    try:
        llm = make_llm(args.backend)
    except ConfigError as exc:
        print(f"Не удалось создать клиент модели: {exc}", file=sys.stderr)
        return 2
    questions = load_questions(args.questions, args.ids.split(",") if args.ids else None, args.limit)
    if not questions:
        print("Нет вопросов для прогона (проверьте --ids и evals/questions.json).", file=sys.stderr)
        return 2

    adapter = make_adapter(args.tools)
    try:
        items = run_all(Agent(llm, adapter), questions)
    finally:
        adapter.close()

    summary = aggregate(items)
    now = datetime.now(timezone.utc)
    details_path = args.details_dir / f"{now:%Y%m%d-%H%M%S}.json"
    row = {
        "date": now.isoformat(timespec="seconds"),
        "prompt_version": PROMPT_VERSION,
        "llm": llm.name,
        "tools_backend": (args.tools or os.getenv("TOOLS_BACKEND") or "mcp").lower(),
        "dataset": datastore.DEFAULT_PATH.name,
        **{k: summary[k] for k in ("n", "errors", "cause_accuracy", "numbers_accuracy", "unverified_share",
                                   "avg_latency_ms", "avg_tokens")},
        "note": args.note,
        "details": str(details_path.relative_to(HERE.parent)) if details_path.is_relative_to(HERE.parent)
        else str(details_path),
    }
    previous = next((r for r in reversed(read_rows(args.results)) if r["llm"] == row["llm"]), None)
    args.details_dir.mkdir(parents=True, exist_ok=True)
    details_path.write_text(json.dumps({"run": row, "summary": summary, "items": items}, ensure_ascii=False,
                                       indent=2), "utf-8")
    append_row(args.results, row)
    print(format_report(summary, row, previous))
    print(f"\n  Таблица: {args.results}\n  Подробности по вопросам: {details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
