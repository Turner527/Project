"""Прогон построения дашбордов по наборам данных: пригоден ли план модели и названы ли в дашборде причины.

Запуск:  python -m evals.run_dashboard_evals [--set demo|flat|all] [--backend stub|gigachat] [--note "..."]

Для каждого набора (демо-витрина банка и три плоских образца из data/samples) дашборд строится так же, как в приложении.
Пять метрик, у каждой набора свой эталон, заложенный в данные:

  plan_valid        — модель вернула план в нужном формате (иначе показан дашборд по умолчанию);
  blocks_kept       — доля предложенных моделью блоков, которые удалось построить по отчёту;
  unverified_share  — доля чисел в текстах плана, которых нет в сводке и данных блоков (проверка кодом);
  causes_named      — доля заложенных в данные причин (сегмент, канал, очередь, отделение), названных в кратком выводе
                      или инсайтах;
  all_kinds         — на дашборде есть блоки всех видов, которые возможны для этого отчёта (плитки, динамика, вклады).

Результат — строка в evals/results_dashboard.csv и разбор по наборам в evals/details/dashboards-<время>.json. В конце
печатается сравнение с предыдущим прогоном той же модели и тех же наборов. На заглушке все метрики 100%: это проверка
конвейера, а не оценка качества GigaChat.
"""
import argparse
import csv
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from agent.dashboard import DashboardAgent
from agent.llm import ConfigError, load_env, make_llm
from agent.loop import extract_json
from agent.prompts import DASHBOARD_PROMPT_VERSION
from agent.schemas import DashboardPlan
from evals import datasets
from evals.run_evals import read_rows
from tools import datastore

HERE = Path(__file__).parent
ANOMALIES = HERE.parent / "data" / "anomalies.json"
RESULTS = HERE / "results_dashboard.csv"
DETAILS_DIR = HERE / "details"
FLAT_SEGMENTS = {"marketing": ["Поиск"], "support": ["Техническая поддержка"],
                 "branches": ["Новосибирск", "Москва Север", "Екатеринбург"]}
SETS = {"demo": [datasets.DEMO], "flat": list(FLAT_SEGMENTS), "all": [datasets.DEMO, *FLAT_SEGMENTS]}
COLUMNS = ["date", "prompt_version", "llm", "set", "n", "plan_valid", "blocks_kept", "unverified_share", "causes_named",
           "all_kinds", "avg_latency_ms", "avg_tokens", "note"]
METRICS = (("plan_valid", "план в нужном формате"), ("blocks_kept", "предложенные блоки построены"),
           ("unverified_share", "числа в текстах без источника"), ("causes_named", "заложенные причины названы"),
           ("all_kinds", "есть блоки всех возможных видов"))


def expected_segments(name: str) -> list[str]:
    """Что заложено в данные набора: сегменты четырёх аномалий демо-витрины или проблемное значение образца."""
    if name == datasets.DEMO:
        return [a["key_segment"] for a in json.loads(ANOMALIES.read_text("utf-8"))["anomalies"]]
    return FLAT_SEGMENTS[name]


def proposed_blocks(result) -> int | None:
    """Сколько блоков предложила модель (по её собственному ответу); None — плана модели не было."""
    if result.plan_source != "model":
        return None
    try:
        return len(DashboardPlan.model_validate_json(extract_json(result.raw_text)).widgets)
    except ValueError:
        return None


def evaluate(result, name: str) -> dict:
    """Оценка одного построенного дашборда по эталону набора. Чистая функция. Метрики про качество плана относятся к
    модели: если план не её (показан дашборд по умолчанию), они равны нулю, а не берутся у запасного плана."""
    segments = expected_segments(name)
    from_model = result.plan_source == "model"
    plan = result.plan or {}
    text = " ".join([plan.get("summary", "")] + [f"{i['title']} {i['text']}" for i in plan.get("insights", [])]).casefold()
    named = [s for s in segments if from_model and s.casefold() in text]
    kinds = {w["type"] for w in result.widgets}
    rep = result.digest["report"]
    required = {"kpi"} | ({"trend"} if rep.get("grain") else set()) | ({"breakdown"} if result.digest["dimensions"] else set())
    proposed = proposed_blocks(result)
    ver = result.verification if from_model else None
    return {
        "dataset": name, "plan_valid": from_model and result.error is None, "error": result.error,
        "proposed": proposed, "widgets": len(result.widgets),
        "blocks_kept": 0.0 if not from_model else (None if not proposed else min(len(result.widgets), proposed) / proposed),
        "checked": ver.checked if ver else 0, "unverified": len(ver.unverified) if ver else 0,
        "causes_named": len(named) / len(segments) if segments else None, "named": named,
        "missing": [s for s in segments if s not in named], "all_kinds": from_model and required <= kinds, "kinds": sorted(kinds),
        "notes": result.notes, "latency_ms": result.latency_ms, "tokens": result.usage.total_tokens,
    }


def _mean(values: list) -> float | None:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def aggregate(items: list[dict]) -> dict:
    checked = sum(i["checked"] for i in items)
    return {
        "n": len(items),
        "plan_valid": _mean([i["plan_valid"] for i in items]),
        "blocks_kept": _mean([i["blocks_kept"] for i in items]),
        "unverified_share": sum(i["unverified"] for i in items) / checked if checked else None,
        "causes_named": _mean([i["causes_named"] for i in items]),
        "all_kinds": _mean([i["all_kinds"] for i in items]),
        "avg_latency_ms": round(_mean([i["latency_ms"] for i in items]) or 0),
        "avg_tokens": round(_mean([i["tokens"] for i in items]) or 0),
    }


def run_sets(llm, names: list[str], workdir: Path, progress=print) -> list[dict]:
    items = []
    for name in names:
        os.environ["KPI_DATA_PATH"] = str(datasets.activate(name, workdir))
        datastore._cache.clear()
        result = DashboardAgent(llm).run(datastore.get_table())
        item = evaluate(result, name)
        items.append(item)
        mark = lambda ok: "—" if ok is None else ("✓" if ok else "✗")  # noqa: E731
        progress(f"{name:<10} формат {mark(item['plan_valid'])}  причины {len(item['named'])}/{len(item['named']) + len(item['missing'])}  "
                 f"блоки всех видов {mark(item['all_kinds'])}  непроверенных чисел: {item['unverified']}  "
                 f"{item['latency_ms'] / 1000:.1f} с" + (f"\n           {item['error']}" if item["error"] else "")
                 + (f"\n           не названы: {', '.join(item['missing'])}" if item["missing"] else ""))
    return items


def _pct(x) -> str:
    return "н/д" if x in (None, "") else f"{float(x) * 100:.1f}%"


def format_report(summary: dict, row: dict, previous: dict | None) -> str:
    lines = ["", f"Прогон {row['date']} · промпт {row['prompt_version']} · {row['llm']} · наборов: {summary['n']}", ""]
    for key, title in METRICS:
        line = f"  {title:<36} {_pct(summary[key]):>7}"
        if previous and previous.get(key) not in (None, "") and summary[key] is not None:
            line += f"   Δ {(summary[key] - float(previous[key])) * 100:+.1f} п.п. к прогону {previous['date']} (промпт {previous['prompt_version']})"
        lines.append(line)
    if previous is None:
        lines.append("\n  Предыдущих прогонов этой модели на этих наборах нет — сравнивать пока не с чем.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Построить дашборды по наборам данных и оценить их по эталону")
    parser.add_argument("--backend", default=None, help="gigachat | stub | auto (по умолчанию по окружению)")
    parser.add_argument("--set", dest="data_set", choices=sorted(SETS), default="all",
                        help="demo — демо-витрина, flat — плоские образцы, all — все наборы")
    parser.add_argument("--note", default="", help="пометка к прогону, например «правка промпта дашборда»")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--details-dir", type=Path, default=DETAILS_DIR)
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    load_env()
    try:
        llm = make_llm(args.backend)
    except ConfigError as exc:
        print(f"Не удалось создать клиент модели: {exc}", file=sys.stderr)
        return 2
    saved = os.environ.get("KPI_DATA_PATH")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            items = run_sets(llm, SETS[args.data_set], Path(tmp))
    finally:
        if saved is None:
            os.environ.pop("KPI_DATA_PATH", None)
        else:
            os.environ["KPI_DATA_PATH"] = saved
        datastore._cache.clear()
    summary = aggregate(items)
    now = datetime.now(timezone.utc)
    row = {"date": now.isoformat(timespec="seconds"), "prompt_version": DASHBOARD_PROMPT_VERSION, "llm": llm.name,
           "set": args.data_set, **{k: v for k, v in summary.items() if k != "n"}, "n": summary["n"], "note": args.note}
    previous = next((r for r in reversed(read_rows(args.results)) if r.get("llm") == llm.name and r.get("set") == args.data_set), None)
    args.details_dir.mkdir(parents=True, exist_ok=True)
    details = args.details_dir / f"dashboards-{now.strftime('%Y%m%dT%H%M%S')}.json"
    details.write_text(json.dumps({"row": row, "items": items}, ensure_ascii=False, indent=2, default=str), "utf-8")
    append_row(args.results, row)
    print(format_report(summary, row, previous))
    print(f"\nСтрока добавлена в {args.results}, разбор по наборам — {details}")
    return 0


def append_row(path: Path, row: dict) -> None:
    """Дописывает строку в csv со своим набором столбцов (общая запись из run_evals знает только столбцы вопросов)."""
    exists = path.exists() and path.stat().st_size > 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in COLUMNS})


if __name__ == "__main__":
    sys.exit(main())
