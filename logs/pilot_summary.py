"""Сводка для пилота по SQLite-журналу.

Запуск:  python -m logs.pilot_summary [--db путь] [--top 5]
"""
import argparse
import re
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from pathlib import Path

from agent.verifier import STATUS_OK
from logs.store import STATUS_ERROR, db_path


def _normalize(question: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", question.lower().replace("ё", "е"))).strip()


def summarize(path: str | Path | None = None, top: int = 5) -> dict:
    with closing(sqlite3.connect(path or db_path())) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute("SELECT * FROM requests ORDER BY ts")]

    total = len(rows)
    errors = sum(1 for r in rows if r["verification_status"] == STATUS_ERROR)
    answered = total - errors
    verified = sum(1 for r in rows if r["verification_status"] == STATUS_OK)
    rated = [r for r in rows if r["rating"] is not None]
    useful = sum(1 for r in rows if r["rating"] == 1)

    originals: dict[str, Counter] = {}
    for r in rows:
        originals.setdefault(_normalize(r["question"]), Counter())[r["question"].strip()] += 1
    frequent = sorted(originals.items(), key=lambda kv: -sum(kv[1].values()))[:top]

    by_client: dict[str, list[str]] = {}
    for r in rows:
        if r["client_id"]:
            by_client.setdefault(r["client_id"], []).append(r["ts"][:10])
    latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    tokens = [r["tokens_total"] for r in rows if r["tokens_total"] is not None]
    return {
        "total": total,
        "errors": errors,
        "answered": answered,
        "verified": verified,
        "verified_share": verified / answered if answered else None,
        "rated": len(rated),
        "rating_coverage": len(rated) / total if total else None,
        "useful": useful,
        "useful_share": useful / len(rated) if rated else None,
        "top_questions": [(counter.most_common(1)[0][0], sum(counter.values())) for _, counter in frequent],
        "users": len(by_client),
        "repeat_users": sum(1 for days in by_client.values() if len(days) >= 2),
        "returned_other_day": sum(1 for days in by_client.values() if len(set(days)) >= 2),
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
        "avg_tokens": round(sum(tokens) / len(tokens)) if tokens else None,
    }


def _pct(x: float | None) -> str:
    return "н/д" if x is None else f"{x * 100:.0f}%"


def format_summary(s: dict) -> str:
    lines = [
        "Сводка пилота",
        "=============",
        f"Запросов всего:                   {s['total']} (из них с ошибкой: {s['errors']})",
        f"Ответов со статусом «проверен»:   {s['verified']} из {s['answered']} ({_pct(s['verified_share'])})",
        f"Оценок пользователей:             {s['rated']} (охват {_pct(s['rating_coverage'])})",
        f"Доля оценок «полезно»:            {_pct(s['useful_share'])} ({s['useful']} из {s['rated']})",
        f"Пользователей:                    {s['users']}",
        f"Повторных пользователей (2+ запроса): {s['repeat_users']}, из них вернулись в другой день: "
        f"{s['returned_other_day']}",
        f"Средняя задержка:                 {s['avg_latency_ms'] if s['avg_latency_ms'] is not None else 'н/д'} мс",
        f"Среднее число токенов на запрос:  {s['avg_tokens'] if s['avg_tokens'] is not None else 'н/д'}",
        "",
        "Самые частые вопросы:",
    ]
    lines += [f"  {n}× {q}" for q, n in s["top_questions"]] or ["  пока нет данных"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Сводка для пилота по журналу запросов")
    parser.add_argument("--db", default=None, help="путь к SQLite (по умолчанию logs/pilot.db или LOG_DB_PATH)")
    parser.add_argument("--top", type=int, default=5, help="сколько частых вопросов показать")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    path = Path(args.db) if args.db else db_path()
    if not path.exists():
        raise SystemExit(f"Журнал {path} не найден: пока не было ни одного запроса.")
    print(format_summary(summarize(path, args.top)))


if __name__ == "__main__":
    main()
