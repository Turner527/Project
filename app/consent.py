"""Согласие на отправку данных в GigaChat: пользователь видит, что именно уйдёт, до первой отправки.

Порядок такой. Страница запрашивает /api/preview: сервер собирает сводку отчёта один раз и запоминает её под отпечатком
отчёта (сборка ограничена по времени и не обязана давать один и тот же результат дважды). Токен предпросмотра — хеш
сводки и каталога, которые уйдут в модель. /api/consent с этим токеном помечает отчёт подтверждённым; дашборд строится по
той же самой запомненной сводке, поэтому в модель уходит ровно то, что пользователь видел.

Согласие хранится в памяти процесса (после перезапуска сервера спросим снова) и привязано к отпечатку отчёта и токену:
другой файл, другие роли или исключённые столбцы, другая сводка — другой токен, и подтверждение не действует.
"""
import hashlib
import json
import threading
from collections import OrderedDict

from agent.prompts import catalog_text
from data.model import Table
from tools import dashboard as D

KEEP = 8  # для скольких последних отчётов помнятся сводка и согласие
CONSENT_DETAIL = ("Перед первой отправкой данных в GigaChat нужно посмотреть и подтвердить, что именно уйдёт. "
                  "Откройте сводку и нажмите «Отправить».")
STALE_DETAIL = "Сводка изменилась, откройте её заново"


def _token(digest_text: str, catalog: str) -> str:
    return hashlib.sha256((digest_text + catalog).encode("utf-8")).hexdigest()


def _about(table: Table, kind: str) -> list[str]:
    lines = [
        "При построении дашборда в GigaChat уходит сводка отчёта (она показана ниже): названия показателей и разрезов, "
        "итоги, ряды по периодам, вклады значений.",
        "Каждый вопрос в чате отправляет в GigaChat текст вопроса, каталог отчёта (показан ниже) и агрегированные "
        "результаты расчётов: числа, названия показателей и значений разрезов.",
        "Строки таблицы и заметки из файла не отправляются никогда.",
        f"Имя файла заменено подписью «{table.public_name}».",
    ]
    if table.spec.excluded:
        lines.append("Столбцы с пометкой «Не отправлять в модель» не участвуют ни в сводке, ни в вопросах. Фильтры по ним "
                     "действуют только у вас: в модель уходит лишь число исключённых строк.")
    if kind == "strategic":
        lines.append("Отчёт в стратегическом формате (листы goals, facts, links): пометить столбцы «Не отправлять» нельзя, "
                     "названия целей, показателей, сегментов и владельцев уходят целиком.")
    return lines


class ConsentGate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._digests: OrderedDict[str, dict] = OrderedDict()  # отпечаток отчёта -> сводка, показанная пользователю
        self._confirmed: OrderedDict[tuple[str, str], bool] = OrderedDict()  # (отпечаток, токен) подтверждены

    def digest(self, table: Table, fingerprint: str) -> dict:
        """Сводка отчёта: первая собранная под этим отпечатком запоминается, дальше отдаётся она же."""
        with self._lock:
            cached = self._digests.get(fingerprint)
        if cached is not None:
            return cached
        built = D.build_digest(table)  # вне замка: сборка может занять десятки секунд, а статус опрашивается постоянно
        with self._lock:
            digest = self._digests.setdefault(fingerprint, built)
            while len(self._digests) > KEEP:
                self._digests.popitem(last=False)
        return digest

    def approved_digest(self, table: Table, fingerprint: str) -> dict | None:
        """Подтверждённая пользователем сводка этого отчёта или None, если подтверждения нет (или оно устарело)."""
        with self._lock:
            digest = self._digests.get(fingerprint)
        if digest is None:
            return None
        token = _token(D.digest_text(digest), catalog_text(table))
        with self._lock:
            return digest if (fingerprint, token) in self._confirmed else None

    def is_confirmed(self, table: Table, fingerprint: str) -> bool:
        return self.approved_digest(table, fingerprint) is not None

    def preview(self, table: Table, fingerprint: str, *, required: bool, kind: str, llm_name: str | None) -> dict:
        digest = self.digest(table, fingerprint)
        text = D.digest_text(digest)
        catalog = catalog_text(table)
        token = _token(text, catalog)
        with self._lock:
            confirmed = (fingerprint, token) in self._confirmed
        return {
            "required": required, "confirmed": confirmed, "digest_text": text,
            "digest_pretty": json.dumps(digest, ensure_ascii=False, indent=2), "catalog": catalog,
            "stats": {"chars": len(text), "metrics": len(digest["metrics"]), "dimensions": len(digest["dimensions"])},
            "excluded_columns": list(table.spec.excluded), "file_name_sent_as": table.public_name, "token": token,
            "report_kind": kind, "llm": llm_name, "about": _about(table, kind),
        }

    def consent(self, table: Table, fingerprint: str, token: str) -> bool:
        """Подтверждает текущий отчёт, если токен совпадает с токеном его нынешней сводки."""
        with self._lock:
            digest = self._digests.get(fingerprint)
        if digest is None or token != _token(D.digest_text(digest), catalog_text(table)):
            return False
        with self._lock:
            self._confirmed[(fingerprint, token)] = True
            while len(self._confirmed) > KEEP:
                self._confirmed.popitem(last=False)
        return True
