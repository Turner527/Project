"""Детерминированная проверка: каждое число в записке должно встречаться в результатах инструментов.

Что проверяется: наличие числа среди числовых полей результатов (или в тексте вопроса) с допуском на
округление, как оно записано в ответе: «86,7» допускает отклонение 0,05, «87» — 0,5. Знак игнорируется.
Что НЕ проверяется: что число относится именно к тому показателю, о котором сказано в тексте.
Не считаются числами: даты и годы, call_id, метки вида S1/30+, целые до 12 без единицы («3 сценария»).
Запись с множителем («2,8 млн», «450 тыс.») сверяется и как есть (данные могут быть уже в миллионах), и в пересчёте
на единицы («2 800 000»); допуск при пересчёте — половина последнего записанного разряда в тех же единицах.
"""
import re
from dataclasses import dataclass, field
from typing import Iterable

STATUS_OK = "проверен"
STATUS_UNVERIFIED = "есть непроверенные числа"

_CALL_ID = re.compile(r"\bc_[0-9a-z_]{4,}\b")
_DATE = re.compile(r"\b(?:19|20)\d{2}-\d{1,2}(?:-\d{1,2})?\b|\b\d{1,2}\.\d{1,2}\.\d{4}\b")
_LABEL = re.compile(r"\b[A-Za-zА-Яа-я]{1,2}\d{1,2}\b|\b\d+\+")
_NUMBER = re.compile(
    r"(?<![\w.,])(?P<sign>[-−–+])?"
    r"(?P<num>\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)"
)
_UNIT_AFTER = re.compile(r"^\s?(?:%|п\.\s?п\.|пп\b|процент|млрд|млн|тыс|пункт|мин\b|₽|руб|раз\b)", re.I)
_MULTIPLIER = re.compile(r"^\s?(тыс|млн|млрд|трлн)", re.I)
_MULTIPLIER_VALUE = {"тыс": 1e3, "млн": 1e6, "млрд": 1e9, "трлн": 1e12}
_TRIVIAL_MAX = 12  # целые до 12 без единицы — счёт, нумерация, месяцы
MAX_SOURCES_PER_NUMBER = 3


@dataclass
class Num:
    raw: str
    value: float  # модуль
    decimals: int
    start: int
    end: int
    multiplier: float = 1.0  # «тыс» — 1e3, «млн» — 1e6, «млрд» — 1e9


@dataclass
class Unverified:
    value: str
    field: str
    context: str


@dataclass
class Verification:
    status: str
    unverified: list[Unverified] = field(default_factory=list)
    checked: int = 0
    sources: dict[str, list[str]] = field(default_factory=dict)  # число из ответа -> call_id, где оно найдено
    unknown_call_ids: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "unverified": [u.__dict__ for u in self.unverified],
            "checked": self.checked,
            "sources": self.sources,
            "unknown_call_ids": self.unknown_call_ids,
        }


def _mask(text: str) -> str:
    """Заменяет call_id, даты и метки пробелами (индексы сохраняются)."""
    for pattern in (_CALL_ID, _DATE, _LABEL):
        text = pattern.sub(lambda m: " " * len(m.group()), text)
    return text


def extract_numbers(text: str, skip_trivial: bool = True) -> list[Num]:
    masked = _mask(text)
    out = []
    for m in _NUMBER.finditer(masked):
        digits = re.sub(r"[   ]", "", m.group("num")).replace(",", ".")
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        value = float(digits)
        if skip_trivial and decimals == 0 and not _UNIT_AFTER.match(masked[m.end():]):
            if value <= _TRIVIAL_MAX or 1990 <= value <= 2100:
                continue
        suffix = _MULTIPLIER.match(masked[m.end():])
        multiplier = _MULTIPLIER_VALUE[suffix.group(1).lower()] if suffix else 1.0
        out.append(Num(text[m.start("num"):m.end("num")], value, decimals, m.start(), m.end(), multiplier))
    return out


def numbers_match(written: Num | tuple[float, int], source: float) -> bool:
    """Совпадает ли записанное число с исходным в пределах округления записи."""
    value, decimals = (written.value, written.decimals) if isinstance(written, Num) else written
    multiplier = written.multiplier if isinstance(written, Num) else 1.0
    tolerance = 0.5 * 10 ** (-decimals) + 1e-9
    if abs(abs(value) - abs(source)) <= tolerance:
        return True
    return multiplier != 1.0 and abs(abs(value) * multiplier - abs(source)) <= tolerance * multiplier


def numeric_leaves(obj) -> Iterable[float]:
    """Все числовые значения вложенного JSON (bool не считается числом)."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from numeric_leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from numeric_leaves(v)


def verify(texts: dict[str, str], tool_calls: list, question: str = "", cited_call_ids: Iterable[str] = ()) -> Verification:
    """texts — поля записки {имя поля: текст}; tool_calls — ToolCallRecord'ы (нужны .call_id и .result)."""
    sources: list[tuple[float, str]] = [(v, rec.call_id) for rec in tool_calls for v in numeric_leaves(rec.result)]
    sources += [(n.value, "вопрос") for n in extract_numbers(question, skip_trivial=False)]

    result = Verification(status=STATUS_OK)
    for field_name, text in texts.items():
        for num in extract_numbers(text):
            result.checked += 1
            found = [cid for value, cid in sources if numbers_match(num, value)]
            if found:
                ids = result.sources.setdefault(num.raw, [])
                fresh = [c for c in dict.fromkeys(found) if c not in ids and c != "вопрос"]
                ids.extend(fresh[: max(0, MAX_SOURCES_PER_NUMBER - len(ids))])
            else:
                ctx = text[max(0, num.start - 30): num.end + 30].replace("\n", " ")
                result.unverified.append(Unverified(num.raw, field_name, ctx.strip()))
    known = {rec.call_id for rec in tool_calls}
    result.unknown_call_ids = [c for c in dict.fromkeys(cited_call_ids) if c not in known]
    if result.unverified:
        result.status = STATUS_UNVERIFIED
    return result
