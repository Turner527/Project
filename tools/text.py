"""Сопоставление названий по основам слов (первые 4 буквы): «розничного» ~ «розничные»."""
import math
import re
from collections import Counter

_STOP = (
    "почему мы нам что как какие какой какая какое делать делаем отстаем отстаём цели цель целью показатель "
    "показателя показатели показателю kpi кпи это для при про над или его они там где стоит нужно надо можно период "
    "месяц месяцев последние последних последний три шесть расскажи объясни покажи отчет отчёт данные данных "
    "изменилось изменение случилось произошло происходит"
).split()
# Распространённые формы тех же слов. Слова из запроса отбрасываются по точной форме или по основе — но основа не
# отбрасывается, если она есть в названиях отчёта: «Показы» (как «показатель») и «Стоимость» (как «стоит») — это
# настоящие названия, их нельзя прятать за стоп-словами.
_STOP_FORMS = (
    "показателем показателями показателях показателей целей целям отчета отчеты периода периоде периодом месяца месяцы "
    "месяце данными данным изменения изменений изменился изменилась изменились происходило отстаёт отстает стоят"
).split()
STOP_WORDS = {w.replace("ё", "е") for w in _STOP + _STOP_FORMS}
STOP_STEMS = {w.replace("ё", "е")[:4] for w in _STOP}


def stems(text: str, *, stop: bool = True, keep: frozenset = frozenset()) -> set[str]:
    """Основы слов (первые 4 буквы). stop=False — без отбрасывания служебных слов (для названий);
    keep — основы, которые не отбрасываются по основе (названия, встречающиеся в отчёте)."""
    out = set()
    for w in re.findall(r"[a-zа-я0-9]+", str(text).lower().replace("ё", "е")):
        if len(w) <= 2:
            continue
        stem = w[:4]
        if stop and (w in STOP_WORDS or (stem in STOP_STEMS and stem not in keep)):
            continue
        out.add(stem)
    return out


def scores(query: str, docs: dict[str, str]) -> list[tuple[str, float]]:
    """Совпадение запроса с названиями: среднее двух долей (с весами IDF) — какая часть названия найдена
    в запросе и какая часть запроса объяснена названием. Полное упоминание названия даёт 1.0."""
    doc_stems = {k: stems(v, stop=False) for k, v in docs.items()}
    df = Counter(s for st in doc_stems.values() for s in st)
    idf = {s: math.log(1 + len(docs) / n) for s, n in df.items()}
    q = {s for s in stems(query, keep=frozenset(df)) if s in idf}  # слова, которых нет ни в одном названии, не штрафуют
    q_total = sum(idf[s] for s in q)
    out = []
    for key, st in doc_stems.items():
        total = sum(idf[s] for s in st)
        if total and q_total:
            hit = sum(idf[s] for s in st & q)
            full_mention = st <= q
            out.append((key, 1.0 if full_mention else round(0.5 * hit / total + 0.5 * hit / q_total, 3)))
    # при равной оценке — более конкретное (длинное) название: «Стоимость заказа» точнее, чем «Заказы»
    return sorted(out, key=lambda kv: (-kv[1], -len(doc_stems[kv[0]]), kv[0]))
