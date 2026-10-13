"""Русская запись чисел и названий в текстах, которые собирает код (заглушка модели, сводка дашборда)."""

MINUS = "−"


def num(x, signed: bool = False) -> str:
    """Число в русской записи с теми же цифрами, что в результате инструмента (знак — для наглядности).
    Крупные числа разбиваются на разряды неразрывным пробелом: «2 819 730»."""
    if x is None:
        return "н/д"
    a = abs(x)
    if float(a).is_integer() and a >= 1000:
        text = f"{int(a):,}".replace(",", " ")
    else:
        whole, _, frac = str(a).partition(".")
        whole = f"{int(whole):,}".replace(",", " ") if len(whole) > 4 else whole
        text = whole + ("," + frac if frac else "")
    if x < 0:
        return f"{MINUS}{text}"
    return f"+{text}" if signed and x > 0 else text


def unit(u: str) -> str:
    """Единица с пробелом перед ней («5 мин»), проценты слитно («9,06%»), без единицы — пустая строка."""
    return "" if not u else u if u == "%" else f" {u}"


def dim_word(name: str) -> str:
    """Название измерения в тексте: «Канал» -> «канал», но «CRM» и «ИП» остаются как есть."""
    return name.lower() if name[1:].islower() else name
