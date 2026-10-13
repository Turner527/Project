"""Структуры, которые модель обязана вернуть JSON-ом: итоговая записка и план дашборда."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Diagnosis(BaseModel):
    summary: str = Field(min_length=1, description="2-4 предложения: что происходит и где")
    root_cause: str = Field(min_length=1, description="Короткое название причины отклонения")
    key_segments: list[str] = Field(default_factory=list, description="Сегменты, дающие основной вклад")


class Scenario(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    effect_estimate: str = Field(min_length=1, description="Оценка эффекта только из чисел инструментов")
    assumptions: list[str] = Field(default_factory=list)


class Memo(BaseModel):
    diagnosis: Diagnosis
    scenarios: list[Scenario] = Field(default_factory=list, max_length=3)
    manual_checks: list[str] = Field(min_length=1)
    used_tool_calls: list[str] = Field(default_factory=list)

    def texts(self) -> dict[str, str]:
        """Все текстовые поля записки: сюда смотрит верификатор чисел."""
        out = {
            "diagnosis.summary": self.diagnosis.summary,
            "diagnosis.root_cause": self.diagnosis.root_cause,
        }
        for i, s in enumerate(self.scenarios, 1):
            out[f"scenarios[{i}].name"] = s.name
            out[f"scenarios[{i}].description"] = s.description
            out[f"scenarios[{i}].effect_estimate"] = s.effect_estimate
            for j, a in enumerate(s.assumptions, 1):
                out[f"scenarios[{i}].assumptions[{j}]"] = a
        for k, m in enumerate(self.manual_checks, 1):
            out[f"manual_checks[{k}]"] = m
        return out


# ---------- план дашборда ----------

WIDGET_TYPES = ("kpi", "trend", "breakdown")
SEVERITIES = ("problem", "risk", "opportunity", "info")
_TYPE_ALIASES = {
    "kpi": "kpi", "card": "kpi", "tile": "kpi", "metric": "kpi", "number": "kpi", "плитка": "kpi", "карточка": "kpi", "показатель": "kpi",
    "trend": "trend", "line": "trend", "linechart": "trend", "line_chart": "trend", "timeseries": "trend", "time_series": "trend",
    "area": "trend", "динамика": "trend", "график": "trend", "тренд": "trend",
    "breakdown": "breakdown", "bar": "breakdown", "bars": "breakdown", "barchart": "breakdown", "bar_chart": "breakdown",
    "contribution": "breakdown", "разбивка": "breakdown", "вклад": "breakdown", "столбцы": "breakdown",
}
_SEVERITY_ALIASES = {
    "problem": "problem", "critical": "problem", "high": "problem", "negative": "problem", "проблема": "problem", "риск_высокий": "problem",
    "risk": "risk", "warning": "risk", "medium": "risk", "warn": "risk", "риск": "risk", "предупреждение": "risk",
    "opportunity": "opportunity", "positive": "opportunity", "good": "opportunity", "возможность": "opportunity", "успех": "opportunity",
    "info": "info", "low": "info", "neutral": "info", "note": "info", "информация": "info", "наблюдение": "info", "инфо": "info",
}


def _cut(value, limit: int):
    """Слишком длинный текст модели обрезается, а не отвергается: ответ с лишним словом не должен ронять весь дашборд."""
    return value.strip()[:limit] if isinstance(value, str) else value


class Widget(BaseModel):
    """Один блок дашборда: что показать. Числа в нём всегда считает код."""

    id: str | None = None
    type: Literal["kpi", "trend", "breakdown"]
    metric: str = Field(min_length=1, description="Название показателя ровно как в сводке")
    title: str = Field(min_length=1, max_length=140)
    note: str = Field(default="", max_length=400, description="Подпись под блоком: что на нём видно")
    dimension: str | None = Field(default=None, description="breakdown: по какому измерению; trend: split_by")
    baseline: str = Field(default="auto", description="auto | plan | prev | yoy")
    where_dimension: str | None = None
    where_value: str | None = None
    last_days: int | None = Field(default=None, ge=1, le=3650)
    last_months: int | None = Field(default=None, ge=1, le=120)

    @field_validator("type", mode="before")
    @classmethod
    def _type(cls, v):
        return _TYPE_ALIASES.get(str(v).strip().lower().replace(" ", "_"), v)

    @field_validator("title", mode="before")
    @classmethod
    def _title(cls, v):
        return _cut(v, 140)

    @field_validator("note", mode="before")
    @classmethod
    def _note(cls, v):
        return _cut(v, 400) if v is not None else ""

    @field_validator("id", "dimension", "where_dimension", "where_value", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        if v is None:
            return None
        text = str(v).strip()
        return text or None

    @field_validator("baseline", mode="before")
    @classmethod
    def _baseline(cls, v):
        return str(v).strip().lower() if v else "auto"

    @field_validator("last_days", "last_months", mode="before")
    @classmethod
    def _positive_or_none(cls, v):
        try:
            return int(v) if v not in (None, "", 0, "0") else None
        except (TypeError, ValueError):
            return None


class Insight(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=700)
    severity: Literal["problem", "risk", "opportunity", "info"] = "info"
    widgets: list[str] = Field(default_factory=list, description="id блоков, на которых это видно")

    @field_validator("title", mode="before")
    @classmethod
    def _title(cls, v):
        return _cut(v, 120)

    @field_validator("text", mode="before")
    @classmethod
    def _text(cls, v):
        return _cut(v, 700)

    @field_validator("severity", mode="before")
    @classmethod
    def _severity(cls, v):
        """Незнакомая важность — не повод терять весь план: инсайт остаётся, помечается как «наблюдение»."""
        return _SEVERITY_ALIASES.get(str(v).strip().lower().replace(" ", "_"), "info") if v else "info"

    @field_validator("widgets", mode="before")
    @classmethod
    def _widgets(cls, v):
        if v is None:
            return []
        return [str(x).strip() for x in (v if isinstance(v, list) else [v]) if str(x).strip()]


class DashboardPlan(BaseModel):
    """Что GigaChat рекомендует показать на дашборде и что он увидел в сводке отчёта."""

    title: str = Field(min_length=1, max_length=140)
    summary: str = Field(min_length=1, max_length=900, description="2-3 предложения: главное в отчёте")
    widgets: list[Widget] = Field(min_length=1, max_length=16)
    insights: list[Insight] = Field(min_length=1, max_length=10)
    next_steps: list[str] = Field(min_length=1, max_length=8, description="Что сделать или проверить дальше")
    questions: list[str] = Field(default_factory=list, max_length=10, description="Вопросы, которые стоит задать аналитику")

    @field_validator("title", mode="before")
    @classmethod
    def _title(cls, v):
        return _cut(v, 140)

    @field_validator("summary", mode="before")
    @classmethod
    def _summary(cls, v):
        return _cut(v, 900)

    @field_validator("next_steps", "questions", mode="before")
    @classmethod
    def _strings(cls, v):
        if v is None:
            return []
        items = v if isinstance(v, list) else [v]
        return [_cut(str(x), 400) for x in items if str(x).strip()]

    def texts(self) -> dict[str, str]:
        """Все тексты плана: сюда смотрит верификатор чисел."""
        out = {"title": self.title, "summary": self.summary}
        for i, w in enumerate(self.widgets, 1):
            out[f"widgets[{i}].title"] = w.title
            if w.note:
                out[f"widgets[{i}].note"] = w.note
        for i, ins in enumerate(self.insights, 1):
            out[f"insights[{i}].title"] = ins.title
            out[f"insights[{i}].text"] = ins.text
        for i, step in enumerate(self.next_steps, 1):
            out[f"next_steps[{i}]"] = step
        for i, q in enumerate(self.questions, 1):
            out[f"questions[{i}]"] = q
        return out
