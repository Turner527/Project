"""Структура итоговой записки; именно её модель обязана вернуть JSON-ом."""
from pydantic import BaseModel, Field


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
