from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.runtime_enums import StrEnum


class Risk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PlanStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    goal: str
    target_files: list[str] = Field(default_factory=list)
    allowed_files: list[str] = Field(default_factory=list)
    tests_to_run: list[str] = Field(default_factory=list)
    expected_result: str
    risk: Risk
    requires_human_approval: bool


class Plan(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: str
    steps: list[PlanStep] = Field(default_factory=list)
    integration_tests: list[str] = Field(default_factory=list)
    rollback_strategy: str
    definition_of_done: list[str] = Field(default_factory=list)
