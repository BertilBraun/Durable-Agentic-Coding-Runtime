from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.runtime_enums import StrEnum


class TaskType(StrEnum):
    BUGFIX = "bugfix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    FRONTEND = "frontend"
    TEST = "test"
    DOCS = "docs"
    UNKNOWN = "unknown"


class TaskRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    raw_request: str
    repo_path: str
    run_id: str | None = None
    docker_image: str | None = None


class TaskContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_type: TaskType
    goal: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    non_goals: list[str] = Field(default_factory=list)
    affected_areas: list[str] = Field(default_factory=list)
    risk_areas: list[str] = Field(default_factory=list)
    tests_expected: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
