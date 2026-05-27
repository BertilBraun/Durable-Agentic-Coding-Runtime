from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.runtime_enums import StrEnum


class WorkerStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    NEEDS_REPLAN = "needs_replan"


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TestResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    command: str
    exit_code: int
    stdout_summary: str
    stderr_summary: str
    passed: bool


class WorkerResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: WorkerStatus
    patch_id: str | None = None
    diff_summary: str
    tests_run: list[str] = Field(default_factory=list)
    test_results: list[TestResult] = Field(default_factory=list)
    discovered_issues: list[str] = Field(default_factory=list)
    confidence: Confidence
    replan_suggestion: str | None = None
