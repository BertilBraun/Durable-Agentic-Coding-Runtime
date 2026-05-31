from __future__ import annotations

from pydantic import Field

from src.models.frozen_base_model import FrozenBaseModel
from src.runtime_enums import StrEnum


class WorkerStatus(StrEnum):
    SUCCESS = 'success'
    FAILED = 'failed'
    BLOCKED = 'blocked'
    NEEDS_REPLAN = 'needs_replan'


class Confidence(StrEnum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'


class TestResult(FrozenBaseModel):
    sequence: int = 0
    command: str
    exit_code: int
    stdout_summary: str
    stderr_summary: str
    passed: bool


class WorkerResult(FrozenBaseModel):
    diff_summary: str
    tests_run: list[str] = Field(default_factory=list)
    test_results: list[TestResult] = Field(default_factory=list)
    discovered_issues: list[str] = Field(default_factory=list)
    replan_suggestion: str | None = None
    patch_id: str | None = None
    confidence: Confidence
    status: WorkerStatus
