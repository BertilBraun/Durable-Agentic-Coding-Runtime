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
    sequence: int = Field(default=0, description='Order in which this verification ran.')
    command: str = Field(description='Exact command that was executed.')
    exit_code: int = Field(description='Process exit code returned by the command.')
    stdout_summary: str = Field(description='Concise summary of relevant stdout evidence.')
    stderr_summary: str = Field(description='Concise summary of relevant stderr evidence.')
    passed: bool = Field(description='Whether the command exited successfully.')


class WorkerResult(FrozenBaseModel):
    diff_summary: str = Field(
        description='What changed in this step, or why no complete change was produced.'
    )
    tests_run: list[str] = Field(
        default_factory=list,
        description='Exact verification commands run by this worker.',
    )
    test_results: list[TestResult] = Field(
        default_factory=list,
        description='Observed results for each verification command.',
    )
    discovered_issues: list[str] = Field(
        default_factory=list,
        description='Evidence-backed issues still requiring attention.',
    )
    replan_suggestion: str | None = Field(
        default=None,
        description='Specific continuation guidance when status is needs_replan.',
    )
    patch_id: str | None = Field(
        default=None,
        description='Optional identifier for an applied patch artifact.',
    )
    confidence: Confidence = Field(
        description=(
            'Confidence that this step is correct based on observed diff and tests: high means '
            'direct evidence covers the step, medium means useful evidence has gaps, low means '
            'evidence is weak, missing, or contradicted.'
        )
    )
    status: WorkerStatus = Field(description='Terminal outcome for this step attempt.')
