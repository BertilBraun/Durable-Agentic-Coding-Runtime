from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.runtime_enums import StrEnum


class ArtifactKind(StrEnum):
    TEST_OUTPUT = 'test_output'
    DIFF = 'diff'
    LOG = 'log'
    SCREENSHOT = 'screenshot'
    REPO_INDEX = 'repo_index'


class ArtifactReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    summary: str
    kind: ArtifactKind


class ContextPack(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_summary: str
    relevant_snippets: list[str] = Field(default_factory=list)
    recent_observations: list[str] = Field(default_factory=list)
    failed_attempt_summaries: list[str] = Field(default_factory=list)
    available_tools: list[str] = Field(default_factory=list)
    budget_remaining: int
