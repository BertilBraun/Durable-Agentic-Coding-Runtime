from __future__ import annotations

from pydantic import Field

from src.models.frozen_base_model import FrozenBaseModel
from src.runtime_enums import StrEnum


class ArtifactKind(StrEnum):
    TEST_OUTPUT = 'test_output'
    DIFF = 'diff'
    LOG = 'log'
    SCREENSHOT = 'screenshot'
    REPO_INDEX = 'repo_index'
    CONTEXT_OVERFLOW = 'context_overflow'


class ArtifactReference(FrozenBaseModel):
    path: str
    summary: str
    kind: ArtifactKind


class ContextSnippet(FrozenBaseModel):
    file_path: str
    start_line: int
    end_line: int
    reason: str


class PackedSnippet(FrozenBaseModel):
    file_path: str
    start_line: int
    end_line: int
    reason: str
    content: str


class ContextPack(FrozenBaseModel):
    task_summary: str
    snippets: list[PackedSnippet] = Field(default_factory=list)
    artifact_references: list[ArtifactReference] = Field(default_factory=list)
    budget_remaining: int
