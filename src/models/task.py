from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from src.models.frozen_base_model import FrozenBaseModel
from src.runtime_enums import StrEnum


class TaskType(StrEnum):
    BUGFIX = 'bugfix'
    FEATURE = 'feature'
    REFACTOR = 'refactor'
    FRONTEND = 'frontend'
    TEST = 'test'
    DOCS = 'docs'
    UNKNOWN = 'unknown'


class HostOrigin(FrozenBaseModel):
    kind: Literal['host'] = 'host'
    repo_path: str


class DockerOrigin(FrozenBaseModel):
    kind: Literal['docker'] = 'docker'
    docker_image: str
    container_repo_path: str = '/testbed'


Origin = Annotated[HostOrigin | DockerOrigin, Field(discriminator='kind')]


class TaskRequest(FrozenBaseModel):
    raw_request: str
    run_id: str | None = None
    origin: Origin


class TaskContract(FrozenBaseModel):
    task_type: TaskType
    goal: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    non_goals: list[str] = Field(default_factory=list)
    affected_areas: list[str] = Field(default_factory=list)
    risk_areas: list[str] = Field(default_factory=list)
    tests_expected: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
