from __future__ import annotations

from pydantic import Field

from src.models.frozen_base_model import FrozenBaseModel
from src.runtime_enums import StrEnum


class Risk(StrEnum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'


class PlanStep(FrozenBaseModel):
    id: str = Field(description='Stable identifier for this independently executable step.')
    goal: str = Field(
        description=(
            'Concrete outcome this step must produce. The worker should complete this step '
            'only, while preserving already completed prior work.'
        )
    )
    target_files: list[str] = Field(
        default_factory=list,
        description=(
            'Files this step is expected to inspect or modify; extra files require evidence.'
        ),
    )
    tests_to_run: list[str] = Field(
        default_factory=list,
        description='Relevant verification commands for this step, not a separate work item.',
    )
    expected_result: str = Field(description='Observable state that means this step is complete.')
    risk: Risk = Field(description='Risk of this step if implemented incorrectly.')
    requires_human_approval: bool = Field(
        description='Whether this step must pause for approval before implementation.'
    )


class Plan(FrozenBaseModel):
    summary: str = Field(description='Short explanation of the overall implementation strategy.')
    steps: list[PlanStep] = Field(
        default_factory=list,
        description='Coherent implementation steps that each run in an independent child workflow.',
    )
    integration_tests: list[str] = Field(
        default_factory=list,
        description='End-to-end verification commands to run after planned implementation work.',
    )
    definition_of_done: list[str] = Field(
        default_factory=list,
        description='Evidence required before the whole patch can be considered complete.',
    )


class PlanContext(FrozenBaseModel):
    summary: str = Field(description='Overall plan summary for orientation.')
    current_step_id: str = Field(description='Identifier of the step the worker must execute.')
    all_step_ids: list[str] = Field(
        default_factory=list,
        description='Ordered identifiers for every step in the plan.',
    )
    completed_step_summaries: list[str] = Field(
        default_factory=list,
        description='Summaries of accepted prior steps already present in the workspace.',
    )
