from __future__ import annotations

from pydantic import Field, model_validator

from src.models.context import ContextSnippet
from src.models.frozen_base_model import FrozenBaseModel
from src.models.repo import RepoIndex
from src.models.reproduction import ReproductionContext
from src.models.task import TaskContract
from src.models.worker import Confidence, TestResult, WorkerStatus
from src.runtime_enums import StrEnum
from src.tools.definitions import PlannerToolCall, ToolName


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
            'Files the implementer should inspect or modify for this step; extra files require '
            'evidence.'
        ),
    )
    context_summary: str = Field(
        default='',
        description='Step-specific code and domain context the implementer should rely on.',
    )
    required_changes: list[str] = Field(
        default_factory=list,
        description='Concrete changes this step must make or verify.',
    )
    out_of_scope: list[str] = Field(
        default_factory=list,
        description='Explicit work the implementer must not do in this step.',
    )
    tests_to_run: list[str] = Field(
        default_factory=list,
        description=(
            'Pytest targets to verify this step, each a file path or node id '
            '(e.g. "pkg/tests/test_mod.py::test_case"); the runner is fixed to '
            '"python -m pytest". Not a separate work item.'
        ),
    )
    expected_result: str = Field(description='Observable state that means this step is complete.')
    risk: Risk = Field(description='Risk of this step if implemented incorrectly.')


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


class ContextRequest(FrozenBaseModel):
    id: str = Field(description='Stable id for this context request.')
    reason: str = Field(
        description='Why this context is needed before planning implementation work.'
    )
    queries: list[str] = Field(
        default_factory=list,
        description='Concrete read-only questions or searches the context gatherer should answer.',
    )
    relevant_files: list[str] = Field(
        default_factory=list,
        description='Files the planner already suspects are relevant.',
    )


class ContextNote(FrozenBaseModel):
    id: str = Field(description='Stable id tying this note to gathered evidence.')
    summary: str = Field(description='Concise evidence summary produced after gathering context.')
    request_reason: str = Field(
        default='',
        description='Original planner reason for the fulfilled context request.',
    )
    request_queries: list[str] = Field(
        default_factory=list,
        description='Original planner queries fulfilled by this note.',
    )
    relevant_files: list[str] = Field(
        default_factory=list,
        description='Files supported by the gathered evidence.',
    )
    snippets: list[ContextSnippet] = Field(
        default_factory=list,
        description='Code snippet references that support this note.',
    )


class StepHistoryEntry(FrozenBaseModel):
    step_id: str
    outcome: WorkerStatus
    confidence: Confidence
    summary: str
    tests: list[TestResult] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)


class PlanningEvidence(FrozenBaseModel):
    diff_summary: str | None = None
    selected_test_results: list[TestResult] = Field(default_factory=list)
    reproduction_target: str | None = None
    reproduction_passed: bool | None = None
    reproduction_stdout_summary: str | None = None
    reproduction_stderr_summary: str | None = None


class PlannerToolObservation(FrozenBaseModel):
    tool_name: ToolName
    stdout: str
    stderr: str
    exit_code: int
    truncated: bool = False


class PlannerState(FrozenBaseModel):
    contract: TaskContract
    repo_index: RepoIndex
    reproduction: ReproductionContext | None = None
    context_notes: list[ContextNote] = Field(default_factory=list)
    tool_observations: list[PlannerToolObservation] = Field(default_factory=list)
    completed_steps: list[StepHistoryEntry] = Field(default_factory=list)
    previous_future_steps: list[PlanStep] = Field(default_factory=list)
    evidence: PlanningEvidence = Field(default_factory=PlanningEvidence)


class PlannerTurn(FrozenBaseModel):
    context_requests: list[ContextRequest] = Field(default_factory=list)
    tool_calls: list[PlannerToolCall] = Field(default_factory=list)
    future_steps: list[PlanStep] = Field(default_factory=list)
    done: bool = False
    done_reason: str | None = None

    @model_validator(mode='before')
    @classmethod
    def normalize_planner_tool_aliases(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        tool_calls = value.get('tool_calls')
        if not isinstance(tool_calls, list):
            return value
        return {
            **value,
            'tool_calls': [_normalize_planner_tool_call(tool_call) for tool_call in tool_calls],
        }


def _normalize_planner_tool_call(tool_call: object) -> object:
    if not isinstance(tool_call, dict):
        return tool_call
    tool_name = tool_call.get('tool_name')
    if tool_name == 'run_shell_command':
        return {**tool_call, 'tool_name': ToolName.RUN_SHELL.value}
    if tool_name != 'read_file':
        return tool_call
    if 'symbol_name' in tool_call and 'file_path' not in tool_call:
        return {
            'tool_name': ToolName.FIND_DEFINITION.value,
            'name': tool_call['symbol_name'],
            'language': tool_call.get('language', 'python'),
        }
    if 'path' in tool_call and 'file_path' not in tool_call:
        return {**tool_call, 'file_path': tool_call['path']}
    return tool_call
