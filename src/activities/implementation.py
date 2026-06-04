from __future__ import annotations

import json
from dataclasses import dataclass
from typing import get_args

from pydantic import Field
from temporal_light import activity

from src.activities.context_gatherer import (
    ContextGatherRequest,
    gather_context,
)
from src.activities.workspace_manager import (
    ToolExecutionRequest,
    ToolResult,
    Workspace,
    run_tool,
)
from src.config import CONFIG, ModelRole
from src.llm.client import LLMUsage, Message, generate_structured
from src.models.context import ContextPack
from src.models.frozen_base_model import FrozenBaseModel
from src.models.plan import PlanContext, PlanStep
from src.models.repo import RepoIndex
from src.models.task import TaskContract
from src.models.worker import Confidence, TestResult, WorkerResult, WorkerStatus
from src.tools.definitions import (
    ApplyPatch,
    GatherContext,
    ImplementationToolCall,
    RunTests,
    WriteFile,
)
from src.tools.handlers import pytest_command

IMPLEMENTATION_SYSTEM_PROMPT = (
    'You execute exactly one planner-selected step. '
    'The workspace already contains accepted prior steps listed in '
    'completed_step_summaries; preserve that work and do not redo it. '
    'Use plan_step.target_files, plan_step.context_summary, '
    'plan_step.required_changes, plan_step.tests_to_run, and plan_step.out_of_scope '
    'as the contract for this turn. Inspect before editing when context is insufficient, '
    'then use the smallest patch that satisfies the current step. Edit with write_file '
    'or apply_patch and run tests with run_tests, passing pytest node ids or file paths as '
    'test_targets (the runner is fixed to "python -m pytest"; never include the runner). '
    'Use run_shell for any other command (status, '
    'diff, search, reading files), writing it for the environment described '
    'in the user payload. Keep changes inside allowed files unless blocked, '
    'and explain why any extra file is needed. Run relevant tests after edits; '
    'inspect failures before editing again. Return done=true with WorkerResult '
    'only for complete, blocked, failed, or needs_replan outcomes. Report '
    'success only with an applied edit or observed test evidence. Do not broaden scope. '
    'Make the fix complete and consistent across the affected component, not just the literal '
    'example in the issue: when your change handles one keyword, path, format, or one direction '
    'of a symmetric operation, check whether sibling keywords, paths, formats, or the reverse '
    'direction need the same treatment for the behavior to be correct in general. '
    'When you add or strengthen tests, prefer self-validating checks over guessed values: '
    'assert round trips for symmetric behavior and invariants otherwise, use a literal '
    'expected value only when the issue or spec states it, and cover multiple inputs and '
    'input counts; before reporting success confirm the tests would fail a trivially-wrong '
    'implementation, not merely that they ran green. '
    'Do not delete or rewrite existing tests unless the step explicitly requires it. '
    'Return concise observations that help the outer planner decide the next future step. '
    'Confidence '
    'means confidence in this step based on observed diff and tests, not how '
    'hard the task felt. Use needs_replan when partial progress exists but '
    'more work is required in the same workspace. Do not fabricate progress, '
    'files, or test results. For bugfix work a failing regression test may '
    'already exist; make it pass by fixing the production code and never '
    'weaken, skip, delete, or otherwise neuter that test to go green. The '
    'environment is persistent across tool calls, so installed dependencies '
    'and prior edits remain available.'
)


IMPLEMENTATION_AVAILABLE_TOOLS: tuple[str, ...] = tuple(
    tool_type.model_fields['tool_name'].default.value
    for tool_type in get_args(ImplementationToolCall)
)


class ImplementationTurnRequest(FrozenBaseModel):
    plan_step: PlanStep
    plan_context: PlanContext | None = None
    context_pack: ContextPack
    task_contract: TaskContract
    workspace_info: Workspace
    repo_index: RepoIndex


class ImplementationAgentTurn(FrozenBaseModel):
    done: bool
    worker_result: WorkerResult | None = None
    tool_calls: list[ImplementationToolCall] = Field(default_factory=list)


@dataclass(frozen=True)
class ImplementationEvidence:
    tests_run: tuple[str, ...]
    test_results: tuple[TestResult, ...]
    saw_diff: bool


async def run_implementation_turn(
    request: ImplementationTurnRequest,
) -> tuple[WorkerResult, LLMUsage]:
    messages = [
        Message(
            role='system',
            content=IMPLEMENTATION_SYSTEM_PROMPT,
            cacheable=True,
        ),
        Message(role='user', content=json.dumps(_llm_user_payload(request))),
    ]
    max_tool_rounds = CONFIG.implementation_max_tool_rounds
    stop_threshold = CONFIG.context_utilization_stop_threshold
    tests_run: list[str] = []
    test_results: list[TestResult] = []
    saw_diff = False
    completed_tool_calls: list[str] = []
    usage = LLMUsage()
    for _ in range(max_tool_rounds):
        completion = await generate_structured(
            role=ModelRole.IMPLEMENTATION,
            messages=messages,
            output_type=ImplementationAgentTurn,
        )
        usage += completion.usage
        agent_turn = completion.output
        if agent_turn.done:
            if agent_turn.worker_result is None:
                raise ValueError('worker_result is required when implementation turn is done')
            return (
                _worker_result_with_evidence(
                    worker_result=agent_turn.worker_result,
                    evidence=ImplementationEvidence(
                        tests_run=tuple(tests_run),
                        test_results=tuple(test_results),
                        saw_diff=saw_diff,
                    ),
                ),
                usage,
            )
        if completion.context_utilization() > stop_threshold:
            return (
                _context_budget_blocked_worker_result(
                    completed_tool_calls=completed_tool_calls,
                    pending_tool_calls=[
                        tool_call.tool_name.value for tool_call in agent_turn.tool_calls
                    ],
                ),
                usage,
            )

        observations: list[str] = []
        for tool_call in agent_turn.tool_calls:
            match tool_call:
                case GatherContext(prompt=prompt):
                    gathered_context, gather_usage = await gather_context(
                        ContextGatherRequest(
                            workspace_info=request.workspace_info,
                            repo_index=request.repo_index,
                            gatherer_prompt=prompt,
                        )
                    )
                    usage += gather_usage
                    observations.append(
                        f'tool={tool_call.tool_name} context_pack:\n'
                        f'{gathered_context.model_dump_json()}'
                    )
                    completed_tool_calls.append(tool_call.tool_name.value)
                case tool:
                    tool_result = await run_tool(
                        ToolExecutionRequest(
                            workspace=request.workspace_info,
                            tool=tool,
                            repo_index=request.repo_index,
                        )
                    )
                    observations.append(f'tool_result:\n{tool_result.model_dump_json()}')
                    completed_tool_calls.append(tool_call.tool_name.value)
                    match tool:
                        case RunTests(test_targets=test_targets):
                            command = pytest_command(test_targets)
                            tests_run.append(command)
                            test_results.append(
                                _test_result_from_tool_result(
                                    command=command,
                                    tool_result=tool_result,
                                    sequence=len(test_results) + 1,
                                )
                            )
                        case WriteFile() | ApplyPatch():
                            saw_diff = saw_diff or tool_result.exit_code == 0
                        case _:
                            pass
        messages.append(Message(role='assistant', content=agent_turn.model_dump_json()))
        messages.append(Message(role='user', content='\n\n'.join(observations)))

    return (
        _tool_rounds_exhausted_worker_result(
            evidence=ImplementationEvidence(
                tests_run=tuple(tests_run),
                test_results=tuple(test_results),
                saw_diff=saw_diff,
            ),
            completed_tool_calls=completed_tool_calls,
        ),
        usage,
    )


def _llm_user_payload(request: ImplementationTurnRequest) -> dict[str, object]:
    completed_step_summaries: list[str] = []
    if request.plan_context is not None:
        completed_step_summaries = request.plan_context.completed_step_summaries
    return {
        'plan_step': request.plan_step.model_dump(mode='json'),
        'completed_step_summaries': completed_step_summaries,
        'context_pack': request.context_pack.model_dump(mode='json'),
        'task_contract': request.task_contract.model_dump(mode='json'),
        'workspace_info': request.workspace_info.model_dump(mode='json'),
        'environment': request.workspace_info.describe_environment(),
        'available_tools': list(IMPLEMENTATION_AVAILABLE_TOOLS),
    }


def _context_budget_blocked_worker_result(
    completed_tool_calls: list[str],
    pending_tool_calls: list[str],
) -> WorkerResult:
    completed_summary = ', '.join(completed_tool_calls) if completed_tool_calls else 'none'
    pending_summary = ', '.join(pending_tool_calls) if pending_tool_calls else 'none'
    return WorkerResult(
        status=WorkerStatus.BLOCKED,
        patch_id=None,
        diff_summary='Implementation stopped before exceeding the context window budget.',
        tests_run=[],
        test_results=[],
        discovered_issues=['context utilization exceeded 80 percent'],
        confidence=Confidence.LOW,
        replan_suggestion=(
            'Context budget exceeded. Completed tool calls: '
            f'{completed_summary}. Pending tool calls: {pending_summary}.'
        ),
    )


def _tool_rounds_exhausted_worker_result(
    evidence: ImplementationEvidence,
    completed_tool_calls: list[str],
) -> WorkerResult:
    completed_summary = ', '.join(completed_tool_calls) if completed_tool_calls else 'none'
    confidence = Confidence.MEDIUM if evidence.saw_diff or evidence.test_results else Confidence.LOW
    return WorkerResult(
        status=WorkerStatus.NEEDS_REPLAN,
        patch_id=None,
        diff_summary=(
            'Partial implementation work is present.'
            if evidence.saw_diff
            else 'Implementation did not complete within the tool-round budget.'
        ),
        tests_run=list(evidence.tests_run),
        test_results=list(evidence.test_results),
        discovered_issues=['maximum implementation tool rounds reached'],
        confidence=confidence,
        replan_suggestion=(
            'Continue this step from the current workspace state. Current workspace may contain '
            'partial edits; inspect git status, git diff, and the relevant files before editing. '
            f'Completed tool calls: {completed_summary}.'
        ),
    )


@activity(retries=0, timeout=120)
async def get_full_diff(workspace: Workspace) -> str:
    return workspace.diff_against_base()


def failed_worker_result(reason: str) -> WorkerResult:
    return WorkerResult(
        status=WorkerStatus.FAILED,
        patch_id=None,
        diff_summary='Implementation did not complete within the iteration budget.',
        tests_run=[],
        test_results=[],
        discovered_issues=[reason],
        confidence=Confidence.LOW,
        replan_suggestion=None,
    )


def _worker_result_with_evidence(
    worker_result: WorkerResult,
    evidence: ImplementationEvidence,
) -> WorkerResult:
    if worker_result.status != WorkerStatus.SUCCESS:
        return worker_result
    test_results = _select_reported_test_results(
        [*worker_result.test_results, *evidence.test_results]
    )
    if test_results:
        tests_run = [test_result.command for test_result in test_results]
    else:
        tests_run = list(dict.fromkeys([*worker_result.tests_run, *evidence.tests_run]))
    has_diff_evidence = evidence.saw_diff
    if not has_diff_evidence and not test_results:
        return failed_worker_result('success result missing diff or test evidence')
    return worker_result.model_copy(
        update={
            'tests_run': tests_run,
            'test_results': test_results,
        }
    )


def _select_reported_test_results(test_results: list[TestResult]) -> list[TestResult]:
    if not test_results:
        return []
    latest_result = test_results[-1]
    if latest_result.passed:
        return [latest_result]
    for index in range(len(test_results) - 2, -1, -1):
        if test_results[index].passed:
            return test_results[index:]
    return test_results


def _test_result_from_tool_result(
    command: str,
    tool_result: ToolResult,
    sequence: int,
) -> TestResult:
    return TestResult(
        sequence=sequence,
        command=command,
        exit_code=tool_result.exit_code,
        stdout_summary=tool_result.stdout,
        stderr_summary=tool_result.stderr,
        passed=tool_result.exit_code == 0,
    )
