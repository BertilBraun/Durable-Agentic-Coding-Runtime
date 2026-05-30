from __future__ import annotations

import json
from dataclasses import dataclass
from typing import get_args

from pydantic import BaseModel, ConfigDict, Field

from src.activities.context_gatherer import (
    ContextGatherRequest,
    gather_context,
)
from src.activities.workspace_manager import (
    ToolExecutionRequest,
    ToolResult,
    WorkspaceInfo,
    run_tool,
)
from src.config import ModelRole, CONFIG
from src.llm.client import Message, generate_structured
from src.models.context import ContextPack
from src.models.plan import PlanStep
from src.models.repo import RepoIndex
from src.models.task import TaskContract
from src.models.worker import Confidence, TestResult, WorkerResult, WorkerStatus
from src.tools.definitions import (
    GatherContext,
    GitDiff,
    ImplementationToolCall,
    RunTests,
)

IMPLEMENTATION_SYSTEM_PROMPT = (
    'You are the implementation worker. Inspect before editing when context '
    'is insufficient, then use the smallest patch that satisfies the current '
    'plan step. Keep changes inside allowed files unless blocked, and explain '
    'why any extra file is needed. Use mutating tools only for the current '
    'step. Run relevant tests after edits; inspect failures before editing '
    'again. Return done=true with WorkerResult only for complete, blocked, '
    'failed, or needs_replan outcomes. Report success only with observed '
    'git_diff or test evidence. Do not fabricate progress, files, or test '
    'results. Each tool call runs in a fresh container, so command-local '
    'setup must be included in the same command.'
)


IMPLEMENTATION_AVAILABLE_TOOLS: tuple[str, ...] = tuple(
    tool_type.model_fields['tool_name'].default.value for tool_type in get_args(ImplementationToolCall)
)


class ImplementationTurnRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_step: PlanStep
    context_pack: ContextPack
    task_contract: TaskContract
    workspace_info: WorkspaceInfo
    repo_index: RepoIndex


class ImplementationAgentTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    done: bool
    worker_result: WorkerResult | None = None
    tool_calls: list[ImplementationToolCall] = Field(default_factory=list)


@dataclass(frozen=True)
class ImplementationEvidence:
    tests_run: tuple[str, ...]
    test_results: tuple[TestResult, ...]
    saw_diff: bool


# TODO very deeply nested.. But a lot of state to pass - so might be fine..
async def run_implementation_turn(request: ImplementationTurnRequest) -> WorkerResult:
    messages = [
        Message(
            role='system',
            content=IMPLEMENTATION_SYSTEM_PROMPT,
        ),
        Message(role='user', content=json.dumps(_llm_user_payload(request))),
    ]
    max_tool_rounds = CONFIG.implementation_max_tool_rounds
    stop_threshold = CONFIG.context_utilization_stop_threshold
    tests_run: list[str] = []
    test_results: list[TestResult] = []
    saw_diff = False
    completed_tool_calls: list[str] = []
    for _ in range(max_tool_rounds):
        completion = await generate_structured(
            role=ModelRole.IMPLEMENTATION,
            messages=messages,
            output_type=ImplementationAgentTurn,
        )
        agent_turn = completion.output
        if completion.result.context_utilization() > stop_threshold:
            return _context_budget_blocked_worker_result(
                completed_tool_calls=completed_tool_calls,
                pending_tool_calls=[tool_call.tool_name.value for tool_call in agent_turn.tool_calls],
            )
        if agent_turn.done:
            if agent_turn.worker_result is None:
                raise ValueError('worker_result is required when implementation turn is done')
            return _worker_result_with_evidence(
                worker_result=agent_turn.worker_result,
                evidence=ImplementationEvidence(
                    tests_run=tuple(tests_run),
                    test_results=tuple(test_results),
                    saw_diff=saw_diff,
                ),
            )

        observations: list[str] = []
        for tool_call in agent_turn.tool_calls:
            match tool_call:
                case GatherContext(prompt=prompt):
                    gathered_context = await gather_context(
                        ContextGatherRequest(
                            workspace_info=request.workspace_info,
                            repo_index=request.repo_index,
                            gatherer_prompt=prompt,
                        )
                    )
                    observations.append(
                        f'tool={tool_call.tool_name} context_pack:\n{gathered_context.model_dump_json()}'
                    )
                    completed_tool_calls.append(tool_call.tool_name.value)
                case tool:
                    tool_result = await run_tool(
                        ToolExecutionRequest(
                            workspace_info=request.workspace_info,
                            tool=tool,
                            repo_index=request.repo_index,
                        )
                    )
                    observations.append(
                        f'tool={tool_call.tool_name} exit_code={tool_result.exit_code}\n'
                        f'stdout:\n{tool_result.stdout}\n'
                        f'stderr:\n{tool_result.stderr}'
                    )
                    completed_tool_calls.append(tool_call.tool_name.value)
                    match tool:
                        case RunTests(command=command):
                            tests_run.append(command)
                            test_results.append(_test_result_from_tool_result(command, tool_result))
                        case GitDiff():
                            saw_diff = saw_diff or bool(tool_result.stdout.strip())
                        case _:
                            pass
        messages.append(Message(role='assistant', content=agent_turn.model_dump_json()))
        messages.append(Message(role='user', content='\n\n'.join(observations)))

    return failed_worker_result('maximum implementation tool rounds reached')


def _llm_user_payload(request: ImplementationTurnRequest) -> dict[str, object]:
    return {
        'plan_step': request.plan_step.model_dump(mode='json'),
        'context_pack': request.context_pack.model_dump(mode='json'),
        'task_contract': request.task_contract.model_dump(mode='json'),
        'workspace_info': request.workspace_info.model_dump(mode='json'),
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


async def get_full_diff(workspace_info: WorkspaceInfo) -> str:
    tool_result = await run_tool(ToolExecutionRequest(workspace_info=workspace_info, tool=GitDiff(path='.')))
    return tool_result.stdout


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
    tests_run = list(dict.fromkeys([*worker_result.tests_run, *evidence.tests_run]))
    test_results = [*worker_result.test_results, *evidence.test_results]
    has_diff_evidence = evidence.saw_diff
    if not has_diff_evidence and not test_results:
        return failed_worker_result('success result missing diff or test evidence')
    return worker_result.model_copy(
        update={
            'tests_run': tests_run,
            'test_results': test_results,
        }
    )


def _test_result_from_tool_result(command: str, tool_result: ToolResult) -> TestResult:
    return TestResult(
        command=command,
        exit_code=tool_result.exit_code,
        stdout_summary=tool_result.stdout,
        stderr_summary=tool_result.stderr,
        passed=tool_result.exit_code == 0,
    )
