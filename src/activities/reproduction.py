from __future__ import annotations

import json
from typing import get_args

from pydantic import Field

from src.activities.context_gatherer import ContextGatherRequest, gather_context
from src.activities.workspace_manager import (
    ToolExecutionRequest,
    ToolResult,
    Workspace,
    run_tool,
)
from src.config import CONFIG, ModelRole
from src.llm.client import LLMUsage, Message, generate_structured
from src.models.frozen_base_model import FrozenBaseModel
from src.models.repo import RepoIndex
from src.models.reproduction import ReproductionResult, ReproductionStatus
from src.models.task import TaskContract
from src.tools.definitions import GatherContext, ImplementationToolCall, RunTests

REPRODUCTION_COMMAND_TIMEOUT_SECONDS = 600

REPRODUCTION_SYSTEM_PROMPT = (
    'You are the reproduction agent for a bugfix task. Your only job is to '
    'demonstrate the bug with a test, never to fix it. Locate the buggy '
    'behavior using run_shell, find_definition / find_callers / find_callees, '
    'and gather_context, then write a single focused regression test that '
    'fails on the current (unfixed) code and isolates exactly the reported '
    'defect. Add the test with write_file or apply_patch and confirm it fails '
    'with run_tests; a failure from an assertion on the buggy behavior is a '
    'real reproduction, while a syntax, import, or collection error is not. Do '
    'not edit production code or weaken any existing test. Return done=true '
    'with a ReproductionResult: set status=reproduced only once you have seen '
    'the new test fail, giving the exact command that runs just that test, the '
    'files you added, and the observed failing output. If the contract is too '
    'vague to pin down, or the failure is flaky or environmental, return '
    'status=could_not_reproduce instead of guessing.'
)


REPRODUCTION_AVAILABLE_TOOLS: tuple[str, ...] = tuple(
    tool_type.model_fields['tool_name'].default.value
    for tool_type in get_args(ImplementationToolCall)
)


class ReproductionTurnRequest(FrozenBaseModel):
    task_contract: TaskContract
    workspace_info: Workspace
    repo_index: RepoIndex


class ReproductionAgentTurn(FrozenBaseModel):
    done: bool
    reproduction_result: ReproductionResult | None = None
    tool_calls: list[ImplementationToolCall] = Field(default_factory=list)


def repro_run_tests(repro_command: str) -> RunTests:
    return RunTests(
        command=repro_command,
        timeout_seconds=REPRODUCTION_COMMAND_TIMEOUT_SECONDS,
        directory='.',
    )


async def reproduce_bug(
    request: ReproductionTurnRequest,
) -> tuple[ReproductionResult, LLMUsage]:
    messages = [
        Message(role='system', content=REPRODUCTION_SYSTEM_PROMPT, cacheable=True),
        Message(role='user', content=json.dumps(_reproduction_user_payload(request))),
    ]
    max_tool_rounds = CONFIG.reproducer_max_tool_rounds
    stop_threshold = CONFIG.context_utilization_stop_threshold
    usage = LLMUsage()
    for _ in range(max_tool_rounds):
        completion = await generate_structured(
            role=ModelRole.REPRODUCER,
            messages=messages,
            output_type=ReproductionAgentTurn,
        )
        usage += completion.usage
        agent_turn = completion.output
        tool_observations, tool_usage = await _run_reproduction_tool_calls(
            request=request,
            tool_calls=agent_turn.tool_calls,
        )
        usage += tool_usage
        if agent_turn.done:
            if agent_turn.reproduction_result is None:
                raise ValueError('reproduction_result is required when reproduction turn is done')
            verified_result = await _verify_reproduction(
                request=request,
                reproduction_result=agent_turn.reproduction_result,
            )
            return verified_result, usage
        if completion.context_utilization() > stop_threshold:
            return (
                _could_not_reproduce('context budget exceeded before reproduction'),
                usage,
            )
        messages.append(Message(role='assistant', content=agent_turn.model_dump_json()))
        messages.append(Message(role='user', content='\n\n'.join(tool_observations)))

    return _could_not_reproduce('maximum reproduction tool rounds reached'), usage


async def _verify_reproduction(
    request: ReproductionTurnRequest,
    reproduction_result: ReproductionResult,
) -> ReproductionResult:
    if reproduction_result.status == ReproductionStatus.COULD_NOT_REPRODUCE:
        return reproduction_result
    tool_result = await run_tool(
        ToolExecutionRequest(
            workspace=request.workspace_info,
            tool=repro_run_tests(reproduction_result.repro_command),
            repo_index=request.repo_index,
        )
    )
    if tool_result.exit_code == 0:
        return _could_not_reproduce(
            'repro command passed on the unfixed tree; not a genuine reproduction'
        )
    if not _is_assertion_failure(tool_result):
        return _could_not_reproduce(
            'repro command failed, but not with a collected test assertion failure'
        )
    return reproduction_result.model_copy(
        update={'failure_evidence': _observed_failure_text(tool_result)}
    )


async def _run_reproduction_tool_calls(
    request: ReproductionTurnRequest,
    tool_calls: list[ImplementationToolCall],
) -> tuple[list[str], LLMUsage]:
    observations: list[str] = []
    usage = LLMUsage()
    for tool_call in tool_calls:
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
            case tool:
                tool_result = await run_tool(
                    ToolExecutionRequest(
                        workspace=request.workspace_info,
                        tool=tool,
                        repo_index=request.repo_index,
                    )
                )
                observations.append(f'tool_result:\n{tool_result.model_dump_json()}')
    return observations, usage


def _is_assertion_failure(tool_result: ToolResult) -> bool:
    output = f'{tool_result.stdout}\n{tool_result.stderr}'.lower()
    non_reproduction_markers = [
        'file or directory not found',
        'collected 0 items',
        'no tests ran',
        'error collecting',
        'syntaxerror',
        'importerror',
        'modulenotfounderror',
    ]
    return tool_result.exit_code == 1 and not any(
        marker in output for marker in non_reproduction_markers
    )


def _observed_failure_text(tool_result: ToolResult) -> str:
    return f'exit_code={tool_result.exit_code}\n{tool_result.stdout}\n{tool_result.stderr}'.strip()


def _could_not_reproduce(reason: str) -> ReproductionResult:
    return ReproductionResult(
        status=ReproductionStatus.COULD_NOT_REPRODUCE,
        repro_command='',
        test_files=[],
        failure_evidence=reason,
    )


def _reproduction_user_payload(request: ReproductionTurnRequest) -> dict[str, object]:
    return {
        'task_contract': request.task_contract.model_dump(mode='json'),
        'workspace_info': request.workspace_info.model_dump(mode='json'),
        'environment': request.workspace_info.describe_environment(),
        'directory_tree': request.repo_index.directory_tree_text(),
        'available_tools': list(REPRODUCTION_AVAILABLE_TOOLS),
    }
