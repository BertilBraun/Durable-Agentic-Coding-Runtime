from __future__ import annotations

import json
from typing import get_args

from pydantic import Field

from src.activities.context_gatherer import ContextGatherRequest, gather_context
from src.activities.test_protection import revert_unauthorized_test_edits, tool_mutates_workspace
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
from src.tools.definitions import GatherContext, ReproductionToolCall, RunTests

REPRODUCTION_COMMAND_TIMEOUT_SECONDS = 600

REPRODUCTION_SYSTEM_PROMPT = (
    'You are the reproduction agent for a bugfix or feature task. Your only job is to '
    'demonstrate the reported defect or missing behavior with a test, never to '
    'implement or fix it. Locate the relevant '
    'behavior using run_shell, find_definition / find_callers / find_callees, '
    'and gather_context, then write a focused pytest regression test '
    'that fails on the current code and isolates exactly the reported '
    'gap (a wrong result for a bug, or the absent capability for a feature). '
    'This test is the project-wide success anchor, so it must cover the WHOLE '
    'requested behavior, not one convenient half: for symmetric behavior '
    '(read/write, encode/decode, serialize/parse) assert the FULL round trip in '
    'both directions, and otherwise assert the invariants the correct behavior '
    'must satisfy, rather than substring or membership presence. Use a literal '
    'expected output or value only when the issue or spec states it explicitly; '
    'never invent the exact bytes or lines yourself, since a guessed expectation '
    'makes a correct fix fail. Write several tests over different inputs and input '
    'counts (for example one, two, and three header rows) so a single example '
    'cannot pass a trivial or partial fix. Write '
    'plain pytest test functions (top-level test_* with assert statements); never '
    'add an "if __name__" / pytest.main / sys.exit block, since run_tests already '
    'runs the file under "python -m pytest". Add the test with the write_regression '
    'tool (it refuses to overwrite an existing file, so pick a NEW path next to the '
    "package's tests so imports resolve); never modify an existing test file. "
    'Confirm it fails with run_tests; a failure from an assertion '
    'on the wrong/missing behavior is a real reproduction, while a syntax, import, or '
    'collection error is not. Do not edit production code or weaken any existing '
    'test. Also identify the existing repository test files most relevant to the '
    'changed area and return them as regression_test_files (whole file paths, not '
    'node ids) so later edits can be checked for regressions. Return done=true with '
    'a ReproductionResult: '
    'set status=reproduced only once you have seen the new test fail, giving '
    'repro_target as the pytest node id that selects just that test '
    '(e.g. "pkg/tests/test_mod.py::test_case"), the files you added in test_files, '
    'the relevant existing test files in regression_test_files, and the '
    'observed failing output. If the contract is too vague to pin down, or the '
    'failure is flaky or environmental, return status=could_not_reproduce '
    'instead of guessing.'
)


REPRODUCTION_AVAILABLE_TOOLS: tuple[str, ...] = tuple(
    tool_type.model_fields['tool_name'].default.value
    for tool_type in get_args(ReproductionToolCall)
)


class ReproductionTurnRequest(FrozenBaseModel):
    task_contract: TaskContract
    workspace_info: Workspace
    repo_index: RepoIndex


class ReproductionAgentTurn(FrozenBaseModel):
    done: bool
    reproduction_result: ReproductionResult | None = None
    tool_calls: list[ReproductionToolCall] = Field(default_factory=list)


def repro_run_tests(repro_target: str) -> RunTests:
    return RunTests(
        test_targets=[repro_target],
        timeout_seconds=REPRODUCTION_COMMAND_TIMEOUT_SECONDS,
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
            tool=repro_run_tests(reproduction_result.repro_target),
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
    tool_calls: list[ReproductionToolCall],
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
                if tool_mutates_workspace(tool):
                    protection_note = await revert_unauthorized_test_edits(
                        workspace=request.workspace_info,
                        repo_index=request.repo_index,
                        allowed_test_files=set(),
                        revert_untracked=False,
                    )
                    if protection_note is not None:
                        observations.append(protection_note)
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
        repro_target='',
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
