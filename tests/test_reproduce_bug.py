from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest
from pydantic import BaseModel
from src.activities import reproduction as reproduction_module
from src.activities.reproduction import (
    ReproductionAgentTurn,
    ReproductionTurnRequest,
    reproduce_bug,
)
from src.activities.workspace_manager import HostWorkspace, ToolExecutionRequest, ToolResult
from src.config import ModelRole
from src.llm.client import LLMUsage, Message, StructuredCompletion
from src.models.repo import RepoIndex
from src.models.reproduction import ReproductionResult, ReproductionStatus
from src.models.task import TaskContract, TaskType


def _workspace() -> HostWorkspace:
    return HostWorkspace(
        run_id='run-1',
        base_sha='basesha',
        base_branch='main',
        current_branch='main',
        repo_path='workspace',
    )


def _request() -> ReproductionTurnRequest:
    return ReproductionTurnRequest(
        task_contract=TaskContract(task_type=TaskType.BUGFIX, goal='Fix the off-by-one'),
        workspace_info=_workspace(),
        repo_index=RepoIndex(),
    )


def _completion(output: BaseModel) -> StructuredCompletion:
    return StructuredCompletion(
        output=output,
        content=output.model_dump_json(),
        model='fake-model',
        context_limit_tokens=100,
        usage=LLMUsage(call_count=1, total_input_tokens=1),
    )


def _patch_generate_structured(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[list[Message], type[BaseModel]], Awaitable[StructuredCompletion]],
) -> None:
    async def fake_generate_structured(
        role: ModelRole,
        messages: list[Message],
        output_type: type[BaseModel],
    ) -> StructuredCompletion:
        return await handler(messages, output_type)

    monkeypatch.setattr(reproduction_module, 'generate_structured', fake_generate_structured)


def _done_turn(status: ReproductionStatus) -> ReproductionAgentTurn:
    return ReproductionAgentTurn(
        done=True,
        reproduction_result=ReproductionResult(
            status=status,
            repro_command='pytest tests/test_bug.py::test_off_by_one',
            test_files=['tests/test_bug.py'],
            failure_evidence='model-claimed failure',
        ),
        tool_calls=[],
    )


@pytest.mark.asyncio
async def test_reproduce_bug_trusts_reproduced_when_command_actually_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(
        messages: list[Message], output_type: type[BaseModel]
    ) -> StructuredCompletion:
        return _completion(_done_turn(ReproductionStatus.REPRODUCED))

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        return ToolResult(stdout='assert 4 == 5', stderr='', exit_code=1, truncated=False)

    _patch_generate_structured(monkeypatch, handler)
    monkeypatch.setattr(reproduction_module, 'run_tool', fake_run_tool)

    result, before_exit_code, _ = await reproduce_bug(_request())

    assert result.status == ReproductionStatus.REPRODUCED
    assert before_exit_code == 1
    assert 'assert 4 == 5' in result.failure_evidence


@pytest.mark.asyncio
async def test_reproduce_bug_rejects_claim_when_command_passes_on_unfixed_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(
        messages: list[Message], output_type: type[BaseModel]
    ) -> StructuredCompletion:
        return _completion(_done_turn(ReproductionStatus.REPRODUCED))

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        return ToolResult(stdout='1 passed', stderr='', exit_code=0, truncated=False)

    _patch_generate_structured(monkeypatch, handler)
    monkeypatch.setattr(reproduction_module, 'run_tool', fake_run_tool)

    result, before_exit_code, _ = await reproduce_bug(_request())

    assert result.status == ReproductionStatus.COULD_NOT_REPRODUCE
    assert before_exit_code == 0


@pytest.mark.asyncio
async def test_reproduce_bug_does_not_run_command_for_could_not_reproduce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_tool_calls = 0

    async def handler(
        messages: list[Message], output_type: type[BaseModel]
    ) -> StructuredCompletion:
        return _completion(_done_turn(ReproductionStatus.COULD_NOT_REPRODUCE))

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        nonlocal run_tool_calls
        run_tool_calls += 1
        return ToolResult(stdout='', stderr='', exit_code=1, truncated=False)

    _patch_generate_structured(monkeypatch, handler)
    monkeypatch.setattr(reproduction_module, 'run_tool', fake_run_tool)

    result, before_exit_code, _ = await reproduce_bug(_request())

    assert result.status == ReproductionStatus.COULD_NOT_REPRODUCE
    assert run_tool_calls == 0
    assert before_exit_code == 0
