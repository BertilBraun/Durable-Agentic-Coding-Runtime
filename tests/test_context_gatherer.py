from collections.abc import Awaitable, Callable

import pytest
from pydantic import BaseModel, ValidationError
from src.activities import context_gatherer as context_gatherer_module
from src.activities.context_gatherer import (
    ContextGathererTurn,
    ContextGatherRequest,
    gather_context,
)
from src.activities.workspace_manager import (
    ContextPackRequest,
    HostWorkspace,
    ToolExecutionRequest,
    ToolResult,
)
from src.config import ModelRole
from src.llm.client import LLMUsage, Message, StructuredCompletion
from src.models.context import ContextPack, PackedSnippet
from src.models.repo import RepoIndex
from src.tools.definitions import ContextGathererToolCallAdapter


def test_context_gatherer_rejects_unknown_tool() -> None:
    with pytest.raises(ValidationError):
        ContextGathererTurn.model_validate(
            {
                'done': False,
                'tool_calls': [
                    {
                        'tool_name': 'delete_everything',
                    }
                ],
            }
        )


def test_context_gatherer_rejects_mutating_tool() -> None:
    with pytest.raises(ValidationError):
        ContextGathererTurn.model_validate(
            {
                'done': False,
                'tool_calls': [
                    {
                        'tool_name': 'write_file',
                        'file_path': 'src/app.py',
                        'content': 'mutating',
                    }
                ],
            }
        )


def test_context_gatherer_tool_conversion_asserts_missing_required_payload_field() -> None:
    with pytest.raises(ValidationError):
        ContextGathererToolCallAdapter.validate_python({'tool_name': 'find_callers'})


def _structured_completion(output: BaseModel, context_utilization: float) -> StructuredCompletion:
    input_tokens = int(context_utilization * 100)
    return StructuredCompletion(
        output=output,
        content=output.model_dump_json(),
        model='fake-model',
        context_limit_tokens=100,
        usage=LLMUsage(call_count=1, total_input_tokens=input_tokens),
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

    monkeypatch.setattr(context_gatherer_module, 'generate_structured', fake_generate_structured)


def _patch_pack_context(
    monkeypatch: pytest.MonkeyPatch,
    captured: list[ContextPackRequest] | None = None,
) -> None:
    async def fake_pack_context(request: ContextPackRequest) -> ContextPack:
        if captured is not None:
            captured.append(request)
        return ContextPack(
            task_summary=request.task_summary,
            snippets=[
                PackedSnippet(
                    file_path=snippet.file_path,
                    start_line=snippet.start_line,
                    end_line=snippet.end_line,
                    reason=snippet.reason,
                    content=f'lines for {snippet.file_path}',
                )
                for snippet in request.snippets
            ],
            artifact_references=[],
            budget_remaining=0,
        )

    monkeypatch.setattr(context_gatherer_module, 'pack_context', fake_pack_context)


@pytest.mark.asyncio
async def test_context_gatherer_sends_only_current_turn_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages: list[list[Message]] = []
    tool_outputs = ['turn one first', 'turn one second', 'turn two first', 'turn two second']
    call_count = 0

    async def handler(
        messages: list[Message], output_type: type[BaseModel]
    ) -> StructuredCompletion:
        nonlocal call_count
        call_count += 1
        captured_messages.append(messages)
        if call_count < 3:
            turn = output_type.model_validate(
                {
                    'done': False,
                    'tool_calls': [
                        {
                            'tool_name': 'run_shell',
                            'command': f'rg pattern-{call_count}-a',
                            'timeout_seconds': 10,
                        },
                        {
                            'tool_name': 'run_shell',
                            'command': f'rg pattern-{call_count}-b',
                            'timeout_seconds': 10,
                        },
                    ],
                }
            )
        else:
            turn = output_type.model_validate(
                {
                    'done': True,
                    'task_summary': 'done',
                    'snippets': [],
                }
            )
        return _structured_completion(turn, context_utilization=0.0)

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        return ToolResult(stdout=tool_outputs.pop(0), stderr='', exit_code=0, truncated=False)

    _patch_generate_structured(monkeypatch, handler)
    monkeypatch.setattr(context_gatherer_module, 'run_tool', fake_run_tool)
    _patch_pack_context(monkeypatch)

    await gather_context(_context_gather_request())

    third_call_last_user_message = captured_messages[2][-1].content
    assert 'turn two first' in third_call_last_user_message
    assert 'turn two second' in third_call_last_user_message
    assert 'turn one first' not in third_call_last_user_message
    assert 'turn one second' not in third_call_last_user_message


@pytest.mark.asyncio
async def test_context_gatherer_emits_snippets_when_context_budget_is_high(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_finalize_message: list[str] = []
    captured_pack_requests: list[ContextPackRequest] = []
    call_count = 0

    async def handler(
        messages: list[Message], output_type: type[BaseModel]
    ) -> StructuredCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            turn = output_type.model_validate(
                {
                    'done': False,
                    'tool_calls': [
                        {
                            'tool_name': 'run_shell',
                            'command': 'rg handler',
                            'timeout_seconds': 10,
                        }
                    ],
                }
            )
            return _structured_completion(turn, context_utilization=0.81)
        captured_finalize_message.append(messages[-1].content)
        turn = output_type.model_validate(
            {
                'done': True,
                'task_summary': 'Auth token handling',
                'snippets': [
                    {
                        'file_path': 'src/auth.py',
                        'start_line': 10,
                        'end_line': 20,
                        'reason': 'token handler',
                    }
                ],
            }
        )
        return _structured_completion(turn, context_utilization=0.0)

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        raise AssertionError(f'run_tool should not run after budget stop: {request.tool}')

    _patch_generate_structured(monkeypatch, handler)
    monkeypatch.setattr(context_gatherer_module, 'run_tool', fake_run_tool)
    _patch_pack_context(monkeypatch, captured_pack_requests)

    context_pack, _ = await gather_context(_context_gather_request())

    assert context_pack.task_summary == 'Auth token handling'
    assert [snippet.file_path for snippet in captured_pack_requests[0].snippets] == ['src/auth.py']
    assert context_pack.snippets[0].reason == 'token handler'
    assert 'context budget is exhausted' in captured_finalize_message[0]


@pytest.mark.asyncio
async def test_context_gatherer_exits_deterministically_above_hard_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    async def handler(
        messages: list[Message], output_type: type[BaseModel]
    ) -> StructuredCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            turn = output_type.model_validate(
                {
                    'done': False,
                    'tool_calls': [
                        {
                            'tool_name': 'run_shell',
                            'command': 'rg handler',
                            'timeout_seconds': 10,
                        }
                    ],
                }
            )
            return _structured_completion(turn, context_utilization=0.0)
        turn = output_type.model_validate(
            {
                'done': False,
                'tool_calls': [
                    {
                        'tool_name': 'run_shell',
                        'command': 'rg again',
                        'timeout_seconds': 10,
                    }
                ],
            }
        )
        return _structured_completion(turn, context_utilization=0.97)

    captured_pack_requests: list[ContextPackRequest] = []

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        return ToolResult(stdout='handler reference line', stderr='', exit_code=0, truncated=False)

    _patch_generate_structured(monkeypatch, handler)
    monkeypatch.setattr(context_gatherer_module, 'run_tool', fake_run_tool)
    _patch_pack_context(monkeypatch, captured_pack_requests)

    context_pack, _ = await gather_context(_context_gather_request())

    assert call_count == 2
    assert context_pack.task_summary == 'Find relevant code'
    assert captured_pack_requests[0].snippets == []
    assert 'handler reference line' in captured_pack_requests[0].overflow_text


def _context_gather_request() -> ContextGatherRequest:
    return ContextGatherRequest(
        workspace_info=HostWorkspace(
            run_id='run-1',
            base_sha='basesha',
            base_branch='main',
            current_branch='main',
            repo_path='workspace',
        ),
        repo_index=RepoIndex(),
        gatherer_prompt='Find relevant code',
    )
