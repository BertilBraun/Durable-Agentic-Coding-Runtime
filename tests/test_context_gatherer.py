from collections.abc import Awaitable, Callable

import pytest
from pydantic import BaseModel, ValidationError
from src.activities import context_gatherer as context_gatherer_module
from src.activities.context_gatherer import (
    ContextGathererTurn,
    ContextGatherRequest,
    gather_context,
)
from src.activities.workspace_manager import ToolExecutionRequest, ToolResult, WorkspaceInfo
from src.config import ModelRole
from src.llm.client import LLMUsage, Message, StructuredCompletion
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
        ContextGathererToolCallAdapter.validate_python({'tool_name': 'find_references'})


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
                            'tool_name': 'search_text',
                            'pattern': f'pattern-{call_count}-a',
                            'directory': '.',
                            'file_glob': '*.py',
                        },
                        {
                            'tool_name': 'search_text',
                            'pattern': f'pattern-{call_count}-b',
                            'directory': '.',
                            'file_glob': '*.py',
                        },
                    ],
                }
            )
        else:
            turn = output_type.model_validate(
                {
                    'done': True,
                    'context_pack': {
                        'task_summary': 'done',
                        'relevant_snippets': [],
                        'recent_observations': [],
                        'failed_attempt_summaries': [],
                        'available_tools': [],
                        'budget_remaining': 1,
                    },
                }
            )
        return _structured_completion(turn, context_utilization=0.0)

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        return ToolResult(stdout=tool_outputs.pop(0), stderr='', exit_code=0, truncated=False)

    _patch_generate_structured(monkeypatch, handler)
    monkeypatch.setattr(context_gatherer_module, 'run_tool', fake_run_tool)

    await gather_context(_context_gather_request())

    third_call_last_user_message = captured_messages[2][-1].content
    assert 'turn two first' in third_call_last_user_message
    assert 'turn two second' in third_call_last_user_message
    assert 'turn one first' not in third_call_last_user_message
    assert 'turn one second' not in third_call_last_user_message


@pytest.mark.asyncio
async def test_context_gatherer_summarizes_when_context_budget_is_high(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_finalize_message: list[str] = []
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
                            'tool_name': 'search_text',
                            'pattern': 'handler',
                            'directory': '.',
                            'file_glob': '*.py',
                        }
                    ],
                }
            )
            return _structured_completion(turn, context_utilization=0.81)
        captured_finalize_message.append(messages[-1].content)
        turn = output_type.model_validate(
            {
                'done': True,
                'context_pack': {
                    'task_summary': 'LLM-summarized auth context',
                    'relevant_snippets': ['summary line 1'],
                    'recent_observations': [],
                    'failed_attempt_summaries': [],
                    'available_tools': ['read_file_range'],
                    'budget_remaining': 0,
                },
            }
        )
        return _structured_completion(turn, context_utilization=0.0)

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        raise AssertionError(f'run_tool should not run after budget stop: {request.tool}')

    _patch_generate_structured(monkeypatch, handler)
    monkeypatch.setattr(context_gatherer_module, 'run_tool', fake_run_tool)

    context_pack, _ = await gather_context(_context_gather_request())

    assert context_pack.task_summary == 'LLM-summarized auth context'
    assert context_pack.relevant_snippets == ['summary line 1']
    assert 'context budget is exhausted' in captured_finalize_message[0]
    assert context_pack.budget_remaining == 0


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
                            'tool_name': 'search_text',
                            'pattern': 'handler',
                            'directory': '.',
                            'file_glob': '*.py',
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
                        'tool_name': 'search_text',
                        'pattern': 'again',
                        'directory': '.',
                        'file_glob': '*.py',
                    }
                ],
            }
        )
        return _structured_completion(turn, context_utilization=0.97)

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        return ToolResult(stdout='handler reference line', stderr='', exit_code=0, truncated=False)

    _patch_generate_structured(monkeypatch, handler)
    monkeypatch.setattr(context_gatherer_module, 'run_tool', fake_run_tool)

    context_pack, _ = await gather_context(_context_gather_request())

    assert call_count == 2
    assert context_pack.task_summary == 'Find relevant code'
    assert context_pack.relevant_snippets == ['handler reference line']
    assert context_pack.budget_remaining == 0


def _context_gather_request() -> ContextGatherRequest:
    return ContextGatherRequest(
        workspace_info=WorkspaceInfo(
            run_id='run-1',
            volume_name='volume',
            worktree_path='workspace',
            branch_name='branch',
        ),
        repo_index=RepoIndex(),
        gatherer_prompt='Find relevant code',
    )
