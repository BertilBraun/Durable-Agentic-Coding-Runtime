import json
from collections.abc import Awaitable, Callable

import pytest
from pydantic import BaseModel, ValidationError
from src.activities import implementation as implementation_module
from src.activities.context_gatherer import ContextGatherRequest
from src.activities.implementation import (
    IMPLEMENTATION_AVAILABLE_TOOLS,
    ImplementationAgentTurn,
    ImplementationTurnRequest,
    run_implementation_turn,
)
from src.activities.workspace_manager import ToolExecutionRequest, ToolResult, WorkspaceInfo
from src.llm.client import LLMResult, Message, StructuredCompletion
from src.llm.config import ModelRole
from src.models.context import ContextPack
from src.models.plan import PlanStep, Risk
from src.models.repo import RepoIndex
from src.models.task import TaskContract, TaskType
from src.models.worker import Confidence, WorkerStatus
from src.tools.definitions import (
    GitDiff,
    ImplementationToolCallAdapter,
    RunTests,
    ToolName,
)


def _structured_completion(output: BaseModel, context_utilization: float) -> StructuredCompletion:
    input_tokens = int(context_utilization * 100)
    return StructuredCompletion(
        output=output,
        result=LLMResult(
            content=output.model_dump_json(),
            model='fake-model',
            input_tokens=input_tokens,
            output_tokens=0,
            cache_read_tokens=0,
            cost_usd=0.0,
            context_limit_tokens=100,
        ),
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

    monkeypatch.setattr(implementation_module, 'generate_structured', fake_generate_structured)


@pytest.mark.asyncio
async def test_implementation_turn_executes_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_names: list[str] = []
    repo_indexes: list[RepoIndex | None] = []

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        tool_names.append(type(request.tool).__name__)
        repo_indexes.append(request.repo_index)
        return ToolResult(stdout='file content', stderr='', exit_code=0, truncated=False)

    call_count = 0

    async def handler(
        messages: list[Message], output_type: type[BaseModel]
    ) -> StructuredCompletion:
        nonlocal call_count
        call_count += 1
        assert output_type.__name__ == 'ImplementationAgentTurn'
        if call_count == 1:
            turn = output_type.model_validate(
                {
                    'done': False,
                    'tool_calls': [
                        {
                            'tool_name': 'read_file_range',
                            'file_path': 'src/app.py',
                            'start_line': 1,
                            'end_line': 5,
                        }
                    ],
                }
            )
        else:
            turn = output_type.model_validate(
                {
                    'done': True,
                    'worker_result': {
                        'status': 'blocked',
                        'patch_id': None,
                        'diff_summary': 'Read target file.',
                        'tests_run': [],
                        'test_results': [],
                        'discovered_issues': [],
                        'confidence': 'low',
                        'replan_suggestion': 'Need an edit or test to complete the step.',
                    },
                }
            )
        return _structured_completion(turn, context_utilization=0.0)

    _patch_generate_structured(monkeypatch, handler)
    monkeypatch.setattr(implementation_module, 'run_tool', fake_run_tool)

    worker_result = await run_implementation_turn(_implementation_request())

    assert worker_result.status == WorkerStatus.BLOCKED
    assert worker_result.confidence == Confidence.LOW
    assert tool_names == ['ReadFileRange']
    assert repo_indexes == [RepoIndex()]


@pytest.mark.asyncio
async def test_implementation_turn_dispatches_gather_context_without_run_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages: list[list[Message]] = []
    captured_gather_prompts: list[str] = []
    call_count = 0

    async def handler(
        messages: list[Message], output_type: type[BaseModel]
    ) -> StructuredCompletion:
        nonlocal call_count
        call_count += 1
        captured_messages.append(messages)
        if call_count == 1:
            turn = output_type.model_validate(
                {
                    'done': False,
                    'tool_calls': [
                        {
                            'tool_name': 'gather_context',
                            'prompt': 'Find auth callers',
                        }
                    ],
                }
            )
        else:
            turn = output_type.model_validate(
                {
                    'done': True,
                    'worker_result': {
                        'status': 'blocked',
                        'patch_id': None,
                        'diff_summary': 'Need implementation after context review.',
                        'tests_run': [],
                        'test_results': [],
                        'discovered_issues': [],
                        'confidence': 'low',
                        'replan_suggestion': 'Use gathered auth context.',
                    },
                }
            )
        return _structured_completion(turn, context_utilization=0.0)

    async def fake_gather_context(request: ContextGatherRequest) -> ContextPack:
        captured_gather_prompts.append(request.gatherer_prompt)
        return ContextPack(
            task_summary='Auth context',
            relevant_snippets=['src/auth.py: token handler'],
            recent_observations=['found caller'],
            failed_attempt_summaries=[],
            available_tools=['read_file_range'],
            budget_remaining=4,
        )

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        raise AssertionError(f'run_tool should not handle gather_context: {request.tool}')

    _patch_generate_structured(monkeypatch, handler)
    monkeypatch.setattr(implementation_module, 'gather_context', fake_gather_context)
    monkeypatch.setattr(implementation_module, 'run_tool', fake_run_tool)

    worker_result = await run_implementation_turn(_implementation_request())

    assert worker_result.status == WorkerStatus.BLOCKED
    assert captured_gather_prompts == ['Find auth callers']
    assert 'Auth context' in captured_messages[1][-1].content
    assert 'src/auth.py: token handler' in captured_messages[1][-1].content


def test_implementation_tool_call_uses_tool_name_enum() -> None:
    tool_call = ImplementationToolCallAdapter.validate_python(
        {
            'tool_name': 'read_file_range',
            'file_path': 'src/app.py',
            'start_line': 1,
            'end_line': 5,
        }
    )

    assert tool_call.tool_name == ToolName.READ_FILE_RANGE


def test_implementation_tool_call_rejects_unknown_tool() -> None:
    with pytest.raises(ValidationError):
        ImplementationAgentTurn.model_validate(
            {
                'done': False,
                'tool_calls': [
                    {
                        'tool_name': 'delete_everything',
                    }
                ],
            }
        )


def test_implementation_tool_call_rejects_missing_required_payload_field() -> None:
    with pytest.raises(ValidationError):
        ImplementationAgentTurn.model_validate(
            {
                'done': False,
                'tool_calls': [
                    {
                        'tool_name': 'read_file_range',
                        'start_line': 1,
                        'end_line': 5,
                    }
                ],
            }
        )


@pytest.mark.asyncio
async def test_implementation_turn_preserves_run_tests_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_timeout_seconds: list[int] = []

    async def handler(
        messages: list[Message], output_type: type[BaseModel]
    ) -> StructuredCompletion:
        turn = output_type.model_validate(
            {
                'done': False,
                'tool_calls': [
                    {
                        'tool_name': 'run_tests',
                        'command': 'pytest -q',
                        'timeout_seconds': 19,
                        'directory': '.',
                    }
                ],
            }
        )
        return _structured_completion(turn, context_utilization=0.0)

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        match request.tool:
            case RunTests(timeout_seconds=timeout_seconds):
                captured_timeout_seconds.append(timeout_seconds)
            case _:
                raise AssertionError(f'Unexpected tool: {request.tool}')
        return ToolResult(stdout='tests failed', stderr='', exit_code=1, truncated=False)

    monkeypatch.setenv('IMPLEMENTATION_MAX_TOOL_ROUNDS', '1')
    _patch_generate_structured(monkeypatch, handler)
    monkeypatch.setattr(implementation_module, 'run_tool', fake_run_tool)

    worker_result = await run_implementation_turn(_implementation_request())

    assert worker_result.status == WorkerStatus.FAILED
    assert captured_timeout_seconds == [19]


@pytest.mark.asyncio
async def test_implementation_turn_adds_run_tests_result_to_success(
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
                            'tool_name': 'run_tests',
                            'command': 'pytest tests/test_app.py -q',
                            'timeout_seconds': 30,
                            'directory': '.',
                        },
                        {
                            'tool_name': 'git_diff',
                            'path': '.',
                        },
                    ],
                }
            )
        else:
            turn = output_type.model_validate(
                {
                    'done': True,
                    'worker_result': {
                        'status': 'success',
                        'patch_id': 'patch-1',
                        'diff_summary': 'Updated app behavior.',
                        'tests_run': [],
                        'test_results': [],
                        'discovered_issues': [],
                        'confidence': 'high',
                        'replan_suggestion': None,
                    },
                }
            )
        return _structured_completion(turn, context_utilization=0.0)

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        match request.tool:
            case RunTests():
                return ToolResult(stdout='1 passed', stderr='', exit_code=0, truncated=False)
            case GitDiff():
                return ToolResult(
                    stdout='diff --git a/app.py b/app.py',
                    stderr='',
                    exit_code=0,
                    truncated=False,
                )
            case _:
                raise AssertionError(f'Unexpected tool: {request.tool}')

    _patch_generate_structured(monkeypatch, handler)
    monkeypatch.setattr(implementation_module, 'run_tool', fake_run_tool)

    worker_result = await run_implementation_turn(_implementation_request())

    assert worker_result.status == WorkerStatus.SUCCESS
    assert worker_result.tests_run == ['pytest tests/test_app.py -q']
    assert len(worker_result.test_results) == 1
    assert worker_result.test_results[0].passed is True
    assert worker_result.test_results[0].stdout_summary == '1 passed'


@pytest.mark.asyncio
async def test_implementation_turn_rejects_success_without_diff_or_test_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(
        messages: list[Message], output_type: type[BaseModel]
    ) -> StructuredCompletion:
        turn = output_type.model_validate(
            {
                'done': True,
                'worker_result': {
                    'status': 'success',
                    'patch_id': 'patch-1',
                    'diff_summary': '',
                    'tests_run': [],
                    'test_results': [],
                    'discovered_issues': [],
                    'confidence': 'high',
                    'replan_suggestion': None,
                },
            }
        )
        return _structured_completion(turn, context_utilization=0.0)

    _patch_generate_structured(monkeypatch, handler)

    worker_result = await run_implementation_turn(_implementation_request())

    assert worker_result.status == WorkerStatus.FAILED
    assert worker_result.discovered_issues == ['success result missing diff or test evidence']


@pytest.mark.asyncio
async def test_implementation_turn_rejects_success_with_only_diff_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(
        messages: list[Message], output_type: type[BaseModel]
    ) -> StructuredCompletion:
        turn = output_type.model_validate(
            {
                'done': True,
                'worker_result': {
                    'status': 'success',
                    'patch_id': 'patch-1',
                    'diff_summary': 'No changes were needed.',
                    'tests_run': [],
                    'test_results': [],
                    'discovered_issues': [],
                    'confidence': 'high',
                    'replan_suggestion': None,
                },
            }
        )
        return _structured_completion(turn, context_utilization=0.0)

    _patch_generate_structured(monkeypatch, handler)

    worker_result = await run_implementation_turn(_implementation_request())

    assert worker_result.status == WorkerStatus.FAILED
    assert worker_result.discovered_issues == ['success result missing diff or test evidence']


@pytest.mark.asyncio
async def test_implementation_turn_blocks_when_context_budget_is_high(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(
        messages: list[Message], output_type: type[BaseModel]
    ) -> StructuredCompletion:
        turn = output_type.model_validate(
            {
                'done': False,
                'tool_calls': [
                    {
                        'tool_name': 'read_file_range',
                        'file_path': 'src/app.py',
                        'start_line': 1,
                        'end_line': 5,
                    }
                ],
            }
        )
        return _structured_completion(turn, context_utilization=0.81)

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        raise AssertionError(f'run_tool should not run after budget block: {request.tool}')

    _patch_generate_structured(monkeypatch, handler)
    monkeypatch.setattr(implementation_module, 'run_tool', fake_run_tool)

    worker_result = await run_implementation_turn(_implementation_request())

    assert worker_result.status == WorkerStatus.BLOCKED
    assert worker_result.confidence == Confidence.LOW
    assert worker_result.replan_suggestion is not None
    assert 'read_file_range' in worker_result.replan_suggestion


@pytest.mark.asyncio
async def test_implementation_turn_user_message_excludes_repo_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_user_payloads: list[dict[str, object]] = []

    async def handler(
        messages: list[Message], output_type: type[BaseModel]
    ) -> StructuredCompletion:
        captured_user_payloads.append(json.loads(messages[1].content))
        turn = output_type.model_validate(
            {
                'done': True,
                'worker_result': {
                    'status': 'blocked',
                    'patch_id': None,
                    'diff_summary': 'Prompt inspected.',
                    'tests_run': [],
                    'test_results': [],
                    'discovered_issues': [],
                    'confidence': 'low',
                    'replan_suggestion': 'Continue.',
                },
            }
        )
        return _structured_completion(turn, context_utilization=0.0)

    _patch_generate_structured(monkeypatch, handler)

    await run_implementation_turn(_implementation_request())

    assert captured_user_payloads == [
        {
            'plan_step': _implementation_request().plan_step.model_dump(mode='json'),
            'context_pack': _implementation_request().context_pack.model_dump(mode='json'),
            'task_contract': _implementation_request().task_contract.model_dump(mode='json'),
            'workspace_info': _implementation_request().workspace_info.model_dump(mode='json'),
            'available_tools': list(IMPLEMENTATION_AVAILABLE_TOOLS),
        }
    ]


def _implementation_request() -> ImplementationTurnRequest:
    return ImplementationTurnRequest(
        plan_step=PlanStep(
            id='step_1',
            goal='Read target file',
            target_files=['src/app.py'],
            tests_to_run=[],
            expected_result='File inspected',
            risk=Risk.LOW,
            requires_human_approval=False,
        ),
        context_pack=ContextPack(
            task_summary='Inspect file',
            relevant_snippets=[],
            recent_observations=[],
            failed_attempt_summaries=[],
            available_tools=[],
            budget_remaining=3,
        ),
        task_contract=TaskContract(
            task_type=TaskType.BUGFIX,
            goal='Inspect file',
            acceptance_criteria=[],
            non_goals=[],
            affected_areas=[],
            risk_areas=[],
            tests_expected=[],
            open_questions=[],
        ),
        workspace_info=WorkspaceInfo(
            run_id='run-1',
            volume_name='volume',
            worktree_path='workspace',
            branch_name='branch',
        ),
        repo_index=RepoIndex(),
    )
