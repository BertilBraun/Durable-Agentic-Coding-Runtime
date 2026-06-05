import pytest
from pydantic import BaseModel
from src.activities import planner as planner_module
from src.activities.planner import plan_next_turn
from src.config import ModelRole
from src.llm.client import LLMUsage, Message, StructuredCompletion
from src.models.context import PackedSnippet
from src.models.plan import (
    ContextNote,
    PlannerState,
    PlannerToolObservation,
    PlannerTurn,
    StepHistoryEntry,
)
from src.models.repo import RepoIndex
from src.models.task import TaskContract, TaskType
from src.models.worker import Confidence, WorkerStatus
from src.tools.definitions import ToolName


def test_planner_state_serializes_normalized_history_and_remaining_work() -> None:
    state = PlannerState(
        contract=TaskContract(task_type=TaskType.BUGFIX, goal='Fix auth'),
        repo_index=RepoIndex(),
        context_notes=[
            ContextNote(
                id='ctx-1',
                summary='Auth handler lives in src/auth.py.',
                relevant_files=['src/auth.py'],
                snippets=[
                    PackedSnippet(
                        file_path='src/auth.py',
                        start_line=1,
                        end_line=4,
                        reason='handler',
                        content='def auth(): ...',
                    )
                ],
            )
        ],
        completed_steps=[
            StepHistoryEntry(
                step_id='step-1',
                outcome=WorkerStatus.SUCCESS,
                confidence=Confidence.HIGH,
                summary='Updated parser.',
            )
        ],
        remaining_work=['Fix missing token case in src/auth.py.'],
    )

    payload = state.model_dump(mode='json')

    assert payload['context_notes'][0]['snippets'][0]['file_path'] == 'src/auth.py'
    assert 'content' not in payload['context_notes'][0]['snippets'][0]
    assert payload['completed_steps'][0]['outcome'] == WorkerStatus.SUCCESS.value
    assert payload['remaining_work'] == ['Fix missing token case in src/auth.py.']


@pytest.mark.asyncio
async def test_plan_next_turn_uses_normalized_state_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages: list[list[Message]] = []
    turn = PlannerTurn(done=True, done_reason='No work remains.')

    async def fake_generate_structured(
        role: ModelRole,
        messages: list[Message],
        output_type: type[BaseModel],
    ) -> StructuredCompletion:
        captured_messages.append(messages)
        assert output_type is PlannerTurn
        return StructuredCompletion(
            output=turn,
            content=turn.model_dump_json(),
            model='fake-model',
            context_limit_tokens=100,
            usage=LLMUsage(call_count=1),
        )

    monkeypatch.setattr(planner_module, 'generate_structured', fake_generate_structured)

    result, usage = await plan_next_turn(
        PlannerState(
            contract=TaskContract(task_type=TaskType.FEATURE, goal='Add auth'),
            repo_index=RepoIndex(overview_text='src/auth.py'),
            context_notes=[
                ContextNote(
                    id='ctx-1',
                    summary='Use src/auth.py',
                    snippets=[
                        PackedSnippet(
                            file_path='src/auth.py',
                            start_line=1,
                            end_line=2,
                            reason='auth entrypoint',
                            content='secret snippet content should not repeat',
                        )
                    ],
                )
            ],
        )
    )

    assert result.done is True
    assert usage.call_count == 1
    user_message = captured_messages[0][-1].content
    assert 'Planner State' in user_message
    assert 'Goal' in user_message
    assert 'Use src/auth.py' in user_message
    assert 'Available Context' in user_message
    assert 'src/auth.py:1-2 - auth entrypoint' in user_message
    assert 'secret snippet content should not repeat' not in user_message


def test_planner_state_prompt_uses_readable_sections() -> None:
    prompt = planner_module._render_planner_state(
        PlannerState(
            contract=TaskContract(
                task_type=TaskType.FEATURE,
                goal='Add auth',
                acceptance_criteria=['Auth works'],
            ),
            repo_index=RepoIndex(overview_text='src/auth.py'),
            context_notes=[
                ContextNote(
                    id='ctx-1',
                    summary='Auth handler found.',
                    request_reason='Need auth code',
                    request_queries=['Read auth handler'],
                    relevant_files=['src/auth.py'],
                    snippets=[
                        PackedSnippet(
                            file_path='src/auth.py',
                            start_line=1,
                            end_line=4,
                            reason='handler',
                            content='def auth(): ...',
                        )
                    ],
                )
            ],
            tool_observations=[
                PlannerToolObservation(
                    tool_name=ToolName.RUN_SHELL,
                    stdout='class Auth: ...',
                    stderr='',
                    exit_code=0,
                    truncated=False,
                )
            ],
        )
    )

    assert prompt.startswith('# Planner State')
    assert '## Goal\n\nAdd auth' in prompt
    assert '## Acceptance Criteria\n\n- Auth works' in prompt
    assert '## Available Context\n\n### ctx-1: Need auth code' in prompt
    assert '- src/auth.py:1-4 - handler' in prompt
    assert '## Planner Tool Observations\n\n### run_shell exit_code=0' in prompt
    assert 'def auth(): ...' not in prompt
