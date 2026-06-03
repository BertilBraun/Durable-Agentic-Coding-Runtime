import pytest
from pydantic import BaseModel
from src.activities import planner as planner_module
from src.activities.planner import PlanRequest, build_plan, plan_next_turn
from src.config import ModelRole
from src.llm.client import LLMUsage, Message, StructuredCompletion
from src.models.context import ContextPack, PackedSnippet
from src.models.plan import (
    ContextNote,
    Plan,
    PlannerState,
    PlannerTurn,
    PlanStep,
    Risk,
    StepHistoryEntry,
)
from src.models.repo import RepoIndex
from src.models.task import TaskContract, TaskType
from src.models.worker import Confidence, WorkerStatus


def _plan() -> Plan:
    return Plan(
        summary='Plan',
        steps=[],
        integration_tests=[],
        definition_of_done=['done'],
    )


@pytest.mark.asyncio
async def test_supplied_context_snippets_appear_in_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_messages: list[list[Message]] = []

    async def fake_generate_structured(
        role: ModelRole,
        messages: list[Message],
        output_type: type[BaseModel],
    ) -> StructuredCompletion:
        captured_messages.append(messages)
        return StructuredCompletion(
            output=_plan(),
            content=_plan().model_dump_json(),
            model='fake-model',
            context_limit_tokens=100,
            usage=LLMUsage(call_count=1),
        )

    monkeypatch.setattr(planner_module, 'generate_structured', fake_generate_structured)

    context = ContextPack(
        task_summary='Auth token handling',
        snippets=[
            PackedSnippet(
                file_path='src/auth.py',
                start_line=10,
                end_line=20,
                reason='token handler',
                content='def validate_token(): ...',
            )
        ],
        budget_remaining=0,
    )

    await build_plan(
        PlanRequest(
            contract=TaskContract(task_type=TaskType.FEATURE, goal='Add auth'),
            repo_index=RepoIndex(),
            worker_results=[],
            context=context,
        ),
    )

    user_message = captured_messages[0][-1].content
    assert 'src/auth.py:10-20' in user_message
    assert 'token handler' in user_message
    assert 'def validate_token(): ...' in user_message


def test_planner_state_serializes_normalized_history_and_future_steps() -> None:
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
        previous_future_steps=[
            PlanStep(
                id='step-2',
                goal='Finish auth fix',
                target_files=['src/auth.py'],
                context_summary='Use gathered auth handler evidence.',
                required_changes=['Fix missing token case.'],
                out_of_scope=['Do not change unrelated middleware.'],
                tests_to_run=['pytest tests/test_auth.py -q'],
                expected_result='Auth test passes.',
                risk=Risk.LOW,
                requires_human_approval=False,
            )
        ],
    )

    payload = state.model_dump(mode='json')

    assert payload['context_notes'][0]['snippets'][0]['file_path'] == 'src/auth.py'
    assert 'content' not in payload['context_notes'][0]['snippets'][0]
    assert payload['completed_steps'][0]['outcome'] == WorkerStatus.SUCCESS.value
    assert payload['previous_future_steps'][0]['required_changes'] == [
        'Fix missing token case.'
    ]


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
    assert 'Normalized planner state' in user_message
    assert 'Use src/auth.py' in user_message
    assert 'Context notes' in user_message
    assert 'src/auth.py' in user_message
    assert 'secret snippet content should not repeat' not in user_message
