import pytest
from src.activities.context_gatherer import context_note_from_pack
from src.activities.workspace_manager import HostWorkspace, Workspace
from src.llm.client import LLMUsage
from src.models.context import ContextPack, PackedSnippet
from src.models.plan import ContextRequest, PlannerState, PlannerTurn, PlanStep, Risk
from src.models.repo import RepoIndex
from src.models.task import TaskContract, TaskType
from src.workflows import replanning_workflow as workflow_module
from src.workflows.replanning_workflow import replanning_workflow


def _contract() -> TaskContract:
    return TaskContract(task_type=TaskType.FEATURE, goal='Do the thing')


def _step() -> PlanStep:
    return PlanStep(
        id='step-1',
        goal='Do step one',
        target_files=['src/app.py'],
        context_summary='App context',
        required_changes=['Change app'],
        tests_to_run=['pytest tests/test_app.py'],
        expected_result='App works',
        risk=Risk.LOW,
    )


def _workspace() -> HostWorkspace:
    return HostWorkspace(
        run_id='run-1',
        execution_id='exec-1',
        base_sha='basesha',
        base_branch='main',
        current_branch='main',
        repo_path='workspace',
    )


@pytest.mark.asyncio
async def test_replanning_workflow_returns_planner_turn_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_state: list[PlannerState] = []
    planner_turn = PlannerTurn(future_steps=[_step()])

    async def fake_plan_next_turn(state: PlannerState) -> tuple[PlannerTurn, LLMUsage]:
        captured_state.append(state)
        return planner_turn, LLMUsage(call_count=1, total_input_tokens=7)

    monkeypatch.setattr(workflow_module, 'plan_next_turn', fake_plan_next_turn)

    result = await replanning_workflow(
        workspace=_workspace().model_dump(mode='json'),
        repo_index=RepoIndex().model_dump(mode='json'),
        max_planner_turns=5,
        planner_state=PlannerState(
            contract=_contract(),
            repo_index=RepoIndex(),
        ).model_dump(mode='json'),
    )

    assert captured_state[0].contract.goal == 'Do the thing'
    assert result['planner_turn'] == planner_turn.model_dump(mode='json')
    assert result['context_notes'] == []
    assert result['context_packs'] == []
    assert result['planner_turn_count'] == 1
    assert result['llm_usage']['call_count'] == 1


@pytest.mark.asyncio
async def test_replanning_workflow_fulfills_context_before_returning_ready_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_states: list[PlannerState] = []
    turns = iter(
        [
            PlannerTurn(
                context_requests=[
                    ContextRequest(
                        id='ctx-1',
                        reason='Need code',
                        queries=['Find app handler'],
                        relevant_files=['src/app.py'],
                    )
                ]
            ),
            PlannerTurn(future_steps=[_step()]),
        ]
    )

    async def fake_plan_next_turn(state: PlannerState) -> tuple[PlannerTurn, LLMUsage]:
        captured_states.append(state)
        return next(turns), LLMUsage(call_count=1, total_input_tokens=7)

    async def fake_fulfill_context_request(
        workspace_info: Workspace,
        repo_index: RepoIndex,
        request: ContextRequest,
    ) -> tuple[object, ContextPack, LLMUsage]:
        context_pack = ContextPack(
            task_summary='Handler lives in src/app.py',
            snippets=[
                PackedSnippet(
                    file_path='src/app.py',
                    start_line=1,
                    end_line=2,
                    reason='handler',
                    content='def handler(): ...',
                )
            ],
            budget_remaining=0,
        )
        return context_note_from_pack(request, context_pack), context_pack, LLMUsage(call_count=1)

    monkeypatch.setattr(workflow_module, 'plan_next_turn', fake_plan_next_turn)
    monkeypatch.setattr(
        workflow_module,
        'fulfill_context_request',
        fake_fulfill_context_request,
    )

    result = await replanning_workflow(
        workspace=_workspace().model_dump(mode='json'),
        repo_index=RepoIndex().model_dump(mode='json'),
        max_planner_turns=5,
        planner_state=PlannerState(
            contract=_contract(),
            repo_index=RepoIndex(),
        ).model_dump(mode='json'),
    )

    assert len(captured_states) == 2
    assert captured_states[1].context_notes[0].summary == 'Handler lives in src/app.py'
    assert captured_states[1].context_notes[0].request_reason == 'Need code'
    assert captured_states[1].context_notes[0].request_queries == ['Find app handler']
    assert captured_states[1].model_dump(mode='json')['context_notes'][0]['request_reason'] == (
        'Need code'
    )
    assert result['planner_turn']['future_steps'][0]['id'] == 'step-1'
    assert result['context_notes'][0]['summary'] == 'Handler lives in src/app.py'
    assert result['context_notes'][0]['request_reason'] == 'Need code'
    assert result['context_notes'][0]['request_queries'] == ['Find app handler']
    assert result['context_packs'][0]['snippets'][0]['content'] == 'def handler(): ...'
    assert result['planner_turn_count'] == 2
    assert result['llm_usage']['call_count'] == 3


@pytest.mark.asyncio
async def test_replanning_workflow_stops_at_planner_turn_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def fake_plan_next_turn(state: PlannerState) -> tuple[PlannerTurn, LLMUsage]:
        nonlocal calls
        calls += 1
        return (
            PlannerTurn(
                context_requests=[
                    ContextRequest(
                        id=f'ctx-{calls}',
                        reason='Need more code',
                        queries=['Find code'],
                    )
                ]
            ),
            LLMUsage(call_count=1),
        )

    async def fake_fulfill_context_request(
        workspace_info: Workspace,
        repo_index: RepoIndex,
        request: ContextRequest,
    ) -> tuple[object, ContextPack, LLMUsage]:
        context_pack = ContextPack(task_summary='Context', snippets=[], budget_remaining=0)
        return context_note_from_pack(request, context_pack), context_pack, LLMUsage(call_count=1)

    monkeypatch.setattr(workflow_module, 'plan_next_turn', fake_plan_next_turn)
    monkeypatch.setattr(
        workflow_module,
        'fulfill_context_request',
        fake_fulfill_context_request,
    )

    result = await replanning_workflow(
        workspace=_workspace().model_dump(mode='json'),
        repo_index=RepoIndex().model_dump(mode='json'),
        max_planner_turns=1,
        planner_state=PlannerState(
            contract=_contract(),
            repo_index=RepoIndex(),
        ).model_dump(mode='json'),
    )

    assert calls == 1
    assert result['planner_turn']['context_requests'][0]['id'] == 'ctx-1'
    assert result['planner_turn_count'] == 1
