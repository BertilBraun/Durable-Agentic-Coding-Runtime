import pytest
from src.activities.context_gatherer import context_note_from_pack
from src.activities.workspace_manager import HostWorkspace
from src.llm.client import LLMUsage
from src.models.context import ContextPack, PackedSnippet
from src.models.plan import (
    ContextRequest,
    PlannerState,
    PlannerTurn,
    PlanStep,
    ReproductionPlanTurn,
    Risk,
)
from src.models.repo import RepoIndex
from src.models.reproduction import ReproductionBrief
from src.models.task import TaskContract, TaskType
from src.tools.definitions import RunShell
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
    planner_turn = PlannerTurn(next_step=_step())

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
async def test_replanning_workflow_reproduction_mode_returns_brief(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repro_turn = ReproductionPlanTurn(
        done=True,
        reproduction_brief=ReproductionBrief(
            summary='round trip',
            is_round_trip=True,
            assertion_guidance='assert read(write(x)) == x',
        ),
        remaining_work=['implement read side'],
    )

    async def fake_plan_reproduction_turn(
        state: PlannerState,
    ) -> tuple[ReproductionPlanTurn, LLMUsage]:
        return repro_turn, LLMUsage(call_count=1)

    async def fail_plan_next_turn(state: PlannerState) -> tuple[PlannerTurn, LLMUsage]:
        raise AssertionError('reproduction mode must not call plan_next_turn')

    monkeypatch.setattr(workflow_module, 'plan_reproduction_turn', fake_plan_reproduction_turn)
    monkeypatch.setattr(workflow_module, 'plan_next_turn', fail_plan_next_turn)

    result = await replanning_workflow(
        workspace=_workspace().model_dump(mode='json'),
        repo_index=RepoIndex().model_dump(mode='json'),
        max_planner_turns=5,
        planner_state=PlannerState(contract=_contract(), repo_index=RepoIndex()).model_dump(
            mode='json'
        ),
        mode='reproduction',
    )

    assert result['planner_turn']['reproduction_brief']['is_round_trip'] is True
    assert result['planner_turn']['remaining_work'] == ['implement read side']


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
            PlannerTurn(next_step=_step()),
        ]
    )

    async def fake_plan_next_turn(state: PlannerState) -> tuple[PlannerTurn, LLMUsage]:
        captured_states.append(state)
        return next(turns), LLMUsage(call_count=1, total_input_tokens=7)

    async def fake_run_child(workflow_name: str, **kwargs: object) -> dict[str, object]:
        assert workflow_name == 'context_gathering_workflow'
        request = ContextRequest.model_validate(kwargs['request'])
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
        note = context_note_from_pack(request, context_pack)
        return {
            'context_note': note.model_dump(mode='json'),
            'context_pack': context_pack.model_dump(mode='json'),
            'llm_usage': LLMUsage(call_count=1).model_dump(mode='json'),
        }

    monkeypatch.setattr(workflow_module, 'plan_next_turn', fake_plan_next_turn)
    monkeypatch.setattr(workflow_module, 'run_child', fake_run_child)

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
    assert result['planner_turn']['next_step']['id'] == 'step-1'
    assert result['context_notes'][0]['summary'] == 'Handler lives in src/app.py'
    assert result['context_notes'][0]['request_reason'] == 'Need code'
    assert result['context_notes'][0]['request_queries'] == ['Find app handler']
    assert result['context_packs'][0]['snippets'][0]['content'] == 'def handler(): ...'
    assert result['planner_turn_count'] == 2
    assert result['llm_usage']['call_count'] == 3


@pytest.mark.asyncio
async def test_replanning_workflow_executes_planner_tool_calls_before_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_states: list[PlannerState] = []
    turns = iter(
        [
            PlannerTurn(
                tool_calls=[
                    RunShell(command='Get-Content src/app.py', timeout_seconds=10),
                ]
            ),
            PlannerTurn(next_step=_step()),
        ]
    )

    async def fake_plan_next_turn(state: PlannerState) -> tuple[PlannerTurn, LLMUsage]:
        captured_states.append(state)
        return next(turns), LLMUsage(call_count=1, total_input_tokens=7)

    async def fake_run_tool(request: object) -> object:
        assert request.tool.command == 'Get-Content src/app.py'
        return workflow_module.ToolResult(
            tool_name=request.tool.tool_name,
            stdout='def handler(): ...',
            stderr='',
            exit_code=0,
            truncated=False,
        )

    monkeypatch.setattr(workflow_module, 'plan_next_turn', fake_plan_next_turn)
    monkeypatch.setattr(workflow_module, 'run_tool', fake_run_tool)

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
    assert captured_states[1].tool_observations[0].tool_name == 'run_shell'
    assert captured_states[1].tool_observations[0].stdout == 'def handler(): ...'
    assert result['planner_turn']['next_step']['id'] == 'step-1'
    assert result['planner_turn_count'] == 2
    assert result['llm_usage']['call_count'] == 2


@pytest.mark.asyncio
async def test_replanning_workflow_fulfills_context_then_executes_tools_in_same_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_states: list[PlannerState] = []
    events: list[str] = []
    turns = iter(
        [
            PlannerTurn(
                context_requests=[
                    ContextRequest(
                        id='ctx-1',
                        reason='Need app context',
                        queries=['Find app handler'],
                        relevant_files=['src/app.py'],
                    )
                ],
                tool_calls=[
                    RunShell(command='Get-Content src/app.py', timeout_seconds=10),
                ],
            ),
            PlannerTurn(next_step=_step()),
        ]
    )

    async def fake_plan_next_turn(state: PlannerState) -> tuple[PlannerTurn, LLMUsage]:
        captured_states.append(state)
        events.append('plan')
        return next(turns), LLMUsage(call_count=1, total_input_tokens=7)

    async def fake_run_child(workflow_name: str, **kwargs: object) -> dict[str, object]:
        events.append('context')
        assert workflow_name == 'context_gathering_workflow'
        request = ContextRequest.model_validate(kwargs['request'])
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
        note = context_note_from_pack(request, context_pack)
        return {
            'context_note': note.model_dump(mode='json'),
            'context_pack': context_pack.model_dump(mode='json'),
            'llm_usage': LLMUsage(call_count=1).model_dump(mode='json'),
        }

    async def fake_run_tool(request: object) -> object:
        events.append('tool')
        assert request.tool.command == 'Get-Content src/app.py'
        return workflow_module.ToolResult(
            tool_name=request.tool.tool_name,
            stdout='def handler(): ...',
            stderr='',
            exit_code=0,
            truncated=False,
        )

    monkeypatch.setattr(workflow_module, 'plan_next_turn', fake_plan_next_turn)
    monkeypatch.setattr(workflow_module, 'run_child', fake_run_child)
    monkeypatch.setattr(workflow_module, 'run_tool', fake_run_tool)

    result = await replanning_workflow(
        workspace=_workspace().model_dump(mode='json'),
        repo_index=RepoIndex().model_dump(mode='json'),
        max_planner_turns=5,
        planner_state=PlannerState(
            contract=_contract(),
            repo_index=RepoIndex(),
        ).model_dump(mode='json'),
    )

    assert events == ['plan', 'context', 'tool', 'plan']
    assert len(captured_states) == 2
    assert captured_states[1].context_notes[0].summary == 'Handler lives in src/app.py'
    assert captured_states[1].tool_observations[0].stdout == 'def handler(): ...'
    assert result['planner_turn']['next_step']['id'] == 'step-1'
    assert result['context_notes'][0]['id'] == 'ctx-1'
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

    async def fake_run_child(workflow_name: str, **kwargs: object) -> dict[str, object]:
        assert workflow_name == 'context_gathering_workflow'
        request = ContextRequest.model_validate(kwargs['request'])
        context_pack = ContextPack(task_summary='Context', snippets=[], budget_remaining=0)
        note = context_note_from_pack(request, context_pack)
        return {
            'context_note': note.model_dump(mode='json'),
            'context_pack': context_pack.model_dump(mode='json'),
            'llm_usage': LLMUsage(call_count=1).model_dump(mode='json'),
        }

    monkeypatch.setattr(workflow_module, 'plan_next_turn', fake_plan_next_turn)
    monkeypatch.setattr(workflow_module, 'run_child', fake_run_child)

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
