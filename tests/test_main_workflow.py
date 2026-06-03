import pytest
from src.activities.context_gatherer import context_note_from_pack
from src.activities.reviewer import ReviewDecision
from src.activities.workspace_manager import (
    HostWorkspace,
    ToolExecutionRequest,
    ToolResult,
    Workspace,
)
from src.config import CONFIG
from src.llm.client import LLMUsage
from src.models.context import ContextPack, PackedSnippet
from src.models.plan import (
    ContextRequest,
    PlanContext,
    PlannerState,
    PlannerTurn,
    PlanStep,
    Risk,
)
from src.models.repo import RepoIndex
from src.models.reproduction import ReproductionContext, ReproductionResult, ReproductionStatus
from src.models.task import Origin, TaskContract, TaskRequest, TaskType
from src.models.worker import Confidence, TestResult, WorkerResult, WorkerStatus
from src.workflows import main_workflow as workflow_module
from src.workflows.main_workflow import (
    PlannerLoopState,
    _run_implementation_child,
    _run_planner_loop,
    _run_replanner_child,
    _run_reproduction_child,
    _select_relevant_test_history,
    main_workflow,
)
from temporal_light.exceptions import WorkflowSuspended


def _usage(count: int = 1) -> LLMUsage:
    return LLMUsage(call_count=count, total_input_tokens=10, total_cost_usd=0.01)


def _workspace() -> HostWorkspace:
    return HostWorkspace(
        run_id='run-1',
        execution_id='exec-1',
        base_sha='basesha',
        base_branch='main',
        current_branch='main',
        repo_path='workspace',
    )


def _ok_tool() -> ToolResult:
    return ToolResult(stdout='', stderr='', exit_code=0, truncated=False)


def _contract(task_type: TaskType = TaskType.FEATURE) -> TaskContract:
    return TaskContract(task_type=task_type, goal='Do the thing')


def _step(step_id: str, confidence: Confidence = Confidence.HIGH) -> PlanStep:
    return PlanStep(
        id=step_id,
        goal=f'Complete {step_id}',
        target_files=[f'{step_id}.py'],
        context_summary=f'Context for {step_id}',
        required_changes=[f'Change {step_id}'],
        out_of_scope=['Do not touch unrelated files'],
        tests_to_run=[],
        expected_result=f'{step_id} complete',
        risk=Risk.LOW,
        requires_human_approval=False,
    )


def _worker_result(
    status: WorkerStatus = WorkerStatus.SUCCESS,
    confidence: Confidence = Confidence.HIGH,
    tests: list[TestResult] | None = None,
) -> WorkerResult:
    return WorkerResult(
        status=status,
        patch_id='patch-1' if status == WorkerStatus.SUCCESS else None,
        diff_summary='did work',
        tests_run=[test.command for test in tests or []],
        test_results=tests or [],
        discovered_issues=[],
        observations=['observed behavior'],
        confidence=confidence,
        replan_suggestion=None,
    )


def _test_result(sequence: int, passed: bool) -> TestResult:
    return TestResult(
        sequence=sequence,
        command=f'pytest test_{sequence}.py',
        exit_code=0 if passed else 1,
        stdout_summary='passed' if passed else 'failed',
        stderr_summary='',
        passed=passed,
    )


def _install_step_branch_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_begin_candidate(workspace_arg: Workspace, candidate_index: int) -> Workspace:
        return workspace_arg.model_copy(update={'current_branch': f'cand-{candidate_index}'})

    async def fake_reset_to_base(workspace_arg: Workspace) -> ToolResult:
        return _ok_tool()

    async def fake_snapshot_candidate_result(workspace_arg: Workspace) -> Workspace:
        return workspace_arg

    async def fake_snapshot_candidate_base(workspace_arg: Workspace) -> Workspace:
        return workspace_arg.model_copy(
            update={'candidate_base_sha': f'base-after-{workspace_arg.current_branch}'}
        )

    monkeypatch.setattr(workflow_module, 'begin_candidate', fake_begin_candidate)
    monkeypatch.setattr(workflow_module, 'reset_to_base', fake_reset_to_base)
    monkeypatch.setattr(
        workflow_module,
        'snapshot_candidate_result',
        fake_snapshot_candidate_result,
    )
    monkeypatch.setattr(workflow_module, 'snapshot_candidate_base', fake_snapshot_candidate_base)


def test_context_note_from_pack_keeps_compact_evidence_only() -> None:
    note = context_note_from_pack(
        ContextRequest(
            id='ctx-1',
            reason='Find parser',
            queries=['Where is parse?'],
            relevant_files=['src/requested.py'],
        ),
        ContextPack(
            task_summary='Parser is in src/parser.py',
            snippets=[
                PackedSnippet(
                    file_path='src/parser.py',
                    start_line=1,
                    end_line=3,
                    reason='parser',
                    content='def parse(): ...',
                )
            ],
            budget_remaining=0,
        ),
    )

    assert note.id == 'ctx-1'
    assert note.summary == 'Parser is in src/parser.py'
    assert note.relevant_files == ['src/requested.py', 'src/parser.py']


def test_planner_loop_state_appends_context_notes_immutably() -> None:
    state = PlannerLoopState(
        workspace_info=_workspace(),
        planner_state=PlannerState(contract=_contract(), repo_index=RepoIndex()),
        context_packs=[],
        latest_future_steps=[],
        worker_results=[],
        usage=LLMUsage(),
    )
    note = context_note_from_pack(
        ContextRequest(id='ctx-1', reason='Find parser'),
        ContextPack(task_summary='Parser context', snippets=[], budget_remaining=0),
    )

    updated = state.with_context(notes=[note], packs=[])

    assert state.planner_state.context_notes == []
    assert updated.planner_state.context_notes == [note]
    assert updated is not state


@pytest.mark.asyncio
async def test_planner_loop_fulfills_context_then_runs_first_future_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner_states: list[PlannerState] = []
    turns = iter(
        [
            PlannerTurn(future_steps=[_step('step-1'), _step('stale-step-2')]),
            PlannerTurn(done=True, done_reason='ready'),
        ]
    )
    executed_steps: list[str] = []

    async def fake_run_replanner_child(
        workspace_info: Workspace,
        repo_index: RepoIndex,
        max_planner_turns: int,
        planner_state: PlannerState,
    ) -> tuple[PlannerTurn, list[object], list[ContextPack], int, LLMUsage]:
        planner_states.append(planner_state)
        context_pack = ContextPack(
            task_summary='Handler context',
            snippets=[
                PackedSnippet(
                    file_path='src/handler.py',
                    start_line=1,
                    end_line=4,
                    reason='handler',
                    content='def handle(): ...',
                )
            ],
            budget_remaining=0,
        )
        notes = []
        packs = []
        if not planner_state.context_notes:
            request = ContextRequest(
                id='ctx-1',
                reason='Need file context',
                queries=['Find handler'],
            )
            notes = [context_note_from_pack(request, context_pack)]
            packs = [context_pack]
        return next(turns), notes, packs, 1, _usage()

    async def fake_run_implementation_child(
        plan_step: PlanStep,
        plan_context: PlanContext,
        context_pack: ContextPack,
        workspace_info: Workspace,
        contract: TaskContract,
        repo_index: RepoIndex,
    ) -> tuple[WorkerResult, LLMUsage]:
        executed_steps.append(plan_step.id)
        assert plan_context.completed_step_summaries == []
        assert context_pack.snippets[0].content == 'def handle(): ...'
        return _worker_result(), _usage()

    async def fake_get_full_diff(workspace_arg: Workspace) -> str:
        return 'diff --git a/app.py b/app.py'

    monkeypatch.setattr(workflow_module, '_run_replanner_child', fake_run_replanner_child)
    monkeypatch.setattr(workflow_module, '_run_implementation_child', fake_run_implementation_child)
    monkeypatch.setattr(workflow_module, 'get_full_diff', fake_get_full_diff)
    _install_step_branch_fakes(monkeypatch)

    result = await _run_planner_loop(
        workspace_info=_workspace(),
        contract=_contract(),
        repo_index=RepoIndex(),
        reproduction=None,
    )

    assert executed_steps == ['step-1']
    assert len(planner_states) == 2
    assert planner_states[1].completed_steps[0].step_id == 'step-1'
    assert result.worker_results[0].observations == ['observed behavior']


@pytest.mark.asyncio
async def test_planner_loop_passes_relevant_context_pack_to_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = iter([PlannerTurn(future_steps=[_step('step-1')]), PlannerTurn(done=True)])
    captured_pack: list[ContextPack] = []

    async def fake_run_replanner_child(
        workspace_info: Workspace,
        repo_index: RepoIndex,
        max_planner_turns: int,
        planner_state: PlannerState,
    ) -> tuple[PlannerTurn, list[object], list[ContextPack], int, LLMUsage]:
        return next(turns), [], [], 1, _usage()

    async def fake_run_implementation_child(
        plan_step: PlanStep,
        plan_context: PlanContext,
        context_pack: ContextPack,
        workspace_info: Workspace,
        contract: TaskContract,
        repo_index: RepoIndex,
    ) -> tuple[WorkerResult, LLMUsage]:
        captured_pack.append(context_pack)
        return _worker_result(), _usage()

    async def fake_get_full_diff(workspace_arg: Workspace) -> str:
        return 'diff --git a/app.py b/app.py'

    monkeypatch.setattr(workflow_module, '_run_replanner_child', fake_run_replanner_child)
    monkeypatch.setattr(workflow_module, '_run_implementation_child', fake_run_implementation_child)
    monkeypatch.setattr(workflow_module, 'get_full_diff', fake_get_full_diff)
    _install_step_branch_fakes(monkeypatch)

    await _run_planner_loop(_workspace(), _contract(), RepoIndex(), None)

    assert captured_pack[0].task_summary == 'Context for step-1'


@pytest.mark.asyncio
async def test_planner_loop_replans_after_one_step_and_does_not_run_stale_second_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = iter(
        [
            PlannerTurn(future_steps=[_step('step-1'), _step('old-step-2')]),
            PlannerTurn(future_steps=[_step('new-step-2')]),
            PlannerTurn(done=True),
        ]
    )
    executed_steps: list[str] = []

    async def fake_run_replanner_child(
        workspace_info: Workspace,
        repo_index: RepoIndex,
        max_planner_turns: int,
        planner_state: PlannerState,
    ) -> tuple[PlannerTurn, list[object], list[ContextPack], int, LLMUsage]:
        return next(turns), [], [], 1, _usage()

    async def fake_run_implementation_child(
        plan_step: PlanStep,
        plan_context: PlanContext,
        context_pack: ContextPack,
        workspace_info: Workspace,
        contract: TaskContract,
        repo_index: RepoIndex,
    ) -> tuple[WorkerResult, LLMUsage]:
        executed_steps.append(plan_step.id)
        return _worker_result(), _usage()

    async def fake_get_full_diff(workspace_arg: Workspace) -> str:
        return 'diff --git a/app.py b/app.py'

    monkeypatch.setattr(workflow_module, '_run_replanner_child', fake_run_replanner_child)
    monkeypatch.setattr(workflow_module, '_run_implementation_child', fake_run_implementation_child)
    monkeypatch.setattr(workflow_module, 'get_full_diff', fake_get_full_diff)
    _install_step_branch_fakes(monkeypatch)

    await _run_planner_loop(_workspace(), _contract(), RepoIndex(), None)

    assert executed_steps == ['step-1', 'new-step-2']


@pytest.mark.asyncio
async def test_needs_replan_advances_workspace_without_completed_step_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner_states: list[PlannerState] = []
    turns = iter(
        [
            PlannerTurn(future_steps=[_step('failed-restore')]),
            PlannerTurn(future_steps=[_step('retry-restore')]),
            PlannerTurn(done=True),
        ]
    )
    implementation_contexts: list[PlanContext] = []
    diff_workspaces: list[Workspace] = []
    worker_results = iter(
        [
            _worker_result(status=WorkerStatus.NEEDS_REPLAN, confidence=Confidence.LOW),
            _worker_result(),
        ]
    )

    async def fake_run_replanner_child(
        workspace_info: Workspace,
        repo_index: RepoIndex,
        max_planner_turns: int,
        planner_state: PlannerState,
    ) -> tuple[PlannerTurn, list[object], list[ContextPack], int, LLMUsage]:
        planner_states.append(planner_state)
        return next(turns), [], [], 1, _usage()

    async def fake_run_implementation_child(
        plan_step: PlanStep,
        plan_context: PlanContext,
        context_pack: ContextPack,
        workspace_info: Workspace,
        contract: TaskContract,
        repo_index: RepoIndex,
    ) -> tuple[WorkerResult, LLMUsage]:
        implementation_contexts.append(plan_context)
        return next(worker_results), _usage()

    async def fake_get_full_diff(workspace_arg: Workspace) -> str:
        diff_workspaces.append(workspace_arg)
        return 'diff --git a/app.py b/app.py'

    monkeypatch.setattr(workflow_module, '_run_replanner_child', fake_run_replanner_child)
    monkeypatch.setattr(workflow_module, '_run_implementation_child', fake_run_implementation_child)
    monkeypatch.setattr(workflow_module, 'get_full_diff', fake_get_full_diff)
    _install_step_branch_fakes(monkeypatch)

    result = await _run_planner_loop(_workspace(), _contract(), RepoIndex(), None)

    assert diff_workspaces[0].current_branch == 'cand-0'
    assert result.workspace_info.current_branch == 'cand-1'
    assert planner_states[1].completed_steps[0].outcome == WorkerStatus.NEEDS_REPLAN
    assert implementation_contexts[1].completed_step_summaries == []


@pytest.mark.asyncio
async def test_failed_step_does_not_advance_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = iter([PlannerTurn(future_steps=[_step('failed-step')]), PlannerTurn(done=True)])
    diff_workspaces: list[Workspace] = []

    async def fake_run_replanner_child(
        workspace_info: Workspace,
        repo_index: RepoIndex,
        max_planner_turns: int,
        planner_state: PlannerState,
    ) -> tuple[PlannerTurn, list[object], list[ContextPack], int, LLMUsage]:
        return next(turns), [], [], 1, _usage()

    async def fake_run_implementation_child(
        plan_step: PlanStep,
        plan_context: PlanContext,
        context_pack: ContextPack,
        workspace_info: Workspace,
        contract: TaskContract,
        repo_index: RepoIndex,
    ) -> tuple[WorkerResult, LLMUsage]:
        return _worker_result(status=WorkerStatus.FAILED, confidence=Confidence.LOW), _usage()

    async def fake_get_full_diff(workspace_arg: Workspace) -> str:
        diff_workspaces.append(workspace_arg)
        return ''

    monkeypatch.setattr(workflow_module, '_run_replanner_child', fake_run_replanner_child)
    monkeypatch.setattr(workflow_module, '_run_implementation_child', fake_run_implementation_child)
    monkeypatch.setattr(workflow_module, 'get_full_diff', fake_get_full_diff)
    _install_step_branch_fakes(monkeypatch)

    result = await _run_planner_loop(_workspace(), _contract(), RepoIndex(), None)

    assert diff_workspaces[0].current_branch == 'main'
    assert result.workspace_info.current_branch == 'main'


@pytest.mark.asyncio
async def test_planner_turn_cap_produces_blocked_result(monkeypatch: pytest.MonkeyPatch) -> None:
    replanner_calls = 0

    async def fake_run_replanner_child(
        workspace_info: Workspace,
        repo_index: RepoIndex,
        max_planner_turns: int,
        planner_state: PlannerState,
    ) -> tuple[PlannerTurn, list[object], list[ContextPack], int, LLMUsage]:
        nonlocal replanner_calls
        replanner_calls += 1
        assert max_planner_turns == 1
        return PlannerTurn(context_requests=[]), [], [], 1, _usage()

    monkeypatch.setattr(workflow_module, '_run_replanner_child', fake_run_replanner_child)
    monkeypatch.setattr(
        workflow_module,
        'CONFIG',
        CONFIG.model_copy(update={'max_planner_turns': 1}),
    )

    result = await _run_planner_loop(_workspace(), _contract(), RepoIndex(), None)

    assert replanner_calls == 1
    assert result.worker_results[-1].status == WorkerStatus.BLOCKED
    assert result.worker_results[-1].diff_summary == 'Planner turn cap reached before done.'


def test_select_relevant_test_history_keeps_last_pass_and_trailing_failures() -> None:
    assert _select_relevant_test_history([_test_result(1, False), _test_result(2, False)]) == [
        _test_result(1, False),
        _test_result(2, False),
    ]
    assert _select_relevant_test_history([_test_result(1, False), _test_result(2, True)]) == [
        _test_result(2, True)
    ]
    assert _select_relevant_test_history(
        [_test_result(1, False), _test_result(2, True), _test_result(3, False)]
    ) == [_test_result(2, True), _test_result(3, False)]
    assert _select_relevant_test_history([_test_result(1, True), _test_result(2, True)]) == [
        _test_result(2, True)
    ]


@pytest.mark.asyncio
async def test_reproduction_failure_is_refreshed_as_planner_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner_states: list[PlannerState] = []
    turns = iter([PlannerTurn(future_steps=[_step('fix')]), PlannerTurn(done=True)])
    repro_calls = 0

    async def fake_run_replanner_child(
        workspace_info: Workspace,
        repo_index: RepoIndex,
        max_planner_turns: int,
        planner_state: PlannerState,
    ) -> tuple[PlannerTurn, list[object], list[ContextPack], int, LLMUsage]:
        planner_states.append(planner_state)
        return next(turns), [], [], 1, _usage()

    async def fake_run_implementation_child(
        plan_step: PlanStep,
        plan_context: PlanContext,
        context_pack: ContextPack,
        workspace_info: Workspace,
        contract: TaskContract,
        repo_index: RepoIndex,
    ) -> tuple[WorkerResult, LLMUsage]:
        return _worker_result(), _usage()

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        nonlocal repro_calls
        repro_calls += 1
        return ToolResult(stdout='still red', stderr='', exit_code=1, truncated=False)

    async def fake_get_full_diff(workspace_arg: Workspace) -> str:
        return 'diff --git a/app.py b/app.py'

    monkeypatch.setattr(workflow_module, '_run_replanner_child', fake_run_replanner_child)
    monkeypatch.setattr(workflow_module, '_run_implementation_child', fake_run_implementation_child)
    monkeypatch.setattr(workflow_module, 'run_tool', fake_run_tool)
    monkeypatch.setattr(workflow_module, 'get_full_diff', fake_get_full_diff)
    _install_step_branch_fakes(monkeypatch)

    await _run_planner_loop(
        _workspace(),
        _contract(TaskType.BUGFIX),
        RepoIndex(),
        ReproductionContext(repro_command='pytest bug.py', failure_evidence='boom'),
    )

    assert repro_calls == 1
    assert planner_states[1].evidence.reproduction_passed is False
    assert planner_states[1].evidence.reproduction_stdout_summary == 'still red'


@pytest.mark.asyncio
async def test_reproduction_child_uses_semantic_child_id(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned_kwargs: list[dict[str, object]] = []

    async def fake_spawn_child(workflow_name: str, **kwargs: object) -> str:
        spawned_kwargs.append({'workflow_name': workflow_name, **kwargs})
        return str(kwargs['child_id'])

    async def fake_wait_for_child(child_id: str) -> dict[str, object]:
        return {
            'reproduction_result': ReproductionResult(
                status=ReproductionStatus.REPRODUCED,
                repro_command='pytest bug.py',
                failure_evidence='boom',
            ).model_dump(mode='json'),
            'llm_usage': _usage().model_dump(mode='json'),
        }

    monkeypatch.setattr(workflow_module, 'spawn_child', fake_spawn_child)
    monkeypatch.setattr(workflow_module, 'wait_for_child', fake_wait_for_child)

    await _run_reproduction_child(_workspace(), _contract(TaskType.BUGFIX), RepoIndex())

    assert spawned_kwargs[0]['workflow_name'] == 'reproduction_workflow'
    assert spawned_kwargs[0]['child_id'] == 'run-1:exec-1:reproduction'


@pytest.mark.asyncio
async def test_implementation_child_includes_rich_step_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned_kwargs: list[dict[str, object]] = []

    async def fake_spawn_child(workflow_name: str, **kwargs: object) -> str:
        spawned_kwargs.append({'workflow_name': workflow_name, **kwargs})
        return str(kwargs['child_id'])

    async def fake_wait_for_child(child_id: str) -> dict[str, object]:
        return {
            'worker_result': _worker_result().model_dump(mode='json'),
            'llm_usage': _usage().model_dump(mode='json'),
        }

    monkeypatch.setattr(workflow_module, 'spawn_child', fake_spawn_child)
    monkeypatch.setattr(workflow_module, 'wait_for_child', fake_wait_for_child)

    await _run_implementation_child(
        plan_step=_step('implement/subtract'),
        plan_context=PlanContext(
            summary='Planner-driven execution',
            current_step_id='implement/subtract',
            all_step_ids=['implement/subtract'],
            completed_step_summaries=['prior: done'],
        ),
        context_pack=ContextPack(
            task_summary='Packed implementation context',
            snippets=[
                PackedSnippet(
                    file_path='src/app.py',
                    start_line=1,
                    end_line=2,
                    reason='target',
                    content='def app(): ...',
                )
            ],
            budget_remaining=3,
        ),
        workspace_info=_workspace().model_copy(update={'current_branch': 'agentic/run-1/cand-0'}),
        contract=_contract(),
        repo_index=RepoIndex(),
    )

    assert spawned_kwargs[0]['workflow_name'] == 'implementation_workflow'
    assert spawned_kwargs[0]['child_id'] == (
        'run-1:exec-1:implementation:agentic-run-1-cand-0:implement-subtract'
    )
    assert spawned_kwargs[0]['step']['context_summary'] == 'Context for implement/subtract'
    assert spawned_kwargs[0]['step']['required_changes'] == ['Change implement/subtract']
    assert spawned_kwargs[0]['plan_context']['completed_step_summaries'] == ['prior: done']
    assert spawned_kwargs[0]['context_pack']['snippets'][0]['content'] == 'def app(): ...'


@pytest.mark.asyncio
async def test_replanner_child_uses_semantic_child_id(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned_kwargs: list[dict[str, object]] = []
    planner_state = PlannerState(contract=_contract(), repo_index=RepoIndex())
    planner_turn = PlannerTurn(future_steps=[_step('step-1')])

    async def fake_spawn_child(workflow_name: str, **kwargs: object) -> str:
        spawned_kwargs.append({'workflow_name': workflow_name, **kwargs})
        return str(kwargs['child_id'])

    async def fake_wait_for_child(child_id: str) -> dict[str, object]:
        return {
            'planner_turn': planner_turn.model_dump(mode='json'),
            'context_notes': [],
            'context_packs': [],
            'planner_turn_count': 1,
            'llm_usage': _usage().model_dump(mode='json'),
        }

    monkeypatch.setattr(workflow_module, 'spawn_child', fake_spawn_child)
    monkeypatch.setattr(workflow_module, 'wait_for_child', fake_wait_for_child)

    turn, notes, packs, turn_count, usage = await _run_replanner_child(
        _workspace(),
        RepoIndex(),
        5,
        planner_state,
    )

    assert turn == planner_turn
    assert notes == []
    assert packs == []
    assert turn_count == 1
    assert usage.call_count == 1
    assert spawned_kwargs[0]['workflow_name'] == 'replanning_workflow'
    assert spawned_kwargs[0]['child_id'] == 'run-1:exec-1:replanning'
    assert spawned_kwargs[0]['workspace']['repo_path'] == 'workspace'
    assert spawned_kwargs[0]['repo_index'] == RepoIndex().model_dump(mode='json')
    assert spawned_kwargs[0]['max_planner_turns'] == 5
    assert spawned_kwargs[0]['planner_state']['contract']['goal'] == 'Do the thing'


def _install_common_workflow_fakes(
    monkeypatch: pytest.MonkeyPatch,
    contract: TaskContract,
    workspace: Workspace,
) -> None:
    async def fake_build_contract(task_request: TaskRequest) -> tuple[TaskContract, LLMUsage]:
        return contract, _usage()

    async def fake_setup_environment(origin: Origin, run_id: str) -> Workspace:
        return workspace

    async def fake_build_repo_index(workspace_arg: Workspace) -> RepoIndex:
        return RepoIndex()

    async def fake_teardown_environment(workspace_arg: Workspace) -> ToolResult:
        return _ok_tool()

    monkeypatch.setattr(workflow_module, 'build_contract', fake_build_contract)
    monkeypatch.setattr(workflow_module, 'setup_environment', fake_setup_environment)
    monkeypatch.setattr(workflow_module, 'build_repo_index', fake_build_repo_index)
    monkeypatch.setattr(workflow_module, 'teardown_environment', fake_teardown_environment)


@pytest.mark.asyncio
async def test_main_workflow_reports_final_reproduction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace()
    _install_common_workflow_fakes(monkeypatch, _contract(TaskType.BUGFIX), workspace)
    _install_step_branch_fakes(monkeypatch)
    turns = iter([PlannerTurn(future_steps=[_step('fix')]), PlannerTurn(done=True)])
    run_tool_calls = 0

    async def fake_run_reproduction_child(
        workspace_info: Workspace,
        contract: TaskContract,
        repo_index: RepoIndex,
    ) -> tuple[ReproductionResult, LLMUsage]:
        return (
            ReproductionResult(
                status=ReproductionStatus.REPRODUCED,
                repro_command='pytest bug.py',
                failure_evidence='boom',
            ),
            _usage(),
        )

    async def fake_run_replanner_child(
        workspace_info: Workspace,
        repo_index: RepoIndex,
        max_planner_turns: int,
        planner_state: PlannerState,
    ) -> tuple[PlannerTurn, list[object], list[ContextPack], int, LLMUsage]:
        return next(turns), [], [], 1, _usage()

    async def fake_run_implementation_child(
        plan_step: PlanStep,
        plan_context: PlanContext,
        context_pack: ContextPack,
        workspace_info: Workspace,
        contract: TaskContract,
        repo_index: RepoIndex,
    ) -> tuple[WorkerResult, LLMUsage]:
        return _worker_result(), _usage()

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        nonlocal run_tool_calls
        run_tool_calls += 1
        exit_code = 1 if run_tool_calls == 2 else 0
        return ToolResult(
            stdout='red' if exit_code else 'green',
            stderr='',
            exit_code=exit_code,
            truncated=False,
        )

    async def fake_get_full_diff(workspace_arg: Workspace) -> str:
        return 'diff --git a/app.py b/app.py'

    async def fake_finalize_winner(workspace_arg: Workspace, winner_branch: str) -> ToolResult:
        return _ok_tool()

    monkeypatch.setattr(workflow_module, '_run_reproduction_child', fake_run_reproduction_child)
    monkeypatch.setattr(workflow_module, '_run_replanner_child', fake_run_replanner_child)
    monkeypatch.setattr(workflow_module, '_run_implementation_child', fake_run_implementation_child)
    monkeypatch.setattr(workflow_module, 'run_tool', fake_run_tool)
    monkeypatch.setattr(workflow_module, 'get_full_diff', fake_get_full_diff)
    monkeypatch.setattr(workflow_module, 'finalize_winner', fake_finalize_winner)

    report = await main_workflow(
        {'raw_request': 'fix bug', 'origin': {'kind': 'host', 'repo_path': 'C:/repo'}}
    )

    assert report['workflow_status'] == 'completed'
    assert report['agent_verdict'] == ReviewDecision.REVISE.value
    assert report['reproduction_passed'] is False
    assert report['official_prediction_emitted'] is True


@pytest.mark.asyncio
async def test_main_workflow_does_not_teardown_while_suspended_on_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torn_down: list[Workspace] = []
    workspace = _workspace()
    _install_common_workflow_fakes(monkeypatch, _contract(), workspace)

    async def fake_spawn_child(workflow_name: str, **kwargs: object) -> str:
        return 'child-1'

    async def fake_wait_for_child(child_id: str) -> dict[str, object]:
        raise WorkflowSuspended('Workflow waiting for child.')

    async def fake_teardown_environment(workspace_arg: Workspace) -> ToolResult:
        torn_down.append(workspace_arg)
        return _ok_tool()

    monkeypatch.setattr(workflow_module, 'spawn_child', fake_spawn_child)
    monkeypatch.setattr(workflow_module, 'wait_for_child', fake_wait_for_child)
    monkeypatch.setattr(workflow_module, 'teardown_environment', fake_teardown_environment)
    _install_step_branch_fakes(monkeypatch)

    with pytest.raises(WorkflowSuspended, match=r'Workflow waiting for child\.'):
        await main_workflow(
            {'raw_request': 'add feature', 'origin': {'kind': 'host', 'repo_path': 'C:/repo'}}
        )

    assert torn_down == []
