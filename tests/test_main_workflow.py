import pytest
from src.activities.complexity_assessor import ComplexityVerdict
from src.activities.plan_reviewer import (
    PlanReviewDecision,
    PlanReviewRequest,
    PlanReviewVerdict,
)
from src.activities.planner import PlanRequest
from src.activities.reviewer import ReviewDecision, ReviewRequest, ReviewVerdict
from src.activities.workspace_manager import (
    HostWorkspace,
    ToolExecutionRequest,
    ToolResult,
    Workspace,
)
from src.config import CONFIG
from src.llm.client import LLMUsage
from src.models.context import ContextPack
from src.models.plan import Plan, PlanStep, Risk
from src.models.repo import RepoIndex
from src.models.reproduction import ReproductionContext, ReproductionResult, ReproductionStatus
from src.models.task import Origin, TaskContract, TaskRequest, TaskType
from src.models.worker import Confidence, WorkerResult, WorkerStatus
from src.workflows.main_workflow import (
    _review_and_revise_plan,
    _run_plan_steps,
    main_workflow,
)
from temporal_light.exceptions import WorkflowSuspended


def _unit_usage() -> LLMUsage:
    return LLMUsage(call_count=1, total_input_tokens=10, total_cost_usd=0.01)


def _workspace() -> HostWorkspace:
    return HostWorkspace(
        run_id='run-1',
        base_sha='basesha',
        base_branch='main',
        current_branch='main',
        repo_path='workspace',
    )


def _ok_result() -> ToolResult:
    return ToolResult(stdout='', stderr='', exit_code=0, truncated=False)


def _worker_result(status: WorkerStatus, replan_suggestion: str | None = None) -> WorkerResult:
    return WorkerResult(
        status=status,
        patch_id='patch-1' if status == WorkerStatus.SUCCESS else None,
        diff_summary='did work',
        tests_run=[],
        test_results=[],
        discovered_issues=[],
        confidence=Confidence.HIGH,
        replan_suggestion=replan_suggestion,
    )


def _contract(task_type: TaskType) -> TaskContract:
    return TaskContract(task_type=task_type, goal='Do the thing')


def _plan_with_step(step_id: str, integration_tests: list[str] | None = None) -> Plan:
    return Plan(
        summary=f'Plan {step_id}',
        steps=[
            PlanStep(
                id=step_id,
                goal='Update generated output',
                target_files=['generated.py'],
                tests_to_run=[],
                expected_result='Generated output updated',
                risk=Risk.LOW,
                requires_human_approval=False,
            )
        ],
        integration_tests=integration_tests or [],
        rollback_strategy='git checkout',
        definition_of_done=['diff reviewed'],
    )


@pytest.mark.asyncio
async def test_run_plan_steps_adds_child_workflow_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_implementation_child(
        plan_step: PlanStep,
        workspace_info: Workspace,
        contract: TaskContract,
        repo_index: RepoIndex,
    ) -> tuple[WorkerResult, LLMUsage]:
        return _worker_result(WorkerStatus.SUCCESS), LLMUsage(
            call_count=3, total_input_tokens=30, total_output_tokens=12, total_cost_usd=0.05
        )

    monkeypatch.setattr(
        'src.workflows.main_workflow._run_implementation_child', fake_run_implementation_child
    )

    _, worker_results, after_exit_code, usage = await _run_plan_steps(
        plan=_plan_with_step('step-1'),
        contract=_contract(TaskType.FEATURE),
        repo_index=RepoIndex(),
        workspace_info=_workspace(),
        reproduction=None,
    )

    assert len(worker_results) == 1
    assert after_exit_code is None
    assert usage.call_count == 3
    assert usage.total_cost_usd == 0.05


@pytest.mark.asyncio
async def test_run_plan_steps_caps_replan_attempts(monkeypatch: pytest.MonkeyPatch) -> None:

    implementation_calls = 0

    async def fake_run_implementation_child(
        plan_step: PlanStep,
        workspace_info: Workspace,
        contract: TaskContract,
        repo_index: RepoIndex,
    ) -> tuple[WorkerResult, LLMUsage]:
        nonlocal implementation_calls
        implementation_calls += 1
        return _worker_result(WorkerStatus.NEEDS_REPLAN, 'try again'), _unit_usage()

    async def fake_build_plan(request: PlanRequest) -> tuple[Plan, LLMUsage]:
        return _plan_with_step('step'), _unit_usage()

    monkeypatch.setattr(
        'src.workflows.main_workflow._run_implementation_child', fake_run_implementation_child
    )
    monkeypatch.setattr('src.workflows.main_workflow.build_plan', fake_build_plan)

    _, worker_results, _, _ = await _run_plan_steps(
        plan=_plan_with_step('step'),
        contract=_contract(TaskType.FEATURE),
        repo_index=RepoIndex(),
        workspace_info=_workspace(),
        reproduction=None,
    )

    assert implementation_calls == CONFIG.max_replan_attempts + 1
    assert worker_results[-1].status == WorkerStatus.BLOCKED


@pytest.mark.asyncio
async def test_run_plan_steps_final_gate_passes_when_repro_and_suite_green(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run_implementation_child(
        plan_step: PlanStep,
        workspace_info: Workspace,
        contract: TaskContract,
        repo_index: RepoIndex,
    ) -> tuple[WorkerResult, LLMUsage]:
        return _worker_result(WorkerStatus.SUCCESS), _unit_usage()

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        return _ok_result()

    monkeypatch.setattr(
        'src.workflows.main_workflow._run_implementation_child', fake_run_implementation_child
    )
    monkeypatch.setattr('src.workflows.main_workflow.run_tool', fake_run_tool)

    _, worker_results, after_exit_code, _ = await _run_plan_steps(
        plan=_plan_with_step('step-1'),
        contract=_contract(TaskType.BUGFIX),
        repo_index=RepoIndex(),
        workspace_info=_workspace(),
        reproduction=ReproductionContext(repro_command='pytest x', failure_evidence='boom'),
    )

    assert after_exit_code == 0
    assert [result.status for result in worker_results] == [WorkerStatus.SUCCESS]


@pytest.mark.asyncio
async def test_run_plan_steps_runs_all_steps_even_after_repro_green(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed_steps: list[str] = []

    async def fake_run_implementation_child(
        plan_step: PlanStep,
        workspace_info: Workspace,
        contract: TaskContract,
        repo_index: RepoIndex,
    ) -> tuple[WorkerResult, LLMUsage]:
        executed_steps.append(plan_step.id)
        return _worker_result(WorkerStatus.SUCCESS), _unit_usage()

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        return _ok_result()

    monkeypatch.setattr(
        'src.workflows.main_workflow._run_implementation_child', fake_run_implementation_child
    )
    monkeypatch.setattr('src.workflows.main_workflow.run_tool', fake_run_tool)

    fix_step = _plan_with_step('fix').steps[0]
    cleanup_step = _plan_with_step('cleanup').steps[0]
    plan = Plan(
        summary='Fix then clean up',
        steps=[fix_step, cleanup_step],
        integration_tests=[],
        rollback_strategy='git checkout',
        definition_of_done=['diff reviewed'],
    )

    _, _, after_exit_code, _ = await _run_plan_steps(
        plan=plan,
        contract=_contract(TaskType.BUGFIX),
        repo_index=RepoIndex(),
        workspace_info=_workspace(),
        reproduction=ReproductionContext(repro_command='pytest x', failure_evidence='boom'),
    )

    assert executed_steps == ['fix', 'cleanup']
    assert after_exit_code == 0


@pytest.mark.asyncio
async def test_run_plan_steps_replans_when_repro_still_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feedbacks: list[str | None] = []

    async def fake_run_implementation_child(
        plan_step: PlanStep,
        workspace_info: Workspace,
        contract: TaskContract,
        repo_index: RepoIndex,
    ) -> tuple[WorkerResult, LLMUsage]:
        return _worker_result(WorkerStatus.SUCCESS), _unit_usage()

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        return ToolResult(stdout='still red', stderr='', exit_code=1, truncated=False)

    async def fake_build_plan(request: PlanRequest) -> tuple[Plan, LLMUsage]:
        feedbacks.append(request.revision_feedback)
        return _plan_with_step('retry'), _unit_usage()

    monkeypatch.setattr(
        'src.workflows.main_workflow._run_implementation_child', fake_run_implementation_child
    )
    monkeypatch.setattr('src.workflows.main_workflow.run_tool', fake_run_tool)
    monkeypatch.setattr('src.workflows.main_workflow.build_plan', fake_build_plan)

    _, worker_results, after_exit_code, _ = await _run_plan_steps(
        plan=_plan_with_step('step-1'),
        contract=_contract(TaskType.BUGFIX),
        repo_index=RepoIndex(),
        workspace_info=_workspace(),
        reproduction=ReproductionContext(repro_command='pytest x', failure_evidence='boom'),
    )

    assert after_exit_code == 1
    assert any('regression test still failing' in (feedback or '') for feedback in feedbacks)
    assert worker_results[-1].status == WorkerStatus.BLOCKED


def _context_pack() -> ContextPack:
    return ContextPack(task_summary='gathered', snippets=[], budget_remaining=0)


@pytest.mark.asyncio
async def test_review_loop_revises_then_accepts_and_uses_revised_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decisions = iter([PlanReviewDecision.REVISE, PlanReviewDecision.ACCEPT])
    build_plan_requests: list[PlanRequest] = []
    gather_calls: list[str] = []

    async def fake_review_plan(
        request: PlanReviewRequest,
    ) -> tuple[PlanReviewVerdict, LLMUsage]:
        decision = next(decisions)
        feedback = '' if decision == PlanReviewDecision.ACCEPT else 'inspect the auth callers'
        return PlanReviewVerdict(decision=decision, feedback=feedback), _unit_usage()

    async def fake_gather_context(request: object) -> tuple[ContextPack, LLMUsage]:
        gather_calls.append('gathered')
        return _context_pack(), _unit_usage()

    async def fake_build_plan(request: PlanRequest) -> tuple[Plan, LLMUsage]:
        build_plan_requests.append(request)
        return _plan_with_step('revised'), _unit_usage()

    monkeypatch.setattr('src.workflows.main_workflow.review_plan', fake_review_plan)
    monkeypatch.setattr('src.workflows.main_workflow.gather_context', fake_gather_context)
    monkeypatch.setattr('src.workflows.main_workflow.build_plan', fake_build_plan)

    plan, usage = await _review_and_revise_plan(
        plan=_plan_with_step('initial'),
        contract=_contract(TaskType.FEATURE),
        repo_index=RepoIndex(),
        workspace_info=_workspace(),
        reproduction=None,
    )

    assert len(build_plan_requests) == 1
    assert build_plan_requests[0].context is not None
    assert build_plan_requests[0].revision_feedback == 'inspect the auth callers'
    assert len(gather_calls) == 1
    assert plan.steps[0].id == 'revised'
    assert usage.call_count == 4


@pytest.mark.asyncio
async def test_review_loop_stops_at_cap_and_keeps_latest_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_plan_count = 0

    async def fake_review_plan(
        request: PlanReviewRequest,
    ) -> tuple[PlanReviewVerdict, LLMUsage]:
        return PlanReviewVerdict(
            decision=PlanReviewDecision.REVISE, feedback='again'
        ), _unit_usage()

    async def fake_gather_context(request: object) -> tuple[ContextPack, LLMUsage]:
        return _context_pack(), _unit_usage()

    async def fake_build_plan(request: PlanRequest) -> tuple[Plan, LLMUsage]:
        nonlocal build_plan_count
        build_plan_count += 1
        return _plan_with_step(f'revised-{build_plan_count}'), _unit_usage()

    monkeypatch.setattr('src.workflows.main_workflow.review_plan', fake_review_plan)
    monkeypatch.setattr('src.workflows.main_workflow.gather_context', fake_gather_context)
    monkeypatch.setattr('src.workflows.main_workflow.build_plan', fake_build_plan)

    plan, _ = await _review_and_revise_plan(
        plan=_plan_with_step('initial'),
        contract=_contract(TaskType.FEATURE),
        repo_index=RepoIndex(),
        workspace_info=_workspace(),
        reproduction=None,
    )

    assert build_plan_count == CONFIG.max_plan_review_rounds
    assert plan.steps[0].id == f'revised-{CONFIG.max_plan_review_rounds}'


def _install_common_workflow_fakes(
    monkeypatch: pytest.MonkeyPatch,
    contract: TaskContract,
    workspace: Workspace,
) -> None:
    async def fake_build_contract(task_request: TaskRequest) -> tuple[TaskContract, LLMUsage]:
        return contract, _unit_usage()

    async def fake_setup_environment(origin: Origin, run_id: str) -> Workspace:
        return workspace

    async def fake_build_repo_index(workspace_arg: Workspace) -> RepoIndex:
        return RepoIndex()

    async def fake_begin_candidate(workspace_arg: Workspace, candidate_index: int) -> Workspace:
        return workspace

    async def fake_snapshot_candidate_base(workspace_arg: Workspace) -> Workspace:
        return workspace

    async def fake_reset_to_base(workspace_arg: Workspace) -> ToolResult:
        return _ok_result()

    async def fake_assess_complexity(
        task_contract: TaskContract,
    ) -> tuple[ComplexityVerdict, LLMUsage]:
        return ComplexityVerdict(requires_human_approval=False, reasoning='narrow'), _unit_usage()

    async def fake_review_plan(
        request: PlanReviewRequest,
    ) -> tuple[PlanReviewVerdict, LLMUsage]:
        return PlanReviewVerdict(decision=PlanReviewDecision.ACCEPT, feedback=''), _unit_usage()

    async def fake_teardown_environment(workspace_arg: Workspace) -> ToolResult:
        return _ok_result()

    monkeypatch.setattr('src.workflows.main_workflow.build_contract', fake_build_contract)
    monkeypatch.setattr('src.workflows.main_workflow.setup_environment', fake_setup_environment)
    monkeypatch.setattr('src.workflows.main_workflow.build_repo_index', fake_build_repo_index)
    monkeypatch.setattr('src.workflows.main_workflow.begin_candidate', fake_begin_candidate)
    monkeypatch.setattr(
        'src.workflows.main_workflow.snapshot_candidate_base', fake_snapshot_candidate_base
    )
    monkeypatch.setattr('src.workflows.main_workflow.reset_to_base', fake_reset_to_base)
    monkeypatch.setattr('src.workflows.main_workflow.assess_complexity', fake_assess_complexity)
    monkeypatch.setattr('src.workflows.main_workflow.review_plan', fake_review_plan)
    monkeypatch.setattr(
        'src.workflows.main_workflow.teardown_environment', fake_teardown_environment
    )


def _reproduction_payload(status: ReproductionStatus, before_exit_code: int) -> dict[str, object]:
    return {
        'reproduction_result': ReproductionResult(
            status=status,
            repro_command='pytest tests/test_bug.py',
            test_files=['tests/test_bug.py'],
            failure_evidence='boom',
        ).model_dump(mode='json'),
        'before_exit_code': before_exit_code,
        'llm_usage': _unit_usage().model_dump(mode='json'),
    }


@pytest.mark.asyncio
async def test_main_workflow_blocks_when_bug_cannot_be_reproduced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace()
    spawned_workflows: list[str] = []
    finalized = False
    _install_common_workflow_fakes(monkeypatch, _contract(TaskType.BUGFIX), workspace)

    async def fake_spawn_child(workflow_name: str, **kwargs: object) -> str:
        spawned_workflows.append(workflow_name)
        return 'repro-1'

    async def fake_wait_for_child(child_id: str) -> dict[str, object]:
        return _reproduction_payload(ReproductionStatus.COULD_NOT_REPRODUCE, 0)

    async def fake_finalize_winner(workspace_arg: Workspace, winner_branch: str) -> ToolResult:
        nonlocal finalized
        finalized = True
        return _ok_result()

    monkeypatch.setattr('src.workflows.main_workflow.spawn_child', fake_spawn_child)
    monkeypatch.setattr('src.workflows.main_workflow.wait_for_child', fake_wait_for_child)
    monkeypatch.setattr('src.workflows.main_workflow.finalize_winner', fake_finalize_winner)

    report = await main_workflow(
        {'raw_request': 'fix bug', 'origin': {'kind': 'host', 'repo_path': 'C:/repo'}}
    )

    assert report['status'] == WorkerStatus.BLOCKED.value
    assert 'could not reproduce' in report['blocked_reason']
    assert spawned_workflows == ['reproduction_workflow']
    assert finalized is False


@pytest.mark.asyncio
async def test_main_workflow_threads_reproduction_evidence_into_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace()
    plan_requests: list[PlanRequest] = []
    _install_common_workflow_fakes(monkeypatch, _contract(TaskType.BUGFIX), workspace)

    async def fake_spawn_child(workflow_name: str, **kwargs: object) -> str:
        if workflow_name == 'reproduction_workflow':
            return 'repro-1'
        return 'impl-1'

    async def fake_wait_for_child(child_id: str) -> dict[str, object]:
        if child_id == 'repro-1':
            return _reproduction_payload(ReproductionStatus.REPRODUCED, 1)
        return {
            'worker_result': _worker_result(WorkerStatus.SUCCESS).model_dump(mode='json'),
            'llm_usage': _unit_usage().model_dump(mode='json'),
        }

    async def fake_build_plan(request: PlanRequest) -> tuple[Plan, LLMUsage]:
        plan_requests.append(request)
        return _plan_with_step('fix-step'), _unit_usage()

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        return _ok_result()

    async def fake_get_full_diff(workspace_arg: Workspace) -> str:
        return 'diff --git a/app.py b/app.py'

    async def fake_review_patch(request: ReviewRequest) -> tuple[ReviewVerdict, LLMUsage]:
        return (
            ReviewVerdict(
                verdict=ReviewDecision.ACCEPT,
                minimality_assessment='minimal',
                recommended_next_action='accept',
            ),
            _unit_usage(),
        )

    async def fake_finalize_winner(workspace_arg: Workspace, winner_branch: str) -> ToolResult:
        return _ok_result()

    monkeypatch.setattr('src.workflows.main_workflow.spawn_child', fake_spawn_child)
    monkeypatch.setattr('src.workflows.main_workflow.wait_for_child', fake_wait_for_child)
    monkeypatch.setattr('src.workflows.main_workflow.build_plan', fake_build_plan)
    monkeypatch.setattr('src.workflows.main_workflow.run_tool', fake_run_tool)
    monkeypatch.setattr('src.workflows.main_workflow.get_full_diff', fake_get_full_diff)
    monkeypatch.setattr('src.workflows.main_workflow.review_patch', fake_review_patch)
    monkeypatch.setattr('src.workflows.main_workflow.finalize_winner', fake_finalize_winner)

    report = await main_workflow(
        {'raw_request': 'fix bug', 'origin': {'kind': 'host', 'repo_path': 'C:/repo'}}
    )

    assert report['status'] == ReviewDecision.ACCEPT.value
    evidence = report['reproduction_evidence']
    assert evidence['failed_before'] is True
    assert evidence['passed_after'] is True
    assert evidence['before_exit_code'] == 1
    assert evidence['after_exit_code'] == 0
    assert plan_requests[0].reproduction is not None


@pytest.mark.asyncio
async def test_main_workflow_does_not_teardown_while_suspended_on_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torn_down: list[Workspace] = []
    workspace = _workspace()
    _install_common_workflow_fakes(monkeypatch, _contract(TaskType.FEATURE), workspace)

    async def fake_build_plan(request: PlanRequest) -> tuple[Plan, LLMUsage]:
        return _plan_with_step('step-1'), _unit_usage()

    async def fake_spawn_child(workflow_name: str, **kwargs: object) -> str:
        return 'child-1'

    async def fake_wait_for_child(child_id: str) -> dict[str, object]:
        raise WorkflowSuspended('Workflow waiting for child.')

    async def fake_teardown_environment(workspace_arg: Workspace) -> ToolResult:
        torn_down.append(workspace_arg)
        return _ok_result()

    monkeypatch.setattr('src.workflows.main_workflow.build_plan', fake_build_plan)
    monkeypatch.setattr('src.workflows.main_workflow.spawn_child', fake_spawn_child)
    monkeypatch.setattr('src.workflows.main_workflow.wait_for_child', fake_wait_for_child)
    monkeypatch.setattr(
        'src.workflows.main_workflow.teardown_environment', fake_teardown_environment
    )

    with pytest.raises(WorkflowSuspended, match=r'Workflow waiting for child\.'):
        await main_workflow(
            {'raw_request': 'add feature', 'origin': {'kind': 'host', 'repo_path': 'C:/repo'}}
        )

    assert torn_down == []


def _review_verdict(decision: ReviewDecision) -> ReviewVerdict:
    return ReviewVerdict(
        verdict=decision,
        minimality_assessment='minimal',
        recommended_next_action='proceed',
    )


class _CandidateRunTracker:
    def __init__(self) -> None:
        self.begin_indices: list[int] = []
        self.reset_count = 0
        self.finalized_branch: str | None = None


def _install_candidate_run_fakes(
    monkeypatch: pytest.MonkeyPatch,
    workspace: HostWorkspace,
    worker_confidence: Confidence,
    verdicts: list[ReviewDecision],
) -> _CandidateRunTracker:
    tracker = _CandidateRunTracker()
    verdict_iter = iter(verdicts)

    async def fake_build_plan(request: PlanRequest) -> tuple[Plan, LLMUsage]:
        return _plan_with_step('step-1'), _unit_usage()

    async def fake_begin_candidate(workspace_arg: Workspace, candidate_index: int) -> Workspace:
        tracker.begin_indices.append(candidate_index)
        return workspace.model_copy(update={'current_branch': f'cand-{candidate_index}'})

    async def fake_reset_to_base(workspace_arg: Workspace) -> ToolResult:
        tracker.reset_count += 1
        return _ok_result()

    async def fake_spawn_child(workflow_name: str, **kwargs: object) -> str:
        return 'impl-1'

    async def fake_wait_for_child(child_id: str) -> dict[str, object]:
        worker_result = _worker_result(WorkerStatus.SUCCESS).model_copy(
            update={'confidence': worker_confidence}
        )
        return {
            'worker_result': worker_result.model_dump(mode='json'),
            'llm_usage': _unit_usage().model_dump(mode='json'),
        }

    async def fake_get_full_diff(workspace_arg: Workspace) -> str:
        return 'diff --git a/app.py b/app.py'

    async def fake_review_patch(request: ReviewRequest) -> tuple[ReviewVerdict, LLMUsage]:
        return _review_verdict(next(verdict_iter)), _unit_usage()

    async def fake_finalize_winner(workspace_arg: Workspace, winner_branch: str) -> ToolResult:
        tracker.finalized_branch = winner_branch
        return _ok_result()

    monkeypatch.setattr('src.workflows.main_workflow.build_plan', fake_build_plan)
    monkeypatch.setattr('src.workflows.main_workflow.begin_candidate', fake_begin_candidate)
    monkeypatch.setattr('src.workflows.main_workflow.reset_to_base', fake_reset_to_base)
    monkeypatch.setattr('src.workflows.main_workflow.spawn_child', fake_spawn_child)
    monkeypatch.setattr('src.workflows.main_workflow.wait_for_child', fake_wait_for_child)
    monkeypatch.setattr('src.workflows.main_workflow.get_full_diff', fake_get_full_diff)
    monkeypatch.setattr('src.workflows.main_workflow.review_patch', fake_review_patch)
    monkeypatch.setattr('src.workflows.main_workflow.finalize_winner', fake_finalize_winner)
    return tracker


@pytest.mark.asyncio
async def test_main_workflow_high_confidence_runs_single_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace()
    _install_common_workflow_fakes(monkeypatch, _contract(TaskType.FEATURE), workspace)
    tracker = _install_candidate_run_fakes(
        monkeypatch,
        workspace,
        worker_confidence=Confidence.HIGH,
        verdicts=[ReviewDecision.ACCEPT],
    )

    report = await main_workflow(
        {'raw_request': 'add feature', 'origin': {'kind': 'host', 'repo_path': 'C:/repo'}}
    )

    assert tracker.begin_indices == [0]
    assert tracker.reset_count == 0
    assert tracker.finalized_branch == 'cand-0'
    assert report['status'] == ReviewDecision.ACCEPT.value


@pytest.mark.asyncio
async def test_main_workflow_low_confidence_escalates_and_finalizes_selected_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace()
    _install_common_workflow_fakes(monkeypatch, _contract(TaskType.FEATURE), workspace)
    tracker = _install_candidate_run_fakes(
        monkeypatch,
        workspace,
        worker_confidence=Confidence.HIGH,
        verdicts=[
            ReviewDecision.REVISE,
            ReviewDecision.REVISE,
            ReviewDecision.ACCEPT,
            ReviewDecision.REVISE,
        ],
    )

    report = await main_workflow(
        {'raw_request': 'add feature', 'origin': {'kind': 'host', 'repo_path': 'C:/repo'}}
    )

    expected_count = CONFIG.candidate_count_low_confidence
    assert tracker.begin_indices == list(range(expected_count))
    assert tracker.reset_count == expected_count - 1
    assert tracker.finalized_branch == 'cand-2'
    assert report['status'] == ReviewDecision.ACCEPT.value
    assert report['chosen_candidate']['index'] == 2
