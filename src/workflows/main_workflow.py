from __future__ import annotations

from dataclasses import dataclass

from temporal_light import run_child, workflow

from src.activities.contract_builder import build_contract
from src.activities.implementation import get_full_diff
from src.activities.repo_indexer import build_repo_index
from src.activities.report_builder import FinalReport
from src.activities.reproduction import repro_run_tests
from src.activities.reviewer import ReviewDecision, ReviewRequest, ReviewVerdict, review_patch
from src.activities.selector import (
    CandidateResult,
    aggregate_candidate_confidence,
    candidate_count_for_confidence,
)
from src.activities.verifier import run_anchor_tests
from src.activities.workspace_manager import (
    ToolExecutionRequest,
    ToolResult,
    Workspace,
    begin_candidate,
    finalize_winner,
    make_run_id,
    reset_to_base,
    run_tool,
    setup_environment,
    snapshot_candidate_base,
    snapshot_candidate_result,
    teardown_environment,
)
from src.config import CONFIG
from src.llm.client import LLMUsage
from src.models.context import ContextPack
from src.models.frozen_base_model import FrozenBaseModel
from src.models.plan import (
    ContextNote,
    Plan,
    PlanContext,
    PlannerState,
    PlannerTurn,
    PlanningEvidence,
    PlanStep,
    ReproductionPlanTurn,
    StepHistoryEntry,
)
from src.models.repo import RepoIndex
from src.models.reproduction import (
    ReproductionBrief,
    ReproductionContext,
    ReproductionEvidence,
    ReproductionResult,
    ReproductionStatus,
    build_reproduction_evidence,
)
from src.models.task import TaskContract, TaskRequest, TaskType
from src.models.worker import Confidence, TestResult, WorkerResult, WorkerStatus

_CONFIDENCE_RANK: dict[Confidence, int] = {
    Confidence.LOW: 0,
    Confidence.MEDIUM: 1,
    Confidence.HIGH: 2,
}

# Reproduction is a red->green anchor test, which only fits tasks that add or correct
# behavior. Refactors preserve behavior (nothing to turn green) and docs/frontend have no
# pytest-observable contract.
_REPRODUCIBLE_TASK_TYPES: frozenset[TaskType] = frozenset({TaskType.BUGFIX, TaskType.FEATURE})


@dataclass(frozen=True)
class StepCandidateRun:
    index: int
    workspace_info: Workspace
    worker_results: list[WorkerResult]
    llm_usage: LLMUsage


class PlannerLoopState(FrozenBaseModel):
    workspace_info: Workspace
    planner_state: PlannerState
    context_packs: list[ContextPack]
    executed_steps: list[PlanStep]
    worker_results: list[WorkerResult]
    usage: LLMUsage
    next_candidate_index: int = 0

    def with_usage(self, usage: LLMUsage) -> PlannerLoopState:
        return self.model_copy(update={'usage': self.usage + usage})

    def with_context(
        self,
        notes: list[ContextNote],
        packs: list[ContextPack],
    ) -> PlannerLoopState:
        return self.model_copy(
            update={
                'planner_state': self.planner_state.model_copy(
                    update={'context_notes': [*self.planner_state.context_notes, *notes]}
                ),
                'context_packs': [*self.context_packs, *packs],
            }
        )

    def with_worker_result(self, worker_result: WorkerResult) -> PlannerLoopState:
        return self.model_copy(update={'worker_results': [*self.worker_results, worker_result]})

    def with_remaining_work(self, remaining_work: list[str]) -> PlannerLoopState:
        return self.model_copy(
            update={
                'planner_state': self.planner_state.model_copy(
                    update={'remaining_work': remaining_work}
                )
            }
        )

    def with_evidence(self, evidence: PlanningEvidence) -> PlannerLoopState:
        return self.model_copy(
            update={'planner_state': self.planner_state.model_copy(update={'evidence': evidence})}
        )


@dataclass(frozen=True)
class PlanExecutionResult:
    plan: Plan
    worker_results: list[WorkerResult]
    llm_usage: LLMUsage
    workspace_info: Workspace


@dataclass(frozen=True)
class BootstrapResult:
    contract: TaskContract
    workspace: Workspace
    repo_index: RepoIndex
    reproduction: ReproductionContext | None
    blocked_reason: str | None
    seed_context_notes: list[ContextNote]
    seed_context_packs: list[ContextPack]
    seed_remaining_work: list[str]
    usage: LLMUsage


@workflow
async def main_workflow(request: dict[str, object]) -> dict[str, object]:
    task_request = TaskRequest.model_validate(request)
    run_id = task_request.run_id or make_run_id()
    usage = LLMUsage()
    bootstrap = await _bootstrap_workflow_state(task_request, run_id)
    usage += bootstrap.usage
    if bootstrap.blocked_reason is not None:
        blocked_report = FinalReport(
            status=WorkerStatus.BLOCKED.value,
            workflow_status='blocked',
            agent_verdict=WorkerStatus.BLOCKED.value,
            reproduction_passed=False,
            official_prediction_emitted=False,
            patch='',
            contract=bootstrap.contract,
            workspace_info=bootstrap.workspace,
            llm_usage=usage,
            blocked_reason=bootstrap.blocked_reason,
        )
        await teardown_environment(bootstrap.workspace)
        return blocked_report.model_dump(mode='json')

    execution_result = await _run_planner_loop(
        workspace_info=bootstrap.workspace,
        contract=bootstrap.contract,
        repo_index=bootstrap.repo_index,
        reproduction=bootstrap.reproduction,
        seed_context_notes=bootstrap.seed_context_notes,
        seed_context_packs=bootstrap.seed_context_packs,
        seed_remaining_work=bootstrap.seed_remaining_work,
    )
    usage += execution_result.llm_usage
    (
        final_reproduction_evidence,
        final_test_results,
        final_blocker,
    ) = await _finalize_or_block(
        workspace=execution_result.workspace_info,
        repo_index=bootstrap.repo_index,
        reproduction=bootstrap.reproduction,
        integration_tests=execution_result.plan.integration_tests,
    )
    if final_blocker is not None:
        execution_result.worker_results.append(final_blocker)
    report, report_usage = await _build_final_report(
        workspace=bootstrap.workspace,
        contract=bootstrap.contract,
        execution_result=execution_result,
        final_reproduction_evidence=final_reproduction_evidence,
        final_test_results=final_test_results,
        final_blocker=final_blocker,
        usage=usage,
    )
    usage += report_usage
    report = report.model_copy(update={'llm_usage': usage})
    await finalize_winner(bootstrap.workspace, execution_result.workspace_info.current_branch)
    await teardown_environment(bootstrap.workspace)
    return report.model_dump(mode='json')


async def _bootstrap_workflow_state(
    task_request: TaskRequest,
    run_id: str,
) -> BootstrapResult:
    usage = LLMUsage()
    contract, contract_usage = await build_contract(task_request)
    usage += contract_usage
    workspace = await setup_environment(task_request.origin, run_id)
    repo_index = await build_repo_index(workspace)
    if contract.task_type not in _REPRODUCIBLE_TASK_TYPES:
        return BootstrapResult(contract, workspace, repo_index, None, None, [], [], [], usage)

    repro_plan, notes, packs, plan_usage = await _run_reproduction_planning_child(
        workspace_info=workspace,
        repo_index=repo_index,
        planner_state=PlannerState(contract=contract, repo_index=repo_index),
    )
    usage += plan_usage
    reproduction_result, reproduction_usage = await _run_reproduction_child(
        workspace_info=workspace,
        contract=contract,
        repo_index=repo_index,
        brief=repro_plan.reproduction_brief,
    )
    usage += reproduction_usage
    if reproduction_result.status == ReproductionStatus.COULD_NOT_REPRODUCE:
        return BootstrapResult(
            contract,
            workspace,
            repo_index,
            None,
            f'could not reproduce the bug: {reproduction_result.failure_evidence}',
            [],
            [],
            [],
            usage,
        )
    reproduction_context = ReproductionContext(
        repro_target=reproduction_result.repro_target,
        failure_evidence=reproduction_result.failure_evidence,
        regression_test_files=reproduction_result.regression_test_files,
    )
    workspace = await snapshot_candidate_base(workspace)
    return BootstrapResult(
        contract,
        workspace,
        repo_index,
        reproduction_context,
        None,
        notes,
        packs,
        repro_plan.remaining_work,
        usage,
    )


async def _run_planner_loop(
    workspace_info: Workspace,
    contract: TaskContract,
    repo_index: RepoIndex,
    reproduction: ReproductionContext | None,
    seed_context_notes: list[ContextNote] | None = None,
    seed_context_packs: list[ContextPack] | None = None,
    seed_remaining_work: list[str] | None = None,
) -> PlanExecutionResult:
    state = PlannerLoopState(
        workspace_info=workspace_info,
        planner_state=PlannerState(
            contract=contract,
            repo_index=repo_index,
            reproduction=reproduction,
            context_notes=seed_context_notes or [],
            remaining_work=seed_remaining_work or [],
        ),
        context_packs=seed_context_packs or [],
        executed_steps=[],
        worker_results=[],
        usage=LLMUsage(),
    )
    hit_cap = True
    planner_turns_remaining = CONFIG.max_planner_turns
    while planner_turns_remaining > 0:
        turn, notes, context_packs, planner_turn_count, turn_usage = await _run_replanner_child(
            workspace_info=state.workspace_info,
            repo_index=repo_index,
            max_planner_turns=planner_turns_remaining,
            planner_state=state.planner_state,
        )
        assert not turn.context_requests, 'replanning_workflow should not return context requests'
        planner_turns_remaining -= planner_turn_count
        state = state.with_usage(turn_usage).with_context(notes, context_packs)
        if turn.next_step is None:
            if turn.done and not _has_outstanding_work(state):
                state = state.with_remaining_work([])
                hit_cap = False
                break
            continue
        state = state.with_remaining_work(turn.remaining_work)
        state = await _run_next_step(state, turn, reproduction)
        state = await _refresh_planning_evidence(state, repo_index, reproduction)
    if hit_cap:
        blocked = WorkerResult(
            status=WorkerStatus.BLOCKED,
            patch_id=None,
            diff_summary='Planner turn cap reached before done.',
            discovered_issues=[f'planner turn cap of {CONFIG.max_planner_turns} reached'],
            confidence=Confidence.LOW,
            replan_suggestion=None,
        )
        state = state.with_worker_result(blocked)
    return _plan_execution_result_from_state(state)


def _has_outstanding_work(state: PlannerLoopState) -> bool:
    if state.planner_state.remaining_work:
        return True
    completed = state.planner_state.completed_steps
    return bool(completed) and completed[-1].outcome == WorkerStatus.NEEDS_REPLAN


def _fold_review_issues(remaining_work: list[str], selected_result: WorkerResult) -> list[str]:
    if selected_result.status != WorkerStatus.NEEDS_REPLAN:
        return remaining_work
    issues = selected_result.discovered_issues or (
        [selected_result.replan_suggestion] if selected_result.replan_suggestion else []
    )
    return [*remaining_work, *issues]


async def _run_next_step(
    state: PlannerLoopState,
    turn: PlannerTurn,
    reproduction: ReproductionContext | None,
) -> PlannerLoopState:
    plan_step = turn.next_step
    assert plan_step is not None, '_run_next_step requires a next_step'
    completed_step_ids = [entry.step_id for entry in state.planner_state.completed_steps]
    step_run, next_candidate_index = await _run_step_with_candidates(
        plan_step=plan_step,
        plan_context=PlanContext(
            summary='Planner-driven execution',
            current_step_id=plan_step.id,
            all_step_ids=[*completed_step_ids, plan_step.id],
            completed_step_summaries=[
                f'{entry.step_id}: {entry.summary}'
                for entry in state.planner_state.completed_steps
                if entry.outcome == WorkerStatus.SUCCESS
            ],
        ),
        context_pack=_context_pack_for_step(plan_step, state.context_packs),
        contract=state.planner_state.contract,
        repo_index=state.planner_state.repo_index,
        workspace_info=state.workspace_info,
        first_candidate_index=state.next_candidate_index,
        reproduction=reproduction,
    )
    selected_result = step_run.worker_results[-1]
    workspace = (
        step_run.workspace_info
        if selected_result.status == WorkerStatus.NEEDS_REPLAN
        else state.workspace_info
    )
    if selected_result.status == WorkerStatus.SUCCESS:
        workspace = await snapshot_candidate_base(step_run.workspace_info)
    history = _record_step_history(plan_step, step_run)
    remaining_work = _fold_review_issues(turn.remaining_work, selected_result)
    return state.model_copy(
        update={
            'workspace_info': workspace,
            'planner_state': state.planner_state.model_copy(
                update={
                    'completed_steps': [*state.planner_state.completed_steps, history],
                    'remaining_work': remaining_work,
                }
            ),
            'executed_steps': [*state.executed_steps, plan_step],
            'worker_results': [*state.worker_results, *step_run.worker_results],
            'usage': state.usage + step_run.llm_usage,
            'next_candidate_index': next_candidate_index,
        }
    )


def _record_step_history(
    plan_step: PlanStep,
    step_run: StepCandidateRun,
) -> StepHistoryEntry:
    worker_result = step_run.worker_results[-1]
    return StepHistoryEntry(
        step_id=plan_step.id,
        outcome=worker_result.status,
        confidence=worker_result.confidence,
        summary=worker_result.diff_summary,
        tests=_select_relevant_test_history(worker_result.test_results),
        observations=worker_result.observations,
    )


def _context_pack_for_step(plan_step: PlanStep, context_packs: list[ContextPack]) -> ContextPack:
    matching_snippets = [
        snippet
        for context_pack in context_packs
        for snippet in context_pack.snippets
        if not plan_step.target_files or snippet.file_path in plan_step.target_files
    ]
    if not matching_snippets:
        matching_snippets = [
            snippet for context_pack in context_packs for snippet in context_pack.snippets
        ]
    summaries = [context_pack.task_summary for context_pack in context_packs]
    task_summary = plan_step.context_summary or '\n'.join(summaries) or plan_step.goal
    return ContextPack(
        task_summary=task_summary,
        snippets=matching_snippets,
        artifact_references=[
            artifact
            for context_pack in context_packs
            for artifact in context_pack.artifact_references
        ],
        budget_remaining=min(
            (context_pack.budget_remaining for context_pack in context_packs),
            default=0,
        ),
    )


async def _refresh_planning_evidence(
    state: PlannerLoopState,
    repo_index: RepoIndex,
    reproduction: ReproductionContext | None,
) -> PlannerLoopState:
    diff_summary = await get_full_diff(state.workspace_info)
    latest_tests = (
        state.planner_state.completed_steps[-1].tests if state.planner_state.completed_steps else []
    )
    evidence = PlanningEvidence(
        diff_summary=diff_summary,
        selected_test_results=latest_tests,
    )
    if reproduction is not None:
        anchor_results = await run_anchor_tests(
            state.workspace_info,
            repo_index,
            reproduction,
            restore_regression_files=False,
        )
        repro_result = anchor_results[0]
        evidence = evidence.model_copy(
            update={
                'reproduction_target': reproduction.repro_target,
                'reproduction_passed': repro_result.passed,
                'reproduction_stdout_summary': repro_result.stdout_summary,
                'reproduction_stderr_summary': repro_result.stderr_summary,
                'selected_test_results': [*latest_tests, *anchor_results],
            }
        )
    return state.with_evidence(evidence)


def _select_relevant_test_history(results: list[TestResult]) -> list[TestResult]:
    last_pass_index = None
    for index, result in enumerate(results):
        if result.passed:
            last_pass_index = index
    if last_pass_index is None:
        return [result for result in results if not result.passed]
    return [
        result
        for index, result in enumerate(results)
        if index == last_pass_index or index > last_pass_index
    ]


async def _finalize_or_block(
    workspace: Workspace,
    repo_index: RepoIndex,
    reproduction: ReproductionContext | None,
    integration_tests: list[str],
) -> tuple[ReproductionEvidence | None, list[TestResult], WorkerResult | None]:
    reproduction_evidence: ReproductionEvidence | None = None
    anchor_results: list[TestResult] = []
    if reproduction is not None:
        anchor_results = await run_anchor_tests(
            workspace,
            repo_index,
            reproduction,
            restore_regression_files=True,
        )
        reproduction_passed = anchor_results[0].passed
        reproduction_evidence = build_reproduction_evidence(
            repro_target=reproduction.repro_target,
            passed_after=reproduction_passed,
        )
        failed_anchor = [result for result in anchor_results if not result.passed]
        if failed_anchor:
            return (
                reproduction_evidence,
                anchor_results,
                WorkerResult(
                    status=WorkerStatus.BLOCKED,
                    patch_id=None,
                    diff_summary='Final reproduction or regression tests failed.',
                    test_results=anchor_results,
                    tests_run=[result.command for result in anchor_results],
                    discovered_issues=[_test_result_summary(result) for result in failed_anchor],
                    confidence=Confidence.LOW,
                    replan_suggestion=None,
                ),
            )
    integration_results: list[TestResult] = list(anchor_results)
    for index, command in enumerate(integration_tests, start=len(anchor_results) + 1):
        suite_result = await _run_repro_target(workspace, repo_index, command)
        integration_results.append(_test_result_from_tool_result(command, suite_result, index))
        if suite_result.exit_code != 0:
            return (
                reproduction_evidence,
                integration_results,
                WorkerResult(
                    status=WorkerStatus.BLOCKED,
                    patch_id=None,
                    diff_summary='Final integration test failed.',
                    test_results=integration_results,
                    tests_run=[result.command for result in integration_results],
                    discovered_issues=[_tool_result_summary(suite_result)],
                    confidence=Confidence.LOW,
                    replan_suggestion=None,
                ),
            )
    return reproduction_evidence, integration_results, None


async def _build_final_report(
    workspace: Workspace,
    contract: TaskContract,
    execution_result: PlanExecutionResult,
    final_reproduction_evidence: ReproductionEvidence | None,
    final_test_results: list[TestResult],
    final_blocker: WorkerResult | None,
    usage: LLMUsage,
) -> tuple[FinalReport, LLMUsage]:
    diff = await get_full_diff(execution_result.workspace_info)
    aggregated_test_results = [
        test_result
        for worker_result in execution_result.worker_results
        for test_result in worker_result.test_results
    ]
    aggregated_test_results.extend(final_test_results)
    if final_blocker is None:
        final_verdict, review_usage = await review_patch(
            ReviewRequest(
                contract=contract,
                plan_step=None,
                diff=diff,
                test_results=aggregated_test_results,
                worker_results=execution_result.worker_results,
                workspace_info=execution_result.workspace_info,
            )
        )
    else:
        review_usage = LLMUsage()
        final_verdict = ReviewVerdict(
            verdict=ReviewDecision.REVISE,
            evidence=[final_blocker.diff_summary],
            blocking_issues=final_blocker.discovered_issues,
            non_blocking_issues=[],
            missing_tests=[],
            regression_risks=[],
            minimality_assessment='final invariant failed',
            recommended_next_action='revise',
        )
    reproduction_passed = (
        final_reproduction_evidence.passed_after
        if final_reproduction_evidence is not None
        else None
    )
    chosen = CandidateResult(
        index=0,
        branch=execution_result.workspace_info.current_branch,
        diff=diff,
        plan=execution_result.plan,
        worker_results=execution_result.worker_results,
        review_verdict=final_verdict,
        confidence=aggregate_candidate_confidence(execution_result.worker_results),
        test_results=aggregated_test_results,
        reproduction_evidence=final_reproduction_evidence,
    )
    report = FinalReport(
        status=final_verdict.verdict.value,
        workflow_status='completed',
        agent_verdict=final_verdict.verdict.value,
        reproduction_passed=reproduction_passed,
        official_prediction_emitted=bool(diff),
        patch=diff,
        contract=contract,
        plan=execution_result.plan,
        worker_results=execution_result.worker_results,
        final_verdict=final_verdict,
        workspace_info=workspace,
        llm_usage=usage,
        reproduction_evidence=final_reproduction_evidence,
        chosen_candidate=chosen,
    )
    return report, review_usage


async def _run_step_with_candidates(
    plan_step: PlanStep,
    plan_context: PlanContext,
    context_pack: ContextPack,
    contract: TaskContract,
    repo_index: RepoIndex,
    workspace_info: Workspace,
    first_candidate_index: int,
    reproduction: ReproductionContext | None,
) -> tuple[StepCandidateRun, int]:
    candidates: list[StepCandidateRun] = []
    next_candidate_index = first_candidate_index
    target_count = 1
    while len(candidates) < target_count:
        if candidates:
            await reset_to_base(workspace_info)
        step_candidate = await _run_step_candidate(
            plan_step=plan_step,
            plan_context=plan_context,
            context_pack=context_pack,
            contract=contract,
            repo_index=repo_index,
            workspace_info=workspace_info,
            candidate_index=next_candidate_index,
            reproduction=reproduction,
        )
        next_candidate_index += 1
        candidates.append(step_candidate)
        if len(candidates) == 1:
            target_count = _step_candidate_target_count(step_candidate)
    return _select_step_candidate(candidates), next_candidate_index


async def _run_step_candidate(
    plan_step: PlanStep,
    plan_context: PlanContext,
    context_pack: ContextPack,
    contract: TaskContract,
    repo_index: RepoIndex,
    workspace_info: Workspace,
    candidate_index: int,
    reproduction: ReproductionContext | None,
) -> StepCandidateRun:
    candidate_workspace = await begin_candidate(workspace_info, candidate_index)
    worker_result, child_usage = await _run_implementation_child(
        plan_step=plan_step,
        plan_context=plan_context,
        context_pack=context_pack,
        workspace_info=candidate_workspace,
        contract=contract,
        repo_index=repo_index,
        reproduction=reproduction,
    )
    candidate_workspace = await snapshot_candidate_result(candidate_workspace)
    return StepCandidateRun(
        index=candidate_index,
        workspace_info=candidate_workspace,
        worker_results=[worker_result],
        llm_usage=child_usage,
    )


def _derive_step_candidate_confidence(candidate: StepCandidateRun) -> Confidence:
    last_result = candidate.worker_results[-1]
    if last_result.status != WorkerStatus.SUCCESS:
        return Confidence.LOW
    return aggregate_candidate_confidence(candidate.worker_results)


def _step_candidate_target_count(candidate: StepCandidateRun) -> int:
    if candidate.worker_results[-1].status != WorkerStatus.SUCCESS:
        return 1
    return candidate_count_for_confidence(_derive_step_candidate_confidence(candidate))


def _select_step_candidate(candidates: list[StepCandidateRun]) -> StepCandidateRun:
    return max(candidates, key=_step_candidate_preference_key)


def _step_candidate_preference_key(candidate: StepCandidateRun) -> tuple[int, int, int, int]:
    last_result = candidate.worker_results[-1]
    return (
        1 if last_result.status == WorkerStatus.SUCCESS else 0,
        _CONFIDENCE_RANK[_derive_step_candidate_confidence(candidate)],
        sum(
            1
            for worker_result in candidate.worker_results
            for test_result in worker_result.test_results
            if test_result.passed
        ),
        -candidate.index,
    )


async def _run_repro_target(
    workspace_info: Workspace,
    repo_index: RepoIndex,
    repro_target: str,
) -> ToolResult:
    return await run_tool(
        ToolExecutionRequest(
            workspace=workspace_info,
            tool=repro_run_tests(repro_target),
            repo_index=repo_index,
        )
    )


async def _run_reproduction_planning_child(
    workspace_info: Workspace,
    repo_index: RepoIndex,
    planner_state: PlannerState,
) -> tuple[ReproductionPlanTurn, list[ContextNote], list[ContextPack], LLMUsage]:
    child_result = await run_child(
        'replanning_workflow',
        workspace=workspace_info.model_dump(mode='json'),
        repo_index=repo_index.model_dump(mode='json'),
        max_planner_turns=CONFIG.max_planner_turns,
        planner_state=planner_state.model_dump(mode='json'),
        mode='reproduction',
    )
    return (
        ReproductionPlanTurn.model_validate(child_result['planner_turn']),
        [ContextNote.model_validate(note) for note in child_result['context_notes']],
        [ContextPack.model_validate(pack) for pack in child_result['context_packs']],
        LLMUsage.model_validate(child_result['llm_usage']),
    )


async def _run_reproduction_child(
    workspace_info: Workspace,
    contract: TaskContract,
    repo_index: RepoIndex,
    brief: ReproductionBrief | None,
) -> tuple[ReproductionResult, LLMUsage]:
    child_result = await run_child(
        'reproduction_workflow',
        workspace=workspace_info.model_dump(mode='json'),
        contract=contract.model_dump(mode='json'),
        repo_index=repo_index.model_dump(mode='json'),
        brief=brief.model_dump(mode='json') if brief is not None else None,
    )
    return (
        ReproductionResult.model_validate(child_result['reproduction_result']),
        LLMUsage.model_validate(child_result['llm_usage']),
    )


async def _run_replanner_child(
    workspace_info: Workspace,
    repo_index: RepoIndex,
    max_planner_turns: int,
    planner_state: PlannerState,
) -> tuple[PlannerTurn, list[ContextNote], list[ContextPack], int, LLMUsage]:
    child_result = await run_child(
        'replanning_workflow',
        workspace=workspace_info.model_dump(mode='json'),
        repo_index=repo_index.model_dump(mode='json'),
        max_planner_turns=max_planner_turns,
        planner_state=planner_state.model_dump(mode='json'),
    )
    return (
        PlannerTurn.model_validate(child_result['planner_turn']),
        [ContextNote.model_validate(note) for note in child_result['context_notes']],
        [ContextPack.model_validate(pack) for pack in child_result['context_packs']],
        int(child_result['planner_turn_count']),
        LLMUsage.model_validate(child_result['llm_usage']),
    )


async def _run_implementation_child(
    plan_step: PlanStep,
    plan_context: PlanContext,
    context_pack: ContextPack,
    workspace_info: Workspace,
    contract: TaskContract,
    repo_index: RepoIndex,
    reproduction: ReproductionContext | None,
) -> tuple[WorkerResult, LLMUsage]:
    child_result = await run_child(
        'implementation_workflow',
        step=plan_step.model_dump(mode='json'),
        plan_context=plan_context.model_dump(mode='json'),
        context_pack=context_pack.model_dump(mode='json'),
        workspace=workspace_info.model_dump(mode='json'),
        contract=contract.model_dump(mode='json'),
        repo_index=repo_index.model_dump(mode='json'),
        reproduction=reproduction.model_dump(mode='json') if reproduction is not None else None,
    )
    return (
        WorkerResult.model_validate(child_result['worker_result']),
        LLMUsage.model_validate(child_result['llm_usage']),
    )


def _plan_execution_result_from_state(state: PlannerLoopState) -> PlanExecutionResult:
    plan = Plan(
        summary='Planner-driven execution',
        steps=state.executed_steps,
        integration_tests=_integration_tests_from_steps(state.executed_steps),
        definition_of_done=['planner done', 'final review complete'],
    )
    return PlanExecutionResult(
        plan=plan,
        worker_results=state.worker_results,
        llm_usage=state.usage,
        workspace_info=state.workspace_info,
    )


def _integration_tests_from_steps(steps: list[PlanStep]) -> list[str]:
    commands: list[str] = []
    for step in steps:
        commands.extend(step.tests_to_run)
    return list(dict.fromkeys(commands))


def _tool_result_summary(result: ToolResult) -> str:
    return f'exit_code={result.exit_code}\nstdout={result.stdout}\nstderr={result.stderr}'.strip()


def _test_result_summary(result: TestResult) -> str:
    return (
        f'command={result.command}\nexit_code={result.exit_code}\n'
        f'stdout={result.stdout_summary}\nstderr={result.stderr_summary}'
    ).strip()


def _test_result_from_tool_result(
    command: str,
    tool_result: ToolResult,
    sequence: int,
) -> TestResult:
    return TestResult(
        sequence=sequence,
        command=command,
        exit_code=tool_result.exit_code,
        stdout_summary=tool_result.stdout,
        stderr_summary=tool_result.stderr,
        passed=tool_result.exit_code == 0,
    )
