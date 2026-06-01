from __future__ import annotations

from temporal_light import spawn_child, wait_for_child, workflow

from src.activities.complexity_assessor import assess_complexity
from src.activities.context_gatherer import ContextGatherRequest, gather_context
from src.activities.contract_builder import build_contract
from src.activities.human_approval import approve_plan_or_replan
from src.activities.implementation import get_full_diff
from src.activities.plan_reviewer import PlanReviewDecision, PlanReviewRequest, review_plan
from src.activities.planner import PlanRequest, build_plan
from src.activities.repo_indexer import build_repo_index
from src.activities.report_builder import FinalReport
from src.activities.reproduction import repro_run_tests
from src.activities.reviewer import ReviewRequest, review_patch
from src.activities.selector import (
    CandidateResult,
    SelectionRequest,
    aggregate_candidate_confidence,
    candidate_count_for_confidence,
    derive_candidate_confidence,
    select_candidate,
)
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
    teardown_environment,
)
from src.config import CONFIG
from src.llm.client import LLMUsage
from src.models.plan import Plan, PlanStep
from src.models.repo import RepoIndex
from src.models.reproduction import (
    ReproductionContext,
    ReproductionEvidence,
    ReproductionResult,
    ReproductionStatus,
    build_reproduction_evidence,
)
from src.models.task import TaskContract, TaskRequest, TaskType
from src.models.worker import Confidence, WorkerResult, WorkerStatus


@workflow
async def main_workflow(request: dict[str, object]) -> dict[str, object]:
    task_request = TaskRequest.model_validate(request)
    run_id = task_request.run_id or make_run_id()
    usage = LLMUsage()

    contract, contract_usage = await build_contract(task_request)
    usage += contract_usage
    workspace = await setup_environment(task_request.origin, run_id)
    repo_index = await build_repo_index(workspace)

    reproduction_context: ReproductionContext | None = None
    if contract.task_type == TaskType.BUGFIX:
        reproduction_result, reproduction_usage = await _run_reproduction_child(
            workspace_info=workspace,
            contract=contract,
            repo_index=repo_index,
        )
        usage += reproduction_usage
        if reproduction_result.status == ReproductionStatus.COULD_NOT_REPRODUCE:
            blocked_report = FinalReport(
                status=WorkerStatus.BLOCKED.value,
                patch='',
                contract=contract,
                workspace_info=workspace,
                llm_usage=usage,
                blocked_reason=(
                    f'could not reproduce the bug: {reproduction_result.failure_evidence}'
                ),
            )
            await teardown_environment(workspace)
            return blocked_report.model_dump(mode='json')
        reproduction_context = ReproductionContext(
            repro_command=reproduction_result.repro_command,
            failure_evidence=reproduction_result.failure_evidence,
        )
        workspace = await snapshot_candidate_base(workspace)

    plan, plan_usage = await build_plan(
        PlanRequest(
            contract=contract,
            repo_index=repo_index,
            worker_results=[],
            revision_feedback=None,
            reproduction=reproduction_context,
        ),
    )
    usage += plan_usage
    plan, review_loop_usage = await _review_and_revise_plan(
        plan=plan,
        contract=contract,
        repo_index=repo_index,
        workspace_info=workspace,
        reproduction=reproduction_context,
    )
    usage += review_loop_usage
    complexity_verdict, complexity_usage = await assess_complexity(contract)
    usage += complexity_usage

    if complexity_verdict.requires_human_approval:
        plan, approval_usage = await approve_plan_or_replan(
            run_id=run_id,
            contract=contract,
            repo_index=repo_index,
            plan=plan,
        )
        usage += approval_usage

    first_candidate, first_usage = await _run_candidate(
        candidate_index=0,
        workspace=workspace,
        plan=plan,
        contract=contract,
        repo_index=repo_index,
        reproduction=reproduction_context,
    )
    usage += first_usage
    candidates = [first_candidate]
    target_count = candidate_count_for_confidence(derive_candidate_confidence(first_candidate))
    while len(candidates) < target_count:
        await reset_to_base(workspace)
        next_candidate, next_usage = await _run_candidate(
            candidate_index=len(candidates),
            workspace=workspace,
            plan=plan,
            contract=contract,
            repo_index=repo_index,
            reproduction=reproduction_context,
        )
        usage += next_usage
        candidates.append(next_candidate)

    chosen = await select_candidate(SelectionRequest(candidates=candidates))
    final_report = FinalReport(
        status=chosen.review_verdict.verdict.value,
        patch=chosen.diff,
        contract=contract,
        plan=chosen.plan,
        worker_results=chosen.worker_results,
        final_verdict=chosen.review_verdict,
        workspace_info=workspace,
        llm_usage=usage,
        reproduction_evidence=chosen.reproduction_evidence,
        chosen_candidate=chosen,
    )
    await finalize_winner(workspace, chosen.branch)
    await teardown_environment(workspace)
    return final_report.model_dump(mode='json')


async def _run_candidate(
    candidate_index: int,
    workspace: Workspace,
    plan: Plan,
    contract: TaskContract,
    repo_index: RepoIndex,
    reproduction: ReproductionContext | None,
) -> tuple[CandidateResult, LLMUsage]:
    usage = LLMUsage()
    candidate_workspace = await begin_candidate(workspace, candidate_index)
    plan, worker_results, reproduction_passed_after, run_usage = await _run_plan_steps(
        plan=plan,
        contract=contract,
        repo_index=repo_index,
        workspace_info=candidate_workspace,
        reproduction=reproduction,
    )
    usage += run_usage

    reproduction_evidence: ReproductionEvidence | None = None
    if reproduction is not None:
        reproduction_evidence = build_reproduction_evidence(
            repro_command=reproduction.repro_command,
            passed_after=bool(reproduction_passed_after),
        )

    diff = await get_full_diff(candidate_workspace)
    aggregated_test_results = [
        test_result
        for worker_result in worker_results
        for test_result in worker_result.test_results
    ]
    final_verdict, review_usage = await review_patch(
        ReviewRequest(
            contract=contract,
            plan_step=None,
            diff=diff,
            test_results=aggregated_test_results,
            worker_results=worker_results,
            workspace_info=candidate_workspace,
        ),
    )
    usage += review_usage
    candidate = CandidateResult(
        index=candidate_index,
        branch=candidate_workspace.current_branch,
        diff=diff,
        plan=plan,
        worker_results=worker_results,
        review_verdict=final_verdict,
        confidence=aggregate_candidate_confidence(worker_results),
        test_results=aggregated_test_results,
        reproduction_evidence=reproduction_evidence,
    )
    return candidate, usage


async def _review_and_revise_plan(
    plan: Plan,
    contract: TaskContract,
    repo_index: RepoIndex,
    workspace_info: Workspace,
    reproduction: ReproductionContext | None,
) -> tuple[Plan, LLMUsage]:
    usage = LLMUsage()
    for _ in range(CONFIG.max_plan_review_rounds):
        verdict, review_usage = await review_plan(
            PlanReviewRequest(
                contract=contract,
                plan=plan,
                repo_index=repo_index,
                reproduction=reproduction,
            ),
        )
        usage += review_usage
        if verdict.decision == PlanReviewDecision.ACCEPT:
            break
        context_pack, gather_usage = await gather_context(
            ContextGatherRequest(
                workspace_info=workspace_info,
                repo_index=repo_index,
                gatherer_prompt=verdict.feedback,
            ),
        )
        usage += gather_usage
        plan, replan_usage = await build_plan(
            PlanRequest(
                contract=contract,
                repo_index=repo_index,
                worker_results=[],
                revision_feedback=verdict.feedback,
                reproduction=reproduction,
                context=context_pack,
            ),
        )
        usage += replan_usage
    return plan, usage


async def _run_plan_steps(
    plan: Plan,
    contract: TaskContract,
    repo_index: RepoIndex,
    workspace_info: Workspace,
    reproduction: ReproductionContext | None,
) -> tuple[Plan, list[WorkerResult], bool | None, LLMUsage]:
    worker_results: list[WorkerResult] = []
    pending_plan_steps = list(plan.steps)
    usage = LLMUsage()
    replan_attempts = 0
    reproduction_passed_after: bool | None = None
    while True:
        if pending_plan_steps:
            plan_step = pending_plan_steps.pop(0)
            worker_result, child_usage = await _run_implementation_child(
                plan_step=plan_step,
                workspace_info=workspace_info,
                contract=contract,
                repo_index=repo_index,
            )
            usage += child_usage
            worker_results.append(worker_result)
            match worker_result.status:
                case WorkerStatus.NEEDS_REPLAN:
                    if replan_attempts >= CONFIG.max_replan_attempts:
                        worker_results.append(_replan_cap_blocked_result())
                        break
                    replan_attempts += 1
                    pending_plan_steps = [
                        _corrective_plan_step(
                            plan_step=plan_step,
                            worker_result=worker_result,
                        )
                    ]
                case WorkerStatus.FAILED | WorkerStatus.BLOCKED:
                    break
                case WorkerStatus.SUCCESS:
                    if reproduction is not None:
                        gate_result = await _run_repro_command(
                            workspace_info, repo_index, reproduction.repro_command
                        )
                        reproduction_passed_after = gate_result.exit_code == 0
            continue
        if reproduction is None:
            break
        repro_result = await _run_repro_command(
            workspace_info, repo_index, reproduction.repro_command
        )
        reproduction_passed_after = repro_result.exit_code == 0
        suite_result = await _run_suite(workspace_info, repo_index, plan.integration_tests)
        if repro_result.exit_code == 0 and suite_result.exit_code == 0:
            break
        if replan_attempts >= CONFIG.max_replan_attempts:
            worker_results.append(_gate_blocked_result(repro_result, suite_result))
            break
        replan_attempts += 1
        plan, replan_usage = await _replan(
            contract=contract,
            repo_index=repo_index,
            worker_results=worker_results,
            revision_feedback=_gate_failure_feedback(repro_result, suite_result),
            reproduction=reproduction,
        )
        usage += replan_usage
        pending_plan_steps = list(plan.steps)
    return plan, worker_results, reproduction_passed_after, usage


async def _replan(
    contract: TaskContract,
    repo_index: RepoIndex,
    worker_results: list[WorkerResult],
    revision_feedback: str | None,
    reproduction: ReproductionContext | None,
) -> tuple[Plan, LLMUsage]:
    return await build_plan(
        PlanRequest(
            contract=contract,
            repo_index=repo_index,
            worker_results=worker_results,
            revision_feedback=revision_feedback,
            reproduction=reproduction,
        ),
    )


async def _run_repro_command(
    workspace_info: Workspace,
    repo_index: RepoIndex,
    repro_command: str,
) -> ToolResult:
    return await run_tool(
        ToolExecutionRequest(
            workspace=workspace_info,
            tool=repro_run_tests(repro_command),
            repo_index=repo_index,
        )
    )


async def _run_suite(
    workspace_info: Workspace,
    repo_index: RepoIndex,
    integration_tests: list[str],
) -> ToolResult:
    last_result = ToolResult(stdout='', stderr='', exit_code=0, truncated=False)
    for command in integration_tests:
        last_result = await _run_repro_command(workspace_info, repo_index, command)
        if last_result.exit_code != 0:
            return last_result
    return last_result


def _gate_failure_feedback(repro_result: ToolResult, suite_result: ToolResult) -> str:
    if repro_result.exit_code != 0:
        return (
            'bug not reproduced: regression test still failing\n'
            f'{repro_result.stdout}\n{repro_result.stderr}'
        ).strip()
    return (
        'regression in pre-existing suite: a previously passing test now fails\n'
        f'{suite_result.stdout}\n{suite_result.stderr}'
    ).strip()


def _corrective_plan_step(plan_step: PlanStep, worker_result: WorkerResult) -> PlanStep:
    feedback = worker_result.replan_suggestion or '; '.join(worker_result.discovered_issues)
    tests_run = ', '.join(worker_result.tests_run) if worker_result.tests_run else 'none'
    issues = '; '.join(worker_result.discovered_issues) or 'none reported'
    prior_summary = worker_result.diff_summary or 'no prior summary'
    return plan_step.model_copy(
        update={
            'id': f'revise-{plan_step.id}',
            'goal': (
                f'Complete this corrective implementation step. Original step goal: '
                f'{plan_step.goal} Review feedback to address: {feedback} '
                f'Prior worker summary: {prior_summary}. Prior tests run: {tests_run}. '
                f'Prior discovered issues: {issues}. Inspect the current workspace state, '
                f'apply only the missing corrections, and rerun the relevant tests.'
            ),
            'expected_result': (
                f'{plan_step.expected_result} The review feedback is resolved and the '
                f'current workspace contains the complete corrected change.'
            ),
        }
    )


def _replan_cap_blocked_result() -> WorkerResult:
    return WorkerResult(
        status=WorkerStatus.BLOCKED,
        patch_id=None,
        diff_summary='Replan attempts exhausted before reaching an acceptable candidate.',
        discovered_issues=[f'replan cap of {CONFIG.max_replan_attempts} attempts reached'],
        confidence=Confidence.LOW,
        replan_suggestion=None,
    )


def _gate_blocked_result(repro_result: ToolResult, suite_result: ToolResult) -> WorkerResult:
    return WorkerResult(
        status=WorkerStatus.BLOCKED,
        patch_id=None,
        diff_summary='Reproduction gate never went green within the replan budget.',
        discovered_issues=[_gate_failure_feedback(repro_result, suite_result)],
        confidence=Confidence.LOW,
        replan_suggestion=None,
    )


async def _run_reproduction_child(
    workspace_info: Workspace,
    contract: TaskContract,
    repo_index: RepoIndex,
) -> tuple[ReproductionResult, LLMUsage]:
    child_id = await spawn_child(
        'reproduction_workflow',
        child_id=_child_workflow_id(workspace_info, 'reproduction'),
        workspace=workspace_info.model_dump(mode='json'),
        contract=contract.model_dump(mode='json'),
        repo_index=repo_index.model_dump(mode='json'),
    )
    child_result = await wait_for_child(child_id)
    assert isinstance(child_result, dict), 'reproduction_workflow returns a dict payload'
    return (
        ReproductionResult.model_validate(child_result['reproduction_result']),
        LLMUsage.model_validate(child_result['llm_usage']),
    )


async def _run_implementation_child(
    plan_step: PlanStep,
    workspace_info: Workspace,
    contract: TaskContract,
    repo_index: RepoIndex,
) -> tuple[WorkerResult, LLMUsage]:
    child_id = await spawn_child(
        'implementation_workflow',
        child_id=_child_workflow_id(
            workspace_info,
            'implementation',
            workspace_info.current_branch,
            plan_step.id,
        ),
        step=plan_step.model_dump(mode='json'),
        workspace=workspace_info.model_dump(mode='json'),
        contract=contract.model_dump(mode='json'),
        repo_index=repo_index.model_dump(mode='json'),
    )
    child_result = await wait_for_child(child_id)
    assert isinstance(child_result, dict), 'implementation_workflow returns a dict payload'
    return (
        WorkerResult.model_validate(child_result['worker_result']),
        LLMUsage.model_validate(child_result['llm_usage']),
    )


def _child_workflow_id(workspace_info: Workspace, *parts: str) -> str:
    safe_parts = [_workflow_id_part(workspace_info.run_id)]
    safe_parts.extend(_workflow_id_part(part) for part in parts)
    return ':'.join(safe_parts)


def _workflow_id_part(value: str) -> str:
    return ''.join(
        character if character.isalnum() or character in '._-' else '-'
        for character in value
    )
