from __future__ import annotations

from temporal_light import workflow

from src.activities.implementation import (
    ImplementationTurnRequest,
    failed_worker_result,
    get_full_diff,
    run_implementation_turn,
)
from src.activities.reviewer import ReviewDecision, ReviewRequest, ReviewVerdict, review_patch
from src.activities.verifier import run_anchor_tests
from src.activities.workspace_manager import Workspace, WorkspaceAdapter
from src.llm.client import LLMUsage
from src.models.context import ContextPack
from src.models.plan import PlanContext, PlanStep
from src.models.repo import RepoIndex
from src.models.reproduction import ReproductionContext
from src.models.task import TaskContract
from src.models.worker import Confidence, TestResult, WorkerResult, WorkerStatus


@workflow
async def implementation_workflow(
    step: dict[str, object],
    workspace: dict[str, object],
    contract: dict[str, object],
    repo_index: dict[str, object],
    plan_context: dict[str, object] | None = None,
    context_pack: dict[str, object] | None = None,
    reproduction: dict[str, object] | None = None,
) -> dict[str, object]:
    plan_step = PlanStep.model_validate(step)
    step_plan_context = (
        PlanContext.model_validate(plan_context) if plan_context is not None else None
    )
    workspace_info = WorkspaceAdapter.validate_python(workspace)
    task_contract = TaskContract.model_validate(contract)
    repository_index = RepoIndex.model_validate(repo_index)
    reproduction_context = (
        ReproductionContext.model_validate(reproduction) if reproduction is not None else None
    )
    step_context_pack = (
        ContextPack.model_validate(context_pack)
        if context_pack is not None
        else _context_pack_from_plan_step(plan_step)
    )
    usage = LLMUsage()

    worker_result, turn_usage = await run_implementation_turn(
        ImplementationTurnRequest(
            plan_step=plan_step,
            plan_context=step_plan_context,
            context_pack=step_context_pack,
            task_contract=task_contract,
            workspace_info=workspace_info,
            repo_index=repository_index,
            reproduction_test_file=_reproduction_test_file(reproduction_context),
        ),
    )
    usage += turn_usage
    if worker_result.status == WorkerStatus.SUCCESS:
        reviewed_result, review_usage = await _review_successful_step(
            plan_step=plan_step,
            workspace_info=workspace_info,
            task_contract=task_contract,
            worker_result=worker_result,
            repo_index=repository_index,
            reproduction=reproduction_context,
        )
        usage += review_usage
        return _packaged_result(reviewed_result, usage)
    return _packaged_result(worker_result, usage)


def _reproduction_test_file(reproduction: ReproductionContext | None) -> str | None:
    if reproduction is None:
        return None
    return reproduction.repro_target.split('::', 1)[0]


def _context_pack_from_plan_step(plan_step: PlanStep) -> ContextPack:
    return ContextPack(
        task_summary=plan_step.context_summary or plan_step.goal,
        snippets=[],
        artifact_references=[],
        budget_remaining=0,
    )


def _packaged_result(worker_result: WorkerResult, usage: LLMUsage) -> dict[str, object]:
    return {
        'worker_result': worker_result.model_dump(mode='json'),
        'llm_usage': usage.model_dump(mode='json'),
    }


async def _review_successful_step(
    plan_step: PlanStep,
    workspace_info: Workspace,
    task_contract: TaskContract,
    worker_result: WorkerResult,
    repo_index: RepoIndex,
    reproduction: ReproductionContext | None,
) -> tuple[WorkerResult, LLMUsage]:
    diff = await get_full_diff(workspace_info)
    review_test_results = await _review_test_results(
        workspace_info=workspace_info,
        repo_index=repo_index,
        reproduction=reproduction,
        worker_result=worker_result,
    )
    review_verdict, review_usage = await review_patch(
        ReviewRequest(
            contract=task_contract,
            plan_step=plan_step,
            diff=diff,
            test_results=review_test_results,
            worker_results=[worker_result],
            workspace_info=workspace_info,
        )
    )
    return _worker_result_from_review(worker_result, review_verdict), review_usage


async def _review_test_results(
    workspace_info: Workspace,
    repo_index: RepoIndex,
    reproduction: ReproductionContext | None,
    worker_result: WorkerResult,
) -> list[TestResult]:
    if reproduction is None:
        return worker_result.test_results
    return await run_anchor_tests(
        workspace_info,
        repo_index,
        reproduction,
        restore_regression_files=False,
    )


def _worker_result_from_review(
    worker_result: WorkerResult,
    review_verdict: ReviewVerdict,
) -> WorkerResult:
    match review_verdict.verdict:
        case ReviewDecision.ACCEPT:
            return worker_result
        case ReviewDecision.REVISE:
            issues = review_verdict.blocking_issues or [review_verdict.recommended_next_action]
            return WorkerResult(
                status=WorkerStatus.NEEDS_REPLAN,
                patch_id=worker_result.patch_id,
                diff_summary=worker_result.diff_summary,
                tests_run=worker_result.tests_run,
                test_results=worker_result.test_results,
                discovered_issues=issues,
                confidence=Confidence.LOW,
                replan_suggestion='; '.join(issues),
            )
        case ReviewDecision.REJECT | ReviewDecision.NEEDS_HUMAN:
            issues = review_verdict.blocking_issues or [review_verdict.recommended_next_action]
            return failed_worker_result('; '.join(issues))
