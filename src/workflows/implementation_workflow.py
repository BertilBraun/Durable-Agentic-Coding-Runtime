from __future__ import annotations

from temporal_light import workflow

from src.activities.context_gatherer import ContextGatherRequest, gather_context
from src.activities.implementation import (
    ImplementationTurnRequest,
    failed_worker_result,
    get_full_diff,
    run_implementation_turn,
)
from src.activities.reviewer import ReviewDecision, ReviewRequest, ReviewVerdict, review_patch
from src.activities.workspace_manager import Workspace, WorkspaceAdapter
from src.llm.client import LLMUsage
from src.models.plan import PlanContext, PlanStep
from src.models.repo import RepoIndex
from src.models.task import TaskContract
from src.models.worker import Confidence, WorkerResult, WorkerStatus


@workflow
async def implementation_workflow(
    step: dict[str, object],
    workspace: dict[str, object],
    contract: dict[str, object],
    repo_index: dict[str, object],
    plan_context: dict[str, object] | None = None,
) -> dict[str, object]:
    plan_step = PlanStep.model_validate(step)
    step_plan_context = (
        PlanContext.model_validate(plan_context) if plan_context is not None else None
    )
    workspace_info = WorkspaceAdapter.validate_python(workspace)
    task_contract = TaskContract.model_validate(contract)
    repository_index = RepoIndex.model_validate(repo_index)
    usage = LLMUsage()

    context_pack, gather_usage = await gather_context(
        ContextGatherRequest(
            workspace_info=workspace_info,
            repo_index=repository_index,
            gatherer_prompt=plan_step.goal,
        ),
    )
    usage += gather_usage

    worker_result, turn_usage = await run_implementation_turn(
        ImplementationTurnRequest(
            plan_step=plan_step,
            plan_context=step_plan_context,
            context_pack=context_pack,
            task_contract=task_contract,
            workspace_info=workspace_info,
            repo_index=repository_index,
        ),
    )
    usage += turn_usage
    if worker_result.status == WorkerStatus.SUCCESS:
        reviewed_result, review_usage = await _review_successful_step(
            plan_step=plan_step,
            workspace_info=workspace_info,
            task_contract=task_contract,
            worker_result=worker_result,
        )
        usage += review_usage
        return _packaged_result(reviewed_result, usage)
    return _packaged_result(worker_result, usage)


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
) -> tuple[WorkerResult, LLMUsage]:
    diff = await get_full_diff(workspace_info)
    review_verdict, review_usage = await review_patch(
        ReviewRequest(
            contract=task_contract,
            plan_step=plan_step,
            diff=diff,
            test_results=worker_result.test_results,
            worker_results=[worker_result],
            workspace_info=workspace_info,
        )
    )
    return _worker_result_from_review(worker_result, review_verdict), review_usage


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
