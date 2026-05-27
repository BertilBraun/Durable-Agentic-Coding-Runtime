from __future__ import annotations

import os

from src.activities.context_gatherer import ContextGatherRequest, gather_context
from src.activities.implementation import (
    ImplementationTurnRequest,
    failed_worker_result,
    get_full_diff,
    run_implementation_turn,
)
from src.activities.repo_indexer import build_repo_index
from src.activities.reviewer import ReviewRequest, review_patch
from src.activities.workspace_manager import WorkspaceInfo
from src.models.plan import PlanStep
from src.models.review import ReviewDecision, ReviewVerdict
from src.models.task import TaskContract
from src.models.worker import WorkerResult, WorkerStatus
from src.workflows.temporal import workflow


@workflow
async def implementation_workflow(
    step: dict[str, object],
    workspace: dict[str, object],
    contract: dict[str, object],
) -> dict[str, object]:
    plan_step = PlanStep.model_validate(step)
    workspace_info = WorkspaceInfo.model_validate(workspace)
    task_contract = TaskContract.model_validate(contract)
    repo_index = await build_repo_index(workspace_info)
    context_pack = await gather_context(
        ContextGatherRequest(
            workspace_info=workspace_info,
            repo_index=repo_index,
            gatherer_prompt=plan_step.goal,
        ),
    )

    max_iterations = int(os.getenv("IMPL_MAX_ITERATIONS", "5"))
    for _ in range(max_iterations):
        worker_result = await run_implementation_turn(
            ImplementationTurnRequest(
                plan_step=plan_step,
                context_pack=context_pack,
                task_contract=task_contract,
                workspace_info=workspace_info,
            ),
        )
        if worker_result.status == WorkerStatus.SUCCESS:
            reviewed_result = await _review_successful_step(
                plan_step=plan_step,
                workspace_info=workspace_info,
                task_contract=task_contract,
                worker_result=worker_result,
            )
            return reviewed_result.model_dump(mode="json")
        if worker_result.status == WorkerStatus.NEEDS_REPLAN:
            return worker_result.model_dump(mode="json")

    return failed_worker_result("maximum implementation iterations reached").model_dump(mode="json")


async def _review_successful_step(
    plan_step: PlanStep,
    workspace_info: WorkspaceInfo,
    task_contract: TaskContract,
    worker_result: WorkerResult,
) -> WorkerResult:
    diff = await get_full_diff(workspace_info)
    review_verdict = await review_patch(
        ReviewRequest(
            contract=task_contract,
            plan_step=plan_step,
            diff=diff,
            test_results=worker_result.test_results,
            worker_results=[worker_result],
            workspace_info=workspace_info,
        )
    )
    return _worker_result_from_review(worker_result, review_verdict)


def _worker_result_from_review(
    worker_result: WorkerResult,
    review_verdict: ReviewVerdict,
) -> WorkerResult:
    match review_verdict.verdict:
        case ReviewDecision.ACCEPT:
            return worker_result
        case ReviewDecision.REVISE | ReviewDecision.REJECT | ReviewDecision.NEEDS_HUMAN:
            issues = review_verdict.blocking_issues or [review_verdict.recommended_next_action]
            return failed_worker_result("; ".join(issues))
