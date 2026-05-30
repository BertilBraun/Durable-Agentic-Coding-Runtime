from __future__ import annotations

from temporal_light import spawn_child, wait_for_child, workflow

from src.activities.complexity_assessor import assess_complexity
from src.activities.contract_builder import build_contract
from src.activities.human_approval import approve_plan_or_replan
from src.activities.implementation import get_full_diff
from src.activities.planner import PlanRequest, build_plan
from src.activities.repo_indexer import build_repo_index
from src.activities.report_builder import (
    FinalReport,
    collect_llm_usage_summary,
    reset_llm_usage_summary,
)
from src.activities.reviewer import ReviewRequest, review_patch
from src.activities.workspace_manager import (
    WorkspaceInfo,
    create_workspace,
    destroy_workspace,
    make_run_id,
)
from src.models.plan import Plan, PlanStep
from src.models.repo import RepoIndex
from src.models.task import TaskContract, TaskRequest
from src.models.worker import WorkerResult, WorkerStatus


@workflow
async def main_workflow(request: dict[str, object]) -> dict[str, object]:
    task_request = TaskRequest.model_validate(request)
    run_id = task_request.run_id or make_run_id()
    await reset_llm_usage_summary()

    contract = await build_contract(task_request)
    workspace_info = await create_workspace(
        run_id, task_request.repo_path, task_request.docker_image
    )
    repo_index = await build_repo_index(workspace_info)

    plan = await build_plan(
        PlanRequest(
            contract=contract,
            repo_index=repo_index,
            worker_results=[],
            human_feedback=None,
        ),
    )
    complexity_verdict = await assess_complexity(contract)

    if complexity_verdict.requires_human_approval:
        plan = await approve_plan_or_replan(
            run_id=run_id,
            contract=contract,
            repo_index=repo_index,
            plan=plan,
        )

    plan, worker_results = await _run_plan_steps(
        plan=plan,
        contract=contract,
        repo_index=repo_index,
        workspace_info=workspace_info,
    )

    diff = await get_full_diff(workspace_info)
    final_verdict = await review_patch(
        ReviewRequest(
            contract=contract,
            plan_step=None,
            diff=diff,
            test_results=[],
            worker_results=worker_results,
            workspace_info=workspace_info,
        ),
    )
    final_report = FinalReport(
        status=final_verdict.verdict.value,
        patch=diff,
        contract=contract,
        plan=plan,
        worker_results=worker_results,
        final_verdict=final_verdict,
        workspace_info=workspace_info,
        llm_usage=await collect_llm_usage_summary(),
    )
    await destroy_workspace(workspace_info)
    return final_report.model_dump(mode='json')


async def _run_plan_steps(
    plan: Plan,
    contract: TaskContract,
    repo_index: RepoIndex,
    workspace_info: WorkspaceInfo,
) -> tuple[Plan, list[WorkerResult]]:
    worker_results: list[WorkerResult] = []
    pending_plan_steps = list(plan.steps)
    while pending_plan_steps:
        plan_step = pending_plan_steps.pop(0)
        worker_result = await _run_implementation_child(
            plan_step=plan_step,
            workspace_info=workspace_info,
            contract=contract,
            repo_index=repo_index,
        )
        worker_results.append(worker_result)
        match worker_result.status:
            case WorkerStatus.NEEDS_REPLAN:
                plan = await build_plan(
                    PlanRequest(
                        contract=contract,
                        repo_index=repo_index,
                        worker_results=worker_results,
                        human_feedback=worker_result.replan_suggestion,
                    ),
                )
                pending_plan_steps = list(plan.steps)
            case WorkerStatus.FAILED | WorkerStatus.BLOCKED:
                break
            case WorkerStatus.SUCCESS:
                pass
    return plan, worker_results


async def _run_implementation_child(
    plan_step: PlanStep,
    workspace_info: WorkspaceInfo,
    contract: TaskContract,
    repo_index: RepoIndex,
) -> WorkerResult:
    child_id = await spawn_child(
        'implementation_workflow',
        step=plan_step.model_dump(mode='json'),
        workspace=workspace_info.model_dump(mode='json'),
        contract=contract.model_dump(mode='json'),
        repo_index=repo_index.model_dump(mode='json'),
    )
    child_result = await wait_for_child(child_id)
    return WorkerResult.model_validate(child_result)
