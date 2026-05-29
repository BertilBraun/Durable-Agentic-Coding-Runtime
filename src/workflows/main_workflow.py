from __future__ import annotations

from src.activities.complexity_assessor import assess_complexity
from src.activities.contract_builder import build_contract
from src.activities.human_approval import HumanPlanPresentationRequest, present_plan_to_human
from src.activities.implementation import get_full_diff
from src.activities.planner import PlanRequest, build_plan
from src.activities.repo_indexer import build_repo_index
from src.activities.report_builder import (
    FinalReportRequest,
    build_final_report,
    collect_llm_usage_summary,
    reset_llm_usage_summary,
)
from src.activities.reviewer import ReviewRequest, review_patch
from src.activities.workspace_manager import create_workspace, destroy_workspace, make_run_id
from src.models.approval import ApprovalDecision, HumanApprovalSignal
from src.models.task import TaskRequest
from src.models.worker import WorkerResult, WorkerStatus
from src.workflows.temporal import spawn_child, wait_for_child, wait_for_signal, workflow


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
        while True:
            await present_plan_to_human(
                HumanPlanPresentationRequest(run_id=run_id, contract=contract, plan=plan),
            )
            signal_payload = await wait_for_signal('human_approval')
            approval = HumanApprovalSignal.model_validate(signal_payload)
            if approval.decision == ApprovalDecision.APPROVE:
                break
            plan = await build_plan(
                PlanRequest(
                    contract=contract,
                    repo_index=repo_index,
                    worker_results=[],
                    human_feedback=approval.feedback,
                ),
            )

    worker_results: list[WorkerResult] = []
    pending_plan_steps = plan.steps
    while pending_plan_steps:
        plan_step = pending_plan_steps.pop(0)
        # TODO extract the child run logic into a helper?
        child_id = await spawn_child(
            'implementation_workflow',
            step=plan_step.model_dump(mode='json'),
            workspace=workspace_info.model_dump(mode='json'),
            contract=contract.model_dump(mode='json'),
            repo_index=repo_index.model_dump(mode='json'),
        )
        child_result = await wait_for_child(child_id)
        worker_result = WorkerResult.model_validate(child_result)
        # TODO until here
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
                pending_plan_steps = plan.steps
            case WorkerStatus.FAILED | WorkerStatus.BLOCKED:
                pending_plan_steps = []  # TODO break here instead - break seems cleaner than mutating the list in place
            case WorkerStatus.SUCCESS:
                pass

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
    final_report = await build_final_report(
        FinalReportRequest(
            patch=diff,
            contract=contract,
            plan=plan,
            worker_results=worker_results,
            final_verdict=final_verdict,
            workspace_info=workspace_info,
            llm_usage=await collect_llm_usage_summary(),
        ),
    )
    await destroy_workspace(workspace_info)
    return final_report.model_dump(mode='json')
