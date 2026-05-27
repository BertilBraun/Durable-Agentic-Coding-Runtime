from __future__ import annotations

from src.activities.complexity_assessor import assess_complexity
from src.activities.contract_builder import build_contract
from src.activities.human_approval import HumanPlanPresentationRequest, present_plan_to_human
from src.activities.implementation import get_full_diff
from src.activities.planner import PlanRequest, build_plan
from src.activities.repo_indexer import build_repo_index
from src.activities.report_builder import FinalReportRequest, build_final_report
from src.activities.reviewer import ReviewRequest, review_patch
from src.activities.workspace_manager import create_workspace, destroy_workspace, make_run_id
from src.models.approval import ApprovalDecision, HumanApprovalSignal
from src.models.task import TaskRequest
from src.models.worker import WorkerResult
from src.workflows.temporal import spawn_child, wait_for_child, wait_for_signal, workflow


@workflow
async def main_workflow(request: dict[str, object]) -> dict[str, object]:
    task_request = TaskRequest.model_validate(request)
    run_id = task_request.run_id or make_run_id()

    contract = await build_contract(task_request)
    workspace_info = await create_workspace(run_id, task_request.repo_path)
    repo_index = await build_repo_index(workspace_info)

    plan = await build_plan(
        PlanRequest(contract=contract, repo_index=repo_index, human_feedback=None),
    )
    complexity_verdict = await assess_complexity(contract)

    if complexity_verdict.requires_human_approval:
        while True:
            await present_plan_to_human(
                HumanPlanPresentationRequest(run_id=run_id, contract=contract, plan=plan),
            )
            signal_payload = await wait_for_signal("human_approval")
            approval = HumanApprovalSignal.model_validate(signal_payload)
            if approval.decision == ApprovalDecision.APPROVE:
                break
            plan = await build_plan(
                PlanRequest(
                    contract=contract, repo_index=repo_index, human_feedback=approval.feedback
                ),
            )

    worker_results: list[WorkerResult] = []
    for plan_step in plan.steps:
        child_id = await spawn_child(
            "implementation_workflow",
            step=plan_step.model_dump(mode="json"),
            workspace=workspace_info.model_dump(mode="json"),
            contract=contract.model_dump(mode="json"),
        )
        child_result = await wait_for_child(child_id)
        worker_results.append(WorkerResult.model_validate(child_result))

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
            contract=contract,
            plan=plan,
            worker_results=worker_results,
            final_verdict=final_verdict,
            workspace_info=workspace_info,
        ),
    )
    await destroy_workspace(workspace_info)
    return final_report.model_dump(mode="json")
