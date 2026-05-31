from __future__ import annotations

from temporal_light import spawn_child, wait_for_child, workflow

from src.activities.complexity_assessor import assess_complexity
from src.activities.contract_builder import build_contract
from src.activities.human_approval import approve_plan_or_replan
from src.activities.implementation import get_full_diff
from src.activities.planner import PlanRequest, build_plan
from src.activities.repo_indexer import build_repo_index
from src.activities.report_builder import FinalReport
from src.activities.reviewer import ReviewRequest, review_patch
from src.activities.workspace_manager import (
    Workspace,
    begin_candidate,
    finalize_winner,
    make_run_id,
    setup_environment,
    teardown_environment,
)
from src.llm.client import LLMUsage
from src.models.plan import Plan, PlanStep
from src.models.repo import RepoIndex
from src.models.task import TaskContract, TaskRequest
from src.models.worker import WorkerResult, WorkerStatus


@workflow
async def main_workflow(request: dict[str, object]) -> dict[str, object]:
    task_request = TaskRequest.model_validate(request)
    run_id = task_request.run_id or make_run_id()
    usage = LLMUsage()

    contract, contract_usage = await build_contract(task_request)
    usage += contract_usage
    workspace = await setup_environment(task_request.origin, run_id)
    repo_index = await build_repo_index(workspace)

    plan, plan_usage = await build_plan(
        PlanRequest(
            contract=contract,
            repo_index=repo_index,
            worker_results=[],
            human_feedback=None,
        ),
    )
    usage += plan_usage
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

    candidate_workspace = await begin_candidate(workspace, 0)
    plan, worker_results, run_usage = await _run_plan_steps(
        plan=plan,
        contract=contract,
        repo_index=repo_index,
        workspace_info=candidate_workspace,
    )
    usage += run_usage

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
    final_report = FinalReport(
        status=final_verdict.verdict.value,
        patch=diff,
        contract=contract,
        plan=plan,
        worker_results=worker_results,
        final_verdict=final_verdict,
        workspace_info=candidate_workspace,
        llm_usage=usage,
    )
    await finalize_winner(candidate_workspace, candidate_workspace.current_branch)
    await teardown_environment(candidate_workspace)
    return final_report.model_dump(mode='json')


async def _run_plan_steps(
    plan: Plan,
    contract: TaskContract,
    repo_index: RepoIndex,
    workspace_info: Workspace,
) -> tuple[Plan, list[WorkerResult], LLMUsage]:
    worker_results: list[WorkerResult] = []
    pending_plan_steps = list(plan.steps)
    usage = LLMUsage()
    while pending_plan_steps:
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
                plan, replan_usage = await build_plan(
                    PlanRequest(
                        contract=contract,
                        repo_index=repo_index,
                        worker_results=worker_results,
                        human_feedback=worker_result.replan_suggestion,
                    ),
                )
                usage += replan_usage
                pending_plan_steps = list(plan.steps)
            case WorkerStatus.FAILED | WorkerStatus.BLOCKED:
                break
            case WorkerStatus.SUCCESS:
                pass
    return plan, worker_results, usage


async def _run_implementation_child(
    plan_step: PlanStep,
    workspace_info: Workspace,
    contract: TaskContract,
    repo_index: RepoIndex,
) -> tuple[WorkerResult, LLMUsage]:
    child_id = await spawn_child(
        'implementation_workflow',
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
