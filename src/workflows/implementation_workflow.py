from __future__ import annotations

import os

from src.activities.context_gatherer import ContextGatherRequest, gather_context
from src.activities.implementation import (
    ImplementationTurnRequest,
    failed_worker_result,
    run_implementation_turn,
)
from src.activities.repo_indexer import build_repo_index
from src.activities.workspace_manager import WorkspaceInfo
from src.models.plan import PlanStep
from src.models.task import TaskContract
from src.models.worker import WorkerStatus
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
        if worker_result.status in {WorkerStatus.SUCCESS, WorkerStatus.NEEDS_REPLAN}:
            return worker_result.model_dump(mode="json")

    return failed_worker_result("maximum implementation iterations reached").model_dump(mode="json")
