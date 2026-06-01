from __future__ import annotations

from temporal_light import workflow

from src.activities.reproduction import ReproductionTurnRequest, reproduce_bug
from src.activities.workspace_manager import WorkspaceAdapter
from src.models.repo import RepoIndex
from src.models.task import TaskContract


@workflow
async def reproduction_workflow(
    workspace: dict[str, object],
    contract: dict[str, object],
    repo_index: dict[str, object],
) -> dict[str, object]:
    workspace_info = WorkspaceAdapter.validate_python(workspace)
    task_contract = TaskContract.model_validate(contract)
    repository_index = RepoIndex.model_validate(repo_index)

    reproduction_result, usage = await reproduce_bug(
        ReproductionTurnRequest(
            task_contract=task_contract,
            workspace_info=workspace_info,
            repo_index=repository_index,
        ),
    )
    return {
        'reproduction_result': reproduction_result.model_dump(mode='json'),
        'llm_usage': usage.model_dump(mode='json'),
    }
