from __future__ import annotations

from temporal_light import workflow

from src.activities.reproduction import ReproductionTurnRequest, reproduce_bug
from src.activities.workspace_manager import WorkspaceAdapter
from src.models.repo import RepoIndex
from src.models.reproduction import ReproductionBrief
from src.models.task import TaskContract


@workflow
async def reproduction_workflow(
    workspace: dict[str, object],
    contract: dict[str, object],
    repo_index: dict[str, object],
    brief: dict[str, object] | None = None,
) -> dict[str, object]:
    workspace_info = WorkspaceAdapter.validate_python(workspace)
    task_contract = TaskContract.model_validate(contract)
    repository_index = RepoIndex.model_validate(repo_index)
    reproduction_brief = ReproductionBrief.model_validate(brief) if brief is not None else None

    reproduction_result, usage = await reproduce_bug(
        ReproductionTurnRequest(
            task_contract=task_contract,
            workspace_info=workspace_info,
            repo_index=repository_index,
            brief=reproduction_brief,
        ),
    )
    return {
        'reproduction_result': reproduction_result.model_dump(mode='json'),
        'llm_usage': usage.model_dump(mode='json'),
    }
