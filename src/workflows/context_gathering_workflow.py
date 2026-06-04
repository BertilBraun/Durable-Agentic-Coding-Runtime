from __future__ import annotations

from temporal_light import workflow

from src.activities.context_gatherer import fulfill_context_request
from src.activities.workspace_manager import WorkspaceAdapter
from src.llm.client import LLMUsage
from src.models.context import ContextPack
from src.models.plan import ContextNote, ContextRequest
from src.models.repo import RepoIndex


@workflow
async def context_gathering_workflow(
    workspace: dict[str, object],
    repo_index: dict[str, object],
    request: dict[str, object],
) -> dict[str, object]:
    workspace_info = WorkspaceAdapter.validate_python(workspace)
    repository_index = RepoIndex.model_validate(repo_index)
    context_request = ContextRequest.model_validate(request)

    note, context_pack, usage = await fulfill_context_request(
        workspace_info=workspace_info,
        repo_index=repository_index,
        request=context_request,
    )

    return {
        'context_note': note.model_dump(mode='json'),
        'context_pack': context_pack.model_dump(mode='json'),
        'llm_usage': usage.model_dump(mode='json'),
    }


def parse_context_gathering_result(
    result: dict[str, object],
) -> tuple[ContextNote, ContextPack, LLMUsage]:
    return (
        ContextNote.model_validate(result['context_note']),
        ContextPack.model_validate(result['context_pack']),
        LLMUsage.model_validate(result['llm_usage']),
    )
