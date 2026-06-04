import pytest

from src.activities.context_gatherer import context_note_from_pack
from src.llm.client import LLMUsage
from src.models.context import ContextPack, PackedSnippet
from src.models.plan import ContextRequest
from src.models.repo import RepoIndex
from src.workflows import context_gathering_workflow as workflow_module
from src.workflows.context_gathering_workflow import context_gathering_workflow


@pytest.mark.asyncio
async def test_context_gathering_workflow_returns_note_pack_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_requests: list[ContextRequest] = []

    async def fake_fulfill_context_request(
        workspace_info: object,
        repo_index: RepoIndex,
        request: ContextRequest,
    ) -> tuple[object, ContextPack, LLMUsage]:
        captured_requests.append(request)
        context_pack = ContextPack(
            task_summary='Parser context gathered',
            snippets=[
                PackedSnippet(
                    file_path='src/app.py',
                    start_line=10,
                    end_line=20,
                    reason='parser entrypoint',
                    content='def parse(): ...',
                )
            ],
            budget_remaining=3,
        )
        return (
            context_note_from_pack(request, context_pack),
            context_pack,
            LLMUsage(call_count=2, total_input_tokens=11),
        )

    monkeypatch.setattr(
        workflow_module,
        'fulfill_context_request',
        fake_fulfill_context_request,
    )

    result = await context_gathering_workflow(
        workspace={
            'kind': 'host',
            'run_id': 'run-1',
            'base_sha': 'base',
            'base_branch': 'main',
            'current_branch': 'main',
            'repo_path': 'workspace',
        },
        repo_index=RepoIndex().model_dump(mode='json'),
        request=ContextRequest(
            id='ctx-1',
            reason='Need parser code',
            queries=['Read parser'],
            relevant_files=['src/app.py'],
        ).model_dump(mode='json'),
    )

    assert captured_requests[0].id == 'ctx-1'
    assert result['context_note']['request_reason'] == 'Need parser code'
    assert result['context_pack']['snippets'][0]['content'] == 'def parse(): ...'
    assert result['llm_usage']['call_count'] == 2
