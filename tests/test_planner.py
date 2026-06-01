import pytest
from pydantic import BaseModel
from src.activities import planner as planner_module
from src.activities.planner import PlanRequest, build_plan
from src.config import ModelRole
from src.llm.client import LLMUsage, Message, StructuredCompletion
from src.models.context import ContextPack, PackedSnippet
from src.models.plan import Plan
from src.models.repo import RepoIndex
from src.models.task import TaskContract, TaskType


def _plan() -> Plan:
    return Plan(
        summary='Plan',
        steps=[],
        integration_tests=[],
        rollback_strategy='git checkout',
        definition_of_done=['done'],
    )


@pytest.mark.asyncio
async def test_supplied_context_snippets_appear_in_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_messages: list[list[Message]] = []

    async def fake_generate_structured(
        role: ModelRole,
        messages: list[Message],
        output_type: type[BaseModel],
    ) -> StructuredCompletion:
        captured_messages.append(messages)
        return StructuredCompletion(
            output=_plan(),
            content=_plan().model_dump_json(),
            model='fake-model',
            context_limit_tokens=100,
            usage=LLMUsage(call_count=1),
        )

    monkeypatch.setattr(planner_module, 'generate_structured', fake_generate_structured)

    context = ContextPack(
        task_summary='Auth token handling',
        snippets=[
            PackedSnippet(
                file_path='src/auth.py',
                start_line=10,
                end_line=20,
                reason='token handler',
                content='def validate_token(): ...',
            )
        ],
        budget_remaining=0,
    )

    await build_plan(
        PlanRequest(
            contract=TaskContract(task_type=TaskType.FEATURE, goal='Add auth'),
            repo_index=RepoIndex(),
            worker_results=[],
            context=context,
        ),
    )

    user_message = captured_messages[0][-1].content
    assert 'src/auth.py:10-20' in user_message
    assert 'token handler' in user_message
    assert 'def validate_token(): ...' in user_message
