import pytest
from src.activities.implementation import ImplementationAgentTurn
from src.llm.client import LLMClient, Message
from src.models.plan import Plan
from src.models.task import TaskContract, TaskType
from src.tools.definitions import ToolName

from tests.fakes.openai_client import FakeAsyncOpenAI


@pytest.mark.asyncio
async def test_test_local_fake_openai_generates_task_contract() -> None:
    llm_client = LLMClient(async_openai_client=FakeAsyncOpenAI())

    result = await llm_client.generate_structured(
        messages=[Message(role='user', content='Fix the test')],
        output_type=TaskContract,
        model='fake-contract-model',
        context_limit_tokens=200_000,
    )

    task_contract = TaskContract.model_validate_json(result.content)
    assert task_contract.task_type == TaskType.BUGFIX


@pytest.mark.asyncio
async def test_test_local_fake_openai_generates_single_step_plan() -> None:
    llm_client = LLMClient(async_openai_client=FakeAsyncOpenAI())

    result = await llm_client.generate_structured(
        messages=[Message(role='user', content='Plan this')],
        output_type=Plan,
        model='fake-planner-model',
        context_limit_tokens=200_000,
    )

    plan = Plan.model_validate_json(result.content)
    assert len(plan.steps) == 1


@pytest.mark.asyncio
async def test_test_local_fake_openai_implementation_first_turn_emits_patch_and_test_tools() -> (
    None
):
    llm_client = LLMClient(async_openai_client=FakeAsyncOpenAI())

    result = await llm_client.generate_structured(
        messages=[Message(role='user', content='Implement smoke patch')],
        output_type=ImplementationAgentTurn,
        model='fake-implementation-model',
        context_limit_tokens=200_000,
    )

    turn = ImplementationAgentTurn.model_validate_json(result.content)
    assert turn.done is False
    assert [tool_call.tool_name for tool_call in turn.tool_calls] == [
        ToolName.APPLY_PATCH,
        ToolName.RUN_TESTS,
    ]


@pytest.mark.asyncio
async def test_test_local_fake_openai_implementation_second_turn_reports_success() -> None:
    llm_client = LLMClient(async_openai_client=FakeAsyncOpenAI())

    result = await llm_client.generate_structured(
        messages=[
            Message(role='user', content='Implement smoke patch'),
            Message(role='assistant', content='{"done":false}'),
            Message(role='user', content='tool=run_tests exit_code=0\nstdout:\n1 passed'),
        ],
        output_type=ImplementationAgentTurn,
        model='fake-implementation-model',
        context_limit_tokens=200_000,
    )

    turn = ImplementationAgentTurn.model_validate_json(result.content)
    assert turn.done is True
    assert turn.worker_result is not None
    assert turn.worker_result.diff_summary == 'Added smoke subtract function and test.'
