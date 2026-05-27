import pytest
from src.activities.implementation import ImplementationAgentTurn
from src.llm.client import LLMClient, Message
from src.llm.config import ModelRole
from src.models.plan import Plan
from src.models.task import TaskContract, TaskType
from src.tools.definitions import ToolName


@pytest.mark.asyncio
async def test_fake_mode_generates_task_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_FAKE_MODE", "1")
    llm_client = LLMClient()

    task_contract = await llm_client.generate_structured(
        role=ModelRole.CONTRACT_BUILDER,
        messages=[Message(role="user", content="Fix the test")],
        output_type=TaskContract,
    )

    assert task_contract.task_type == TaskType.BUGFIX


@pytest.mark.asyncio
async def test_fake_mode_generates_single_step_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_FAKE_MODE", "1")
    llm_client = LLMClient()

    plan = await llm_client.generate_structured(
        role=ModelRole.PLANNER,
        messages=[Message(role="user", content="Plan this")],
        output_type=Plan,
    )

    assert len(plan.steps) == 1


@pytest.mark.asyncio
async def test_fake_mode_implementation_first_turn_emits_patch_and_test_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_FAKE_MODE", "1")
    llm_client = LLMClient()

    turn = await llm_client.generate_structured(
        role=ModelRole.IMPLEMENTATION,
        messages=[Message(role="user", content="Implement smoke patch")],
        output_type=ImplementationAgentTurn,
    )

    assert turn.done is False
    assert [tool_call.tool_name for tool_call in turn.tool_calls] == [
        ToolName.APPLY_PATCH,
        ToolName.RUN_TESTS,
        ToolName.GIT_DIFF,
    ]


@pytest.mark.asyncio
async def test_fake_mode_implementation_second_turn_reports_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_FAKE_MODE", "1")
    llm_client = LLMClient()

    turn = await llm_client.generate_structured(
        role=ModelRole.IMPLEMENTATION,
        messages=[
            Message(role="user", content="Implement smoke patch"),
            Message(role="assistant", content='{"done":false}'),
            Message(role="user", content="tool=run_tests exit_code=0\nstdout:\n1 passed"),
        ],
        output_type=ImplementationAgentTurn,
    )

    assert turn.done is True
    assert turn.worker_result is not None
    assert turn.worker_result.diff_summary == "Added smoke subtract function and test."
