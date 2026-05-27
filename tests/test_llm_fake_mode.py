import pytest
from src.llm.client import LLMClient, Message
from src.llm.config import ModelRole
from src.models.plan import Plan
from src.models.task import TaskContract, TaskType


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
