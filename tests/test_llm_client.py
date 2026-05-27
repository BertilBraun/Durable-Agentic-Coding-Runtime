import json
from typing import ClassVar

import pytest
from src.llm.client import LLMClient, LLMUsageLedger, Message
from src.llm.config import ModelConfiguration, ModelRole
from src.models.approval import ComplexityVerdict


class FakeUsage:
    prompt_tokens = 10
    completion_tokens = 4


class FakeChoiceMessage:
    content = json.dumps({"requires_human_approval": False, "reasoning": "Narrow bugfix."})


class FakeChoice:
    message = FakeChoiceMessage()


class FakeCompletion:
    usage = FakeUsage()
    choices: ClassVar[list[FakeChoice]] = [FakeChoice()]


class FakeCompletions:
    async def create(self, **keyword_arguments: object) -> FakeCompletion:
        return FakeCompletion()


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeCompletions()


class FakeAsyncOpenAI:
    def __init__(self) -> None:
        self.chat = FakeChat()


@pytest.mark.asyncio
async def test_structured_generation_parses_model_and_records_usage() -> None:
    llm_client = LLMClient(
        model_configuration=ModelConfiguration(
            contract_builder_model="contract",
            planner_model="planner",
            complexity_assessor_model="complexity",
            context_gatherer_model="context",
            implementation_model="implementation",
            reviewer_model="review",
            summarizer_model="summary",
        ),
        usage_ledger=LLMUsageLedger(),
        async_openai_client=FakeAsyncOpenAI(),
    )

    verdict = await llm_client.generate_structured(
        role=ModelRole.COMPLEXITY_ASSESSOR,
        messages=[Message(role="user", content="Assess this task")],
        output_type=ComplexityVerdict,
    )

    assert verdict.requires_human_approval is False
    assert llm_client.usage_ledger.total_input_tokens == 10
    assert llm_client.usage_ledger.total_output_tokens == 4
