from __future__ import annotations

from src.activities.temporal import durable_activity
from src.llm.client import LLMClient, Message
from src.llm.config import ModelRole
from src.llm.prompts import system_prompt_for_role
from src.models.approval import ComplexityVerdict
from src.models.task import TaskContract


@durable_activity(retries=2, timeout=120, backoff_seconds=10)
async def assess_complexity(contract: TaskContract) -> ComplexityVerdict:
    llm_client = LLMClient()
    return await llm_client.generate_structured(
        role=ModelRole.COMPLEXITY_ASSESSOR,
        messages=[
            Message(
                role="system",
                content=system_prompt_for_role(ModelRole.COMPLEXITY_ASSESSOR),
            ),
            Message(role="user", content=contract.model_dump_json()),
        ],
        output_type=ComplexityVerdict,
    )
