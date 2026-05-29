from __future__ import annotations

from src.activities.temporal import durable_activity
from src.llm.client import LLMClient, Message
from src.llm.config import ModelRole
from src.llm.prompts import system_prompt_for_role
from src.models.task import TaskContract, TaskRequest


@durable_activity(retries=2, timeout=120, backoff_seconds=10)
async def build_contract(request: TaskRequest) -> TaskContract:
    llm_client = LLMClient()
    return await llm_client.generate_structured(
        role=ModelRole.CONTRACT_BUILDER,
        messages=[
            Message(
                role='system',
                content=system_prompt_for_role(ModelRole.CONTRACT_BUILDER),
            ),
            Message(role='user', content=request.model_dump_json()),
        ],
        output_type=TaskContract,
    )
