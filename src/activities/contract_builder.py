from __future__ import annotations

from src.activities.temporal import durable_activity
from src.llm.client import LLMClient, Message
from src.llm.config import ModelRole
from src.models.task import TaskContract, TaskRequest


@durable_activity(retries=2, timeout=120, backoff_seconds=10)
async def build_contract(request: TaskRequest) -> TaskContract:
    llm_client = LLMClient()
    return await llm_client.generate_structured(
        role=ModelRole.CONTRACT_BUILDER,
        messages=[
            Message(
                role="system",
                content=(
                    "You convert a raw software engineering request into a precise task "
                    "contract. Include goal, acceptance criteria, non-goals, affected "
                    "areas, risk areas, expected tests, and open questions."
                ),
            ),
            Message(role="user", content=request.model_dump_json()),
        ],
        output_type=TaskContract,
    )
