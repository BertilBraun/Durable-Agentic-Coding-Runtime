from __future__ import annotations

from src.llm.client import Message, generate_structured
from src.llm.config import ModelRole
from src.llm.prompts import system_prompt_for_role
from src.models.task import TaskContract, TaskRequest


async def build_contract(request: TaskRequest) -> TaskContract:
    completion = await generate_structured(
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
    return completion.output
