from __future__ import annotations

from src.llm.client import Message, generate_structured
from src.llm.config import ModelRole
from src.llm.prompts import system_prompt_for_role
from src.models.approval import ComplexityVerdict
from src.models.task import TaskContract


async def assess_complexity(contract: TaskContract) -> ComplexityVerdict:
    completion = await generate_structured(
        role=ModelRole.COMPLEXITY_ASSESSOR,
        messages=[
            Message(
                role='system',
                content=system_prompt_for_role(ModelRole.COMPLEXITY_ASSESSOR),
            ),
            Message(role='user', content=contract.model_dump_json()),
        ],
        output_type=ComplexityVerdict,
    )
    return completion.output
