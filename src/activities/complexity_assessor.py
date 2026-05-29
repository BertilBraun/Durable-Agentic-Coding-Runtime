from __future__ import annotations

from src.llm.client import Message, generate_structured
from src.llm.config import ModelRole
from src.models.approval import ComplexityVerdict
from src.models.task import TaskContract

COMPLEXITY_ASSESSOR_SYSTEM_PROMPT = (
    'You are the complexity assessor. Return one ComplexityVerdict using '
    'only the task contract and repository evidence. Require human approval '
    'for likely broad diffs, public APIs, migrations, authentication, '
    'security, data integrity, ambiguous acceptance criteria, feature work, '
    'or breaking-change risk. Do not assume safety from missing evidence; '
    'explain uncertainty in the reasoning. Stop after the verdict.'
)


async def assess_complexity(contract: TaskContract) -> ComplexityVerdict:
    completion = await generate_structured(
        role=ModelRole.COMPLEXITY_ASSESSOR,
        messages=[
            Message(role='system', content=COMPLEXITY_ASSESSOR_SYSTEM_PROMPT),
            Message(role='user', content=contract.model_dump_json()),
        ],
        output_type=ComplexityVerdict,
    )
    return completion.output
