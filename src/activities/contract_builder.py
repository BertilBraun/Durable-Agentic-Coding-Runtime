from __future__ import annotations

from src.llm.client import Message, generate_structured
from src.llm.config import ModelRole
from src.models.task import TaskContract, TaskRequest

CONTRACT_BUILDER_SYSTEM_PROMPT = (
    'You are the contract builder. Convert the raw request into one '
    'TaskContract using only the request text and supplied repository '
    'evidence. State the goal, acceptance criteria, non-goals, affected '
    'areas, risks, expected tests, and open questions as concrete, '
    'verifiable items. Do not invent files, behavior, or requirements. '
    'When scope or intent is ambiguous, record the ambiguity in open '
    'questions instead of guessing.'
)


async def build_contract(request: TaskRequest) -> TaskContract:
    completion = await generate_structured(
        role=ModelRole.CONTRACT_BUILDER,
        messages=[
            Message(role='system', content=CONTRACT_BUILDER_SYSTEM_PROMPT),
            Message(role='user', content=request.model_dump_json()),
        ],
        output_type=TaskContract,
    )
    return completion.output
