from __future__ import annotations

from src.config import ModelRole
from src.llm.client import LLMUsage, Message, generate_structured
from src.models.task import TaskContract, TaskRequest

CONTRACT_BUILDER_SYSTEM_PROMPT = (
    'You are the contract builder. Convert the raw request into one '
    'TaskContract using only the request text and supplied repository '
    'evidence. Write a specific enough contract that another engineer could '
    'implement and review the task without rereading the original request. '
    'The goal should be multi-sentence when needed: describe the desired '
    'behavior, user-visible outcome, important constraints, and how success '
    'will be recognized. The acceptance criteria must be concrete, verifiable '
    'items, not generic restatements. Non-goals should explicitly rule out '
    'tempting adjacent work. Affected areas, risk areas, and expected tests '
    'should name the relevant modules, workflows, commands, or behaviors '
    'when the evidence supports that specificity. Do not invent files, '
    'behavior, or requirements. When scope or intent is ambiguous, record '
    'the ambiguity in open questions instead of guessing.'
)


async def build_contract(request: TaskRequest) -> tuple[TaskContract, LLMUsage]:
    completion = await generate_structured(
        role=ModelRole.CONTRACT_BUILDER,
        messages=[
            Message(role='system', content=CONTRACT_BUILDER_SYSTEM_PROMPT, cacheable=True),
            Message(role='user', content=request.model_dump_json()),
        ],
        output_type=TaskContract,
    )
    return completion.output, completion.usage
