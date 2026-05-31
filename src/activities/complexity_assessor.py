from __future__ import annotations

from src.config import CONFIG, ModelRole
from src.llm.client import LLMUsage, Message, generate_structured
from src.models.frozen_base_model import FrozenBaseModel
from src.models.task import TaskContract

COMPLEXITY_ASSESSOR_SYSTEM_PROMPT = (
    'You are the complexity assessor. Return one ComplexityVerdict using '
    'only the task contract and repository evidence. Human approval is '
    'expensive, so default to not requiring it. Require it only for the few '
    'genuinely high-stakes cases: a complete or sweeping refactor, a highly '
    'complex task, or a task so ambiguous that the intended outcome cannot be '
    'pinned down from the contract. Ordinary feature work, bug fixes, and '
    'contained changes do not require approval. When unsure, do not require '
    'approval. Explain the decision in the reasoning. Stop after the verdict.'
)


class ComplexityVerdict(FrozenBaseModel):
    reasoning: str
    requires_human_approval: bool


async def assess_complexity(contract: TaskContract) -> tuple[ComplexityVerdict, LLMUsage]:
    if not CONFIG.human_approval_enabled:
        verdict = ComplexityVerdict(
            requires_human_approval=False,
            reasoning='Human approval disabled by configuration.',
        )
        return verdict, LLMUsage()
    completion = await generate_structured(
        role=ModelRole.COMPLEXITY_ASSESSOR,
        messages=[
            Message(role='system', content=COMPLEXITY_ASSESSOR_SYSTEM_PROMPT, cacheable=True),
            Message(role='user', content=contract.model_dump_json()),
        ],
        output_type=ComplexityVerdict,
    )
    return completion.output, completion.usage
