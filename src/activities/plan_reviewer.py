from __future__ import annotations

from pydantic import Field

from src.config import ModelRole
from src.llm.client import LLMUsage, Message, generate_structured
from src.models.frozen_base_model import FrozenBaseModel
from src.models.plan import Plan
from src.models.repo import RepoIndex
from src.models.reproduction import ReproductionContext
from src.models.task import TaskContract
from src.runtime_enums import StrEnum

PLAN_REVIEWER_SYSTEM_PROMPT = (
    'You are the plan reviewer. Judge the plan against the task contract and '
    'repository overview before any implementation starts. Check that every '
    'acceptance criterion is covered by a step, that there are no missing or '
    'duplicated steps, and that each step is sized for roughly 5 to 10 minutes '
    'of focused work — flag oversized or undersized steps. Flag risky or '
    'unjustified changes and any change that strays outside the affected areas. '
    'Reject plans that split one tiny behavior into separate create-test, '
    'implement, and run-tests steps; those belong in one implementation step. '
    'For bugfix tasks, when a failing regression test already exists, the plan '
    'must make that command pass without weakening, skipping, or deleting it, '
    'and must not plan a separate reproduction step. Return decision=accept when '
    'the plan is good enough to implement (not perfect) and leave feedback '
    'empty; return decision=revise with concrete, actionable feedback that the '
    'planner can act on to close the gaps you found.'
)


class PlanReviewDecision(StrEnum):
    ACCEPT = 'accept'
    REVISE = 'revise'


class PlanReviewVerdict(FrozenBaseModel):
    decision: PlanReviewDecision
    blocking_issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    feedback: str


class PlanReviewRequest(FrozenBaseModel):
    contract: TaskContract
    plan: Plan
    repo_index: RepoIndex
    reproduction: ReproductionContext | None = None


async def review_plan(request: PlanReviewRequest) -> tuple[PlanReviewVerdict, LLMUsage]:
    completion = await generate_structured(
        role=ModelRole.PLAN_REVIEWER,
        messages=[
            Message(role='system', content=PLAN_REVIEWER_SYSTEM_PROMPT, cacheable=True),
            Message(
                role='user',
                content=(
                    f'Contract:\n{request.contract.model_dump_json()}\n\n'
                    f'Plan:\n{request.plan.model_dump_json()}\n\n'
                    f'Repository overview:\n{request.repo_index.directory_tree_text()}\n\n'
                    f'{_reproduction_guidance(request.reproduction)}'
                ),
            ),
        ],
        output_type=PlanReviewVerdict,
    )
    return completion.output, completion.usage


def _reproduction_guidance(reproduction: ReproductionContext | None) -> str:
    if reproduction is None:
        return 'This is not a bugfix task with an existing regression test.'
    return (
        'A failing regression test already reproduces the bug. It runs with: '
        f'{reproduction.repro_command}\n'
        f'Observed failure:\n{reproduction.failure_evidence}\n'
        'The plan must make this command pass without weakening, skipping, or deleting the '
        'test, and must not plan a separate reproduction step.'
    )
