from __future__ import annotations

from pydantic import Field

from src.activities.workspace_manager import Workspace
from src.config import ModelRole
from src.llm.client import LLMUsage, Message, generate_structured
from src.models.frozen_base_model import FrozenBaseModel
from src.models.plan import PlanStep
from src.models.task import TaskContract
from src.models.worker import TestResult, WorkerResult
from src.runtime_enums import StrEnum

REVIEWER_SYSTEM_PROMPT = (
    'You are the reviewer. Lead with blocking issues. Judge contract '
    'compliance, test adequacy, minimality, and regression risk using only '
    'diff, test, worker, and acceptance-criteria evidence. Ground every '
    'finding in observed evidence. Treat tests as inadequate when they rely on '
    'substring or membership assertions instead of round-trip or invariant checks, cover '
    'only a single happy-path example, or exercise just one side of a symmetric '
    'behavior (for example write without the matching read); such a test passing '
    'is weak evidence. When the diff edits a test, distinguish a justified correction '
    '(a wrong expected value re-derived from the issue or spec, with coverage preserved) '
    'from capitulation (weakening assertions or deleting tests to go green), and treat '
    'capitulation as a blocking issue. '
    'Request revision when evidence is missing, '
    'tests are inadequate, or the patch changes unrelated behavior. Do not '
    'invent validation or assume unrun tests passed. For final workflow review, '
    'distinguish whether the workflow produced a patch, reproduction passed, '
    'tests passed, and the patch is acceptable.'
)


class ReviewDecision(StrEnum):
    ACCEPT = 'accept'
    REVISE = 'revise'
    REJECT = 'reject'
    NEEDS_HUMAN = 'needs_human'


class ReviewVerdict(FrozenBaseModel):
    evidence: list[str] = Field(
        default_factory=list,
        description='Observed diff, worker, and test evidence used for the verdict.',
    )
    blocking_issues: list[str] = Field(
        default_factory=list,
        description=(
            'Issues that must be fixed before acceptance; accept requires this to be empty.'
        ),
    )
    non_blocking_issues: list[str] = Field(
        default_factory=list,
        description='Minor issues that do not prevent acceptance.',
    )
    missing_tests: list[str] = Field(
        default_factory=list,
        description=(
            'Required verification evidence that is absent. Missing required tests should produce '
            'a revise verdict unless acceptance criteria are otherwise proven.'
        ),
    )
    regression_risks: list[str] = Field(
        default_factory=list,
        description='Specific regression risks observed from the diff or test evidence.',
    )
    minimality_assessment: str = Field(
        description='Whether the diff is scoped to the contract and plan step.'
    )
    recommended_next_action: str = Field(
        description=(
            'Concrete next action: accept, revise with specific fixes, reject, or ask human.'
        )
    )
    verdict: ReviewDecision = Field(
        description=(
            'accept only when there are no blocking issues and required evidence is sufficient; '
            'revise when the current workspace can likely be corrected; reject for unusable work.'
        )
    )


class ReviewRequest(FrozenBaseModel):
    contract: TaskContract
    plan_step: PlanStep | None = None
    diff: str
    test_results: list[TestResult] = Field(default_factory=list)
    worker_results: list[WorkerResult] = Field(default_factory=list)
    workspace_info: Workspace


async def review_patch(request: ReviewRequest) -> tuple[ReviewVerdict, LLMUsage]:
    completion = await generate_structured(
        role=ModelRole.REVIEWER,
        messages=[
            Message(role='system', content=REVIEWER_SYSTEM_PROMPT, cacheable=True),
            Message(role='user', content=request.model_dump_json()),
        ],
        output_type=ReviewVerdict,
    )
    return completion.output, completion.usage
