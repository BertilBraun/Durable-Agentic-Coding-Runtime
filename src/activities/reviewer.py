from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.activities.workspace_manager import WorkspaceInfo
from src.config import ModelRole
from src.llm.client import Message, generate_structured
from src.models.plan import PlanStep
from src.models.task import TaskContract
from src.models.worker import TestResult, WorkerResult
from src.runtime_enums import StrEnum

REVIEWER_SYSTEM_PROMPT = (
    'You are the reviewer. Lead with blocking issues. Judge contract '
    'compliance, test adequacy, minimality, and regression risk using only '
    'diff, test, worker, and acceptance-criteria evidence. Ground every '
    'finding in observed evidence. Request revision when evidence is missing, '
    'tests are inadequate, or the patch changes unrelated behavior. Do not '
    'invent validation or assume unrun tests passed.'
)


class ReviewDecision(StrEnum):
    ACCEPT = 'accept'
    REVISE = 'revise'
    REJECT = 'reject'
    NEEDS_HUMAN = 'needs_human'


class ReviewVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence: list[str] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    non_blocking_issues: list[str] = Field(default_factory=list)
    missing_tests: list[str] = Field(default_factory=list)
    regression_risks: list[str] = Field(default_factory=list)
    minimality_assessment: str
    recommended_next_action: str
    verdict: ReviewDecision


class ReviewRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract: TaskContract
    plan_step: PlanStep | None = None
    diff: str
    test_results: list[TestResult] = Field(default_factory=list)
    worker_results: list[WorkerResult] = Field(default_factory=list)
    workspace_info: WorkspaceInfo


async def review_patch(request: ReviewRequest) -> ReviewVerdict:
    completion = await generate_structured(
        role=ModelRole.REVIEWER,
        messages=[
            Message(role='system', content=REVIEWER_SYSTEM_PROMPT),
            Message(role='user', content=request.model_dump_json()),
        ],
        output_type=ReviewVerdict,
    )
    return completion.output
