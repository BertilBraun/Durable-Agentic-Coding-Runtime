from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.activities.temporal import durable_activity
from src.activities.workspace_manager import WorkspaceInfo
from src.llm.client import LLMClient, Message
from src.llm.config import ModelRole
from src.llm.prompts import system_prompt_for_role
from src.models.plan import PlanStep
from src.models.review import ReviewVerdict
from src.models.task import TaskContract
from src.models.worker import TestResult, WorkerResult


class ReviewRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract: TaskContract
    plan_step: PlanStep | None = None
    diff: str
    test_results: list[TestResult] = Field(default_factory=list)
    worker_results: list[WorkerResult] = Field(default_factory=list)
    workspace_info: WorkspaceInfo


@durable_activity(retries=2, timeout=180, backoff_seconds=10)
async def review_patch(request: ReviewRequest) -> ReviewVerdict:
    llm_client = LLMClient()
    return await llm_client.generate_structured(
        role=ModelRole.REVIEWER,
        messages=[
            Message(
                role='system',
                content=system_prompt_for_role(ModelRole.REVIEWER),
            ),
            Message(role='user', content=request.model_dump_json()),
        ],
        output_type=ReviewVerdict,
    )
