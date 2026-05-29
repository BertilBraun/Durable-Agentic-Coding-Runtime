from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from temporal_light import activity

from src.activities.reviewer import ReviewVerdict
from src.activities.workspace_manager import WorkspaceInfo
from src.llm.client import LLMClient, LLMUsageSummary
from src.models.plan import Plan
from src.models.task import TaskContract
from src.models.worker import WorkerResult


class FinalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    patch: str
    contract: TaskContract
    plan: Plan
    worker_results: list[WorkerResult] = Field(default_factory=list)
    final_verdict: ReviewVerdict
    workspace_info: WorkspaceInfo
    llm_usage: LLMUsageSummary


# TODO global state in the LLMClient will not persist across activity invocations, need to store in durable storage and pass around


@activity(retries=0, timeout=30)
async def reset_llm_usage_summary() -> None:
    LLMClient.reset_global_usage()


@activity(retries=0, timeout=30)
async def collect_llm_usage_summary() -> LLMUsageSummary:
    return LLMClient.global_usage_summary()
