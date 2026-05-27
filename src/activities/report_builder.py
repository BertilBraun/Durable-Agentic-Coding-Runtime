from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.activities.temporal import durable_activity
from src.activities.workspace_manager import WorkspaceInfo
from src.llm.client import LLMClient, LLMUsageSummary
from src.models.plan import Plan
from src.models.review import ReviewVerdict
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


class FinalReportRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    patch: str
    contract: TaskContract
    plan: Plan
    worker_results: list[WorkerResult] = Field(default_factory=list)
    final_verdict: ReviewVerdict
    workspace_info: WorkspaceInfo
    llm_usage: LLMUsageSummary


@durable_activity(retries=0, timeout=30)
async def build_final_report(request: FinalReportRequest) -> FinalReport:
    return FinalReport(
        status=request.final_verdict.verdict.value,
        patch=request.patch,
        contract=request.contract,
        plan=request.plan,
        worker_results=request.worker_results,
        final_verdict=request.final_verdict,
        workspace_info=request.workspace_info,
        llm_usage=request.llm_usage,
    )


@durable_activity(retries=0, timeout=30)
async def reset_llm_usage_summary() -> None:
    LLMClient.reset_global_usage()


@durable_activity(retries=0, timeout=30)
async def collect_llm_usage_summary() -> LLMUsageSummary:
    return LLMClient.global_usage_summary()
