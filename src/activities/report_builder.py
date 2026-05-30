from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.activities.reviewer import ReviewVerdict
from src.activities.workspace_manager import WorkspaceInfo
from src.llm.client import LLMUsage
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
    llm_usage: LLMUsage
