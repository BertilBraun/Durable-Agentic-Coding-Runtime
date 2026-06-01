from __future__ import annotations

from pydantic import Field

from src.activities.reviewer import ReviewVerdict
from src.activities.workspace_manager import Workspace
from src.llm.client import LLMUsage
from src.models.frozen_base_model import FrozenBaseModel
from src.models.plan import Plan
from src.models.reproduction import ReproductionEvidence
from src.models.task import TaskContract
from src.models.worker import WorkerResult


class FinalReport(FrozenBaseModel):
    status: str
    patch: str
    contract: TaskContract
    plan: Plan
    worker_results: list[WorkerResult] = Field(default_factory=list)
    final_verdict: ReviewVerdict
    workspace_info: Workspace
    llm_usage: LLMUsage
    reproduction_evidence: ReproductionEvidence | None = None
