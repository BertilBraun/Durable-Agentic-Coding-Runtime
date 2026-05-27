from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.activities.temporal import durable_activity
from src.llm.client import LLMClient, Message
from src.llm.config import ModelRole
from src.models.plan import Plan
from src.models.repo import RepoIndex
from src.models.task import TaskContract


class PlanRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract: TaskContract
    repo_index: RepoIndex
    human_feedback: str | None = None


@durable_activity(retries=2, timeout=180, backoff_seconds=10)
async def build_plan(request: PlanRequest) -> Plan:
    llm_client = LLMClient()
    revision_guidance = request.human_feedback or "No human revision guidance provided."
    return await llm_client.generate_structured(
        role=ModelRole.PLANNER,
        messages=[
            Message(
                role="system",
                content=(
                    "Build a minimal implementation plan from the contract and repository "
                    "index. Keep steps scoped, include allowed files, tests, risks, rollback, "
                    "and definition of done. Apply revision guidance when provided."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Revision guidance: {revision_guidance}\n\n"
                    f"Contract:\n{request.contract.model_dump_json()}\n\n"
                    f"Repo index:\n{request.repo_index.model_dump_json()}"
                ),
            ),
        ],
        output_type=Plan,
    )
