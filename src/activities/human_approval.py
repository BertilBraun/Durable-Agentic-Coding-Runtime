from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.activities.temporal import durable_activity
from src.models.plan import Plan
from src.models.task import TaskContract


class HumanPlanPresentationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    contract: TaskContract
    plan: Plan
    artifacts_directory: str = "/artifacts"


@durable_activity(retries=0, timeout=30)
async def present_plan_to_human(request: HumanPlanPresentationRequest) -> str:
    artifact_directory = Path(request.artifacts_directory) / request.run_id
    artifact_directory.mkdir(parents=True, exist_ok=True)
    plan_path = artifact_directory / "plan_for_approval.json"
    plan_path.write_text(
        request.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return str(plan_path)
