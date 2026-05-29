from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.activities.temporal import durable_activity
from src.activities.workspace_manager import ARTIFACTS_ROOT_ENVIRONMENT_NAME, DEFAULT_ARTIFACTS_ROOT
from src.models.plan import Plan
from src.models.task import TaskContract


class HumanPlanPresentationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    contract: TaskContract
    plan: Plan


@durable_activity(retries=0, timeout=30)
async def present_plan_to_human(request: HumanPlanPresentationRequest) -> str:
    artifacts_root = os.getenv(ARTIFACTS_ROOT_ENVIRONMENT_NAME, DEFAULT_ARTIFACTS_ROOT)
    artifact_directory = Path(artifacts_root) / request.run_id
    artifact_directory.mkdir(parents=True, exist_ok=True)
    plan_path = artifact_directory / "plan_for_approval.json"
    plan_path.write_text(
        request.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return str(plan_path)
