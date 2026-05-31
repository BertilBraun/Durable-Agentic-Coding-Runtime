from __future__ import annotations

from pathlib import Path

from temporal_light import activity, wait_for_signal

from src.activities.planner import PlanRequest, build_plan
from src.config import CONFIG
from src.llm.client import LLMUsage
from src.models.approval import ApprovalDecision, HumanApprovalSignal
from src.models.frozen_base_model import FrozenBaseModel
from src.models.plan import Plan
from src.models.repo import RepoIndex
from src.models.task import TaskContract


class HumanPlanPresentationRequest(FrozenBaseModel):
    run_id: str
    contract: TaskContract
    plan: Plan


@activity(retries=0, timeout=30)
async def present_plan_to_human(request: HumanPlanPresentationRequest) -> str:
    artifacts_root = CONFIG.artifacts_root
    artifact_directory = Path(artifacts_root) / request.run_id
    artifact_directory.mkdir(parents=True, exist_ok=True)
    plan_path = artifact_directory / 'plan_for_approval.json'
    plan_path.write_text(
        request.model_dump_json(indent=2),
        encoding='utf-8',
    )
    return str(plan_path)


async def approve_plan_or_replan(
    run_id: str,
    contract: TaskContract,
    repo_index: RepoIndex,
    plan: Plan,
) -> tuple[Plan, LLMUsage]:
    usage = LLMUsage()
    while True:
        await present_plan_to_human(
            HumanPlanPresentationRequest(run_id=run_id, contract=contract, plan=plan),
        )
        signal_payload = await wait_for_signal('human_approval')
        approval = HumanApprovalSignal.model_validate(signal_payload)
        if approval.decision == ApprovalDecision.APPROVE:
            break
        plan, plan_usage = await build_plan(
            PlanRequest(
                contract=contract,
                repo_index=repo_index,
                worker_results=[],
                human_feedback=approval.feedback,
            ),
        )
        usage += plan_usage

    return plan, usage
