from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.llm.client import Message, generate_structured
from src.llm.config import ModelRole
from src.models.plan import Plan
from src.models.repo import RepoIndex
from src.models.task import TaskContract
from src.models.worker import WorkerResult

PLANNER_SYSTEM_PROMPT = (
    'You are the planner. Build a minimal Plan from the task contract, '
    'repository index, context, and revision guidance. Use small, reviewable '
    'steps with explicit allowed files, expected tests, risk, rollback '
    'strategy, and definition of done. Avoid unrelated refactors and broad '
    'cleanup. If evidence is insufficient, plan an inspection step instead '
    'of inventing implementation details.'
)


class PlanRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract: TaskContract
    repo_index: RepoIndex
    worker_results: list[WorkerResult]
    human_feedback: str | None = None


async def build_plan(request: PlanRequest) -> Plan:
    revision_guidance = request.human_feedback or 'No human revision guidance provided.'
    worker_results_json = [
        worker_result.model_dump(mode='json') for worker_result in request.worker_results
    ]
    completion = await generate_structured(
        role=ModelRole.PLANNER,
        messages=[
            Message(role='system', content=PLANNER_SYSTEM_PROMPT),
            Message(
                role='user',
                content=(
                    f'Revision guidance: {revision_guidance}\n\n'
                    f'Contract:\n{request.contract.model_dump_json()}\n\n'
                    f'Repo index:\n{request.repo_index.model_dump_json()}\n\n'
                    f'Worker results so far:\n{worker_results_json}'
                ),
            ),
        ],
        output_type=Plan,
    )
    return completion.output
