from __future__ import annotations

from src.config import ModelRole
from src.llm.client import LLMUsage, Message, generate_structured
from src.models.frozen_base_model import FrozenBaseModel
from src.models.plan import Plan
from src.models.repo import RepoIndex
from src.models.reproduction import ReproductionContext
from src.models.task import TaskContract
from src.models.worker import WorkerResult

# Future: lift this task-type guidance into a structured, deterministic step
# template (reproduction-before-repair, generated regression tests, rollback
# checkpoints) instead of relying on prompt wording — see PLAN.md milestone 2.
PLANNER_SYSTEM_PROMPT = (
    'You are the planner. Build a minimal Plan from the task contract, '
    'repository index, context, and revision guidance. Use detailed, '
    'reviewable steps with explicit allowed files, expected tests, risk, '
    'rollback strategy, and definition of done. Each implementation step '
    'should be sized for roughly 5 to 10 minutes of focused work by a '
    'standard developer: substantial enough to produce a coherent patch, '
    'small enough to review and retry. Do not split one tiny behavior into '
    'separate test and implementation steps; a simple function plus its '
    'regression test usually belongs in one step. Split by meaningful '
    'subtasks such as independent behavior areas, nontrivial functions, '
    'integration surfaces, or risky migrations. Avoid unrelated refactors '
    'and broad cleanup. If evidence is insufficient, plan an inspection step '
    'instead of inventing implementation details. For bugfix tasks a failing '
    'regression test usually already exists (its command and observed failure '
    'are given in the revision guidance); when it does, plan the fix that '
    'makes that command pass and never weaken, skip, or delete the test, and '
    'do not plan a separate reproduction step.'
)


class PlanRequest(FrozenBaseModel):
    contract: TaskContract
    repo_index: RepoIndex
    worker_results: list[WorkerResult]
    human_feedback: str | None = None
    reproduction: ReproductionContext | None = None


async def build_plan(request: PlanRequest) -> tuple[Plan, LLMUsage]:
    revision_guidance = request.human_feedback or 'No human revision guidance provided.'
    worker_results_json = [
        worker_result.model_dump(mode='json') for worker_result in request.worker_results
    ]
    completion = await generate_structured(
        role=ModelRole.PLANNER,
        messages=[
            Message(role='system', content=PLANNER_SYSTEM_PROMPT, cacheable=True),
            Message(
                role='user',
                content=(
                    f'Revision guidance: {revision_guidance}\n\n'
                    f'{_reproduction_guidance(request.reproduction)}\n\n'
                    f'Contract:\n{request.contract.model_dump_json()}\n\n'
                    f'Repository tree:\n{request.repo_index.directory_tree_text()}\n\n'
                    f'Worker results so far:\n{worker_results_json}'
                ),
            ),
        ],
        output_type=Plan,
    )
    return completion.output, completion.usage


def _reproduction_guidance(reproduction: ReproductionContext | None) -> str:
    if reproduction is None:
        return 'No reproduction test exists yet.'
    return (
        'A failing regression test already exists and reproduces the bug. It runs with: '
        f'{reproduction.repro_command}\n'
        f'Observed failure:\n{reproduction.failure_evidence}\n'
        'Plan the fix that makes this command pass without weakening, skipping, or deleting '
        'the test.'
    )
