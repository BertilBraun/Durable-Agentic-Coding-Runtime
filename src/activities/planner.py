from __future__ import annotations

from src.config import ModelRole
from src.llm.client import LLMUsage, Message, generate_structured
from src.models.context import ContextPack
from src.models.frozen_base_model import FrozenBaseModel
from src.models.plan import ContextNote, Plan, PlannerState, PlannerTurn
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
    'and definition of done. Each implementation step runs as an independent '
    'child workflow with its own context gathering, implementation loop, and '
    'review. Because each step has that fixed overhead, each step should be '
    'sized for roughly 15 to 20 minutes of focused work by a '
    'standard developer: substantial enough to produce a coherent patch, '
    'small enough to review and retry. For small or localized changes, combine '
    'inspection, test updates, implementation, and verification in one step. '
    'Do not create inspection-only steps unless inspection is the whole task or '
    'the repository is too ambiguous to safely name files. '
    'Do not emit separate create-test, implement, and run-tests steps for one '
    'tiny behavior; a simple function plus its regression test belongs in one '
    'step. Split by meaningful '
    'subtasks such as independent behavior areas, nontrivial functions, '
    'integration surfaces, or risky migrations. Avoid unrelated refactors '
    'and broad cleanup. If evidence is insufficient, plan an inspection step '
    'instead of inventing implementation details. For bugfix tasks a failing '
    'regression test usually already exists (its command and observed failure '
    'are given in the revision guidance); when it does, plan the fix that '
    'makes that command pass and never weaken, skip, or delete the test, and '
    'do not plan a separate reproduction step.'
)

PLANNER_TURN_SYSTEM_PROMPT = (
    'You are the workflow coordinator. You do not edit code.\n'
    'You receive normalized state, not a chat transcript.\n'
    'If relevant files/functions are not known well enough to create concrete implementation '
    'steps, request context instead of guessing.\n'
    'If you request context, make each request concrete and do not output implementation steps '
    'that depend on missing context.\n'
    'When enough context exists, output only future steps. Never repeat completed steps.\n'
    'Each future step must be independently executable and must include target files, '
    'step-specific context, required changes, tests to run, expected result, and out-of-scope '
    'constraints.\n'
    'Prefer one substantial concrete step over artificial inspect/create-test/run-test splits.\n'
    'Very concrete test-first steps are allowed when useful, but the step must say exactly what '
    'behavior, file, and failure mode is expected.\n'
    'For bugfixes, use reproduction evidence as the central source of truth.\n'
    'Do not ask workers to weaken, skip, delete, or rewrite existing tests to get green.\n'
    'Set done=true only when no implementation work remains and current evidence supports final '
    'verification.'
)


class PlanRequest(FrozenBaseModel):
    contract: TaskContract
    repo_index: RepoIndex
    worker_results: list[WorkerResult]
    revision_feedback: str | None = None
    reproduction: ReproductionContext | None = None
    context: ContextPack | None = None


async def build_plan(request: PlanRequest) -> tuple[Plan, LLMUsage]:
    revision_guidance = request.revision_feedback or 'No revision guidance provided.'
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
                    f'{_context_guidance(request.context)}\n\n'
                    f'Worker results so far:\n{worker_results_json}'
                ),
            ),
        ],
        output_type=Plan,
    )
    return completion.output, completion.usage


async def plan_next_turn(state: PlannerState) -> tuple[PlannerTurn, LLMUsage]:
    completion = await generate_structured(
        role=ModelRole.PLANNER,
        messages=[
            Message(role='system', content=PLANNER_TURN_SYSTEM_PROMPT, cacheable=True),
            Message(role='user', content=_render_planner_state(state)),
        ],
        output_type=PlannerTurn,
    )
    return completion.output, completion.usage


def _render_planner_state(state: PlannerState) -> str:
    context_notes_json = [_render_context_note(note) for note in state.context_notes]
    completed_steps_json = [
        step.model_dump(mode='json') for step in state.completed_steps
    ]
    previous_future_steps = [
        {
            'id': step.id,
            'goal': step.goal,
            'target_files': step.target_files,
            'expected_result': step.expected_result,
        }
        for step in state.previous_future_steps
    ]
    reproduction_payload = None
    if state.reproduction is not None:
        reproduction_payload = state.reproduction.model_dump(mode='json')
    return (
        'Normalized planner state.\n\n'
        f'Contract:\n{state.contract.model_dump_json()}\n\n'
        f'Reproduction:\n{reproduction_payload}\n\n'
        f'Repository tree:\n{state.repo_index.directory_tree_text()}\n\n'
        f'Context notes:\n{context_notes_json}\n\n'
        f'Completed step history:\n{completed_steps_json}\n\n'
        f'Previous future step summary:\n{previous_future_steps}\n\n'
        f'Current evidence:\n{state.evidence.model_dump(mode="json")}'
    )


def _render_context_note(note: ContextNote) -> dict[str, object]:
    return {
        'id': note.id,
        'summary': note.summary,
        'relevant_files': note.relevant_files,
        'snippet_references': [
            {
                'file_path': snippet.file_path,
                'start_line': snippet.start_line,
                'end_line': snippet.end_line,
                'reason': snippet.reason,
            }
            for snippet in note.snippets
        ],
    }


def _context_guidance(context: ContextPack | None) -> str:
    if context is None or not context.snippets:
        return 'No gathered code context.'
    rendered_snippets = '\n\n'.join(
        f'{snippet.file_path}:{snippet.start_line}-{snippet.end_line} '
        f'({snippet.reason})\n{snippet.content}'
        for snippet in context.snippets
    )
    return f'Gathered code context ({context.task_summary}):\n{rendered_snippets}'


def _reproduction_guidance(reproduction: ReproductionContext | None) -> str:
    if reproduction is None:
        return (
            'No reproduction test exists yet. If this is revision guidance from a prior '
            'worker result, plan only the corrective remaining work; do not repeat '
            'completed steps that already produced an acceptable diff.'
        )
    return (
        'A failing regression test already exists and reproduces the bug. It runs with: '
        f'{reproduction.repro_command}\n'
        f'Observed failure:\n{reproduction.failure_evidence}\n'
        'Plan the fix that makes this command pass without weakening, skipping, or deleting '
        'the test.'
    )
