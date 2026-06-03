from __future__ import annotations

from src.config import ModelRole
from src.llm.client import LLMUsage, Message, generate_structured
from src.models.plan import ContextNote, PlannerState, PlannerTurn

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
    completed_steps_json = [step.model_dump(mode='json') for step in state.completed_steps]
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
        'Step attempt history. Entries with outcome=success are accepted immutable completed '
        'steps; non-success entries are failed attempts for diagnosis, not completed work:\n'
        f'{completed_steps_json}\n\n'
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
