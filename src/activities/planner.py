from __future__ import annotations

import json

from src.config import ModelRole
from src.llm.client import LLMUsage, Message, generate_structured
from src.models.plan import ContextNote, PlannerState, PlannerToolObservation, PlannerTurn

PLANNER_TURN_SYSTEM_PROMPT = (
    'You are the workflow coordinator. You do not edit code.\n'
    'You receive normalized state, not a chat transcript.\n'
    'If relevant files/functions are not known well enough to create concrete implementation '
    'steps, request context instead of guessing.\n'
    'You may request read-only tool calls to inspect concrete code before planning. Tool calls '
    'must never mutate files, run tests, or write patches.\n'
    'Do not request context already covered by Context notes. Treat fulfilled request queries, '
    'relevant files, and snippet references as available evidence; ask for more context only when '
    'you need different concrete files, functions, or behavior.\n'
    'If you request context, make each request concrete and do not output implementation steps '
    'that depend on missing context.\n'
    'When enough context exists, output only future steps. Never repeat completed steps.\n'
    'Each future step must be independently executable and must include target files, '
    'step-specific context, required changes, tests to run, expected result, and out-of-scope '
    'constraints.\n'
    'Prefer one substantial concrete step over artificial inspect/create-test/run-test splits.\n'
    'Scope steps for a complete, consistent fix, not just the literal example: when the issue is '
    'one instance of a general defect (one keyword, format, or direction of a symmetric '
    'operation), require handling the sibling cases the corrected behavior implies.\n'
    'Very concrete test-first steps are allowed when useful, but the step must say exactly what '
    'behavior, file, and failure mode is expected.\n'
    'When a step specifies tests, require strong, self-validating ones: prefer round trips and '
    'invariants over literal values, ground any literal expectation in the issue or spec rather '
    'than guessing it, and cover multiple inputs and input counts instead of a single example.\n'
    'A red test is ambiguous: the fault may be the test, not the code. If an implemented step '
    'keeps a self-authored test red, request the test and implementation as context and judge '
    'whether the test is unrepresentative or encodes a misunderstanding; if so, schedule a step '
    'to correct the test, re-derived from the issue or spec with coverage preserved, rather than '
    'only changing the production code.\n'
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
    return '\n\n'.join(
        [
            'Planner State',
            _render_contract(state),
            _section('Reproduction', _json_or_none(state.reproduction)),
            _section('Repository Tree', state.repo_index.directory_tree_text() or '(empty)'),
            _section('Available Context', _render_context_notes(state.context_notes)),
            _section(
                'Planner Tool Observations',
                _render_tool_observations(state.tool_observations),
            ),
            _section('Step Attempt History', _render_completed_steps(state)),
            _section('Previous Future Step Summary', _render_previous_future_steps(state)),
            _section('Current Evidence', _json(state.evidence.model_dump(mode='json'))),
            _section(
                'Allowed Next Outputs',
                '\n'.join(
                    [
                        '- read-only tool calls to inspect concrete code',
                        '- context requests for missing relevant ranges',
                        '- future implementation steps',
                        '- done=true when no work remains',
                    ]
                ),
            ),
        ]
    )


def _render_contract(state: PlannerState) -> str:
    contract = state.contract
    return '\n'.join(
        [
            'Goal',
            contract.goal,
            '',
            'Task Type',
            contract.task_type.value,
            '',
            _section('Acceptance Criteria', _bullet_list(contract.acceptance_criteria)),
            _section('Non Goals', _bullet_list(contract.non_goals)),
            _section('Affected Areas', _bullet_list(contract.affected_areas)),
            _section('Risk Areas', _bullet_list(contract.risk_areas)),
            _section('Tests Expected', _bullet_list(contract.tests_expected)),
            _section('Open Questions', _bullet_list(contract.open_questions)),
        ]
    )


def _render_context_notes(notes: list[ContextNote]) -> str:
    if not notes:
        return '(none)'
    return '\n\n'.join(_render_context_note_text(note) for note in notes)


def _render_context_note_text(note: ContextNote) -> str:
    request_reason = note.request_reason or '(reason unavailable)'
    lines = [f'{note.id}: {request_reason}']
    if note.request_queries:
        lines.append('queries:')
        lines.extend(f'- {query}' for query in note.request_queries)
    if note.relevant_files:
        lines.append('relevant files:')
        lines.extend(f'- {file_path}' for file_path in note.relevant_files)
    if note.snippets:
        lines.append('snippets:')
        lines.extend(f'- {_snippet_reference(snippet)}' for snippet in note.snippets)
    lines.append('summary:')
    lines.append(note.summary)
    return '\n'.join(lines)


def _snippet_reference(snippet: object) -> str:
    return (
        f'{snippet.file_path}:{snippet.start_line}-{snippet.end_line} - {snippet.reason}'
    )


def _render_tool_observations(observations: list[PlannerToolObservation]) -> str:
    if not observations:
        return '(none)'
    return '\n\n'.join(_render_tool_observation(observation) for observation in observations)


def _render_tool_observation(observation: PlannerToolObservation) -> str:
    lines = [
        f'{observation.tool_name.value} exit_code={observation.exit_code}',
        'stdout:',
        observation.stdout or '(empty)',
    ]
    if observation.stderr:
        lines.extend(['stderr:', observation.stderr])
    if observation.truncated:
        lines.append('output truncated: true')
    return '\n'.join(lines)


def _render_completed_steps(state: PlannerState) -> str:
    if not state.completed_steps:
        return '(none)'
    lines = [
        'Entries with outcome=success are accepted immutable completed steps; non-success '
        'entries are failed attempts for diagnosis, not completed work.'
    ]
    for step in state.completed_steps:
        lines.append(
            f'- {step.step_id}: outcome={step.outcome.value}, '
            f'confidence={step.confidence.value}, summary={step.summary}'
        )
        if step.observations:
            lines.append(f'  observations: {_json(step.observations)}')
        if step.tests:
            lines.append(f'  tests: {_json([test.model_dump(mode="json") for test in step.tests])}')
    return '\n'.join(lines)


def _render_previous_future_steps(state: PlannerState) -> str:
    if not state.previous_future_steps:
        return '(none)'
    lines = []
    for step in state.previous_future_steps:
        lines.append(
            f'- {step.id}: {step.goal}; target_files={_json(step.target_files)}; '
            f'expected_result={step.expected_result}'
        )
    return '\n'.join(lines)


def _section(title: str, body: str) -> str:
    return f'{title}\n{body}'


def _bullet_list(items: list[str]) -> str:
    if not items:
        return '(none)'
    return '\n'.join(f'- {item}' for item in items)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True)


def _json_or_none(value: object | None) -> str:
    if value is None:
        return '(none)'
    if hasattr(value, 'model_dump'):
        return _json(value.model_dump(mode='json'))
    return _json(value)


def _render_context_note(note: ContextNote) -> dict[str, object]:
    return {
        'id': note.id,
        'summary': note.summary,
        'fulfilled_request': {
            'reason': note.request_reason,
            'queries': note.request_queries,
        },
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
