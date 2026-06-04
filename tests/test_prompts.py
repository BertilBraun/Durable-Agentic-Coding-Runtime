from src.activities.contract_builder import CONTRACT_BUILDER_SYSTEM_PROMPT
from src.activities.implementation import IMPLEMENTATION_SYSTEM_PROMPT
from src.activities.planner import PLANNER_TURN_SYSTEM_PROMPT
from src.activities.reproduction import REPRODUCTION_SYSTEM_PROMPT
from src.activities.reviewer import REVIEWER_SYSTEM_PROMPT


def test_contract_builder_prompt_requires_detailed_contract_fields() -> None:
    assert 'multi-sentence' in CONTRACT_BUILDER_SYSTEM_PROMPT
    assert 'specific enough' in CONTRACT_BUILDER_SYSTEM_PROMPT
    assert 'acceptance criteria' in CONTRACT_BUILDER_SYSTEM_PROMPT


def test_planner_turn_prompt_requires_normalized_future_step_contract() -> None:
    assert 'normalized state' in PLANNER_TURN_SYSTEM_PROMPT
    assert 'request context instead of guessing' in PLANNER_TURN_SYSTEM_PROMPT
    assert 'output only future steps' in PLANNER_TURN_SYSTEM_PROMPT
    assert 'Never repeat completed steps' in PLANNER_TURN_SYSTEM_PROMPT
    assert 'target files' in PLANNER_TURN_SYSTEM_PROMPT
    assert 'out-of-scope' in PLANNER_TURN_SYSTEM_PROMPT


def test_implementation_prompt_scopes_worker_to_current_step() -> None:
    assert 'exactly one planner-selected step' in IMPLEMENTATION_SYSTEM_PROMPT
    assert 'preserve that work and do not redo it' in IMPLEMENTATION_SYSTEM_PROMPT
    assert 'plan_step.required_changes' in IMPLEMENTATION_SYSTEM_PROMPT
    assert 'plan_step.out_of_scope' in IMPLEMENTATION_SYSTEM_PROMPT
    assert 'Confidence means confidence in this step' in IMPLEMENTATION_SYSTEM_PROMPT


def test_implementation_prompt_requires_self_checked_strong_tests() -> None:
    assert 'self-validating checks' in IMPLEMENTATION_SYSTEM_PROMPT
    assert 'round trips' in IMPLEMENTATION_SYSTEM_PROMPT
    assert 'trivially-wrong implementation' in IMPLEMENTATION_SYSTEM_PROMPT


def test_planner_prompt_requires_strong_test_specs() -> None:
    assert 'self-validating' in PLANNER_TURN_SYSTEM_PROMPT
    assert 'multiple inputs and input counts' in PLANNER_TURN_SYSTEM_PROMPT
    assert 'round trips' in PLANNER_TURN_SYSTEM_PROMPT


def test_planner_prompt_allows_rechecking_a_faulty_test() -> None:
    assert 'A red test is ambiguous' in PLANNER_TURN_SYSTEM_PROMPT
    assert 'unrepresentative or encodes a misunderstanding' in PLANNER_TURN_SYSTEM_PROMPT


def test_reproduction_prompt_requires_strong_pytest_tests() -> None:
    assert 'self-validating checks' in REPRODUCTION_SYSTEM_PROMPT
    assert 'never invent' in REPRODUCTION_SYSTEM_PROMPT
    assert 'pytest.main' in REPRODUCTION_SYSTEM_PROMPT


def test_reviewer_prompt_flags_weak_tests_and_capitulation() -> None:
    assert 'substring or membership assertions' in REVIEWER_SYSTEM_PROMPT
    assert 'single happy-path example' in REVIEWER_SYSTEM_PROMPT
    assert 'one side of a symmetric' in REVIEWER_SYSTEM_PROMPT
    assert 'capitulation' in REVIEWER_SYSTEM_PROMPT
