from src.activities.contract_builder import CONTRACT_BUILDER_SYSTEM_PROMPT
from src.activities.implementation import IMPLEMENTATION_SYSTEM_PROMPT
from src.activities.planner import PLANNER_SYSTEM_PROMPT


def test_contract_builder_prompt_requires_detailed_contract_fields() -> None:
    assert 'multi-sentence' in CONTRACT_BUILDER_SYSTEM_PROMPT
    assert 'specific enough' in CONTRACT_BUILDER_SYSTEM_PROMPT
    assert 'acceptance criteria' in CONTRACT_BUILDER_SYSTEM_PROMPT


def test_planner_prompt_discourages_tiny_test_then_code_steps() -> None:
    assert '15 to 20 minutes' in PLANNER_SYSTEM_PROMPT
    assert 'Do not emit separate create-test, implement, and run-tests steps' in (
        PLANNER_SYSTEM_PROMPT
    )
    assert 'regression test belongs in one step' in PLANNER_SYSTEM_PROMPT


def test_implementation_prompt_scopes_worker_to_current_step() -> None:
    assert 'one step of a larger coding plan' in IMPLEMENTATION_SYSTEM_PROMPT
    assert 'preserve that work and do not redo it' in IMPLEMENTATION_SYSTEM_PROMPT
    assert 'execute the current plan step only' in IMPLEMENTATION_SYSTEM_PROMPT
    assert 'Confidence means confidence in this step' in IMPLEMENTATION_SYSTEM_PROMPT
