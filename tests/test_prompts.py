from src.activities.contract_builder import CONTRACT_BUILDER_SYSTEM_PROMPT
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
