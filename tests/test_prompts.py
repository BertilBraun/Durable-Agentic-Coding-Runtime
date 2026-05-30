from src.activities.contract_builder import CONTRACT_BUILDER_SYSTEM_PROMPT
from src.activities.planner import PLANNER_SYSTEM_PROMPT


def test_contract_builder_prompt_requires_detailed_contract_fields() -> None:
    assert 'multi-sentence' in CONTRACT_BUILDER_SYSTEM_PROMPT
    assert 'specific enough' in CONTRACT_BUILDER_SYSTEM_PROMPT
    assert 'acceptance criteria' in CONTRACT_BUILDER_SYSTEM_PROMPT


def test_planner_prompt_discourages_tiny_test_then_code_steps() -> None:
    assert '5 to 10 minutes' in PLANNER_SYSTEM_PROMPT
    assert 'Do not split' in PLANNER_SYSTEM_PROMPT
    assert 'test and implementation' in PLANNER_SYSTEM_PROMPT
