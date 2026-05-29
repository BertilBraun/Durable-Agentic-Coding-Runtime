from src.llm.config import ModelRole
from src.llm.prompts import system_prompt_for_role


def test_contract_builder_prompt_is_evidence_bound() -> None:
    prompt = system_prompt_for_role(ModelRole.CONTRACT_BUILDER)

    assert 'TaskContract' in prompt
    assert 'Do not invent' in prompt
    assert 'open questions' in prompt


def test_complexity_assessor_prompt_handles_uncertainty() -> None:
    prompt = system_prompt_for_role(ModelRole.COMPLEXITY_ASSESSOR)

    assert 'ComplexityVerdict' in prompt
    assert 'human approval' in prompt
    assert 'missing evidence' in prompt


def test_planner_prompt_defines_reviewable_steps() -> None:
    prompt = system_prompt_for_role(ModelRole.PLANNER)

    assert 'small, reviewable' in prompt
    assert 'allowed files' in prompt
    assert 'rollback' in prompt
    assert 'definition of done' in prompt
    assert 'unrelated refactors' in prompt


def test_implementation_prompt_defines_bounded_agent_behavior() -> None:
    prompt = system_prompt_for_role(ModelRole.IMPLEMENTATION)

    assert 'Inspect before editing' in prompt
    assert 'smallest patch' in prompt
    assert 'allowed files' in prompt
    assert 'Run relevant tests' in prompt
    assert 'observed git_diff or test evidence' in prompt
    assert 'mutating tools' in prompt
    assert 'fresh container' in prompt


def test_context_gatherer_prompt_forbids_mutating_tools() -> None:
    prompt = system_prompt_for_role(ModelRole.CONTEXT_GATHERER)

    assert 'Use only read_file_range, search_text, find_symbol, and find_references' in prompt
    assert 'never request mutating tools' in prompt
    assert 'done=true with ContextPack' in prompt
    assert 'Avoid repeating observations' in prompt


def test_reviewer_prompt_prioritizes_evidence_and_blockers() -> None:
    prompt = system_prompt_for_role(ModelRole.REVIEWER)

    assert 'Lead with blocking issues' in prompt
    assert 'contract compliance' in prompt
    assert 'test adequacy' in prompt
    assert 'Ground every finding' in prompt
    assert 'Request revision' in prompt


def test_summarizer_prompt_forbids_invented_validation() -> None:
    prompt = system_prompt_for_role(ModelRole.SUMMARIZER)

    assert 'using only recorded' in prompt
    assert 'tests run' in prompt
    assert 'unresolved risks' in prompt
    assert 'Do not invent validation' in prompt
