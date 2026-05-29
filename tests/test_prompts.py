from src.llm.config import ModelRole
from src.llm.prompts import system_prompt_for_role


def test_implementation_prompt_defines_bounded_agent_behavior() -> None:
    prompt = system_prompt_for_role(ModelRole.IMPLEMENTATION)

    assert "inspect, edit, diff, and test" in prompt
    assert "Return done=true with WorkerResult only" in prompt
    assert "mutating tools" in prompt
    assert "fresh container" in prompt


def test_context_gatherer_prompt_forbids_mutating_tools() -> None:
    prompt = system_prompt_for_role(ModelRole.CONTEXT_GATHERER)

    assert "Use only read_file_range, search_text, find_symbol, and find_references" in prompt
    assert "Do not call mutating tools" in prompt
