import pytest
from pydantic import TypeAdapter, ValidationError
from src.tools.definitions import (
    ContextGathererToolCall,
    FindCallers,
    ImplementationToolCall,
    RunShell,
    RunTests,
    WriteFile,
)


def test_tool_parameter_schema_includes_descriptions() -> None:
    schema = RunShell.model_json_schema()

    assert schema['properties']['command']['description']
    assert schema['properties']['timeout_seconds']['description']


def test_implementation_tool_union_uses_openai_compatible_any_of_schema() -> None:
    schema = TypeAdapter(ImplementationToolCall).json_schema()

    assert 'anyOf' in schema
    assert 'oneOf' not in schema
    assert 'discriminator' not in schema


def test_implementation_tool_call_rejects_missing_required_parameters() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(ImplementationToolCall).validate_python(
            {
                'tool_name': 'run_shell',
                'command': 'ls',
            }
        )


def test_context_gatherer_cannot_call_mutating_tools_by_construction() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(ContextGathererToolCall).validate_python(
            {
                'tool_name': 'write_file',
                'file_path': 'src/app.py',
                'content': 'mutating',
            }
        )


def test_implementation_tool_union_parses_runtime_tool_models() -> None:
    assert TypeAdapter(ImplementationToolCall).validate_python(
        {
            'tool_name': 'run_shell',
            'command': 'rg needle src',
            'timeout_seconds': 15,
        }
    ) == RunShell(command='rg needle src', timeout_seconds=15)
    assert TypeAdapter(ImplementationToolCall).validate_python(
        {
            'tool_name': 'write_file',
            'file_path': 'src/app.py',
            'content': 'updated',
        }
    ) == WriteFile(file_path='src/app.py', content='updated')
    assert TypeAdapter(ImplementationToolCall).validate_python(
        {
            'tool_name': 'run_tests',
            'test_targets': ['tests/test_app.py'],
            'timeout_seconds': 60,
        }
    ) == RunTests(test_targets=['tests/test_app.py'], timeout_seconds=60)


def test_context_gatherer_tool_union_uses_runtime_tool_models() -> None:
    tool_call = TypeAdapter(ContextGathererToolCall).validate_python(
        {
            'tool_name': 'find_callers',
            'symbol_name': 'handle_request',
        }
    )

    assert tool_call == FindCallers(symbol_name='handle_request')
