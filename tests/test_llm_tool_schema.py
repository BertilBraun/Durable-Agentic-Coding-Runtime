import pytest
from pydantic import TypeAdapter, ValidationError
from src.tools.definitions import ReadFileRange, RunTests, SearchText, WriteFile
from src.tools.llm_schema import (
    ContextGathererToolCall,
    ImplementationToolCall,
    ImplementationToolCallAdapter,
    SearchTextToolCall,
    context_gatherer_tool_from_llm_tool_call,
    implementation_tool_from_llm_tool_call,
)


def test_tool_parameter_schema_includes_descriptions() -> None:
    schema = SearchTextToolCall.model_json_schema()

    assert schema["properties"]["pattern"]["description"]
    assert schema["properties"]["directory"]["description"]
    assert schema["properties"]["file_glob"]["description"]


def test_implementation_tool_union_uses_openai_compatible_any_of_schema() -> None:
    schema = ImplementationToolCallAdapter.json_schema()

    assert "anyOf" in schema
    assert "oneOf" not in schema
    assert "discriminator" not in schema


def test_implementation_tool_call_rejects_missing_required_parameters() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(ImplementationToolCall).validate_python(
            {
                "tool_name": "read_file_range",
                "file_path": "src/app.py",
            }
        )


def test_context_gatherer_cannot_call_mutating_tools_by_construction() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(ContextGathererToolCall).validate_python(
            {
                "tool_name": "write_file",
                "file_path": "src/app.py",
                "content": "mutating",
            }
        )


def test_implementation_tool_conversion_uses_typed_models() -> None:
    assert implementation_tool_from_llm_tool_call(
        TypeAdapter(ImplementationToolCall).validate_python(
            {
                "tool_name": "read_file_range",
                "file_path": "src/app.py",
                "start_line": 10,
                "end_line": 20,
            }
        )
    ) == ReadFileRange(file_path="src/app.py", start_line=10, end_line=20)
    assert implementation_tool_from_llm_tool_call(
        TypeAdapter(ImplementationToolCall).validate_python(
            {
                "tool_name": "write_file",
                "file_path": "src/app.py",
                "content": "updated",
            }
        )
    ) == WriteFile(file_path="src/app.py", content="updated")
    assert implementation_tool_from_llm_tool_call(
        TypeAdapter(ImplementationToolCall).validate_python(
            {
                "tool_name": "run_tests",
                "command": "pytest -q",
                "timeout_seconds": 60,
                "directory": ".",
            }
        )
    ) == RunTests(command="pytest -q", timeout_seconds=60, directory=".")


def test_context_gatherer_tool_conversion_uses_shared_models() -> None:
    tool_call = TypeAdapter(ContextGathererToolCall).validate_python(
        {
            "tool_name": "search_text",
            "pattern": "class Handler",
            "directory": "src",
            "file_glob": "*.py",
        }
    )

    assert context_gatherer_tool_from_llm_tool_call(tool_call) == SearchText(
        pattern="class Handler",
        directory="src",
        file_glob="*.py",
    )
