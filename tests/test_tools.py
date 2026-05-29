import pytest
from src.tools.definitions import (
    ApplyPatch,
    GitCommit,
    GitDiff,
    ReadFileRange,
    RunLint,
    RunTests,
    SearchText,
    ToolName,
    WriteFile,
    tool_definition_for_tool,
)
from src.tools.handlers import command_for_tool


def test_tool_definitions_have_stable_names() -> None:
    read_file_range = ReadFileRange(file_path="src/app.py", start_line=1, end_line=10)

    tool_definition = tool_definition_for_tool(read_file_range)

    assert tool_definition.name == ToolName.READ_FILE_RANGE


def test_tool_definition_serializes_schema() -> None:
    search_text = SearchText(pattern="class Parser", directory="src", file_glob="*.py")

    tool_definition = tool_definition_for_tool(search_text)

    assert "pattern" in tool_definition.field_names


def test_mutating_tools_are_identified() -> None:
    assert tool_definition_for_tool(ApplyPatch(patch="--- a/file\n+++ b/file")).mutates_workspace
    assert not tool_definition_for_tool(GitDiff(path=".")).mutates_workspace


def test_git_commit_command_stages_new_files() -> None:
    command = command_for_tool(GitCommit(message="commit message"))

    assert command == ["sh", "-lc", "git add -A && git commit -m 'commit message'"]


def test_tool_command_rejects_absolute_file_path() -> None:
    with pytest.raises(ValueError, match="workspace-relative"):
        command_for_tool(ReadFileRange(file_path="/etc/passwd", start_line=1, end_line=1))


def test_tool_command_rejects_parent_traversal_file_path() -> None:
    with pytest.raises(ValueError, match="workspace-relative"):
        command_for_tool(WriteFile(file_path="../outside.txt", content="escape"))


def test_tool_command_rejects_parent_traversal_directory() -> None:
    with pytest.raises(ValueError, match="workspace-relative"):
        command_for_tool(SearchText(pattern="needle", directory="src/../..", file_glob="*.py"))


def test_tool_command_allows_current_directory_path() -> None:
    command = command_for_tool(RunLint(path="."))

    assert command == ["ruff", "check", "."]


def test_run_tests_command_runs_from_requested_directory() -> None:
    command = command_for_tool(
        RunTests(command="pytest -q", timeout_seconds=60, directory="examples/smoke")
    )

    assert command == ["sh", "-lc", "cd examples/smoke && pytest -q"]


def test_run_tests_command_rejects_parent_traversal_directory() -> None:
    with pytest.raises(ValueError, match="workspace-relative"):
        command_for_tool(RunTests(command="pytest -q", timeout_seconds=60, directory="../outside"))
