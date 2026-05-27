from src.tools.definitions import (
    ApplyPatch,
    GitCommit,
    GitDiff,
    ReadFileRange,
    SearchText,
    ToolName,
    tool_definition_for_tool,
)


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
    from src.tools.handlers import command_for_tool

    command = command_for_tool(GitCommit(message="commit message"))

    assert command == ["sh", "-lc", "git add -A && git commit -m 'commit message'"]
