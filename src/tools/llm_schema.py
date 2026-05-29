from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from src.tools.definitions import (
    ApplyPatch,
    FindReferences,
    FindSymbol,
    GatherContext,
    GitDiff,
    GitStatus,
    ReadFileRange,
    RunLint,
    RunTests,
    RunTypecheck,
    SearchText,
    Tool,
    ToolName,
    WriteFile,
)


class ReadFileRangeToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: Literal[ToolName.READ_FILE_RANGE]
    file_path: str = Field(description="Workspace-relative file path to read.")
    start_line: int = Field(description="First 1-based line number to read.")
    end_line: int = Field(description="Last 1-based line number to read.")


class SearchTextToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: Literal[ToolName.SEARCH_TEXT]
    pattern: str = Field(description="Search pattern. Use ripgrep-compatible syntax.")
    directory: str = Field(description="Workspace-relative directory to search.")
    file_glob: str = Field(description='File glob filter, for example "*.py".')


class WriteFileToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: Literal[ToolName.WRITE_FILE]
    file_path: str = Field(description="Workspace-relative file path to replace.")
    content: str = Field(description="Complete file content to write.")


class ApplyPatchToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: Literal[ToolName.APPLY_PATCH]
    patch: str = Field(description="Unified diff patch to apply from the workspace root.")


class GitDiffToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: Literal[ToolName.GIT_DIFF]
    path: str = Field(description='Workspace-relative path to diff, or "." for all changes.')


class GitStatusToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: Literal[ToolName.GIT_STATUS]
    path: str = Field(description='Workspace-relative path to inspect, or "." for all changes.')


class RunTestsToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: Literal[ToolName.RUN_TESTS]
    command: str = Field(description="Test command to run, including needed setup for this call.")
    timeout_seconds: int = Field(description="Maximum seconds to allow the test command to run.")
    directory: str = Field(description='Workspace-relative directory to run from, or ".".')


class RunLintToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: Literal[ToolName.RUN_LINT]
    path: str = Field(description='Workspace-relative path to lint, or ".".')


class RunTypecheckToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: Literal[ToolName.RUN_TYPECHECK]
    path: str = Field(description='Workspace-relative path to typecheck, or ".".')


class FindSymbolToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: Literal[ToolName.FIND_SYMBOL]
    name: str = Field(description="Exact symbol name to look up in the repository index.")
    language: str = Field(description='Source language, for example "python" or "javascript".')


class FindReferencesToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: Literal[ToolName.FIND_REFERENCES]
    symbol_name: str = Field(description="Exact symbol name whose references should be found.")
    file_path: str = Field(description="Workspace-relative file path where the symbol is defined.")


class GatherContextToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: Literal[ToolName.GATHER_CONTEXT]
    prompt: str = Field(description="Focused read-only context request for the context gatherer.")


ImplementationToolCall: TypeAlias = (
    ReadFileRangeToolCall
    | SearchTextToolCall
    | WriteFileToolCall
    | ApplyPatchToolCall
    | GitDiffToolCall
    | GitStatusToolCall
    | RunTestsToolCall
    | RunLintToolCall
    | RunTypecheckToolCall
    | FindSymbolToolCall
    | FindReferencesToolCall
    | GatherContextToolCall
)

ContextGathererToolCall: TypeAlias = (
    ReadFileRangeToolCall | SearchTextToolCall | FindSymbolToolCall | FindReferencesToolCall
)

ImplementationToolCallAdapter: TypeAdapter[ImplementationToolCall] = TypeAdapter(
    ImplementationToolCall
)
ContextGathererToolCallAdapter: TypeAdapter[ContextGathererToolCall] = TypeAdapter(
    ContextGathererToolCall
)


def implementation_tool_from_llm_tool_call(tool_call: ImplementationToolCall) -> Tool:
    match tool_call:
        case ReadFileRangeToolCall(file_path=file_path, start_line=start_line, end_line=end_line):
            return ReadFileRange(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
            )
        case SearchTextToolCall(pattern=pattern, directory=directory, file_glob=file_glob):
            return SearchText(pattern=pattern, directory=directory, file_glob=file_glob)
        case WriteFileToolCall(file_path=file_path, content=content):
            return WriteFile(file_path=file_path, content=content)
        case ApplyPatchToolCall(patch=patch):
            return ApplyPatch(patch=patch)
        case GitDiffToolCall(path=path):
            return GitDiff(path=path)
        case GitStatusToolCall(path=path):
            return GitStatus(path=path)
        case RunTestsToolCall(
            command=command, timeout_seconds=timeout_seconds, directory=directory
        ):
            return RunTests(
                command=command,
                timeout_seconds=timeout_seconds,
                directory=directory,
            )
        case RunLintToolCall(path=path):
            return RunLint(path=path)
        case RunTypecheckToolCall(path=path):
            return RunTypecheck(path=path)
        case FindSymbolToolCall(name=name, language=language):
            return FindSymbol(name=name, language=language)
        case FindReferencesToolCall(symbol_name=symbol_name, file_path=file_path):
            return FindReferences(symbol_name=symbol_name, file_path=file_path)
        case GatherContextToolCall(prompt=prompt):
            return GatherContext(prompt=prompt)


def context_gatherer_tool_from_llm_tool_call(tool_call: ContextGathererToolCall) -> Tool:
    match tool_call:
        case ReadFileRangeToolCall(file_path=file_path, start_line=start_line, end_line=end_line):
            return ReadFileRange(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
            )
        case SearchTextToolCall(pattern=pattern, directory=directory, file_glob=file_glob):
            return SearchText(pattern=pattern, directory=directory, file_glob=file_glob)
        case FindSymbolToolCall(name=name, language=language):
            return FindSymbol(name=name, language=language)
        case FindReferencesToolCall(symbol_name=symbol_name, file_path=file_path):
            return FindReferences(symbol_name=symbol_name, file_path=file_path)
