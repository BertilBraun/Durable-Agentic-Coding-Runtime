from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from src.runtime_enums import StrEnum


class ToolName(StrEnum):
    READ_FILE_RANGE = 'read_file_range'
    SEARCH_TEXT = 'search_text'
    WRITE_FILE = 'write_file'
    APPLY_PATCH = 'apply_patch'
    GIT_DIFF = 'git_diff'
    GIT_STATUS = 'git_status'
    GIT_COMMIT = 'git_commit'
    RUN_TESTS = 'run_tests'
    RUN_LINT = 'run_lint'
    RUN_TYPECHECK = 'run_typecheck'
    FIND_SYMBOL = 'find_symbol'
    FIND_REFERENCES = 'find_references'
    GATHER_CONTEXT = 'gather_context'


class ToolBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')


class ReadFileRange(ToolBase):
    tool_name: Literal[ToolName.READ_FILE_RANGE] = Field(
        default=ToolName.READ_FILE_RANGE,
        description='Tool name tag.',
    )
    file_path: str = Field(description='Workspace-relative file path to read.')
    start_line: int = Field(description='First 1-based line number to read.')
    end_line: int = Field(description='Last 1-based line number to read.')


class SearchText(ToolBase):
    tool_name: Literal[ToolName.SEARCH_TEXT] = Field(
        default=ToolName.SEARCH_TEXT,
        description='Tool name tag.',
    )
    pattern: str = Field(description='Search pattern. Use ripgrep-compatible syntax.')
    directory: str = Field(description='Workspace-relative directory to search.')
    file_glob: str = Field(description='File glob filter, for example "*.py".')


class WriteFile(ToolBase):
    tool_name: Literal[ToolName.WRITE_FILE] = Field(
        default=ToolName.WRITE_FILE,
        description='Tool name tag.',
    )
    file_path: str = Field(description='Workspace-relative file path to replace.')
    content: str = Field(description='Complete file content to write.')


class ApplyPatch(ToolBase):
    tool_name: Literal[ToolName.APPLY_PATCH] = Field(
        default=ToolName.APPLY_PATCH,
        description='Tool name tag.',
    )
    patch: str = Field(description='Unified diff patch to apply from the workspace root.')


class GitDiff(ToolBase):
    tool_name: Literal[ToolName.GIT_DIFF] = Field(
        default=ToolName.GIT_DIFF,
        description='Tool name tag.',
    )
    path: str = Field(description='Workspace-relative path to diff, or "." for all changes.')


class GitStatus(ToolBase):
    tool_name: Literal[ToolName.GIT_STATUS] = Field(
        default=ToolName.GIT_STATUS,
        description='Tool name tag.',
    )
    path: str = Field(description='Workspace-relative path to inspect, or "." for all changes.')


class GitCommit(ToolBase):
    tool_name: Literal[ToolName.GIT_COMMIT] = Field(
        default=ToolName.GIT_COMMIT,
        description='Tool name tag.',
    )
    message: str = Field(description='Commit message.')


class RunTests(ToolBase):
    tool_name: Literal[ToolName.RUN_TESTS] = Field(
        default=ToolName.RUN_TESTS,
        description='Tool name tag.',
    )
    command: str = Field(description='Test command to run, including needed setup for this call.')
    timeout_seconds: int = Field(description='Maximum seconds to allow the test command to run.')
    directory: str = Field(description='Workspace-relative directory to run from, or ".".')


class RunLint(ToolBase):
    tool_name: Literal[ToolName.RUN_LINT] = Field(
        default=ToolName.RUN_LINT,
        description='Tool name tag.',
    )
    path: str = Field(description='Workspace-relative path to lint, or ".".')


class RunTypecheck(ToolBase):
    tool_name: Literal[ToolName.RUN_TYPECHECK] = Field(
        default=ToolName.RUN_TYPECHECK,
        description='Tool name tag.',
    )
    path: str = Field(description='Workspace-relative path to typecheck, or ".".')


class FindSymbol(ToolBase):
    tool_name: Literal[ToolName.FIND_SYMBOL] = Field(
        default=ToolName.FIND_SYMBOL,
        description='Tool name tag.',
    )
    name: str = Field(description='Exact symbol name to look up in the repository index.')
    language: str = Field(description='Source language, for example "python" or "javascript".')


class FindReferences(ToolBase):
    tool_name: Literal[ToolName.FIND_REFERENCES] = Field(
        default=ToolName.FIND_REFERENCES,
        description='Tool name tag.',
    )
    symbol_name: str = Field(description='Exact symbol name whose references should be found.')
    file_path: str = Field(description='Workspace-relative file path where the symbol is defined.')


class GatherContext(ToolBase):
    tool_name: Literal[ToolName.GATHER_CONTEXT] = Field(
        default=ToolName.GATHER_CONTEXT,
        description='Focused read-only context request for the context gatherer.',
    )
    prompt: str = Field(description='Focused read-only context request for the context gatherer.')


Tool = (
    ReadFileRange
    | SearchText
    | WriteFile
    | ApplyPatch
    | GitDiff
    | GitStatus
    | GitCommit
    | RunTests
    | RunLint
    | RunTypecheck
    | FindSymbol
    | FindReferences
    | GatherContext
)

ImplementationToolCall: TypeAlias = Tool

ContextGathererToolCall: TypeAlias = ReadFileRange | SearchText | FindSymbol | FindReferences

ImplementationToolCallAdapter: TypeAdapter[ImplementationToolCall] = TypeAdapter(
    ImplementationToolCall
)
ContextGathererToolCallAdapter: TypeAdapter[ContextGathererToolCall] = TypeAdapter(
    ContextGathererToolCall
)
