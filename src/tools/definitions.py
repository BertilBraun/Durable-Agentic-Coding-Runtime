from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ToolFieldDefinition:
    name: str
    description: str
    required: bool


@dataclass(frozen=True)
class ToolDefinition:
    name: ToolName
    description: str
    fields: tuple[ToolFieldDefinition, ...]
    mutates_workspace: bool

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)


# TODO can we define a class ToolBase(BaseModel) which defines the model_config? And have all tools inherif from it instead of the BaseModel?


class ReadFileRange(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    tool_name: Literal[ToolName.READ_FILE_RANGE] = Field(
        default=ToolName.READ_FILE_RANGE,
        description='Tool name tag.',
    )
    file_path: str = Field(description='Workspace-relative file path to read.')
    start_line: int = Field(description='First 1-based line number to read.')
    end_line: int = Field(description='Last 1-based line number to read.')


class SearchText(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    tool_name: Literal[ToolName.SEARCH_TEXT] = Field(
        default=ToolName.SEARCH_TEXT,
        description='Tool name tag.',
    )
    pattern: str = Field(description='Search pattern. Use ripgrep-compatible syntax.')
    directory: str = Field(description='Workspace-relative directory to search.')
    file_glob: str = Field(description='File glob filter, for example "*.py".')


class WriteFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    tool_name: Literal[ToolName.WRITE_FILE] = Field(
        default=ToolName.WRITE_FILE,
        description='Tool name tag.',
    )
    file_path: str = Field(description='Workspace-relative file path to replace.')
    content: str = Field(description='Complete file content to write.')


class ApplyPatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    tool_name: Literal[ToolName.APPLY_PATCH] = Field(
        default=ToolName.APPLY_PATCH,
        description='Tool name tag.',
    )
    patch: str = Field(description='Unified diff patch to apply from the workspace root.')


class GitDiff(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    tool_name: Literal[ToolName.GIT_DIFF] = Field(
        default=ToolName.GIT_DIFF,
        description='Tool name tag.',
    )
    path: str = Field(description='Workspace-relative path to diff, or "." for all changes.')


class GitStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    tool_name: Literal[ToolName.GIT_STATUS] = Field(
        default=ToolName.GIT_STATUS,
        description='Tool name tag.',
    )
    path: str = Field(description='Workspace-relative path to inspect, or "." for all changes.')


class GitCommit(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    tool_name: Literal[ToolName.GIT_COMMIT] = Field(
        default=ToolName.GIT_COMMIT,
        description='Tool name tag.',
    )
    message: str = Field(description='Commit message.')


class RunTests(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    tool_name: Literal[ToolName.RUN_TESTS] = Field(
        default=ToolName.RUN_TESTS,
        description='Tool name tag.',
    )
    command: str = Field(description='Test command to run, including needed setup for this call.')
    timeout_seconds: int = Field(description='Maximum seconds to allow the test command to run.')
    directory: str = Field(description='Workspace-relative directory to run from, or ".".')


class RunLint(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    tool_name: Literal[ToolName.RUN_LINT] = Field(
        default=ToolName.RUN_LINT,
        description='Tool name tag.',
    )
    path: str = Field(description='Workspace-relative path to lint, or ".".')


class RunTypecheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    tool_name: Literal[ToolName.RUN_TYPECHECK] = Field(
        default=ToolName.RUN_TYPECHECK,
        description='Tool name tag.',
    )
    path: str = Field(description='Workspace-relative path to typecheck, or ".".')


class FindSymbol(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    tool_name: Literal[ToolName.FIND_SYMBOL] = Field(
        default=ToolName.FIND_SYMBOL,
        description='Tool name tag.',
    )
    name: str = Field(description='Exact symbol name to look up in the repository index.')
    language: str = Field(description='Source language, for example "python" or "javascript".')


class FindReferences(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    tool_name: Literal[ToolName.FIND_REFERENCES] = Field(
        default=ToolName.FIND_REFERENCES,
        description='Tool name tag.',
    )
    symbol_name: str = Field(description='Exact symbol name whose references should be found.')
    file_path: str = Field(description='Workspace-relative file path where the symbol is defined.')


class GatherContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    tool_name: Literal[ToolName.GATHER_CONTEXT] = Field(
        default=ToolName.GATHER_CONTEXT,
        description='Tool name tag.',
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

# TODO when
ImplementationToolCall: TypeAlias = Tool

ContextGathererToolCall: TypeAlias = ReadFileRange | SearchText | FindSymbol | FindReferences

ImplementationToolCallAdapter: TypeAdapter[ImplementationToolCall] = TypeAdapter(
    ImplementationToolCall
)
ContextGathererToolCallAdapter: TypeAdapter[ContextGathererToolCall] = TypeAdapter(
    ContextGathererToolCall
)


# TODO this function is only ever used to look up the name of the tool, which is already in the tool object as the tool_name field, so we can remove this function and just use the tool_name field directly instead of looking up the tool definition every time we need the name - that way we avoid unnecessary lookups and simplify the code - if we need more information about the tool in the future we can always add a lookup function for that specific information, but for now we should just use the tool_name field directly
def tool_definition_for_tool(tool: Tool) -> ToolDefinition:
    match tool:
        case ReadFileRange():
            return ToolDefinition(
                name=ToolName.READ_FILE_RANGE,
                description='Read an inclusive line range from a file.',
                fields=(
                    ToolFieldDefinition('file_path', 'Workspace-relative file path.', True),
                    ToolFieldDefinition('start_line', 'First line to read.', True),
                    ToolFieldDefinition('end_line', 'Last line to read.', True),
                ),
                mutates_workspace=False,
            )
        case SearchText():
            return ToolDefinition(
                name=ToolName.SEARCH_TEXT,
                description='Search text in workspace files.',
                fields=(
                    ToolFieldDefinition('pattern', 'Search pattern.', True),
                    ToolFieldDefinition('directory', 'Workspace-relative directory.', True),
                    ToolFieldDefinition('file_glob', 'File glob filter.', True),
                ),
                mutates_workspace=False,
            )
        case WriteFile():
            return _single_path_definition(
                ToolName.WRITE_FILE, 'Write complete file content.', True
            )
        case ApplyPatch():
            return ToolDefinition(
                name=ToolName.APPLY_PATCH,
                description='Apply a unified diff patch.',
                fields=(ToolFieldDefinition('patch', 'Unified diff content.', True),),
                mutates_workspace=True,
            )
        case GitDiff():
            return _single_path_definition(ToolName.GIT_DIFF, 'Show git diff.', False)
        case GitStatus():
            return _single_path_definition(ToolName.GIT_STATUS, 'Show git status.', False)
        case GitCommit():
            return ToolDefinition(
                name=ToolName.GIT_COMMIT,
                description='Commit current workspace changes.',
                fields=(ToolFieldDefinition('message', 'Commit message.', True),),
                mutates_workspace=True,
            )
        case RunTests():
            return ToolDefinition(
                name=ToolName.RUN_TESTS,
                description='Run a test command.',
                fields=(
                    ToolFieldDefinition('command', 'Test command.', True),
                    ToolFieldDefinition('timeout_seconds', 'Command timeout.', True),
                    ToolFieldDefinition(
                        'directory',
                        'Workspace-relative directory to run the command from.',
                        False,
                    ),
                ),
                mutates_workspace=False,
            )
        case RunLint():
            return _single_path_definition(ToolName.RUN_LINT, 'Run ruff checks.', False)
        case RunTypecheck():
            return _single_path_definition(
                ToolName.RUN_TYPECHECK, 'Run project type checks.', False
            )
        case FindSymbol():
            return ToolDefinition(
                name=ToolName.FIND_SYMBOL,
                description='Find indexed symbols by name.',
                fields=(
                    ToolFieldDefinition('name', 'Symbol name.', True),
                    ToolFieldDefinition('language', 'Programming language.', True),
                ),
                mutates_workspace=False,
            )
        case FindReferences():
            return ToolDefinition(
                name=ToolName.FIND_REFERENCES,
                description='Find references to a symbol.',
                fields=(
                    ToolFieldDefinition('symbol_name', 'Symbol name.', True),
                    ToolFieldDefinition('file_path', 'Starting file path.', True),
                ),
                mutates_workspace=False,
            )
        case GatherContext():
            return ToolDefinition(
                name=ToolName.GATHER_CONTEXT,
                description='Delegate context gathering to the cheap-model agent.',
                fields=(ToolFieldDefinition('prompt', 'Gathering prompt.', True),),
                mutates_workspace=False,
            )
        case _:
            raise AssertionError(f'No tool definition for tool type: {type(tool).__name__}')


def _single_path_definition(
    name: ToolName, description: str, mutates_workspace: bool
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        fields=(ToolFieldDefinition('path', 'Workspace-relative path.', True),),
        mutates_workspace=mutates_workspace,
    )
