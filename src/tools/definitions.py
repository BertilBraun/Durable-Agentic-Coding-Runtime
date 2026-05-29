from __future__ import annotations

from dataclasses import dataclass

from src.runtime_enums import StrEnum


class ToolName(StrEnum):
    READ_FILE_RANGE = "read_file_range"
    SEARCH_TEXT = "search_text"
    WRITE_FILE = "write_file"
    APPLY_PATCH = "apply_patch"
    GIT_DIFF = "git_diff"
    GIT_STATUS = "git_status"
    GIT_COMMIT = "git_commit"
    RUN_TESTS = "run_tests"
    RUN_LINT = "run_lint"
    RUN_TYPECHECK = "run_typecheck"
    FIND_SYMBOL = "find_symbol"
    FIND_REFERENCES = "find_references"
    GATHER_CONTEXT = "gather_context"


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


@dataclass(frozen=True)
class ReadFileRange:
    file_path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class SearchText:
    pattern: str
    directory: str
    file_glob: str


@dataclass(frozen=True)
class WriteFile:
    file_path: str
    content: str


@dataclass(frozen=True)
class ApplyPatch:
    patch: str


@dataclass(frozen=True)
class GitDiff:
    path: str


@dataclass(frozen=True)
class GitStatus:
    path: str


@dataclass(frozen=True)
class GitCommit:
    message: str


@dataclass(frozen=True)
class RunTests:
    command: str
    timeout_seconds: int


@dataclass(frozen=True)
class RunLint:
    path: str


@dataclass(frozen=True)
class RunTypecheck:
    path: str


@dataclass(frozen=True)
class FindSymbol:
    name: str
    language: str


@dataclass(frozen=True)
class FindReferences:
    symbol_name: str
    file_path: str


@dataclass(frozen=True)
class GatherContext:
    prompt: str


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


def tool_definition_for_tool(tool: Tool) -> ToolDefinition:
    match tool:
        case ReadFileRange():
            return ToolDefinition(
                name=ToolName.READ_FILE_RANGE,
                description="Read an inclusive line range from a file.",
                fields=(
                    ToolFieldDefinition("file_path", "Workspace-relative file path.", True),
                    ToolFieldDefinition("start_line", "First line to read.", True),
                    ToolFieldDefinition("end_line", "Last line to read.", True),
                ),
                mutates_workspace=False,
            )
        case SearchText():
            return ToolDefinition(
                name=ToolName.SEARCH_TEXT,
                description="Search text in workspace files.",
                fields=(
                    ToolFieldDefinition("pattern", "Search pattern.", True),
                    ToolFieldDefinition("directory", "Workspace-relative directory.", True),
                    ToolFieldDefinition("file_glob", "File glob filter.", True),
                ),
                mutates_workspace=False,
            )
        case WriteFile():
            return _single_path_definition(
                ToolName.WRITE_FILE, "Write complete file content.", True
            )
        case ApplyPatch():
            return ToolDefinition(
                name=ToolName.APPLY_PATCH,
                description="Apply a unified diff patch.",
                fields=(ToolFieldDefinition("patch", "Unified diff content.", True),),
                mutates_workspace=True,
            )
        case GitDiff():
            return _single_path_definition(ToolName.GIT_DIFF, "Show git diff.", False)
        case GitStatus():
            return _single_path_definition(ToolName.GIT_STATUS, "Show git status.", False)
        case GitCommit():
            return ToolDefinition(
                name=ToolName.GIT_COMMIT,
                description="Commit current workspace changes.",
                fields=(ToolFieldDefinition("message", "Commit message.", True),),
                mutates_workspace=True,
            )
        case RunTests():
            return ToolDefinition(
                name=ToolName.RUN_TESTS,
                description="Run a test command.",
                fields=(
                    ToolFieldDefinition("command", "Test command.", True),
                    ToolFieldDefinition("timeout_seconds", "Command timeout.", True),
                ),
                mutates_workspace=False,
            )
        case RunLint():
            return _single_path_definition(ToolName.RUN_LINT, "Run ruff checks.", False)
        case RunTypecheck():
            return _single_path_definition(
                ToolName.RUN_TYPECHECK, "Run project type checks.", False
            )
        case FindSymbol():
            return ToolDefinition(
                name=ToolName.FIND_SYMBOL,
                description="Find indexed symbols by name.",
                fields=(
                    ToolFieldDefinition("name", "Symbol name.", True),
                    ToolFieldDefinition("language", "Programming language.", True),
                ),
                mutates_workspace=False,
            )
        case FindReferences():
            return ToolDefinition(
                name=ToolName.FIND_REFERENCES,
                description="Find references to a symbol.",
                fields=(
                    ToolFieldDefinition("symbol_name", "Symbol name.", True),
                    ToolFieldDefinition("file_path", "Starting file path.", True),
                ),
                mutates_workspace=False,
            )
        case GatherContext():
            return ToolDefinition(
                name=ToolName.GATHER_CONTEXT,
                description="Delegate context gathering to the cheap-model agent.",
                fields=(ToolFieldDefinition("prompt", "Gathering prompt.", True),),
                mutates_workspace=False,
            )
        case _:
            raise AssertionError(f"No tool definition for tool type: {type(tool).__name__}")


def _single_path_definition(
    name: ToolName, description: str, mutates_workspace: bool
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        fields=(ToolFieldDefinition("path", "Workspace-relative path.", True),),
        mutates_workspace=mutates_workspace,
    )
