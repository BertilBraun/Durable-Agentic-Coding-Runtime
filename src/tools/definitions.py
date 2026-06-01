from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import Field, TypeAdapter

from src.models.frozen_base_model import FrozenBaseModel
from src.runtime_enums import StrEnum


class ToolName(StrEnum):
    WRITE_FILE = 'write_file'
    APPLY_PATCH = 'apply_patch'
    RUN_TESTS = 'run_tests'
    RUN_LINT = 'run_lint'
    RUN_SHELL = 'run_shell'
    FIND_DEFINITION = 'find_definition'
    FIND_CALLERS = 'find_callers'
    FIND_CALLEES = 'find_callees'
    GATHER_CONTEXT = 'gather_context'


class ToolBase(FrozenBaseModel):
    pass


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


class RunShell(ToolBase):
    tool_name: Literal[ToolName.RUN_SHELL] = Field(
        default=ToolName.RUN_SHELL,
        description='Tool name tag.',
    )
    command: str = Field(description='Shell command to run from the repository root.')
    timeout_seconds: int = Field(description='Maximum seconds to allow the command to run.')


class FindDefinition(ToolBase):
    tool_name: Literal[ToolName.FIND_DEFINITION] = Field(
        default=ToolName.FIND_DEFINITION,
        description='Tool name tag.',
    )
    name: str = Field(description='Exact symbol name to resolve to its definition site(s).')
    language: str = Field(description='Source language, for example "python" or "javascript".')


class FindCallers(ToolBase):
    tool_name: Literal[ToolName.FIND_CALLERS] = Field(
        default=ToolName.FIND_CALLERS,
        description='Tool name tag.',
    )
    symbol_name: str = Field(description='Exact symbol name whose call sites should be found.')


class FindCallees(ToolBase):
    tool_name: Literal[ToolName.FIND_CALLEES] = Field(
        default=ToolName.FIND_CALLEES,
        description='Tool name tag.',
    )
    file_path: str = Field(description='Workspace-relative file path where the symbol is defined.')
    symbol_name: str = Field(description='Exact symbol name whose callees should be found.')


class GatherContext(ToolBase):
    tool_name: Literal[ToolName.GATHER_CONTEXT] = Field(
        default=ToolName.GATHER_CONTEXT,
        description='Focused read-only context request for the context gatherer.',
    )
    prompt: str = Field(description='Focused read-only context request for the context gatherer.')


Tool = (
    WriteFile
    | ApplyPatch
    | RunTests
    | RunLint
    | RunShell
    | FindDefinition
    | FindCallers
    | FindCallees
    | GatherContext
)

ImplementationToolCall: TypeAlias = Tool

ContextGathererToolCall: TypeAlias = RunShell | FindDefinition | FindCallers | FindCallees

ImplementationToolCallAdapter: TypeAdapter[ImplementationToolCall] = TypeAdapter(
    ImplementationToolCall
)
ContextGathererToolCallAdapter: TypeAdapter[ContextGathererToolCall] = TypeAdapter(
    ContextGathererToolCall
)
