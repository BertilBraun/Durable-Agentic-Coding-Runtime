from __future__ import annotations

import base64
import shlex
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

from src.tools.definitions import (
    ApplyPatch,
    FindCallees,
    FindCallers,
    FindDefinition,
    RunLint,
    RunShell,
    RunTests,
    Tool,
    WriteFile,
)

if TYPE_CHECKING:
    from src.activities.workspace_manager import Workspace


def command_for_tool(tool: Tool, workspace: Workspace) -> list[str]:
    _validate_tool_paths(tool)
    match tool:
        case WriteFile(file_path=file_path, content=content):
            encoded_content = base64.b64encode(content.encode('utf-8')).decode('ascii')
            quoted_path = shlex.quote(file_path)
            write_command = (
                f'mkdir -p $(dirname {quoted_path}) && '
                f'printf %s {encoded_content} | base64 -d > {quoted_path}'
            )
            return ['sh', '-lc', write_command]
        case ApplyPatch(patch=patch):
            encoded_patch = base64.b64encode(patch.encode('utf-8')).decode('ascii')
            return ['sh', '-lc', f'printf %s {encoded_patch} | base64 -d | git apply -']
        case RunTests(command=command, directory=directory):
            quoted_directory = shlex.quote(directory)
            return ['sh', '-lc', f'cd {quoted_directory} && {command}']
        case RunLint(path=path):
            return ['ruff', 'check', path]
        case RunShell(command=command):
            return workspace.shell_invocation(command)
        case FindDefinition() | FindCallers() | FindCallees():
            raise AssertionError('Index tools must be served from the repo index, not a command')
        case _:
            raise ValueError(f'No command defined for tool: {type(tool).__name__}')


def _validate_tool_paths(tool: Tool) -> None:
    match tool:
        case RunLint(path=path) | WriteFile(file_path=path) | RunTests(directory=path):
            _validate_workspace_relative_path(path)
        case _:
            return


def _validate_workspace_relative_path(path: str) -> None:
    if path in ('', '.'):
        return
    posix_path = PurePosixPath(path)
    windows_path = PureWindowsPath(path)
    has_parent_traversal = '..' in posix_path.parts or '..' in windows_path.parts
    if posix_path.is_absolute() or windows_path.is_absolute() or has_parent_traversal:
        raise ValueError(f'Path must be workspace-relative and cannot escape workspace: {path}')
