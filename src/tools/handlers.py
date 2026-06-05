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
    ReadFile,
    RunShell,
    RunTests,
    Tool,
    WriteFile,
    WriteRegression,
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
        case WriteRegression(file_path=file_path, content=content):
            encoded_content = base64.b64encode(content.encode('utf-8')).decode('ascii')
            quoted_path = shlex.quote(file_path)
            write_command = (
                f'if [ -e {quoted_path} ]; then '
                'echo "regression test file already exists" >&2; exit 1; fi; '
                f'mkdir -p $(dirname {quoted_path}) && '
                f'printf %s {encoded_content} | base64 -d > {quoted_path}'
            )
            return ['sh', '-lc', write_command]
        case ApplyPatch(patch=patch):
            encoded_patch = base64.b64encode(patch.encode('utf-8')).decode('ascii')
            return ['sh', '-lc', f'printf %s {encoded_patch} | base64 -d | git apply -']
        case RunTests(test_targets=test_targets):
            return workspace.shell_invocation(pytest_command(test_targets))
        case RunShell(command=command):
            return workspace.shell_invocation(command)
        case ReadFile(file_path=file_path, start_line=start_line, end_line=end_line):
            return workspace.shell_invocation(_read_file_command(file_path, start_line, end_line))
        case FindDefinition() | FindCallers() | FindCallees():
            raise AssertionError('Index tools must be served from the repo index, not a command')
        case _:
            raise ValueError(f'No command defined for tool: {type(tool).__name__}')


def pytest_command(test_targets: list[str]) -> str:
    return ' '.join(['python', '-m', 'pytest', *(shlex.quote(target) for target in test_targets)])


def _read_file_command(file_path: str, start_line: int, end_line: int | None) -> str:
    path_argument = shlex.quote(file_path)
    start_argument = str(max(1, start_line))
    end_argument = '' if end_line is None else str(max(start_line, end_line))
    code = (
        'from pathlib import Path; import sys; '
        'path=Path(sys.argv[1]); start=int(sys.argv[2]); '
        'end=int(sys.argv[3]) if sys.argv[3] else None; '
        "lines=path.read_text(encoding='utf-8', errors='replace').splitlines(True); "
        'sys.stdout.write("".join(lines[start-1:end]))'
    )
    return (
        f'python -c {shlex.quote(code)} {path_argument} '
        f'{shlex.quote(start_argument)} {shlex.quote(end_argument)}'
    )


def _validate_tool_paths(tool: Tool) -> None:
    match tool:
        case WriteFile(file_path=path):
            _validate_workspace_relative_path(path)
        case WriteRegression(file_path=path):
            _validate_workspace_relative_path(path)
        case ReadFile(file_path=path):
            _validate_workspace_relative_path(path)
        case RunTests(test_targets=test_targets):
            for target in test_targets:
                _validate_workspace_relative_path(target.split('::', 1)[0])
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
