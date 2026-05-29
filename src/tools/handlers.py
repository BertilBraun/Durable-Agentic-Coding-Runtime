from __future__ import annotations

import base64
import shlex
from pathlib import PurePosixPath, PureWindowsPath

from src.tools.definitions import (
    ApplyPatch,
    FindReferences,
    FindSymbol,
    GitCommit,
    GitDiff,
    GitStatus,
    ReadFileRange,
    RunLint,
    RunTests,
    RunTypecheck,
    SearchText,
    Tool,
    WriteFile,
)


def command_for_tool(tool: Tool) -> list[str]:
    _validate_tool_paths(tool)
    match tool:
        case ReadFileRange(file_path=file_path, start_line=start_line, end_line=end_line):
            quoted_path = shlex.quote(file_path)
            return ['sh', '-lc', f"sed -n '{start_line},{end_line}p' {quoted_path}"]
        case SearchText(pattern=pattern, directory=directory, file_glob=file_glob):
            return [
                'sh',
                '-lc',
                f'rg {shlex.quote(pattern)} {shlex.quote(directory)} -g {shlex.quote(file_glob)}',
            ]
        case WriteFile(file_path=file_path, content=content):
            encoded_content = base64.b64encode(content.encode('utf-8')).decode('ascii')
            quoted_path = shlex.quote(file_path)
            write_command = f'mkdir -p $(dirname {quoted_path}) && printf %s {encoded_content} | base64 -d > {quoted_path}'
            return [
                'sh',
                '-lc',
                write_command,
            ]
        case ApplyPatch(patch=patch):
            encoded_patch = base64.b64encode(patch.encode('utf-8')).decode('ascii')
            return ['sh', '-lc', f'printf %s {encoded_patch} | base64 -d | git apply -']
        case GitDiff(path=path):
            return ['git', 'diff', '--', path]
        case GitStatus(path=path):
            return ['git', 'status', '--short', '--', path]
        case GitCommit(message=message):
            return ['sh', '-lc', f'git add -A && git commit -m {shlex.quote(message)}']
        case RunTests(command=command, directory=directory):
            quoted_directory = shlex.quote(directory)
            return ['sh', '-lc', f'cd {quoted_directory} && {command}']
        case RunLint(path=path):
            return ['ruff', 'check', path]
        case RunTypecheck(path=path):
            return ['sh', '-lc', f'python -m mypy {shlex.quote(path)}']
        case FindSymbol():
            raise AssertionError('FindSymbol must be served from the repo index, not via Docker')
        case FindReferences(symbol_name=symbol_name):
            return ['rg', symbol_name, '.']
        case _:
            raise ValueError(f'No command defined for tool: {type(tool).__name__}')


def _validate_tool_paths(tool: Tool) -> None:
    match tool:
        case (
            ReadFileRange(file_path=path)
            | GitStatus(path=path)
            | GitDiff(path=path)
            | RunLint(path=path)
            | RunTypecheck(path=path)
            | WriteFile(file_path=path)
            | FindReferences(file_path=path)
            | RunTests(directory=path)
            | SearchText(directory=path)
        ):
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
