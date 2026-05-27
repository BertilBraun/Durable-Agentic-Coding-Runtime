from __future__ import annotations

import base64
import shlex
from pathlib import PurePosixPath, PureWindowsPath

from src.tools.definitions import (
    ApplyPatch,
    FindReferences,
    FindSymbol,
    GatherContext,
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
            return ["sh", "-lc", f"sed -n '{start_line},{end_line}p' {quoted_path}"]
        case SearchText(pattern=pattern, directory=directory, file_glob=file_glob):
            return [
                "sh",
                "-lc",
                f"rg {shlex.quote(pattern)} {shlex.quote(directory)} -g {shlex.quote(file_glob)}",
            ]
        case WriteFile(file_path=file_path, content=content):
            encoded_content = base64.b64encode(content.encode("utf-8")).decode("ascii")
            quoted_path = shlex.quote(file_path)
            write_command = (
                f"mkdir -p $(dirname {quoted_path}) && "
                f"printf %s {encoded_content} | base64 -d > {quoted_path}"
            )
            return [
                "sh",
                "-lc",
                write_command,
            ]
        case ApplyPatch(patch=patch):
            encoded_patch = base64.b64encode(patch.encode("utf-8")).decode("ascii")
            return ["sh", "-lc", f"printf %s {encoded_patch} | base64 -d | git apply -"]
        case GitDiff(path=path):
            return ["git", "diff", "--", path]
        case GitStatus(path=path):
            return ["git", "status", "--short", "--", path]
        case GitCommit(message=message):
            return ["sh", "-lc", f"git add -A && git commit -m {shlex.quote(message)}"]
        case RunTests(command=command):
            return ["sh", "-lc", command]
        case RunLint(path=path):
            return ["ruff", "check", path]
        case RunTypecheck(path=path):
            return ["sh", "-lc", f"python -m mypy {shlex.quote(path)}"]
        case FindSymbol(name=name):
            quoted_name = shlex.quote(name)
            return [
                "sh",
                "-lc",
                f"rg 'class {quoted_name}|def {quoted_name}|function {quoted_name}' .",
            ]
        case FindReferences(symbol_name=symbol_name):
            return ["rg", symbol_name, "."]
        case GatherContext(prompt=prompt):
            return ["sh", "-lc", f"printf %s {shlex.quote(prompt)}"]


def _validate_tool_paths(tool: Tool) -> None:
    match tool:
        case ReadFileRange(file_path=file_path):
            _validate_workspace_relative_path(file_path)
        case SearchText(directory=directory):
            _validate_workspace_relative_path(directory)
        case WriteFile(file_path=file_path):
            _validate_workspace_relative_path(file_path)
        case GitDiff(path=path):
            _validate_workspace_relative_path(path)
        case GitStatus(path=path):
            _validate_workspace_relative_path(path)
        case RunLint(path=path):
            _validate_workspace_relative_path(path)
        case RunTypecheck(path=path):
            _validate_workspace_relative_path(path)
        case FindReferences(file_path=file_path):
            _validate_workspace_relative_path(file_path)
        case _:
            return


def _validate_workspace_relative_path(path: str) -> None:
    if path in ("", "."):
        return
    posix_path = PurePosixPath(path)
    windows_path = PureWindowsPath(path)
    has_parent_traversal = ".." in posix_path.parts or ".." in windows_path.parts
    if posix_path.is_absolute() or windows_path.is_absolute() or has_parent_traversal:
        raise ValueError(f"Path must be workspace-relative and cannot escape workspace: {path}")
