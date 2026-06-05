from __future__ import annotations

import shlex
from pathlib import PurePosixPath

from src.activities.workspace_manager import (
    ToolExecutionRequest,
    ToolResult,
    Workspace,
    run_tool,
)
from src.models.repo import RepoIndex
from src.tools.definitions import ApplyPatch, RunShell, Tool, WriteFile, WriteRegression

_PROTECTION_TIMEOUT_SECONDS = 60


def tool_mutates_workspace(tool: Tool) -> bool:
    return isinstance(tool, WriteFile | WriteRegression | ApplyPatch | RunShell)


def is_test_path(path: str) -> bool:
    pure_path = PurePosixPath(path)
    if 'tests' in pure_path.parts or 'test' in pure_path.parts:
        return True
    return pure_path.name.startswith('test_') or pure_path.name.endswith('_test.py')


async def revert_unauthorized_test_edits(
    workspace: Workspace,
    repo_index: RepoIndex,
    allowed_test_files: set[str],
    revert_untracked: bool,
) -> str | None:
    """Revert edits to existing test files, returning a note when anything was reverted.

    ``allowed_test_files`` are exempt. ``revert_untracked`` also removes new test files
    (the implementer may only touch the reproduction file); the reproducer keeps the new
    test it is creating, so it reverts modifications to tracked tests only.
    """
    status = await _run_shell(workspace, repo_index, 'git status --porcelain')
    modified: list[str] = []
    untracked: list[str] = []
    for line in status.stdout.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if not is_test_path(path) or path in allowed_test_files:
            continue
        if line[:2] == '??':
            if revert_untracked:
                untracked.append(path)
        else:
            modified.append(path)
    if not modified and not untracked:
        return None
    commands: list[str] = []
    if modified:
        commands.append('git checkout -- ' + ' '.join(shlex.quote(path) for path in modified))
    if untracked:
        commands.append('git clean -f -- ' + ' '.join(shlex.quote(path) for path in untracked))
    await _run_shell(workspace, repo_index, '; '.join(commands))
    reverted = ', '.join(modified + untracked)
    return (
        f'BLOCKED: reverted unauthorized changes to existing test files ({reverted}). '
        'You may not modify or delete existing test files; fix the production code instead.'
    )


async def _run_shell(workspace: Workspace, repo_index: RepoIndex, command: str) -> ToolResult:
    return await run_tool(
        ToolExecutionRequest(
            workspace=workspace,
            tool=RunShell(command=command, timeout_seconds=_PROTECTION_TIMEOUT_SECONDS),
            repo_index=repo_index,
        )
    )
