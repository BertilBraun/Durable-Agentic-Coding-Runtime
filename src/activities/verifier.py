from __future__ import annotations

import shlex

from src.activities.reproduction import repro_run_tests
from src.activities.workspace_manager import (
    ToolExecutionRequest,
    ToolResult,
    Workspace,
    run_tool,
)
from src.models.repo import RepoIndex
from src.models.reproduction import ReproductionContext
from src.models.worker import TestResult
from src.tools.definitions import RunShell, RunTests

_ANCHOR_TEST_TIMEOUT_SECONDS = 600


async def run_anchor_tests(
    workspace: Workspace,
    repo_index: RepoIndex,
    reproduction: ReproductionContext,
    restore_regression_files: bool = True,
) -> list[TestResult]:
    """Run the reproduction target plus the repo regression files as ground truth.

    When ``restore_regression_files`` is set, regression files are reverted to their
    committed content first so an agent edit (via any tool, including the shell) cannot
    mask a regression. This must only happen once the agent loop is finished, since a
    mid-run revert would silently change the model's assumptions.
    """
    if restore_regression_files:
        await _restore_regression_files(workspace, repo_index, reproduction.regression_test_files)
    results: list[TestResult] = []
    repro_result = await _run(workspace, repo_index, repro_run_tests(reproduction.repro_target))
    results.append(_test_result(reproduction.repro_target, repro_result, len(results) + 1))
    if reproduction.regression_test_files:
        regression_result = await _run(
            workspace,
            repo_index,
            RunTests(
                test_targets=list(reproduction.regression_test_files),
                timeout_seconds=_ANCHOR_TEST_TIMEOUT_SECONDS,
            ),
        )
        command = ' '.join(reproduction.regression_test_files)
        results.append(_test_result(command, regression_result, len(results) + 1))
    return results


async def _restore_regression_files(
    workspace: Workspace,
    repo_index: RepoIndex,
    regression_test_files: list[str],
) -> None:
    if not regression_test_files:
        return
    quoted = ' '.join(shlex.quote(path) for path in regression_test_files)
    await _run(
        workspace,
        repo_index,
        RunShell(command=f'git checkout HEAD -- {quoted}', timeout_seconds=60),
    )


async def _run(
    workspace: Workspace,
    repo_index: RepoIndex,
    tool: RunTests | RunShell,
) -> ToolResult:
    return await run_tool(
        ToolExecutionRequest(workspace=workspace, tool=tool, repo_index=repo_index)
    )


def _test_result(command: str, tool_result: ToolResult, sequence: int) -> TestResult:
    return TestResult(
        sequence=sequence,
        command=command,
        exit_code=tool_result.exit_code,
        stdout_summary=tool_result.stdout,
        stderr_summary=tool_result.stderr,
        passed=tool_result.exit_code == 0,
    )
