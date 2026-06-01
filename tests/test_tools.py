import pytest
from src.activities import workspace_manager
from src.activities.workspace_manager import DockerWorkspace, HostWorkspace
from src.tools.definitions import (
    ApplyPatch,
    FindCallers,
    RunLint,
    RunShell,
    RunTests,
    ToolName,
    WriteFile,
)
from src.tools.handlers import command_for_tool


def _host_workspace() -> HostWorkspace:
    return HostWorkspace(
        run_id='run-1',
        base_sha='basesha',
        base_branch='main',
        current_branch='main',
        repo_path='workspace',
    )


def _docker_workspace() -> DockerWorkspace:
    return DockerWorkspace(
        run_id='run-1',
        base_sha='basesha',
        base_branch='main',
        current_branch='main',
        container_id='container-1',
        container_repo_path='/repo',
    )


def test_tool_exposes_stable_tool_name() -> None:
    run_shell = RunShell(command='ls', timeout_seconds=10)

    assert run_shell.tool_name == ToolName.RUN_SHELL


def test_docker_workspace_shell_invocation_activates_testbed_conda_env() -> None:
    command = command_for_tool(
        RunShell(command='git status', timeout_seconds=5), _docker_workspace()
    )

    assert command == [
        'sh',
        '-lc',
        'export PATH=/opt/miniconda3/envs/testbed/bin:$PATH && git status',
    ]


def test_host_workspace_shell_invocation_selects_shell_for_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workspace_manager, '_host_is_windows', lambda: False)
    posix_command = command_for_tool(RunShell(command='ls', timeout_seconds=5), _host_workspace())

    monkeypatch.setattr(workspace_manager, '_host_is_windows', lambda: True)
    windows_command = command_for_tool(RunShell(command='ls', timeout_seconds=5), _host_workspace())

    assert posix_command == ['sh', '-lc', 'ls']
    assert windows_command == ['powershell', '-NoProfile', '-Command', 'ls']


def test_tool_command_rejects_absolute_file_path() -> None:
    with pytest.raises(ValueError, match='workspace-relative'):
        command_for_tool(WriteFile(file_path='/etc/passwd', content='x'), _host_workspace())


def test_tool_command_rejects_parent_traversal_file_path() -> None:
    with pytest.raises(ValueError, match='workspace-relative'):
        command_for_tool(WriteFile(file_path='../outside.txt', content='escape'), _host_workspace())


def test_tool_command_rejects_parent_traversal_directory() -> None:
    with pytest.raises(ValueError, match='workspace-relative'):
        command_for_tool(
            RunTests(command='pytest', timeout_seconds=60, directory='src/../..'),
            _host_workspace(),
        )


def test_tool_command_allows_current_directory_path() -> None:
    command = command_for_tool(RunLint(path='.'), _host_workspace())

    assert command == ['ruff', 'check', '.']


def test_run_tests_command_runs_from_requested_directory() -> None:
    command = command_for_tool(
        RunTests(command='pytest -q', timeout_seconds=60, directory='examples/smoke'),
        _docker_workspace(),
    )

    assert command == [
        'sh',
        '-lc',
        'export PATH=/opt/miniconda3/envs/testbed/bin:$PATH && cd examples/smoke && pytest -q',
    ]


def test_run_tests_command_rejects_parent_traversal_directory() -> None:
    with pytest.raises(ValueError, match='workspace-relative'):
        command_for_tool(
            RunTests(command='pytest -q', timeout_seconds=60, directory='../outside'),
            _docker_workspace(),
        )


def test_apply_patch_tool_name() -> None:
    assert ApplyPatch(patch='--- a/file\n+++ b/file').tool_name == ToolName.APPLY_PATCH


def test_index_tool_must_be_served_from_index() -> None:
    with pytest.raises(AssertionError, match='repo index'):
        command_for_tool(FindCallers(symbol_name='handle'), _docker_workspace())
