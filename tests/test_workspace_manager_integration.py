import os
import subprocess
from pathlib import Path

import pytest
from src.activities.workspace_manager import (
    ToolExecutionRequest,
    run_tool,
    setup_environment,
    teardown_environment,
)
from src.models.task import HostOrigin
from src.tools.definitions import ReadFileRange, WriteFile

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv('RUN_HOST_TESTS') != '1',
    reason='Set RUN_HOST_TESTS=1 to run host workspace integration tests (requires a POSIX shell).',
)
@pytest.mark.asyncio
async def test_host_workspace_runs_tools_in_place(tmp_path: Path) -> None:
    repository_path = tmp_path / 'source'
    repository_path.mkdir()
    (repository_path / 'hello.txt').write_text('hello\n', encoding='utf-8')
    _run_git(repository_path, 'init')
    _run_git(repository_path, 'config', 'user.email', 'test@example.com')
    _run_git(repository_path, 'config', 'user.name', 'Test User')
    _run_git(repository_path, 'add', '.')
    _run_git(repository_path, 'commit', '-m', 'initial')

    workspace = await setup_environment(HostOrigin(repo_path=str(repository_path)), 'integration')
    candidate_workspace = workspace.begin_candidate(0)
    try:
        read_result = await run_tool(
            ToolExecutionRequest(
                workspace=candidate_workspace,
                tool=ReadFileRange(file_path='hello.txt', start_line=1, end_line=1),
            )
        )
        write_result = await run_tool(
            ToolExecutionRequest(
                workspace=candidate_workspace,
                tool=WriteFile(file_path='created.txt', content='created\n'),
            )
        )
    finally:
        await teardown_environment(candidate_workspace)

    assert read_result.stdout == 'hello\n'
    assert write_result.exit_code == 0
    assert (repository_path / 'created.txt').read_text(encoding='utf-8') == 'created\n'


def _run_git(repository_path: Path, *arguments: str) -> None:
    subprocess.run(
        ['git', *arguments],
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    )
