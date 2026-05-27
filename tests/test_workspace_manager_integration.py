import os
import subprocess
from pathlib import Path

import pytest
from src.activities.workspace_manager import create_workspace, destroy_workspace, run_tool
from src.tools.definitions import ReadFileRange, WriteFile

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_DOCKER_TESTS") != "1",
    reason="Set RUN_DOCKER_TESTS=1 to run Docker workspace integration tests.",
)
@pytest.mark.asyncio
async def test_workspace_runs_tools_in_fresh_container(tmp_path: Path) -> None:
    source_repository_path = tmp_path / "source"
    source_repository_path.mkdir()
    (source_repository_path / "hello.txt").write_text("hello\n", encoding="utf-8")
    _run_git(source_repository_path, "init")
    _run_git(source_repository_path, "config", "user.email", "test@example.com")
    _run_git(source_repository_path, "config", "user.name", "Test User")
    _run_git(source_repository_path, "add", ".")
    _run_git(source_repository_path, "commit", "-m", "initial")

    workspace_info = await create_workspace("integration-test", str(source_repository_path))
    try:
        read_result = await run_tool(
            workspace_info,
            ReadFileRange(file_path="hello.txt", start_line=1, end_line=1),
        )
        write_result = await run_tool(
            workspace_info,
            WriteFile(file_path="created.txt", content="created\n"),
        )
    finally:
        await destroy_workspace(workspace_info)

    assert read_result.stdout == "hello\n"
    assert write_result.exit_code == 0


def _run_git(repository_path: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    )
