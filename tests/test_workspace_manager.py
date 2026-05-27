import pytest
from src.activities.workspace_manager import WorkspaceInfo, run_tool
from src.tools.definitions import RunTests


class FakeContainer:
    def __init__(self) -> None:
        self.timeout_seconds: int | None = None
        self.removed = False

    def wait(self, timeout: int | None = None) -> dict[str, int]:
        self.timeout_seconds = timeout
        return {"StatusCode": 0}

    def logs(self, stdout: bool, stderr: bool) -> bytes:
        if stdout:
            return b"ok\n"
        return b""

    def remove(self, force: bool) -> None:
        self.removed = force


class FakeContainers:
    def __init__(self, container: FakeContainer) -> None:
        self.container = container

    def run(self, **keyword_arguments: object) -> FakeContainer:
        return self.container


class FakeDockerClient:
    def __init__(self, container: FakeContainer) -> None:
        self.containers = FakeContainers(container)


@pytest.mark.asyncio
async def test_run_tool_applies_tool_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    container = FakeContainer()
    monkeypatch.setattr(
        "src.activities.workspace_manager._docker_client",
        lambda: FakeDockerClient(container),
    )

    result = await run_tool(
        WorkspaceInfo(
            run_id="run-1",
            volume_name="volume",
            worktree_path="workspace",
            branch_name="branch",
        ),
        RunTests(command="pytest", timeout_seconds=17),
    )

    assert result.exit_code == 0
    assert container.timeout_seconds == 17
    assert container.removed is True
