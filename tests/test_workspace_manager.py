from pathlib import Path

import pytest
from src.activities.workspace_manager import ToolExecutionRequest, WorkspaceInfo, run_tool
from src.models.context import ArtifactKind
from src.models.repo import FileEntry, Language, RepoIndex, Symbol, SymbolKind
from src.tools.definitions import FindReferences, FindSymbol, GitStatus, RunTests


class FakeContainer:
    def __init__(self, stdout: bytes = b"ok\n", stderr: bytes = b"") -> None:
        self.timeout_seconds: int | None = None
        self.removed = False
        self.stdout = stdout
        self.stderr = stderr

    def wait(self, timeout: int | None = None) -> dict[str, int]:
        self.timeout_seconds = timeout
        return {"StatusCode": 0}

    def logs(self, stdout: bool, stderr: bool) -> bytes:
        if stdout:
            return self.stdout
        return self.stderr

    def remove(self, force: bool) -> None:
        self.removed = force


class WaitFailingContainer(FakeContainer):
    def wait(self, timeout: int | None = None) -> dict[str, int]:
        self.timeout_seconds = timeout
        raise TimeoutError("container timed out")


class FakeContainers:
    def __init__(self, container: FakeContainer) -> None:
        self.container = container

    def run(self, **keyword_arguments: object) -> FakeContainer:
        return self.container


class FakeDockerClient:
    def __init__(self, container: FakeContainer) -> None:
        self.containers = FakeContainers(container)


class CyclingContainers:
    def __init__(self, containers: list[FakeContainer]) -> None:
        self.containers = containers

    def run(self, **keyword_arguments: object) -> FakeContainer:
        return self.containers.pop(0)


class CyclingDockerClient:
    def __init__(self, containers: list[FakeContainer]) -> None:
        self.containers = CyclingContainers(containers)


@pytest.mark.asyncio
async def test_run_tool_applies_tool_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    container = FakeContainer()
    monkeypatch.setattr(
        "src.activities.workspace_manager._docker_client",
        lambda: FakeDockerClient(container),
    )

    result = await run_tool(
        ToolExecutionRequest(
            workspace_info=WorkspaceInfo(
                run_id="run-1",
                volume_name="volume",
                worktree_path="workspace",
                branch_name="branch",
            ),
            tool=RunTests(command="pytest", timeout_seconds=17),
        ),
    )

    assert result.exit_code == 0
    assert container.timeout_seconds == 17
    assert container.removed is True


@pytest.mark.asyncio
async def test_run_tool_removes_container_when_wait_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = WaitFailingContainer()
    monkeypatch.setattr(
        "src.activities.workspace_manager._docker_client",
        lambda: FakeDockerClient(container),
    )

    with pytest.raises(TimeoutError, match="container timed out"):
        await run_tool(
            ToolExecutionRequest(
                workspace_info=WorkspaceInfo(
                    run_id="run-1",
                    volume_name="volume",
                    worktree_path="workspace",
                    branch_name="branch",
                ),
                tool=RunTests(command="pytest", timeout_seconds=17),
            ),
        )

    assert container.timeout_seconds == 17
    assert container.removed is True


@pytest.mark.asyncio
async def test_run_tool_writes_large_stdout_to_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    container = FakeContainer(stdout=b"x" * 20_001)
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setattr(
        "src.activities.workspace_manager._docker_client",
        lambda: FakeDockerClient(container),
    )

    result = await run_tool(
        ToolExecutionRequest(
            workspace_info=WorkspaceInfo(
                run_id="run-large",
                volume_name="volume",
                worktree_path="workspace",
                branch_name="branch",
            ),
            tool=RunTests(command="pytest", timeout_seconds=17),
        ),
    )

    assert result.truncated is True
    assert len(result.stdout) < 1_000
    assert len(result.artifacts) == 1
    artifact_reference = result.artifacts[0]
    assert artifact_reference.kind == ArtifactKind.TEST_OUTPUT
    artifact_path = Path(artifact_reference.path)
    assert artifact_path.parent.name == "run-large"
    assert artifact_path.name.startswith("run_tests-")
    assert artifact_path.name.endswith("-stdout.log")
    assert artifact_path.read_text(encoding="utf-8") == "x" * 20_001


@pytest.mark.asyncio
async def test_run_tool_writes_large_stderr_to_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    container = FakeContainer(stderr=b"e" * 20_001)
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setattr(
        "src.activities.workspace_manager._docker_client",
        lambda: FakeDockerClient(container),
    )

    result = await run_tool(
        ToolExecutionRequest(
            workspace_info=WorkspaceInfo(
                run_id="run-large",
                volume_name="volume",
                worktree_path="workspace",
                branch_name="branch",
            ),
            tool=GitStatus(path="."),
        ),
    )

    assert result.truncated is True
    assert len(result.stderr) < 1_000
    assert len(result.artifacts) == 1
    artifact_reference = result.artifacts[0]
    assert artifact_reference.kind == ArtifactKind.LOG
    artifact_path = Path(artifact_reference.path)
    assert artifact_path.parent.name == "run-large"
    assert artifact_path.name.startswith("git_status-")
    assert artifact_path.name.endswith("-stderr.log")
    assert artifact_path.read_text(encoding="utf-8") == "e" * 20_001


@pytest.mark.asyncio
async def test_run_tool_large_output_artifacts_do_not_collide_within_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    containers = [
        FakeContainer(stdout=b"a" * 20_001),
        FakeContainer(stdout=b"b" * 20_001),
    ]
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setattr(
        "src.activities.workspace_manager._docker_client",
        lambda: CyclingDockerClient(containers),
    )
    workspace_info = WorkspaceInfo(
        run_id="run-large",
        volume_name="volume",
        worktree_path="workspace",
        branch_name="branch",
    )

    first_result = await run_tool(
        ToolExecutionRequest(
            workspace_info=workspace_info,
            tool=RunTests(command="pytest tests/test_a.py", timeout_seconds=17),
        ),
    )
    second_result = await run_tool(
        ToolExecutionRequest(
            workspace_info=workspace_info,
            tool=RunTests(command="pytest tests/test_b.py", timeout_seconds=17),
        ),
    )

    first_path = Path(first_result.artifacts[0].path)
    second_path = Path(second_result.artifacts[0].path)
    assert first_path != second_path
    assert first_path.read_text(encoding="utf-8") == "a" * 20_001
    assert second_path.read_text(encoding="utf-8") == "b" * 20_001


def test_tool_execution_request_preserves_tool_type_after_json_round_trip() -> None:
    request = ToolExecutionRequest(
        workspace_info=WorkspaceInfo(
            run_id="run-1",
            volume_name="volume",
            worktree_path="workspace",
            branch_name="branch",
        ),
        tool=GitStatus(path="."),
    )

    restored_request = ToolExecutionRequest.model_validate(request.model_dump(mode="json"))

    assert restored_request.tool == GitStatus(path=".")


@pytest.mark.asyncio
async def test_run_tool_finds_python_symbol_from_repo_index(tmp_path: Path) -> None:
    repository_index = RepoIndex(
        symbols=[
            Symbol(
                name="Parser",
                kind=SymbolKind.CLASS,
                file_path="src/parser.py",
                start_line=3,
                end_line=8,
                language=Language.PYTHON,
            )
        ]
    )

    result = await run_tool(
        ToolExecutionRequest(
            workspace_info=WorkspaceInfo(
                run_id="run-1",
                volume_name="volume",
                worktree_path=str(tmp_path),
                branch_name="branch",
            ),
            tool=FindSymbol(name="Parser", language="python"),
            repo_index=repository_index,
        )
    )

    assert result.exit_code == 0
    assert "src/parser.py:3-8 class Parser" in result.stdout


@pytest.mark.asyncio
async def test_run_tool_finds_tsx_references_from_indexed_files(tmp_path: Path) -> None:
    component_path = tmp_path / "src" / "component.tsx"
    component_path.parent.mkdir()
    component_path.write_text(
        "export function Widget() {\n"
        "  return null;\n"
        "}\n"
        "\n"
        "export function App() {\n"
        "  return <Widget />;\n"
        "}\n",
        encoding="utf-8",
    )
    repository_index = RepoIndex(
        file_tree=[
            FileEntry(
                path="src/component.tsx",
                language=Language.TSX,
                size_bytes=component_path.stat().st_size,
            )
        ],
        symbols=[
            Symbol(
                name="Widget",
                kind=SymbolKind.FUNCTION,
                file_path="src/component.tsx",
                start_line=1,
                end_line=3,
                language=Language.TSX,
            )
        ],
    )

    result = await run_tool(
        ToolExecutionRequest(
            workspace_info=WorkspaceInfo(
                run_id="run-1",
                volume_name="volume",
                worktree_path=str(tmp_path),
                branch_name="branch",
            ),
            tool=FindReferences(symbol_name="Widget", file_path="src/component.tsx"),
            repo_index=repository_index,
        )
    )

    assert result.exit_code == 0
    assert "src/component.tsx:6:  return <Widget />;" in result.stdout
