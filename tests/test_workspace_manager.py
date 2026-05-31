from pathlib import Path

import pytest
from src.activities.workspace_manager import (
    CommandResult,
    HostWorkspace,
    ToolExecutionRequest,
    run_tool,
)
from src.config import CONFIG
from src.models.context import ArtifactKind
from src.models.repo import FileEntry, Language, RepoIndex, Symbol, SymbolKind
from src.tools.definitions import FindReferences, FindSymbol, GitStatus, RunTests, ToolName


def _host_workspace(run_id: str = 'run-1', repo_path: str = 'workspace') -> HostWorkspace:
    return HostWorkspace(
        run_id=run_id,
        base_sha='basesha',
        base_branch='main',
        current_branch='agentic/run-1/cand-0',
        repo_path=repo_path,
    )


@pytest.mark.asyncio
async def test_run_tool_passes_tool_timeout_to_run_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, int | None] = {}

    def fake_run_command(
        self: HostWorkspace, command: list[str], timeout: int | None = None
    ) -> CommandResult:
        captured['timeout'] = timeout
        return CommandResult(stdout='ok\n', stderr='', exit_code=0)

    monkeypatch.setattr(HostWorkspace, 'run_command', fake_run_command)

    result = await run_tool(
        ToolExecutionRequest(
            workspace=_host_workspace(),
            tool=RunTests(command='pytest', timeout_seconds=17, directory='.'),
        ),
    )

    assert result.exit_code == 0
    assert result.tool_name == ToolName.RUN_TESTS
    assert captured['timeout'] == 17


@pytest.mark.asyncio
async def test_run_tool_propagates_run_command_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_command(
        self: HostWorkspace, command: list[str], timeout: int | None = None
    ) -> CommandResult:
        raise TimeoutError('command timed out')

    monkeypatch.setattr(HostWorkspace, 'run_command', fake_run_command)

    with pytest.raises(TimeoutError, match='command timed out'):
        await run_tool(
            ToolExecutionRequest(
                workspace=_host_workspace(),
                tool=RunTests(command='pytest', timeout_seconds=17, directory='.'),
            ),
        )


@pytest.mark.asyncio
async def test_run_tool_writes_large_stdout_to_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('ARTIFACTS_ROOT', str(tmp_path / 'artifacts'))

    def fake_run_command(
        self: HostWorkspace, command: list[str], timeout: int | None = None
    ) -> CommandResult:
        return CommandResult(stdout='x' * 20_001, stderr='', exit_code=0)

    monkeypatch.setattr(HostWorkspace, 'run_command', fake_run_command)

    result = await run_tool(
        ToolExecutionRequest(
            workspace=_host_workspace(run_id='run-large'),
            tool=RunTests(command='pytest', timeout_seconds=17, directory='.'),
        ),
    )

    assert result.truncated is True
    assert len(result.stdout) < CONFIG.tool_output_max_characters
    assert 'characters omitted' in result.stdout
    assert len(result.artifacts) == 1
    artifact_reference = result.artifacts[0]
    assert artifact_reference.kind == ArtifactKind.TEST_OUTPUT
    artifact_path = Path(artifact_reference.path)
    assert artifact_path.parent.name == 'run-large'
    assert artifact_path.name.startswith('run_tests-')
    assert artifact_path.name.endswith('-stdout.log')
    assert artifact_path.read_text(encoding='utf-8') == 'x' * 20_001


@pytest.mark.asyncio
async def test_run_tool_writes_large_stderr_to_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('ARTIFACTS_ROOT', str(tmp_path / 'artifacts'))

    def fake_run_command(
        self: HostWorkspace, command: list[str], timeout: int | None = None
    ) -> CommandResult:
        return CommandResult(stdout='', stderr='e' * 20_001, exit_code=0)

    monkeypatch.setattr(HostWorkspace, 'run_command', fake_run_command)

    result = await run_tool(
        ToolExecutionRequest(
            workspace=_host_workspace(run_id='run-large'),
            tool=GitStatus(path='.'),
        ),
    )

    assert result.truncated is True
    assert len(result.stderr) < CONFIG.tool_output_max_characters
    assert 'characters omitted' in result.stderr
    assert len(result.artifacts) == 1
    artifact_reference = result.artifacts[0]
    assert artifact_reference.kind == ArtifactKind.LOG
    artifact_path = Path(artifact_reference.path)
    assert artifact_path.parent.name == 'run-large'
    assert artifact_path.name.startswith('git_status-')
    assert artifact_path.name.endswith('-stderr.log')
    assert artifact_path.read_text(encoding='utf-8') == 'e' * 20_001


@pytest.mark.asyncio
async def test_run_tool_large_output_artifacts_do_not_collide_within_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('ARTIFACTS_ROOT', str(tmp_path / 'artifacts'))
    outputs = ['a' * 20_001, 'b' * 20_001]

    def fake_run_command(
        self: HostWorkspace, command: list[str], timeout: int | None = None
    ) -> CommandResult:
        return CommandResult(stdout=outputs.pop(0), stderr='', exit_code=0)

    monkeypatch.setattr(HostWorkspace, 'run_command', fake_run_command)
    workspace = _host_workspace(run_id='run-large')

    first_result = await run_tool(
        ToolExecutionRequest(
            workspace=workspace,
            tool=RunTests(command='pytest tests/test_a.py', timeout_seconds=17, directory='.'),
        ),
    )
    second_result = await run_tool(
        ToolExecutionRequest(
            workspace=workspace,
            tool=RunTests(command='pytest tests/test_b.py', timeout_seconds=17, directory='.'),
        ),
    )

    first_path = Path(first_result.artifacts[0].path)
    second_path = Path(second_result.artifacts[0].path)
    assert first_path != second_path
    assert first_path.read_text(encoding='utf-8') == 'a' * 20_001
    assert second_path.read_text(encoding='utf-8') == 'b' * 20_001


def test_tool_execution_request_preserves_tool_type_after_json_round_trip() -> None:
    request = ToolExecutionRequest(
        workspace=_host_workspace(),
        tool=GitStatus(path='.'),
    )

    restored_request = ToolExecutionRequest.model_validate(request.model_dump(mode='json'))

    assert restored_request.tool == GitStatus(path='.')


def test_tool_execution_request_preserves_workspace_subclass_after_round_trip() -> None:
    request = ToolExecutionRequest(
        workspace=_host_workspace(),
        tool=GitStatus(path='.'),
    )

    restored_request = ToolExecutionRequest.model_validate(request.model_dump(mode='json'))

    assert isinstance(restored_request.workspace, HostWorkspace)
    assert restored_request.workspace.repo_path == 'workspace'


@pytest.mark.asyncio
async def test_run_tool_finds_python_symbol_from_repo_index() -> None:
    repository_index = RepoIndex(
        symbols=[
            Symbol(
                name='Parser',
                kind=SymbolKind.CLASS,
                file_path='src/parser.py',
                start_line=3,
                end_line=8,
                language=Language.PYTHON,
            )
        ]
    )

    result = await run_tool(
        ToolExecutionRequest(
            workspace=_host_workspace(),
            tool=FindSymbol(name='Parser', language='python'),
            repo_index=repository_index,
        )
    )

    assert result.exit_code == 0
    assert 'src/parser.py:3-8 class Parser' in result.stdout


@pytest.mark.asyncio
async def test_run_tool_finds_tsx_references_from_indexed_files(tmp_path: Path) -> None:
    component_path = tmp_path / 'src' / 'component.tsx'
    component_path.parent.mkdir()
    component_path.write_text(
        'export function Widget() {\n'
        '  return null;\n'
        '}\n'
        '\n'
        'export function App() {\n'
        '  return <Widget />;\n'
        '}\n',
        encoding='utf-8',
    )
    repository_index = RepoIndex(
        file_tree=[
            FileEntry(
                path='src/component.tsx',
                language=Language.TSX,
                size_bytes=component_path.stat().st_size,
            )
        ],
        symbols=[
            Symbol(
                name='Widget',
                kind=SymbolKind.FUNCTION,
                file_path='src/component.tsx',
                start_line=1,
                end_line=3,
                language=Language.TSX,
            )
        ],
    )

    result = await run_tool(
        ToolExecutionRequest(
            workspace=_host_workspace(repo_path=str(tmp_path)),
            tool=FindReferences(symbol_name='Widget', file_path='src/component.tsx'),
            repo_index=repository_index,
        )
    )

    assert result.exit_code == 0
    assert 'src/component.tsx:6:  return <Widget />;' in result.stdout
