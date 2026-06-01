from pathlib import Path

import docker
import pytest
from src.activities.workspace_manager import (
    CommandResult,
    ContextPackRequest,
    DockerWorkspace,
    HostWorkspace,
    ToolExecutionRequest,
    _setup_docker_workspace,
    pack_context,
    run_tool,
)
from src.config import CONFIG
from src.models.context import ArtifactKind, ContextSnippet
from src.models.repo import (
    Language,
    Reference,
    ReferenceKind,
    RepoIndex,
    Symbol,
    SymbolKind,
)
from src.tools.definitions import (
    FindCallers,
    FindDefinition,
    RunShell,
    RunTests,
    ToolName,
    WriteFile,
)


def _host_workspace(run_id: str = 'run-1', repo_path: str = 'workspace') -> HostWorkspace:
    return HostWorkspace(
        run_id=run_id,
        base_sha='basesha',
        base_branch='main',
        current_branch='agentic/run-1/cand-0',
        repo_path=repo_path,
    )


def test_begin_candidate_branches_off_candidate_base_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run_command(
        self: HostWorkspace, command: list[str], timeout: int | None = None
    ) -> CommandResult:
        commands.append(command)
        return CommandResult(stdout='', stderr='', exit_code=0)

    monkeypatch.setattr(HostWorkspace, 'run_command', fake_run_command)
    workspace = _host_workspace().model_copy(update={'candidate_base_sha': 'snapsha'})

    workspace.begin_candidate(1)

    assert commands == [['git', 'checkout', '-B', 'agentic/run-1/cand-1', 'snapsha']]


def test_begin_candidate_falls_back_to_base_sha_without_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run_command(
        self: HostWorkspace, command: list[str], timeout: int | None = None
    ) -> CommandResult:
        commands.append(command)
        return CommandResult(stdout='', stderr='', exit_code=0)

    monkeypatch.setattr(HostWorkspace, 'run_command', fake_run_command)

    _host_workspace().begin_candidate(0)

    assert commands == [['git', 'checkout', '-B', 'agentic/run-1/cand-0', 'basesha']]


def test_snapshot_candidate_base_commits_working_tree_off_base_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = {'git write-tree': 'treesha\n', 'git commit-tree': 'snapsha\n'}

    def fake_run_command(
        self: HostWorkspace, command: list[str], timeout: int | None = None
    ) -> CommandResult:
        for prefix, stdout in outputs.items():
            if ' '.join(command).startswith(prefix):
                return CommandResult(stdout=stdout, stderr='', exit_code=0)
        return CommandResult(stdout='', stderr='', exit_code=0)

    monkeypatch.setattr(HostWorkspace, 'run_command', fake_run_command)

    snapshot = _host_workspace().snapshot_candidate_base()

    assert snapshot.candidate_base_sha == 'snapsha'


def test_reset_to_base_targets_candidate_base(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def fake_run_command(
        self: HostWorkspace, command: list[str], timeout: int | None = None
    ) -> CommandResult:
        commands.append(command)
        return CommandResult(stdout='', stderr='', exit_code=0)

    monkeypatch.setattr(HostWorkspace, 'run_command', fake_run_command)
    workspace = _host_workspace().model_copy(update={'candidate_base_sha': 'snapsha'})

    workspace.reset_to_base()

    assert commands[0] == ['git', 'reset', '--hard', 'snapsha']


def test_setup_docker_workspace_uses_existing_local_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_calls: list[str] = []

    class FakeImages:
        def get(self, image_name: str) -> object:
            image_calls.append(f'get:{image_name}')
            return object()

        def pull(self, image_name: str) -> object:
            image_calls.append(f'pull:{image_name}')
            return object()

    class FakeContainers:
        def run(self, **keyword_arguments: object) -> object:
            assert keyword_arguments['image'] == 'sweb.eval.test:latest'
            return type('Container', (), {'id': 'container-1'})()

    class FakeDockerClient:
        images = FakeImages()
        containers = FakeContainers()

    def fake_run_command(
        self: DockerWorkspace, command: list[str], timeout: int | None = None
    ) -> CommandResult:
        command_text = ' '.join(command)
        if command_text == 'git status --porcelain':
            return CommandResult(stdout='', stderr='', exit_code=0)
        if command_text == 'git rev-parse HEAD':
            return CommandResult(stdout='basesha\n', stderr='', exit_code=0)
        if command_text == 'git symbolic-ref --short -q HEAD':
            return CommandResult(stdout='main\n', stderr='', exit_code=0)
        raise AssertionError(f'unexpected command: {command}')

    monkeypatch.setattr('src.activities.workspace_manager._docker_client', FakeDockerClient)
    monkeypatch.setattr(DockerWorkspace, 'run_command', fake_run_command)

    workspace = _setup_docker_workspace(
        docker_image='sweb.eval.test:latest',
        container_repo_path='/testbed',
        run_id='run-1',
    )

    assert image_calls == ['get:sweb.eval.test:latest']
    assert workspace.container_id == 'container-1'


def test_setup_docker_workspace_reports_missing_local_image_without_pull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_calls: list[str] = []

    class FakeImages:
        def get(self, image_name: str) -> object:
            image_calls.append(f'get:{image_name}')
            raise docker.errors.ImageNotFound('missing')

        def pull(self, image_name: str) -> object:
            image_calls.append(f'pull:{image_name}')
            return object()

    class FakeDockerClient:
        images = FakeImages()

    monkeypatch.setattr('src.activities.workspace_manager._docker_client', FakeDockerClient)

    with pytest.raises(RuntimeError, match='Build the SWE-bench image locally'):
        _setup_docker_workspace(
            docker_image='sweb.eval.x86_64.astropy__astropy-12907:latest',
            container_repo_path='/testbed',
            run_id='run-1',
        )

    assert image_calls == ['get:sweb.eval.x86_64.astropy__astropy-12907:latest']


@pytest.mark.asyncio
async def test_run_shell_dispatches_through_shell_invocation_with_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_command(
        self: HostWorkspace, command: list[str], timeout: int | None = None
    ) -> CommandResult:
        captured['command'] = command
        captured['timeout'] = timeout
        return CommandResult(stdout='listing\n', stderr='', exit_code=0)

    monkeypatch.setattr(HostWorkspace, 'run_command', fake_run_command)

    result = await run_tool(
        ToolExecutionRequest(
            workspace=_host_workspace(),
            tool=RunShell(command='ls -la', timeout_seconds=23),
        ),
    )

    assert result.exit_code == 0
    assert result.tool_name == ToolName.RUN_SHELL
    assert captured['command'] == _host_workspace().shell_invocation('ls -la')
    assert captured['timeout'] == 23


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
async def test_run_tool_writes_host_file_without_shell(tmp_path: Path) -> None:
    repository_path = tmp_path / 'repo'
    repository_path.mkdir()

    result = await run_tool(
        ToolExecutionRequest(
            workspace=_host_workspace(repo_path=str(repository_path)),
            tool=WriteFile(file_path='test_app.py', content='import unittest\n'),
        )
    )

    assert result.exit_code == 0
    assert result.tool_name == ToolName.WRITE_FILE
    assert (repository_path / 'test_app.py').read_text(encoding='utf-8') == 'import unittest\n'


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
            tool=RunShell(command='git status', timeout_seconds=10),
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
    assert artifact_path.name.startswith('run_shell-')
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
        tool=RunShell(command='git status', timeout_seconds=10),
    )

    restored_request = ToolExecutionRequest.model_validate(request.model_dump(mode='json'))

    assert restored_request.tool == RunShell(command='git status', timeout_seconds=10)


def test_tool_execution_request_preserves_workspace_subclass_after_round_trip() -> None:
    request = ToolExecutionRequest(
        workspace=_host_workspace(),
        tool=RunShell(command='git status', timeout_seconds=10),
    )

    restored_request = ToolExecutionRequest.model_validate(request.model_dump(mode='json'))

    assert isinstance(restored_request.workspace, HostWorkspace)
    assert restored_request.workspace.repo_path == 'workspace'


@pytest.mark.asyncio
async def test_pack_context_reads_host_workspace_file_without_shell(tmp_path: Path) -> None:
    repository_path = tmp_path / 'repo'
    repository_path.mkdir()
    (repository_path / 'app.py').write_text(
        'def add(first_number: int, second_number: int) -> int:\n'
        '    return first_number + second_number\n',
        encoding='utf-8',
    )

    context_pack = await pack_context(
        ContextPackRequest(
            workspace=_host_workspace(repo_path=str(repository_path)),
            task_summary='Read add implementation',
            snippets=[
                ContextSnippet(
                    file_path='app.py',
                    start_line=1,
                    end_line=2,
                    reason='Contains add implementation.',
                )
            ],
        )
    )

    assert context_pack.snippets[0].content == (
        'def add(first_number: int, second_number: int) -> int:\n'
        '    return first_number + second_number\n'
    )


@pytest.mark.asyncio
async def test_pack_context_records_missing_host_snippet_as_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository_path = tmp_path / 'repo'
    artifacts_path = tmp_path / 'artifacts'
    repository_path.mkdir()
    monkeypatch.setenv('ARTIFACTS_ROOT', str(artifacts_path))

    context_pack = await pack_context(
        ContextPackRequest(
            workspace=_host_workspace(run_id='run-missing', repo_path=str(repository_path)),
            task_summary='Read generated test file',
            snippets=[
                ContextSnippet(
                    file_path='test_app.py',
                    start_line=1,
                    end_line=10,
                    reason='Contains generated subtract tests.',
                )
            ],
        )
    )

    assert context_pack.snippets == []
    assert len(context_pack.artifact_references) == 1
    artifact_reference = context_pack.artifact_references[0]
    assert artifact_reference.kind == ArtifactKind.CONTEXT_OVERFLOW
    assert 'missing context snippet' in artifact_reference.summary
    assert 'test_app.py' in Path(artifact_reference.path).read_text(encoding='utf-8')


@pytest.mark.asyncio
async def test_run_tool_finds_definition_from_repo_index() -> None:
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
            tool=FindDefinition(name='Parser', language='python'),
            repo_index=repository_index,
        )
    )

    assert result.exit_code == 0
    assert result.tool_name == ToolName.FIND_DEFINITION
    assert 'src/parser.py:3-8 class Parser' in result.stdout


@pytest.mark.asyncio
async def test_run_tool_finds_callers_from_repo_index_excluding_mentions() -> None:
    repository_index = RepoIndex(
        references=[
            Reference(
                symbol_name='build_parser',
                file_path='src/app.py',
                line=12,
                kind=ReferenceKind.CALL,
            ),
            Reference(
                symbol_name='build_parser',
                file_path='src/notes.py',
                line=4,
                kind=ReferenceKind.MENTION,
            ),
        ]
    )

    result = await run_tool(
        ToolExecutionRequest(
            workspace=_host_workspace(),
            tool=FindCallers(symbol_name='build_parser'),
            repo_index=repository_index,
        )
    )

    assert result.exit_code == 0
    assert result.tool_name == ToolName.FIND_CALLERS
    assert result.stdout == 'src/app.py:12: build_parser'
