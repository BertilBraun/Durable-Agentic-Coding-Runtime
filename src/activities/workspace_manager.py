from __future__ import annotations

import re
import subprocess
import tarfile
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal, NamedTuple, TypeVar

import docker
from pydantic import Field, TypeAdapter
from temporal_light import activity

from src.config import CONFIG
from src.models.context import ArtifactKind, ArtifactReference
from src.models.frozen_base_model import FrozenBaseModel
from src.models.repo import Language, RepoIndex, Symbol
from src.models.task import DockerOrigin, HostOrigin, Origin
from src.tools.definitions import FindReferences, FindSymbol, RunTests, Tool, ToolName
from src.tools.handlers import command_for_tool


class CommandResult(NamedTuple):
    stdout: str
    stderr: str
    exit_code: int


def _candidate_branch(run_id: str, candidate_index: int) -> str:
    return f'agentic/{run_id}/cand-{candidate_index}'


Self = TypeVar('Self', bound='_Workspace')


class _Workspace(FrozenBaseModel):
    run_id: str
    base_sha: str
    base_branch: str | None = None
    current_branch: str

    def run_command(self, command: list[str], timeout: int | None = None) -> CommandResult:
        raise NotImplementedError

    def teardown(self) -> None:
        raise NotImplementedError

    def _run_checked(self, command: list[str]) -> str:
        result = self.run_command(command)
        if result.exit_code != 0:
            raise RuntimeError(
                f'Command failed ({result.exit_code}): {" ".join(command)}\n{result.stderr}'
            )
        return result.stdout

    def begin_candidate(self: Self, candidate_index: int) -> Self:
        branch = _candidate_branch(self.run_id, candidate_index)
        self._run_checked(['git', 'checkout', '-B', branch, self.base_sha])
        return self.model_copy(update={'current_branch': branch})

    def reset_to_base(self) -> None:
        self._run_checked(['git', 'reset', '--hard', self.base_sha])
        self._run_checked(['git', 'clean', '-fd'])

    def diff_against_base(self) -> str:
        return self.run_command(['git', 'diff', self.base_sha]).stdout

    def finalize_to_base(self, winner_branch: str, cleanup_branches: bool) -> None:
        if self.base_branch is not None:
            self._run_checked(['git', 'checkout', self.base_branch])
        else:
            self._run_checked(['git', 'checkout', self.base_sha])
        self._run_checked(['git', 'checkout', winner_branch, '--', '.'])
        self._run_checked(['git', 'reset'])
        if cleanup_branches:
            self._delete_candidate_branches()

    def _delete_candidate_branches(self) -> None:
        listed = self._run_checked(
            [
                'git',
                'for-each-ref',
                '--format=%(refname:short)',
                f'refs/heads/agentic/{self.run_id}/',
            ]
        )
        for branch in listed.splitlines():
            if branch.strip():
                self._run_checked(['git', 'branch', '-D', branch.strip()])


class HostWorkspace(_Workspace):
    kind: Literal['host'] = 'host'
    repo_path: str

    def run_command(self, command: list[str], timeout: int | None = None) -> CommandResult:
        completed = subprocess.run(
            command,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CommandResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
        )

    def teardown(self) -> None:
        return


class DockerWorkspace(_Workspace):
    kind: Literal['docker'] = 'docker'
    container_id: str
    container_repo_path: str

    def run_command(self, command: list[str], timeout: int | None = None) -> CommandResult:
        full_command = ['timeout', str(timeout), *command] if timeout is not None else command
        container = _docker_client().containers.get(self.container_id)
        exec_result = container.exec_run(
            full_command,
            workdir=self.container_repo_path,
            demux=True,
        )
        stdout_bytes, stderr_bytes = exec_result.output
        return CommandResult(
            stdout=stdout_bytes.decode('utf-8', errors='replace') if stdout_bytes else '',
            stderr=stderr_bytes.decode('utf-8', errors='replace') if stderr_bytes else '',
            exit_code=exec_result.exit_code,
        )

    def teardown(self) -> None:
        container = _docker_client().containers.get(self.container_id)
        container.stop(timeout=10)
        container.remove(force=True)


Workspace = Annotated[HostWorkspace | DockerWorkspace, Field(discriminator='kind')]
WorkspaceAdapter: TypeAdapter[Workspace] = TypeAdapter(Workspace)


class ToolResult(FrozenBaseModel):
    tool_name: ToolName | None = None
    stdout: str
    stderr: str
    exit_code: int
    truncated: bool
    artifacts: list[ArtifactReference] = Field(default_factory=list)


class ToolExecutionRequest(FrozenBaseModel):
    workspace: Workspace
    tool: Tool
    repo_index: RepoIndex | None = None


@activity(retries=0, timeout=600)
async def setup_environment(origin: Origin, run_id: str) -> Workspace:
    match origin:
        case HostOrigin(repo_path=repo_path):
            return _setup_host_workspace(repo_path=repo_path, run_id=run_id)
        case DockerOrigin(docker_image=docker_image, container_repo_path=container_repo_path):
            return _setup_docker_workspace(
                docker_image=docker_image,
                container_repo_path=container_repo_path,
                run_id=run_id,
            )


def _setup_host_workspace(repo_path: str, run_id: str) -> HostWorkspace:
    workspace = HostWorkspace(
        run_id=run_id,
        base_sha='',
        base_branch=None,
        current_branch='',
        repo_path=repo_path,
    )
    status = workspace.run_command(['git', 'status', '--porcelain'])
    if status.stdout.strip():
        raise RuntimeError(f'Working tree is not clean at {repo_path}:\n{status.stdout}')
    head_probe = workspace._run_checked(['git', 'rev-parse', 'HEAD'])
    base_sha = head_probe.strip()
    branch_probe = workspace.run_command(['git', 'symbolic-ref', '--short', '-q', 'HEAD'])
    base_branch = branch_probe.stdout.strip()
    return workspace.model_copy(
        update={
            'base_sha': base_sha,
            'base_branch': base_branch or None,
            'current_branch': base_branch or base_sha,
        }
    )


def _setup_docker_workspace(
    docker_image: str, container_repo_path: str, run_id: str
) -> DockerWorkspace:
    docker_client = _docker_client()
    docker_client.images.pull(docker_image)
    container = docker_client.containers.run(
        image=docker_image,
        command=['sleep', 'infinity'],
        detach=True,
        working_dir=container_repo_path,
    )
    workspace = DockerWorkspace(
        run_id=run_id,
        base_sha='',
        base_branch=None,
        current_branch='',
        container_id=str(container.id),
        container_repo_path=container_repo_path,
    )
    status = workspace.run_command(['git', 'status', '--porcelain'])
    if status.stdout.strip():
        raise RuntimeError(f'Working tree is not clean at {container_repo_path}:\n{status.stdout}')
    head_probe = workspace._run_checked(['git', 'rev-parse', 'HEAD'])
    base_sha = head_probe.strip()
    branch_probe = workspace.run_command(['git', 'symbolic-ref', '--short', '-q', 'HEAD'])
    base_branch = branch_probe.stdout.strip() or None
    return workspace.model_copy(
        update={
            'base_sha': base_sha,
            'base_branch': base_branch,
            'current_branch': base_branch or base_sha,
        }
    )


@activity(retries=0, timeout=300)
async def run_tool(request: ToolExecutionRequest) -> ToolResult:
    indexed_result = _indexed_tool_result(request)
    if indexed_result is not None:
        return indexed_result
    command_result = request.workspace.run_command(
        command_for_tool(request.tool),
        timeout=_tool_timeout_seconds(request.tool),
    )
    stdout_reference = _write_large_output_artifact(
        request=request,
        stream_name='stdout',
        output=command_result.stdout,
    )
    stderr_reference = _write_large_output_artifact(
        request=request,
        stream_name='stderr',
        output=command_result.stderr,
    )
    artifacts = [
        artifact_reference
        for artifact_reference in (stdout_reference, stderr_reference)
        if artifact_reference is not None
    ]
    return ToolResult(
        tool_name=request.tool.tool_name,
        stdout=_compact_output(command_result.stdout),
        stderr=_compact_output(command_result.stderr),
        exit_code=command_result.exit_code,
        truncated=bool(artifacts),
        artifacts=artifacts,
    )


@activity(retries=0, timeout=120)
async def begin_candidate(workspace: Workspace, candidate_index: int) -> Workspace:
    return workspace.begin_candidate(candidate_index)


@activity(retries=0, timeout=120)
async def finalize_winner(workspace: Workspace, winner_branch: str) -> ToolResult:
    workspace.finalize_to_base(
        winner_branch=winner_branch,
        cleanup_branches=CONFIG.cleanup_candidate_branches,
    )
    return ToolResult(stdout='', stderr='', exit_code=0, truncated=False)


@activity(retries=0, timeout=120)
async def teardown_environment(workspace: Workspace) -> ToolResult:
    workspace.teardown()
    return ToolResult(stdout='', stderr='', exit_code=0, truncated=False)


def _indexed_tool_result(request: ToolExecutionRequest) -> ToolResult | None:
    if request.repo_index is None:
        return None
    match request.tool:
        case FindSymbol(name=name, language=language):
            return _find_indexed_symbol(request.repo_index, name, language)
        case FindReferences(symbol_name=symbol_name):
            match request.workspace:
                case HostWorkspace(repo_path=repo_path):
                    return _find_indexed_references(
                        workspace_path=Path(repo_path),
                        repository_index=request.repo_index,
                        symbol_name=symbol_name,
                    )
                case _:
                    return None
        case _:
            return None


def _find_indexed_symbol(repository_index: RepoIndex, name: str, language: str) -> ToolResult:
    matching_symbols = [
        symbol
        for symbol in repository_index.symbols
        if symbol.name == name and (not language or symbol.language == Language(language))
    ]
    lines = [_format_symbol(symbol) for symbol in matching_symbols]
    return ToolResult(
        tool_name=ToolName.FIND_SYMBOL,
        stdout='\n'.join(lines),
        stderr='',
        exit_code=0,
        truncated=False,
    )


def _find_indexed_references(
    workspace_path: Path,
    repository_index: RepoIndex,
    symbol_name: str,
) -> ToolResult:
    reference_lines: list[str] = []
    symbol_pattern = re.compile(rf'\b{re.escape(symbol_name)}\b')
    indexed_paths = {
        file_entry.path
        for file_entry in repository_index.file_tree
        if file_entry.language != Language.UNKNOWN
    }
    for relative_path in sorted(indexed_paths):
        file_path = workspace_path / relative_path
        if not file_path.is_file():
            continue
        for line_number, line in enumerate(
            file_path.read_text(encoding='utf-8', errors='replace').splitlines(),
            start=1,
        ):
            if symbol_pattern.search(line):
                reference_lines.append(f'{relative_path}:{line_number}:{line}')
    return ToolResult(
        tool_name=ToolName.FIND_REFERENCES,
        stdout='\n'.join(reference_lines),
        stderr='',
        exit_code=0,
        truncated=False,
    )


def _format_symbol(symbol: Symbol) -> str:
    location = f'{symbol.file_path}:{symbol.start_line}-{symbol.end_line}'
    return f'{location} {symbol.kind.value} {symbol.name}'


def _tool_timeout_seconds(tool: Tool) -> int | None:
    match tool:
        case RunTests(timeout_seconds=timeout_seconds):
            return timeout_seconds
        case _:
            return None


def make_run_id() -> str:
    return str(uuid.uuid4())


@contextmanager
def extract_docker_repo_snapshot(workspace: DockerWorkspace) -> Iterator[Path]:
    container = _docker_client().containers.get(workspace.container_id)
    archive_stream, _ = container.get_archive(workspace.container_repo_path)
    with tempfile.TemporaryDirectory() as temporary_directory:
        archive_path = Path(temporary_directory) / 'snapshot.tar'
        with archive_path.open('wb') as archive_file:
            for chunk in archive_stream:
                archive_file.write(chunk)
        extraction_path = Path(temporary_directory) / 'snapshot'
        with tarfile.open(archive_path) as archive:
            archive.extractall(extraction_path)
        repository_name = Path(workspace.container_repo_path).name
        yield extraction_path / repository_name


def _docker_client() -> docker.DockerClient:
    return docker.from_env()


def _write_large_output_artifact(
    request: ToolExecutionRequest,
    stream_name: str,
    output: str,
) -> ArtifactReference | None:
    if len(output) <= CONFIG.tool_output_max_characters:
        return None
    tool_name = request.tool.tool_name.value
    artifact_filename = f'{tool_name}-{uuid.uuid4()}-{stream_name}.log'
    artifact_path = Path(_artifacts_root()) / request.workspace.run_id / artifact_filename
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(output, encoding='utf-8')
    return ArtifactReference(
        path=artifact_path.as_posix(),
        summary=f'{stream_name} output stored in artifact ({len(output)} characters)',
        kind=_artifact_kind_for_output(request.tool),
    )


def _artifact_kind_for_output(tool: Tool) -> ArtifactKind:
    match tool:
        case RunTests():
            return ArtifactKind.TEST_OUTPUT
        case _:
            return ArtifactKind.LOG


def _artifacts_root() -> str:
    return CONFIG.artifacts_root


def _compact_output(output: str) -> str:
    if len(output) <= CONFIG.tool_output_max_characters:
        return output
    head_size = CONFIG.tool_output_compact_head_characters
    tail_size = CONFIG.tool_output_compact_tail_characters
    head = output[:head_size]
    tail = output[-tail_size:]
    omitted = len(output) - head_size - tail_size
    return f'{head}\n\n[... {omitted} characters omitted ...]\n\n{tail}'
