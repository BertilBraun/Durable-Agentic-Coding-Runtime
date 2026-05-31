from __future__ import annotations

import shlex
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Annotated, Literal, NamedTuple, TypeVar

import docker
from pydantic import Field, TypeAdapter
from temporal_light import activity

from src.config import CONFIG
from src.models.context import (
    ArtifactKind,
    ArtifactReference,
    ContextPack,
    ContextSnippet,
    PackedSnippet,
)
from src.models.frozen_base_model import FrozenBaseModel
from src.models.repo import Language, Reference, ReferenceKind, RepoIndex, Symbol
from src.models.task import DockerOrigin, HostOrigin, Origin
from src.tools.definitions import (
    FindCallees,
    FindCallers,
    FindDefinition,
    RunShell,
    RunTests,
    Tool,
    ToolName,
)
from src.tools.handlers import command_for_tool

POSIX_ENVIRONMENT_DESCRIPTION = (
    'OS: Linux/Unix. Shell: POSIX `sh` (commands run as `sh -lc <command>`). '
    'Run from the repository root. Chain statements with `;`, `&&`, `||`, or newlines; '
    'multi-line commands work directly. Quote arguments with single quotes and escape a '
    "literal single quote as '\\''. Use forward-slash paths and redirect to /dev/null."
)
WINDOWS_ENVIRONMENT_DESCRIPTION = (
    'OS: Windows. Shell: Windows PowerShell (commands run as '
    '`powershell -NoProfile -Command <command>`). Run from the repository root. Separate '
    'statements with `;` or newlines and use a backtick (`) for line continuation. Quote '
    "arguments with single quotes and escape a literal single quote by doubling it (''). "
    'Cmdlets use Verb-Noun names (Get-Content, Select-String); redirect to $null, not /dev/null.'
)


def _host_is_windows() -> bool:
    return sys.platform.startswith('win')


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

    def shell_invocation(self, command: str) -> list[str]:
        raise NotImplementedError

    def describe_environment(self) -> str:
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

    def shell_invocation(self, command: str) -> list[str]:
        if _host_is_windows():
            return ['powershell', '-NoProfile', '-Command', command]
        return ['sh', '-lc', command]

    def describe_environment(self) -> str:
        if _host_is_windows():
            return WINDOWS_ENVIRONMENT_DESCRIPTION
        return POSIX_ENVIRONMENT_DESCRIPTION


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

    def shell_invocation(self, command: str) -> list[str]:
        return ['sh', '-lc', command]

    def describe_environment(self) -> str:
        return POSIX_ENVIRONMENT_DESCRIPTION


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
        command_for_tool(request.tool, request.workspace),
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


class ContextPackRequest(FrozenBaseModel):
    workspace: Workspace
    task_summary: str
    snippets: list[ContextSnippet] = Field(default_factory=list)
    overflow_text: str | None = None


@activity(retries=0, timeout=120)
async def pack_context(request: ContextPackRequest) -> ContextPack:
    budget = CONFIG.context_pack_max_characters
    packed_snippets: list[PackedSnippet] = []
    artifacts: list[ArtifactReference] = []
    used_characters = 0
    for snippet in request.snippets:
        content = _read_snippet_lines(request.workspace, snippet)
        if used_characters + len(content) <= budget:
            packed_snippets.append(
                PackedSnippet(
                    file_path=snippet.file_path,
                    start_line=snippet.start_line,
                    end_line=snippet.end_line,
                    reason=snippet.reason,
                    content=content,
                )
            )
            used_characters += len(content)
        else:
            artifacts.append(
                _write_context_overflow_artifact(
                    run_id=request.workspace.run_id,
                    summary=(
                        f'{snippet.file_path}:{snippet.start_line}-{snippet.end_line} '
                        'exceeded the context pack budget'
                    ),
                    content=content,
                )
            )
    if request.overflow_text:
        artifacts.append(
            _write_context_overflow_artifact(
                run_id=request.workspace.run_id,
                summary='context gatherer observations (budget exhausted before curation)',
                content=request.overflow_text,
            )
        )
    return ContextPack(
        task_summary=request.task_summary,
        snippets=packed_snippets,
        artifact_references=artifacts,
        budget_remaining=max(0, budget - used_characters),
    )


def _read_snippet_lines(workspace: Workspace, snippet: ContextSnippet) -> str:
    start_line = max(1, snippet.start_line)
    end_line = max(start_line, snippet.end_line)
    quoted_path = shlex.quote(snippet.file_path)
    result = workspace.run_command(['sh', '-lc', f'sed -n {start_line},{end_line}p {quoted_path}'])
    return result.stdout


def _write_context_overflow_artifact(run_id: str, summary: str, content: str) -> ArtifactReference:
    artifact_filename = f'context-{uuid.uuid4()}.log'
    artifact_path = Path(_artifacts_root()) / run_id / artifact_filename
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(content, encoding='utf-8')
    return ArtifactReference(
        path=artifact_path.as_posix(),
        summary=summary,
        kind=ArtifactKind.CONTEXT_OVERFLOW,
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
        case FindDefinition(name=name, language=language):
            return _find_indexed_definition(request.repo_index, name, language)
        case FindCallers(symbol_name=symbol_name):
            return _find_indexed_callers(request.repo_index, symbol_name)
        case FindCallees(file_path=file_path, symbol_name=symbol_name):
            return _find_indexed_callees(request.repo_index, file_path, symbol_name)
        case _:
            return None


def _find_indexed_definition(repository_index: RepoIndex, name: str, language: str) -> ToolResult:
    matching_symbols = [
        symbol
        for symbol in repository_index.symbols
        if symbol.name == name and (not language or symbol.language == Language(language))
    ]
    lines = [_format_symbol(symbol) for symbol in matching_symbols]
    return ToolResult(
        tool_name=ToolName.FIND_DEFINITION,
        stdout='\n'.join(lines),
        stderr='',
        exit_code=0,
        truncated=False,
    )


def _find_indexed_callers(repository_index: RepoIndex, symbol_name: str) -> ToolResult:
    callers = [
        reference
        for reference in repository_index.references
        if reference.symbol_name == symbol_name and reference.kind == ReferenceKind.CALL
    ]
    lines = [_format_reference(reference) for reference in callers]
    return ToolResult(
        tool_name=ToolName.FIND_CALLERS,
        stdout='\n'.join(lines),
        stderr='',
        exit_code=0,
        truncated=False,
    )


def _find_indexed_callees(
    repository_index: RepoIndex,
    file_path: str,
    symbol_name: str,
) -> ToolResult:
    definition_ranges = [
        (symbol.start_line, symbol.end_line)
        for symbol in repository_index.symbols
        if symbol.name == symbol_name and symbol.file_path == file_path
    ]
    callees = [
        reference
        for reference in repository_index.references
        if reference.kind == ReferenceKind.CALL
        and reference.file_path == file_path
        and any(start <= reference.line <= end for start, end in definition_ranges)
    ]
    lines = [_format_reference(reference) for reference in callees]
    return ToolResult(
        tool_name=ToolName.FIND_CALLEES,
        stdout='\n'.join(lines),
        stderr='',
        exit_code=0,
        truncated=False,
    )


def _format_symbol(symbol: Symbol) -> str:
    location = f'{symbol.file_path}:{symbol.start_line}-{symbol.end_line}'
    return f'{location} {symbol.kind.value} {symbol.name}'


def _format_reference(reference: Reference) -> str:
    return f'{reference.file_path}:{reference.line}: {reference.symbol_name}'


def _tool_timeout_seconds(tool: Tool) -> int | None:
    match tool:
        case RunTests(timeout_seconds=timeout_seconds) | RunShell(timeout_seconds=timeout_seconds):
            return timeout_seconds
        case _:
            return None


def make_run_id() -> str:
    return str(uuid.uuid4())


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
