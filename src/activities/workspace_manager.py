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
from src.models.repo import RepoIndex
from src.models.task import DockerOrigin, HostOrigin, Origin
from src.tools.definitions import (
    FindCallees,
    FindCallers,
    FindDefinition,
    RunShell,
    RunTests,
    Tool,
    ToolName,
    WriteFile,
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
    execution_id: str = ''
    base_sha: str
    base_branch: str | None = None
    current_branch: str
    candidate_base_sha: str | None = None

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

    @property
    def _candidate_base(self) -> str:
        return self.candidate_base_sha or self.base_sha

    def begin_candidate(self: Self, candidate_index: int) -> Self:
        branch = _candidate_branch(self.run_id, candidate_index)
        self._run_checked(['git', 'checkout', '-B', branch, self._candidate_base])
        return self.model_copy(update={'current_branch': branch})

    def reset_to_base(self) -> None:
        self._run_checked(['git', 'reset', '--hard', self._candidate_base])
        self._run_checked(['git', 'clean', '-fd'])

    def snapshot_candidate_base(self: Self) -> Self:
        self._run_checked(['git', 'add', '-A'])
        tree = self._run_checked(['git', 'write-tree']).strip()
        snapshot_sha = self._run_checked(
            ['git', 'commit-tree', tree, '-p', self.base_sha, '-m', 'agentic candidate base']
        ).strip()
        return self.model_copy(update={'candidate_base_sha': snapshot_sha})

    def snapshot_candidate_result(self: Self) -> Self:
        self._run_checked(['git', 'add', '-A'])
        tree = self._run_checked(['git', 'write-tree']).strip()
        result_sha = self._run_checked(
            [
                'git',
                'commit-tree',
                tree,
                '-p',
                self._candidate_base,
                '-m',
                'agentic candidate result',
            ]
        ).strip()
        self._run_checked(['git', 'reset', '--hard', result_sha])
        return self

    def diff_against_base(self) -> str:
        return self.run_command(['git', 'diff', self.base_sha]).stdout

    def finalize_to_base(self, winner_branch: str, cleanup_branches: bool) -> None:
        self._run_checked(['git', 'reset', '--hard'])
        self._run_checked(['git', 'clean', '-fd'])
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
        return ['sh', '-lc', f'export PATH=/opt/miniconda3/envs/testbed/bin:$PATH && {command}']

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
        execution_id=make_run_id(),
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
    try:
        docker_client.images.get(docker_image)
    except docker.errors.ImageNotFound as error:
        raise RuntimeError(
            'Missing Docker image for docker workspace: '
            f'{docker_image}. Build the SWE-bench image locally before running generation.'
        ) from error
    container = docker_client.containers.run(
        image=docker_image,
        command=['sleep', 'infinity'],
        detach=True,
        working_dir=container_repo_path,
        name=_swe_bench_container_name(run_id),
        labels=_swe_bench_container_labels(run_id),
    )
    workspace = DockerWorkspace(
        run_id=run_id,
        execution_id=make_run_id(),
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


def _swe_bench_container_name(run_id: str) -> str:
    return f'agentic-swe-bench-{_docker_name_part(run_id)}-{make_run_id()}'


def _docker_name_part(value: str) -> str:
    return ''.join(character.lower() if character.isalnum() else '-' for character in value).strip(
        '-'
    )


def _swe_bench_container_labels(run_id: str) -> dict[str, str]:
    return {
        'com.docker.compose.project': 'agentic-swe-bench',
        'com.docker.compose.service': 'workspace',
        'agentic.workflow.run_id': run_id,
        'agentic.workflow.kind': 'swe-bench',
    }


@activity(retries=0, timeout=300)
async def run_tool(request: ToolExecutionRequest) -> ToolResult:
    indexed_result = _indexed_tool_result(request)
    if indexed_result is not None:
        return indexed_result
    host_result = _host_tool_result(request)
    if host_result is not None:
        return host_result
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
        try:
            content = _read_snippet_lines(request.workspace, snippet)
        except FileNotFoundError:
            artifacts.append(_missing_snippet_artifact(request.workspace.run_id, snippet))
            continue
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
    if isinstance(workspace, HostWorkspace):
        return _read_host_snippet_lines(workspace, snippet.file_path, start_line, end_line)
    quoted_path = shlex.quote(snippet.file_path)
    result = workspace.run_command(['sh', '-lc', f'sed -n {start_line},{end_line}p {quoted_path}'])
    return result.stdout


def _read_host_snippet_lines(
    workspace: HostWorkspace,
    file_path: str,
    start_line: int,
    end_line: int,
) -> str:
    target_path = _host_workspace_file_path(workspace, file_path)
    lines = target_path.read_text(encoding='utf-8').splitlines(keepends=True)
    return ''.join(lines[start_line - 1 : end_line])


def _host_workspace_file_path(workspace: HostWorkspace, file_path: str) -> Path:
    repo_root = Path(workspace.repo_path).resolve()
    target_path = (repo_root / file_path).resolve()
    try:
        target_path.relative_to(repo_root)
    except ValueError as error:
        raise ValueError(f'Path escapes workspace: {file_path}') from error
    return target_path


def _host_tool_result(request: ToolExecutionRequest) -> ToolResult | None:
    if not isinstance(request.workspace, HostWorkspace):
        return None
    match request.tool:
        case WriteFile(file_path=file_path, content=content):
            target_path = _host_workspace_file_path(request.workspace, file_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding='utf-8')
            return ToolResult(
                tool_name=ToolName.WRITE_FILE,
                stdout='',
                stderr='',
                exit_code=0,
                truncated=False,
            )
        case _:
            return None


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


def _missing_snippet_artifact(run_id: str, snippet: ContextSnippet) -> ArtifactReference:
    return _write_context_overflow_artifact(
        run_id=run_id,
        summary=(
            f'missing context snippet: {snippet.file_path}:{snippet.start_line}-{snippet.end_line}'
        ),
        content=(
            'Context snippet file was requested but was not present in the workspace.\n'
            f'file_path: {snippet.file_path}\n'
            f'line_range: {snippet.start_line}-{snippet.end_line}\n'
            f'reason: {snippet.reason}\n'
        ),
    )


@activity(retries=0, timeout=120)
async def begin_candidate(workspace: Workspace, candidate_index: int) -> Workspace:
    return workspace.begin_candidate(candidate_index)


@activity(retries=0, timeout=120)
async def reset_to_base(workspace: Workspace) -> ToolResult:
    workspace.reset_to_base()
    return ToolResult(stdout='', stderr='', exit_code=0, truncated=False)


@activity(retries=0, timeout=120)
async def snapshot_candidate_base(workspace: Workspace) -> Workspace:
    return workspace.snapshot_candidate_base()


@activity(retries=0, timeout=120)
async def snapshot_candidate_result(workspace: Workspace) -> Workspace:
    return workspace.snapshot_candidate_result()


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
            return _find_definition_with_search(request.workspace, name, language)
        case FindCallers(symbol_name=symbol_name):
            return _find_callers_with_search(request.workspace, symbol_name)
        case FindCallees(file_path=file_path, symbol_name=symbol_name):
            return _find_callees_with_search(request.workspace, file_path, symbol_name)
        case _:
            return None


def _find_definition_with_search(workspace: Workspace, name: str, language: str) -> ToolResult:
    if language and language != 'python':
        return ToolResult(
            tool_name=ToolName.FIND_DEFINITION,
            stdout='',
            stderr=f'find_definition currently supports python only, got: {language}',
            exit_code=1,
            truncated=False,
        )
    pattern = rf'^[[:space:]]*(def|class)[[:space:]]+{_regex_word(name)}([^[:alnum:]_]|$)'
    return _run_search_tool(
        workspace=workspace,
        tool_name=ToolName.FIND_DEFINITION,
        command=_python_search_command(workspace, pattern),
    )


def _find_callers_with_search(workspace: Workspace, symbol_name: str) -> ToolResult:
    return _run_search_tool(
        workspace=workspace,
        tool_name=ToolName.FIND_CALLERS,
        command=_python_search_command(
            workspace,
            rf'(^|[^[:alnum:]_]){_regex_word(symbol_name)}([^[:alnum:]_]|$)',
        ),
    )


def _find_callees_with_search(
    workspace: Workspace,
    file_path: str,
    symbol_name: str,
) -> ToolResult:
    del symbol_name
    return _run_search_tool(
        workspace=workspace,
        tool_name=ToolName.FIND_CALLEES,
        command=_single_file_search_command(
            workspace,
            r'[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\(',
            file_path,
        ),
    )


def _run_search_tool(workspace: Workspace, tool_name: ToolName, command: list[str]) -> ToolResult:
    command_result = workspace.run_command(command, timeout=30)
    exit_code = 0 if command_result.exit_code == 1 else command_result.exit_code
    return ToolResult(
        tool_name=tool_name,
        stdout=_compact_output(command_result.stdout),
        stderr=_compact_output(command_result.stderr),
        exit_code=exit_code,
        truncated=False,
    )


def _python_search_command(workspace: Workspace, pattern: str) -> list[str]:
    if isinstance(workspace, DockerWorkspace):
        return ['grep', '-RInE', '--include=*.py', pattern, '.']
    return ['rg', '--line-number', pattern, '--glob', '*.py']


def _single_file_search_command(workspace: Workspace, pattern: str, file_path: str) -> list[str]:
    if isinstance(workspace, DockerWorkspace):
        return ['grep', '-nE', pattern, file_path]
    return ['rg', '--line-number', pattern, file_path]


def _regex_word(value: str) -> str:
    escaped = ''.join(
        character if character.isalnum() or character == '_' else f'\\{character}'
        for character in value
    )
    return escaped


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
