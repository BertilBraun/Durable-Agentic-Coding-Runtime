from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import docker
import docker.models.containers
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from src.activities.temporal import durable_activity
from src.models.context import ArtifactKind, ArtifactReference
from src.models.repo import Language, RepoIndex, Symbol
from src.tools.definitions import (
    ApplyPatch,
    FindReferences,
    FindSymbol,
    GatherContext,
    GitCommit,
    GitDiff,
    GitStatus,
    ReadFileRange,
    RunLint,
    RunTests,
    RunTypecheck,
    SearchText,
    Tool,
    ToolName,
    WriteFile,
    tool_definition_for_tool,
)
from src.tools.handlers import command_for_tool

WORKSPACE_IMAGE_ENVIRONMENT_NAME = "WORKSPACE_IMAGE"
DEFAULT_WORKSPACE_IMAGE = "durable-agentic-workspace:latest"
CONTAINER_WORKSPACE_PATH = "/workspace/repository"
MAX_OUTPUT_CHARACTERS = 20_000
COMPACT_OUTPUT_CHARACTERS = 800
ARTIFACTS_ROOT_ENVIRONMENT_NAME = "ARTIFACTS_ROOT"
DEFAULT_ARTIFACTS_ROOT = "/artifacts"


class WorkspaceInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    volume_name: str
    worktree_path: str
    branch_name: str
    workspace_image: str | None = None
    container_repo_path: str = CONTAINER_WORKSPACE_PATH


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    stdout: str
    stderr: str
    exit_code: int
    truncated: bool
    artifacts: list[ArtifactReference] = []


class ToolExecutionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    workspace_info: WorkspaceInfo
    tool: Tool
    repo_index: RepoIndex | None = None

    @field_serializer("tool")
    def serialize_tool(self, tool: Tool) -> dict[str, object]:
        return {
            "tool_name": tool_definition_for_tool(tool).name,
            "payload": asdict(tool),
        }

    @field_validator("tool", mode="before")
    @classmethod
    def restore_tool(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        tool_name = value.get("tool_name")
        payload = value.get("payload")
        if not isinstance(tool_name, str) or not isinstance(payload, dict):
            return value
        return _tool_from_serialized_payload(ToolName(tool_name), payload)


@durable_activity(retries=0, timeout=300)
async def create_workspace(
    run_id: str, repo_path: str, docker_image: str | None = None
) -> WorkspaceInfo:
    docker_client = _docker_client()
    volume_name = f"agentic-coding-{run_id}"
    branch_name = f"agentic-coding/{run_id}"
    docker_client.volumes.create(name=volume_name)
    workspace_root = Path(os.getenv("WORKSPACE_ROOT", ".agentic-workspaces")).resolve()
    worktree_path = workspace_root / run_id / "repository"
    if worktree_path.exists():
        shutil.rmtree(worktree_path)
    worktree_path.mkdir(parents=True, exist_ok=True)

    if docker_image is not None:
        # SWE-bench path: repo is pre-materialized at /testbed inside the official image.
        # Pull the image, copy /testbed to the host worktree, then branch for agent changes.
        docker_client.images.pull(docker_image)
        docker_client.containers.run(
            image=docker_image,
            command=[
                "sh",
                "-lc",
                f"cp -rp /testbed/. /target/ && cd /target && git checkout -b {branch_name}",
            ],
            remove=True,
            volumes={str(worktree_path): {"bind": "/target", "mode": "rw"}},
        )
        return WorkspaceInfo(
            run_id=run_id,
            volume_name=volume_name,
            worktree_path=str(worktree_path),
            branch_name=branch_name,
            workspace_image=docker_image,
            container_repo_path="/testbed",
        )

    repository_source_path = os.path.abspath(repo_path)
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    docker_client.containers.run(
        image=_workspace_image(),
        command=[
            "sh",
            "-lc",
            (
                "rm -rf /target/repository && "
                "git clone /source /target/repository && "
                "cd /target/repository && "
                f"git checkout -b {branch_name}"
            ),
        ],
        remove=True,
        volumes={
            repository_source_path: {"bind": "/source", "mode": "ro"},
            str(worktree_path.parent): {"bind": "/target", "mode": "rw"},
        },
    )
    return WorkspaceInfo(
        run_id=run_id,
        volume_name=volume_name,
        worktree_path=str(worktree_path),
        branch_name=branch_name,
    )


@durable_activity(retries=0, timeout=300)
async def run_tool(request: ToolExecutionRequest) -> ToolResult:
    indexed_result = _indexed_tool_result(request)
    if indexed_result is not None:
        return indexed_result
    docker_client = _docker_client()
    container_repo_path = request.workspace_info.container_repo_path
    container = docker_client.containers.run(
        image=request.workspace_info.workspace_image or _workspace_image(),
        command=command_for_tool(request.tool),
        detach=True,
        working_dir=container_repo_path,
        volumes={
            request.workspace_info.worktree_path: {
                "bind": container_repo_path,
                "mode": "rw",
            }
        },
    )
    try:
        wait_result = container.wait(timeout=_tool_timeout_seconds(request.tool))
        stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
    finally:
        container.remove(force=True)
    stdout_reference = _write_large_output_artifact(
        request=request,
        stream_name="stdout",
        output=stdout,
    )
    stderr_reference = _write_large_output_artifact(
        request=request,
        stream_name="stderr",
        output=stderr,
    )
    artifacts = [
        artifact_reference
        for artifact_reference in (stdout_reference, stderr_reference)
        if artifact_reference is not None
    ]
    return ToolResult(
        stdout=_compact_output(stdout),
        stderr=_compact_output(stderr),
        exit_code=int(wait_result.get("StatusCode", 1)),
        truncated=bool(artifacts),
        artifacts=artifacts,
    )


def _tool_from_serialized_payload(tool_name: ToolName, payload: dict[str, Any]) -> Tool:
    match tool_name:
        case ToolName.READ_FILE_RANGE:
            return ReadFileRange(**payload)
        case ToolName.SEARCH_TEXT:
            return SearchText(**payload)
        case ToolName.WRITE_FILE:
            return WriteFile(**payload)
        case ToolName.APPLY_PATCH:
            return ApplyPatch(**payload)
        case ToolName.GIT_DIFF:
            return GitDiff(**payload)
        case ToolName.GIT_STATUS:
            return GitStatus(**payload)
        case ToolName.GIT_COMMIT:
            return GitCommit(**payload)
        case ToolName.RUN_TESTS:
            return RunTests(**payload)
        case ToolName.RUN_LINT:
            return RunLint(**payload)
        case ToolName.RUN_TYPECHECK:
            return RunTypecheck(**payload)
        case ToolName.FIND_SYMBOL:
            return FindSymbol(**payload)
        case ToolName.FIND_REFERENCES:
            return FindReferences(**payload)
        case ToolName.GATHER_CONTEXT:
            return GatherContext(**payload)
        case _:
            raise ValueError(f"Cannot deserialize unknown tool: {tool_name}")


def _indexed_tool_result(request: ToolExecutionRequest) -> ToolResult | None:
    if request.repo_index is None:
        return None
    match request.tool:
        case FindSymbol(name=name, language=language):
            return _find_indexed_symbol(request.repo_index, name, language)
        case FindReferences(symbol_name=symbol_name):
            return _find_indexed_references(
                workspace_path=Path(request.workspace_info.worktree_path),
                repository_index=request.repo_index,
                symbol_name=symbol_name,
            )
        case _:
            return None


def _find_indexed_symbol(repository_index: RepoIndex, name: str, language: str) -> ToolResult:
    matching_symbols = [
        symbol
        for symbol in repository_index.symbols
        if symbol.name == name and (not language or symbol.language == Language(language))
    ]
    lines = [_format_symbol(symbol) for symbol in matching_symbols]
    return ToolResult(stdout="\n".join(lines), stderr="", exit_code=0, truncated=False)


def _find_indexed_references(
    workspace_path: Path,
    repository_index: RepoIndex,
    symbol_name: str,
) -> ToolResult:
    reference_lines: list[str] = []
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
            file_path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if symbol_name in line:
                reference_lines.append(f"{relative_path}:{line_number}:{line}")
    return ToolResult(stdout="\n".join(reference_lines), stderr="", exit_code=0, truncated=False)


def _format_symbol(symbol: Symbol) -> str:
    return (
        f"{symbol.file_path}:{symbol.start_line}-{symbol.end_line} "
        f"{symbol.kind.value} {symbol.name}"
    )


def _tool_timeout_seconds(tool: Tool) -> int | None:
    match tool:
        case RunTests(timeout_seconds=timeout_seconds):
            return timeout_seconds
        case _:
            return None


@durable_activity(retries=0, timeout=120)
async def destroy_workspace(workspace_info: WorkspaceInfo) -> ToolResult:
    docker_client = _docker_client()
    volume = docker_client.volumes.get(workspace_info.volume_name)
    volume.remove(force=True)
    workspace_path = Path(workspace_info.worktree_path)
    if workspace_path.exists():
        shutil.rmtree(workspace_path.parent)
    return ToolResult(stdout="", stderr="", exit_code=0, truncated=False)


def make_run_id() -> str:
    return str(uuid.uuid4())


def _workspace_image() -> str:
    return os.getenv(WORKSPACE_IMAGE_ENVIRONMENT_NAME, DEFAULT_WORKSPACE_IMAGE)


def _docker_client() -> docker.DockerClient:
    return docker.from_env()


def _write_large_output_artifact(
    request: ToolExecutionRequest,
    stream_name: str,
    output: str,
) -> ArtifactReference | None:
    if len(output) <= MAX_OUTPUT_CHARACTERS:
        return None
    tool_name = tool_definition_for_tool(request.tool).name.value
    artifact_filename = f"{tool_name}-{uuid.uuid4()}-{stream_name}.log"
    artifact_path = Path(_artifacts_root()) / request.workspace_info.run_id / artifact_filename
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(output, encoding="utf-8")
    return ArtifactReference(
        path=artifact_path.as_posix(),
        summary=f"{stream_name} output stored in artifact ({len(output)} characters)",
        kind=_artifact_kind_for_output(request.tool),
    )


def _artifact_kind_for_output(tool: Tool) -> ArtifactKind:
    match tool:
        case RunTests():
            return ArtifactKind.TEST_OUTPUT
        case _:
            return ArtifactKind.LOG


def _artifacts_root() -> str:
    return os.getenv(ARTIFACTS_ROOT_ENVIRONMENT_NAME, DEFAULT_ARTIFACTS_ROOT)


def _compact_output(output: str) -> str:
    if len(output) <= MAX_OUTPUT_CHARACTERS:
        return output
    return output[:COMPACT_OUTPUT_CHARACTERS]
