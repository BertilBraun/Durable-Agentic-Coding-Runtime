from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.activities.temporal import durable_activity
from src.tools.definitions import RunTests, Tool
from src.tools.handlers import command_for_tool

WORKSPACE_IMAGE_ENVIRONMENT_NAME = "WORKSPACE_IMAGE"
DEFAULT_WORKSPACE_IMAGE = "durable-agentic-workspace:latest"
CONTAINER_WORKSPACE_PATH = "/workspace/repository"
MAX_OUTPUT_CHARACTERS = 20_000


class WorkspaceInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    volume_name: str
    worktree_path: str
    branch_name: str


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    stdout: str
    stderr: str
    exit_code: int
    truncated: bool


@durable_activity(retries=0, timeout=300)
async def create_workspace(run_id: str, repo_path: str) -> WorkspaceInfo:
    docker_client = _docker_client()
    volume_name = f"agentic-coding-{run_id}"
    branch_name = f"agentic-coding/{run_id}"
    docker_client.volumes.create(name=volume_name)
    repository_source_path = os.path.abspath(repo_path)
    workspace_root = Path(os.getenv("WORKSPACE_ROOT", ".agentic-workspaces")).resolve()
    worktree_path = workspace_root / run_id / "repository"
    if worktree_path.exists():
        shutil.rmtree(worktree_path)
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "sh",
        "-lc",
        (
            "rm -rf /target/repository && "
            "git clone /source /target/repository && "
            "cd /target/repository && "
            f"git checkout -b {branch_name}"
        ),
    ]
    docker_client.containers.run(
        image=_workspace_image(),
        command=command,
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
async def run_tool(workspace_info: WorkspaceInfo, tool: Tool) -> ToolResult:
    docker_client = _docker_client()
    container = docker_client.containers.run(
        image=_workspace_image(),
        command=command_for_tool(tool),
        detach=True,
        working_dir=CONTAINER_WORKSPACE_PATH,
        volumes={workspace_info.worktree_path: {"bind": CONTAINER_WORKSPACE_PATH, "mode": "rw"}},
    )
    wait_result = container.wait(timeout=_tool_timeout_seconds(tool))
    stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
    stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
    container.remove(force=True)
    truncated_stdout = _truncate(stdout)
    truncated_stderr = _truncate(stderr)
    return ToolResult(
        stdout=truncated_stdout,
        stderr=truncated_stderr,
        exit_code=int(wait_result.get("StatusCode", 1)),
        truncated=truncated_stdout != stdout or truncated_stderr != stderr,
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


def _docker_client() -> object:
    import docker

    return docker.from_env()


def _truncate(output: str) -> str:
    if len(output) <= MAX_OUTPUT_CHARACTERS:
        return output
    return output[:MAX_OUTPUT_CHARACTERS]
