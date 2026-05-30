from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from temporal_light import Client, WorkflowFailedError

from src.models.task import TaskRequest
from src.models.worker import WorkerResult


class SmokeWorkflowInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: TaskRequest


class SmokeWorkflowRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    workflow_name: str
    workflow_input: SmokeWorkflowInput


class SmokeWorkflowResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    workflow_id: str
    status: str
    changed_diff: bool = False
    test_result_passed: bool = False


class WorkflowCompletedEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal['workflow_completed']
    result: SmokeWorkflowFinalResult


class SmokeWorkflowFinalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    worker_results: list[WorkerResult]


async def run_smoke_workflow(
    temporal_api_url: str,
    temporal_database_url: str,
    workspace_image: str,
    timeout_seconds: int,
) -> SmokeWorkflowResult:
    with tempfile.TemporaryDirectory() as temporary_directory:
        repository_path = _create_smoke_repository(Path(temporary_directory))
        worker_process = _start_worker(
            temporal_database_url=temporal_database_url,
            workspace_image=workspace_image,
        )
        try:
            return await _start_and_wait_for_workflow(
                temporal_api_url=temporal_api_url,
                repository_path=repository_path,
                timeout_seconds=timeout_seconds,
            )
        finally:
            worker_process.terminate()
            worker_process.wait(timeout=10)


def _start_worker(temporal_database_url: str, workspace_image: str) -> subprocess.Popen[str]:
    environment = {
        **os.environ,
        'TEMPORAL_DATABASE_URL': temporal_database_url,
        'WORKSPACE_IMAGE': workspace_image,
        'WORKSPACE_ROOT': '.agentic-workspaces',
    }
    return subprocess.Popen(
        [sys.executable, '-m', 'src.worker'],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _create_smoke_repository(temporary_directory: Path) -> Path:
    repository_path = temporary_directory / 'repo'
    repository_path.mkdir()
    (repository_path / 'app.py').write_text(
        'def add(first_number: int, second_number: int) -> int:\n    return first_number + second_number\n',
        encoding='utf-8',
    )
    _run_git(repository_path, 'init')
    _run_git(repository_path, 'config', 'user.email', 'test@example.com')
    _run_git(repository_path, 'config', 'user.name', 'Test User')
    _run_git(repository_path, 'add', '.')
    _run_git(repository_path, 'commit', '-m', 'initial')
    return repository_path


def _run_git(repository_path: Path, *arguments: str) -> None:
    subprocess.run(
        ['git', *arguments],
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    )


async def _start_and_wait_for_workflow(
    temporal_api_url: str,
    repository_path: Path,
    timeout_seconds: int,
) -> SmokeWorkflowResult:
    task_request = TaskRequest(
        raw_request='Add a subtract function to the app.py in the repo. Commit your work.',
        repo_path=str(repository_path),
        run_id='smoke-live',
    )
    client = Client(temporal_api_url)
    handle = await client.start(
        'main_workflow',
        request=task_request.model_dump(mode='json'),
    )
    try:
        workflow_result = await handle.result(timeout=timeout_seconds)
    except TimeoutError:
        return SmokeWorkflowResult(workflow_id=handle.workflow_id, status='timeout')
    except WorkflowFailedError:
        return SmokeWorkflowResult(workflow_id=handle.workflow_id, status='failed')
    return _smoke_result_from_workflow_result(
        SmokeWorkflowFinalResult.model_validate(workflow_result),
        workflow_id=handle.workflow_id,
    )


def _smoke_result_from_workflow_result(
    workflow_result: SmokeWorkflowFinalResult,
    workflow_id: str = '',
) -> SmokeWorkflowResult:
    completed_event = WorkflowCompletedEvent(type='workflow_completed', result=workflow_result)
    changed_diff = any(
        worker_result.diff_summary.strip() and 'no-op' not in worker_result.diff_summary.lower()
        for worker_result in completed_event.result.worker_results
    )
    test_result_passed = any(
        test_result.passed
        for worker_result in completed_event.result.worker_results
        for test_result in worker_result.test_results
    )
    status = 'completed' if changed_diff and test_result_passed else 'verification_failed'
    return SmokeWorkflowResult(
        workflow_id=workflow_id,
        status=status,
        changed_diff=changed_diff,
        test_result_passed=test_result_passed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--temporal-api-url', default='http://localhost:8080')
    parser.add_argument(
        '--temporal-database-url',
        default='postgresql://tl:changeme@localhost:5432/temporal_light',
    )
    parser.add_argument('--workspace-image', default='durable-agentic-workspace:latest')
    parser.add_argument('--timeout-seconds', type=int, default=120)
    arguments = parser.parse_args()
    result = asyncio.run(
        run_smoke_workflow(
            temporal_api_url=arguments.temporal_api_url,
            temporal_database_url=arguments.temporal_database_url,
            workspace_image=arguments.workspace_image,
            timeout_seconds=arguments.timeout_seconds,
        )
    )
    print(result.model_dump_json(indent=2))


if __name__ == '__main__':
    main()
