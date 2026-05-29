from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import docker
import docker.models.containers
import httpx
from pydantic import BaseModel, ConfigDict, Field

from src.llm.client import Message, generate
from src.llm.config import ModelRole


class SweBenchInstance(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    repo: str
    problem_statement: str
    base_commit: str | None = None
    language: str | None = None
    fail_to_pass: list[str] = Field(default_factory=list)
    pass_to_pass: list[str] = Field(default_factory=list)
    docker_image: str | None = None


class WorkflowStartResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    workflow_id: str


class WorkflowStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    result: dict[str, object] | None = None


class WorkflowUsageSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_cost_usd: float
    call_count: int


class EvaluationTaskResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    status: str
    resolved: bool
    cost_usd: float
    llm_calls: int
    wall_clock_seconds: float
    reason: str | None = None
    patch: str | None = None


class LlmCallsPerTask(BaseModel):
    model_config = ConfigDict(frozen=True)

    resolved: float
    failed: float


class SkipReasonSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: str
    count: int


class EvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_results: list[EvaluationTaskResult]
    baseline_results: list[EvaluationTaskResult]
    resolved: int
    failed: int
    skipped: int
    skip_reasons: list[SkipReasonSummary]
    resolved_percent: float
    skipped_percent: float
    cost_per_resolved_task: float
    llm_calls_per_task: LlmCallsPerTask
    baseline_resolved_percent: float
    delta: float


class PatchApplicationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    applied: bool
    exit_code: int
    stdout: str
    stderr: str


class OracleCommandResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    command: str
    exit_code: int
    stdout: str
    stderr: str


class OracleResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    resolved: bool
    reason: str | None
    command_results: list[OracleCommandResult] = Field(default_factory=list)


SUPPORTED_LANGUAGES = frozenset({'python', 'typescript', 'javascript', 'js', 'ts'})
TERMINAL_WORKFLOW_STATUSES = frozenset({'completed', 'failed'})
SWE_BENCH_WORKDIR = '/testbed'
WORKFLOW_POLL_INTERVAL_SECONDS = 5
WORKFLOW_POLL_TIMEOUT_SECONDS = 7200


async def run_evaluation(
    instances_path: Path,
    temporal_api_url: str,
    output_path: Path,
    limit: int | None,
    docker_client: docker.DockerClient,
    supported_only: bool = False,
) -> EvaluationReport:
    instances = _select_evaluation_instances(
        instances=_load_instances(instances_path),
        limit=limit,
        supported_only=supported_only,
    )
    task_results: list[EvaluationTaskResult] = []
    baseline_results: list[EvaluationTaskResult] = []

    for instance in instances:
        if not _is_supported_instance(instance):
            task_results.append(_skipped_result(instance, 'unsupported_language'))
            continue
        task_results.append(await _run_framework_task(instance, temporal_api_url, docker_client))
        baseline_results.append(await _run_baseline_task(instance, docker_client))

    report = _build_report(task_results=task_results, baseline_results=baseline_results)
    output_path.write_text(report.model_dump_json(indent=2), encoding='utf-8')
    return report


def _load_instances(instances_path: Path) -> list[SweBenchInstance]:
    raw_instances = json.loads(instances_path.read_text(encoding='utf-8'))
    return [SweBenchInstance.model_validate(raw_instance) for raw_instance in raw_instances]


def _is_supported_instance(instance: SweBenchInstance) -> bool:
    if instance.language is None:
        return True
    return instance.language.lower() in SUPPORTED_LANGUAGES


def _select_evaluation_instances(
    instances: list[SweBenchInstance],
    limit: int | None,
    supported_only: bool,
) -> list[SweBenchInstance]:
    selected_instances: list[SweBenchInstance] = []
    eligible_count = 0
    for instance in instances:
        supported_instance = _is_supported_instance(instance)
        if supported_only and not supported_instance:
            continue
        selected_instances.append(instance)
        if supported_instance:
            eligible_count += 1
        if limit is not None and eligible_count >= limit:
            break
    return selected_instances


def _pull_official_image(instance: SweBenchInstance, docker_client: docker.DockerClient) -> None:
    image = _official_image(instance)
    docker_client.images.pull(image)


def _start_official_container(
    instance: SweBenchInstance, docker_client: docker.DockerClient
) -> str:
    image = _official_image(instance)
    container = docker_client.containers.run(
        image=image,
        command='sleep infinity',
        detach=True,
        working_dir=SWE_BENCH_WORKDIR,
    )
    return str(container.id)


def _stop_and_remove_container(container_id: str, docker_client: docker.DockerClient) -> None:
    container = docker_client.containers.get(container_id)
    container.stop(timeout=10)
    container.remove(force=True)


def _official_image(instance: SweBenchInstance) -> str:
    if instance.docker_image is None:
        raise ValueError(f'SWE-bench instance {instance.instance_id} is missing docker_image')
    return instance.docker_image


def _apply_patch_to_container(
    container_id: str,
    patch: str,
    docker_client: docker.DockerClient,
) -> PatchApplicationResult:
    if not patch.strip():
        raise ValueError('SWE-bench patch cannot be empty')
    encoded_patch = base64.b64encode(patch.encode()).decode()
    container = docker_client.containers.get(container_id)
    result = container.exec_run(
        ['sh', '-lc', f'echo {encoded_patch} | base64 -d | git apply -'],
        workdir=SWE_BENCH_WORKDIR,
        demux=True,
    )
    stdout_bytes, stderr_bytes = result.output
    stdout = stdout_bytes.decode('utf-8', errors='replace') if stdout_bytes else ''
    stderr = stderr_bytes.decode('utf-8', errors='replace') if stderr_bytes else ''
    return PatchApplicationResult(
        applied=result.exit_code == 0,
        exit_code=result.exit_code,
        stdout=stdout,
        stderr=stderr,
    )


def _run_oracle(
    container_id: str,
    instance: SweBenchInstance,
    docker_client: docker.DockerClient,
) -> OracleResult:
    command_results: list[OracleCommandResult] = []
    fail_to_pass_results = [
        _run_oracle_command(container_id, test_identifier, docker_client)
        for test_identifier in instance.fail_to_pass
    ]
    command_results.extend(fail_to_pass_results)
    if any(command_result.exit_code != 0 for command_result in fail_to_pass_results):
        return OracleResult(
            resolved=False,
            reason='fail_to_pass_failed',
            command_results=command_results,
        )

    pass_to_pass_results = [
        _run_oracle_command(container_id, test_identifier, docker_client)
        for test_identifier in instance.pass_to_pass
    ]
    command_results.extend(pass_to_pass_results)
    if any(command_result.exit_code != 0 for command_result in pass_to_pass_results):
        return OracleResult(
            resolved=False,
            reason='pass_to_pass_failed',
            command_results=command_results,
        )
    return OracleResult(resolved=True, reason=None, command_results=command_results)


def _evaluate_patch_with_oracle(
    instance: SweBenchInstance,
    patch: str,
    docker_client: docker.DockerClient,
) -> OracleResult:
    _pull_official_image(instance, docker_client)
    container_id = _start_official_container(instance, docker_client)
    try:
        materialized_patch = _materialize_patch(patch)
        patch_application = _apply_patch_to_container(
            container_id=container_id,
            patch=materialized_patch,
            docker_client=docker_client,
        )
        if not patch_application.applied:
            return OracleResult(resolved=False, reason='patch_apply_failed')
        return _run_oracle(
            container_id=container_id,
            instance=instance,
            docker_client=docker_client,
        )
    finally:
        _stop_and_remove_container(container_id, docker_client)


def _materialize_patch(patch: str) -> str:
    if patch.startswith('@'):
        return Path(patch[1:]).read_text(encoding='utf-8')
    return patch


def _run_oracle_command(
    container_id: str,
    test_identifier: str,
    docker_client: docker.DockerClient,
) -> OracleCommandResult:
    command = f'pytest {test_identifier}'
    container = docker_client.containers.get(container_id)
    result = container.exec_run(
        ['sh', '-lc', command],
        workdir=SWE_BENCH_WORKDIR,
        demux=True,
    )
    stdout_bytes, stderr_bytes = result.output
    stdout = stdout_bytes.decode('utf-8', errors='replace') if stdout_bytes else ''
    stderr = stderr_bytes.decode('utf-8', errors='replace') if stderr_bytes else ''
    return OracleCommandResult(
        command=command,
        exit_code=result.exit_code,
        stdout=stdout,
        stderr=stderr,
    )


def _extract_patch_from_workflow_result(workflow_result: dict[str, object]) -> str:
    inline_patch = workflow_result.get('patch')
    if isinstance(inline_patch, str) and inline_patch.strip():
        return inline_patch
    patch_artifact = workflow_result.get('patch_artifact')
    if isinstance(patch_artifact, dict):
        artifact_path = patch_artifact.get('path')
        if isinstance(artifact_path, str) and artifact_path.strip():
            return f'@{artifact_path}'
    raise ValueError('Workflow result did not include a patch')


def _skipped_result(instance: SweBenchInstance, reason: str) -> EvaluationTaskResult:
    return EvaluationTaskResult(
        instance_id=instance.instance_id,
        status='skipped',
        resolved=False,
        cost_usd=0.0,
        llm_calls=0,
        wall_clock_seconds=0.0,
        reason=reason,
        patch=None,
    )


async def _run_framework_task(
    instance: SweBenchInstance,
    temporal_api_url: str,
    docker_client: docker.DockerClient,
) -> EvaluationTaskResult:
    started_at = time.monotonic()
    async with httpx.AsyncClient(timeout=60) as http_client:
        response = await http_client.post(
            f'{temporal_api_url.rstrip("/")}/workflows',
            json={
                'workflow_name': 'main_workflow',
                'workflow_input': {
                    'request': {
                        'raw_request': instance.problem_statement,
                        'repo_path': instance.repo,
                        'docker_image': instance.docker_image,
                        'run_id': instance.instance_id,
                    }
                },
            },
        )
        response.raise_for_status()
        workflow_start = WorkflowStartResponse.model_validate(response.json())
        workflow_status = await _poll_workflow(
            http_client, temporal_api_url, workflow_start.workflow_id
        )

    if workflow_status.status != 'completed' or workflow_status.result is None:
        return EvaluationTaskResult(
            instance_id=instance.instance_id,
            status=workflow_status.status,
            resolved=False,
            cost_usd=0.0,
            llm_calls=0,
            wall_clock_seconds=time.monotonic() - started_at,
            reason='workflow_failed_or_incomplete',
            patch=None,
        )
    workflow_usage = _usage_from_workflow_result(workflow_status.result)
    try:
        patch = _extract_patch_from_workflow_result(workflow_status.result)
    except ValueError:
        return EvaluationTaskResult(
            instance_id=instance.instance_id,
            status='failed',
            resolved=False,
            cost_usd=workflow_usage.total_cost_usd,
            llm_calls=workflow_usage.call_count,
            wall_clock_seconds=time.monotonic() - started_at,
            reason='workflow_patch_missing',
            patch=None,
        )
    oracle_result = _evaluate_patch_with_oracle(
        instance=instance,
        patch=patch,
        docker_client=docker_client,
    )
    return EvaluationTaskResult(
        instance_id=instance.instance_id,
        status='resolved' if oracle_result.resolved else 'failed',
        resolved=oracle_result.resolved,
        cost_usd=workflow_usage.total_cost_usd,
        llm_calls=workflow_usage.call_count,
        wall_clock_seconds=time.monotonic() - started_at,
        reason=oracle_result.reason,
        patch=patch,
    )


async def _poll_workflow(
    http_client: httpx.AsyncClient,
    temporal_api_url: str,
    workflow_id: str,
) -> WorkflowStatusResponse:
    elapsed_seconds = 0
    while elapsed_seconds < WORKFLOW_POLL_TIMEOUT_SECONDS:
        response = await http_client.get(f'{temporal_api_url.rstrip("/")}/workflows/{workflow_id}')
        response.raise_for_status()
        workflow_status = WorkflowStatusResponse.model_validate(response.json())
        if workflow_status.status in TERMINAL_WORKFLOW_STATUSES:
            return workflow_status
        await asyncio.sleep(WORKFLOW_POLL_INTERVAL_SECONDS)
        elapsed_seconds += WORKFLOW_POLL_INTERVAL_SECONDS
    raise TimeoutError(
        f'Workflow {workflow_id} did not complete within {WORKFLOW_POLL_TIMEOUT_SECONDS} seconds'
    )


async def _run_baseline_task(
    instance: SweBenchInstance,
    docker_client: docker.DockerClient,
) -> EvaluationTaskResult:
    started_at = time.monotonic()
    patch_response = await generate(
        role=ModelRole.IMPLEMENTATION,
        messages=[
            Message(
                role='system',
                content='Produce a minimal unified diff patch for this SWE-bench task.',
            ),
            Message(role='user', content=instance.problem_statement),
        ],
    )
    patch = _extract_patch_from_llm_response(patch_response.content)
    oracle_result = _evaluate_patch_with_oracle(
        instance=instance,
        patch=patch,
        docker_client=docker_client,
    )
    return EvaluationTaskResult(
        instance_id=instance.instance_id,
        status='resolved' if oracle_result.resolved else 'failed',
        resolved=oracle_result.resolved,
        cost_usd=patch_response.cost_usd,
        llm_calls=1,
        wall_clock_seconds=time.monotonic() - started_at,
        reason=oracle_result.reason,
        patch=patch,
    )


def _extract_patch_from_llm_response(response_content: str) -> str:
    diff_fence_start = response_content.find('```diff')
    if diff_fence_start >= 0:
        patch_start = response_content.find('\n', diff_fence_start)
        patch_end = response_content.find('```', patch_start + 1)
        if patch_start >= 0 and patch_end >= 0:
            return response_content[patch_start + 1 : patch_end]
    return response_content


def _build_report(
    task_results: list[EvaluationTaskResult],
    baseline_results: list[EvaluationTaskResult],
) -> EvaluationReport:
    total_count = len(task_results)
    skipped_results = [
        task_result for task_result in task_results if task_result.status == 'skipped'
    ]
    skipped_count = len(skipped_results)
    resolved_results = [task_result for task_result in task_results if task_result.resolved]
    failed_results = [
        task_result
        for task_result in task_results
        if task_result.status != 'skipped' and not task_result.resolved
    ]
    baseline_resolved_results = [
        task_result for task_result in baseline_results if task_result.resolved
    ]
    eligible_count = max(total_count - skipped_count, 1)
    resolved_percent = (len(resolved_results) / eligible_count) * 100
    baseline_resolved_percent = (len(baseline_resolved_results) / eligible_count) * 100
    resolved_cost = sum(task_result.cost_usd for task_result in resolved_results)
    cost_per_resolved_task = resolved_cost / max(len(resolved_results), 1)
    return EvaluationReport(
        task_results=task_results,
        baseline_results=baseline_results,
        resolved=len(resolved_results),
        failed=len(failed_results),
        skipped=skipped_count,
        skip_reasons=_skip_reason_summaries(skipped_results),
        resolved_percent=resolved_percent,
        skipped_percent=(skipped_count / max(total_count, 1)) * 100,
        cost_per_resolved_task=cost_per_resolved_task,
        llm_calls_per_task=LlmCallsPerTask(
            resolved=_average_llm_calls(resolved_results),
            failed=_average_llm_calls(failed_results),
        ),
        baseline_resolved_percent=baseline_resolved_percent,
        delta=resolved_percent - baseline_resolved_percent,
    )


def _average_llm_calls(task_results: list[EvaluationTaskResult]) -> float:
    if not task_results:
        return 0.0
    return sum(task_result.llm_calls for task_result in task_results) / len(task_results)


def _skip_reason_summaries(
    skipped_results: list[EvaluationTaskResult],
) -> list[SkipReasonSummary]:
    reasons = sorted(
        {task_result.reason for task_result in skipped_results if task_result.reason is not None}
    )
    return [
        SkipReasonSummary(
            reason=reason,
            count=len(
                [task_result for task_result in skipped_results if task_result.reason == reason]
            ),
        )
        for reason in reasons
    ]


def _usage_from_workflow_result(workflow_result: dict[str, object]) -> WorkflowUsageSummary:
    raw_usage = workflow_result.get('llm_usage')
    if isinstance(raw_usage, dict):
        return WorkflowUsageSummary.model_validate(raw_usage)
    return WorkflowUsageSummary(total_cost_usd=0.0, call_count=0)


def _docker_client() -> docker.DockerClient:
    return docker.from_env()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--instances', required=True)
    parser.add_argument('--temporal-api-url', required=True)
    parser.add_argument('--output', default='swe_bench_results.json')
    parser.add_argument('--subset', type=int, default=None)
    parser.add_argument('--five-task-subset', action='store_true')
    arguments = parser.parse_args()
    report = asyncio.run(
        run_evaluation(
            instances_path=Path(arguments.instances),
            temporal_api_url=arguments.temporal_api_url,
            output_path=Path(arguments.output),
            limit=5 if arguments.five_task_subset else arguments.subset,
            docker_client=_docker_client(),
            supported_only=arguments.five_task_subset,
        )
    )
    print(report.model_dump_json(indent=2))


if __name__ == '__main__':
    main()
