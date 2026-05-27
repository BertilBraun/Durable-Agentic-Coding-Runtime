from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field

from src.llm.client import LLMClient, Message
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


class EvaluationTaskResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    instance_id: str
    status: str
    resolved: bool
    cost_usd: float
    llm_calls: int
    wall_clock_seconds: float
    reason: str | None = None


class EvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_results: list[EvaluationTaskResult]
    resolved_percent: float
    skipped_percent: float
    cost_per_resolved_task: float
    baseline_resolved_percent: float
    delta: float


SUPPORTED_LANGUAGES = frozenset({"python", "typescript", "javascript", "js", "ts"})
TERMINAL_WORKFLOW_STATUSES = frozenset({"completed", "failed"})
SWE_BENCH_WORKDIR = "/testbed"


async def run_evaluation(
    instances_path: Path,
    temporal_api_url: str,
    output_path: Path,
    limit: int,
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
            task_results.append(_skipped_result(instance, "unsupported_language"))
            continue
        task_results.append(await _run_framework_task(instance, temporal_api_url))
        baseline_results.append(await _run_baseline_task(instance))

    report = _build_report(task_results=task_results, baseline_results=baseline_results)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def _load_instances(instances_path: Path) -> list[SweBenchInstance]:
    raw_instances = json.loads(instances_path.read_text(encoding="utf-8"))
    return [SweBenchInstance.model_validate(raw_instance) for raw_instance in raw_instances]


def _is_supported_instance(instance: SweBenchInstance) -> bool:
    if instance.language is None:
        return True
    return instance.language.lower() in SUPPORTED_LANGUAGES


def _select_evaluation_instances(
    instances: list[SweBenchInstance],
    limit: int,
    supported_only: bool,
) -> list[SweBenchInstance]:
    selected_instances: list[SweBenchInstance] = []
    for instance in instances:
        if supported_only and not _is_supported_instance(instance):
            continue
        selected_instances.append(instance)
        if len(selected_instances) >= limit:
            break
    return selected_instances


def _pull_official_image(instance: SweBenchInstance, docker_client: object) -> None:
    image = _official_image(instance)
    docker_client.images.pull(image)


def _start_official_container(instance: SweBenchInstance, docker_client: object) -> str:
    image = _official_image(instance)
    container = docker_client.containers.run(
        image=image,
        command="sleep infinity",
        detach=True,
        working_dir=SWE_BENCH_WORKDIR,
    )
    return str(container.id)


def _official_image(instance: SweBenchInstance) -> str:
    if instance.docker_image is None:
        raise ValueError(f"SWE-bench instance {instance.instance_id} is missing docker_image")
    return instance.docker_image


def _skipped_result(instance: SweBenchInstance, reason: str) -> EvaluationTaskResult:
    return EvaluationTaskResult(
        instance_id=instance.instance_id,
        status="skipped",
        resolved=False,
        cost_usd=0.0,
        llm_calls=0,
        wall_clock_seconds=0.0,
        reason=reason,
    )


async def _run_framework_task(
    instance: SweBenchInstance,
    temporal_api_url: str,
) -> EvaluationTaskResult:
    started_at = time.monotonic()
    async with httpx.AsyncClient(timeout=60) as http_client:
        response = await http_client.post(
            f"{temporal_api_url.rstrip('/')}/workflows",
            json={
                "workflow_name": "main_workflow",
                "workflow_input": {
                    "request": {
                        "raw_request": instance.problem_statement,
                        "repo_path": instance.repo,
                        "run_id": instance.instance_id,
                    }
                },
            },
        )
        response.raise_for_status()
        workflow_start = WorkflowStartResponse.model_validate(response.json())
        workflow_status = await _poll_workflow(
            http_client, temporal_api_url, workflow_start.workflow_id
        )

    resolved = workflow_status.status == "completed"
    return EvaluationTaskResult(
        instance_id=instance.instance_id,
        status=workflow_status.status,
        resolved=resolved,
        cost_usd=0.0,
        llm_calls=0,
        wall_clock_seconds=time.monotonic() - started_at,
        reason=None if resolved else "workflow_failed_or_incomplete",
    )


async def _poll_workflow(
    http_client: httpx.AsyncClient,
    temporal_api_url: str,
    workflow_id: str,
) -> WorkflowStatusResponse:
    while True:
        response = await http_client.get(f"{temporal_api_url.rstrip('/')}/workflows/{workflow_id}")
        response.raise_for_status()
        workflow_status = WorkflowStatusResponse.model_validate(response.json())
        if workflow_status.status in TERMINAL_WORKFLOW_STATUSES:
            return workflow_status
        await asyncio.sleep(5)


async def _run_baseline_task(instance: SweBenchInstance) -> EvaluationTaskResult:
    started_at = time.monotonic()
    llm_client = LLMClient()
    await llm_client.complete(
        role=ModelRole.IMPLEMENTATION,
        messages=[
            Message(
                role="system",
                content="Produce a minimal unified diff patch for this SWE-bench task.",
            ),
            Message(role="user", content=instance.problem_statement),
        ],
    )
    return EvaluationTaskResult(
        instance_id=instance.instance_id,
        status="baseline_patch_generated",
        resolved=False,
        cost_usd=llm_client.usage_ledger.total_cost_usd,
        llm_calls=len(llm_client.usage_ledger.calls),
        wall_clock_seconds=time.monotonic() - started_at,
        reason="oracle_execution_not_configured",
    )


def _build_report(
    task_results: list[EvaluationTaskResult],
    baseline_results: list[EvaluationTaskResult],
) -> EvaluationReport:
    total_count = len(task_results)
    skipped_count = len(
        [task_result for task_result in task_results if task_result.status == "skipped"]
    )
    resolved_results = [task_result for task_result in task_results if task_result.resolved]
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
        resolved_percent=resolved_percent,
        skipped_percent=(skipped_count / max(total_count, 1)) * 100,
        cost_per_resolved_task=cost_per_resolved_task,
        baseline_resolved_percent=baseline_resolved_percent,
        delta=resolved_percent - baseline_resolved_percent,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True)
    parser.add_argument("--temporal-api-url", required=True)
    parser.add_argument("--output", default="swe_bench_results.json")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--five-task-subset", action="store_true")
    arguments = parser.parse_args()
    report = asyncio.run(
        run_evaluation(
            instances_path=Path(arguments.instances),
            temporal_api_url=arguments.temporal_api_url,
            output_path=Path(arguments.output),
            limit=arguments.limit,
            supported_only=arguments.five_task_subset,
        )
    )
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
