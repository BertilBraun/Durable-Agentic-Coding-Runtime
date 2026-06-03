from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import docker
from pydantic import Field
from temporal_light import Client, WorkflowFailedError

from src.config import ModelRole
from src.llm.client import Message, generate_structured
from src.models.frozen_base_model import FrozenBaseModel

DEFAULT_DATASET_NAME = 'princeton-nlp/SWE-bench_Lite'
DEFAULT_SPLIT = 'test'
DEFAULT_MODEL_NAME = 'agentic-coding-runtime'
DEFAULT_PREDICTIONS_ROOT = Path('predictions')
DEFAULT_TEMPORAL_API_URL = 'http://localhost:8080'
DEFAULT_TEMPORAL_DATABASE_URL = 'postgresql://tl:changeme@localhost:5432/temporal_light'
DEFAULT_CONTAINER_REPO_PATH = '/testbed'
DEFAULT_WORKFLOW_TIMEOUT_SECONDS = 7200
DEFAULT_MAX_WORKERS = 1
DEFAULT_IMAGE_TAG = 'latest'
DEFAULT_ENV_IMAGE_TAG = 'latest'
PYTHON_LANGUAGE = 'python'
UNKNOWN_LANGUAGE = 'unknown'


class SweBenchInstance(FrozenBaseModel):
    instance_id: str
    repo: str
    problem_statement: str
    base_commit: str
    version: str | None = None
    issue_id: str | int | None = None
    issue_url: str | None = None
    pr_url: str | None = None
    gold_patch: str | None = None
    test_patch: str | None = None
    fail_to_pass: list[str] = Field(default_factory=list)
    pass_to_pass: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    language: str = PYTHON_LANGUAGE


class WorkflowUsageSummary(FrozenBaseModel):
    total_cost_usd: float = 0.0
    call_count: int = 0


class PatchComparison(FrozenBaseModel):
    summary: str
    likely_equivalent: bool
    missing_from_model_patch: list[str] = Field(default_factory=list)
    extra_in_model_patch: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)


class PredictionRecord(FrozenBaseModel):
    instance_id: str
    model_name_or_path: str
    model_patch: str
    status: str
    dataset_name: str
    split: str
    run_id: str
    workflow_run_id: str
    docker_image: str
    container_repo_path: str
    cost: float = 0.0
    llm_calls: int = 0
    wall_clock_seconds: float = 0.0
    reason: str | None = None
    workflow_status: str | None = None
    agent_verdict: str | None = None
    reproduction_passed: bool | None = None
    official_prediction_emitted: bool = False
    started_at: str | None = None
    completed_at: str | None = None
    instance: dict[str, object] = Field(default_factory=dict)
    gold_patch: str | None = None
    patch_comparison: PatchComparison | None = None


DatasetLoader = Callable[[str, str], Iterable[dict[str, object]]]
ClientFactory = Callable[[str], Client]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
DockerImageChecker = Callable[[list[str]], None]
PatchComparator = Callable[..., Awaitable[PatchComparison]]


def load_swe_bench_instances(
    dataset_name: str,
    split: str,
    limit: int | None,
    dataset_loader: DatasetLoader | None = None,
) -> list[SweBenchInstance]:
    loader = dataset_loader or _load_dataset_rows
    rows = list(loader(dataset_name, split))
    rows = [row for row in rows if _is_python_dataset_row(row, dataset_name)]
    if limit is not None and len(rows) < limit:
        raise ValueError(
            f'Requested {limit} Python SWE-bench instances, but only found {len(rows)}.'
        )
    if limit is not None:
        rows = rows[:limit]
    return [_instance_from_dataset_row(row, dataset_name) for row in rows]


def _load_dataset_rows(dataset_name: str, split: str) -> Iterable[dict[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            'Missing eval dependency `datasets`. Install with the eval dependency group.'
        ) from error
    dataset = load_dataset(dataset_name, split=split)
    return [dict(row) for row in dataset]


def _instance_from_dataset_row(row: dict[str, object], dataset_name: str) -> SweBenchInstance:
    return SweBenchInstance(
        instance_id=str(row['instance_id']),
        repo=str(row['repo']),
        problem_statement=str(row['problem_statement']),
        base_commit=str(row['base_commit']),
        version=_optional_string(row.get('version')),
        issue_id=row.get('issue_id') if row.get('issue_id') is not None else None,
        issue_url=_optional_string(row.get('issue_url')),
        pr_url=_optional_string(row.get('pr_url')),
        gold_patch=_optional_string(row.get('patch')),
        test_patch=_optional_string(row.get('test_patch')),
        fail_to_pass=_coerce_test_list(row.get('FAIL_TO_PASS', [])),
        pass_to_pass=_coerce_test_list(row.get('PASS_TO_PASS', [])),
        difficulty=_optional_string(row.get('difficulty')),
        language=_language_from_dataset_row(row, dataset_name),
    )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_test_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            return []
        value = json.loads(stripped_value)
    if not isinstance(value, list):
        raise ValueError(f'Expected SWE-bench test list, got {type(value).__name__}')
    return [str(item) for item in value]


def _is_python_dataset_row(row: dict[str, object], dataset_name: str) -> bool:
    return _language_from_dataset_row(row, dataset_name) == PYTHON_LANGUAGE


def _language_from_dataset_row(row: dict[str, object], dataset_name: str) -> str:
    language = row.get('language')
    if language is None:
        if _is_known_python_swe_bench_dataset(dataset_name):
            return PYTHON_LANGUAGE
        return UNKNOWN_LANGUAGE
    return str(language).strip().lower()


def _is_known_python_swe_bench_dataset(dataset_name: str) -> bool:
    return dataset_name in {
        'princeton-nlp/SWE-bench',
        'princeton-nlp/SWE-bench_Lite',
        'princeton-nlp/SWE-bench_Verified',
    }


async def generate_predictions(
    dataset_name: str,
    split: str,
    subset: int | None,
    temporal_api_url: str,
    predictions_dir: Path,
    run_id: str,
    model_name_or_path: str,
    workflow_timeout_seconds: int,
    force: bool = False,
    dataset_loader: DatasetLoader | None = None,
    client_factory: ClientFactory = Client,
    docker_image_checker: DockerImageChecker | None = None,
    patch_comparator: PatchComparator | None = None,
) -> Path:
    docker_image_checker = docker_image_checker or ensure_docker_images_available
    instances = load_swe_bench_instances(
        dataset_name=dataset_name,
        split=split,
        limit=subset,
        dataset_loader=dataset_loader,
    )
    run_predictions_dir = predictions_dir / run_id
    run_predictions_dir.mkdir(parents=True, exist_ok=True)
    pending_instances = [
        instance
        for instance in instances
        if force or not _prediction_path(run_predictions_dir, instance.instance_id).exists()
    ]
    docker_image_checker(
        [_docker_image_for_instance(instance.instance_id) for instance in pending_instances]
    )
    prediction_records: list[PredictionRecord] = []
    for instance in instances:
        prediction_path = _prediction_path(run_predictions_dir, instance.instance_id)
        if prediction_path.exists() and not force:
            prediction_records.append(_read_prediction(prediction_path))
            continue
        prediction = await _run_agent_prediction(
            instance=instance,
            dataset_name=dataset_name,
            split=split,
            temporal_api_url=temporal_api_url,
            run_id=run_id,
            model_name_or_path=model_name_or_path,
            workflow_timeout_seconds=workflow_timeout_seconds,
            client_factory=client_factory,
        )
        if prediction.model_patch and instance.gold_patch:
            comparison = await (patch_comparator or compare_patch_to_gold)(
                instance_id=instance.instance_id,
                problem_statement=instance.problem_statement,
                model_patch=prediction.model_patch,
                gold_patch=instance.gold_patch,
            )
            prediction = prediction.model_copy(
                update={
                    'gold_patch': instance.gold_patch,
                    'patch_comparison': comparison,
                }
            )
        _write_prediction(prediction_path, prediction)
        prediction_records.append(prediction)
    return write_predictions_jsonl(run_predictions_dir, prediction_records)


def ensure_docker_images_available(docker_images: list[str]) -> None:
    docker_client = docker.from_env()
    missing_images: list[str] = []
    for docker_image in docker_images:
        try:
            docker_client.images.get(docker_image)
        except docker.errors.ImageNotFound:
            missing_images.append(docker_image)
    if missing_images:
        missing_list = ', '.join(missing_images)
        instance_ids = ', '.join(_instance_id_from_docker_image(image) for image in missing_images)
        raise RuntimeError(
            'Missing required SWE-bench Docker image(s): '
            f'{missing_list}. Build them locally before generation, preferably from WSL/Linux, '
            'then rerun this command. Example: '
            'uv run --group eval python -m swebench.harness.prepare_images '
            f'--dataset_name {DEFAULT_DATASET_NAME} --split {DEFAULT_SPLIT} '
            f'--instance_ids {instance_ids} --max_workers 1 '
            '--tag latest --env_image_tag latest'
        )


async def compare_patch_to_gold(
    instance_id: str,
    problem_statement: str,
    model_patch: str,
    gold_patch: str,
) -> PatchComparison:
    completion = await generate_structured(
        role=ModelRole.REVIEWER,
        messages=[
            Message(
                role='system',
                content=(
                    'You compare a generated SWE-bench patch with the official gold patch '
                    'for post-run introspection only. Do not grade by tests and do not suggest '
                    'changes for resubmission. Identify likely semantic gaps, extra behavior, '
                    'and risks using only the problem statement and the two diffs. The generated '
                    'patch can be different from the gold patch and still be likely equivalent.'
                ),
                cacheable=True,
            ),
            Message(
                role='user',
                content=json.dumps(
                    {
                        'instance_id': instance_id,
                        'problem_statement': problem_statement,
                        'model_patch': model_patch,
                        'gold_patch': gold_patch,
                    }
                ),
            ),
        ],
        output_type=PatchComparison,
    )
    return completion.output


def prepare_swe_bench_images(
    dataset_name: str,
    split: str,
    instance_ids: Sequence[str],
    max_workers: int,
    tag: str,
    env_image_tag: str,
    command_runner: CommandRunner = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        '-m',
        'swebench.harness.prepare_images',
        '--dataset_name',
        dataset_name,
        '--split',
        split,
        '--instance_ids',
        *instance_ids,
        '--max_workers',
        str(max_workers),
        '--tag',
        tag,
        '--env_image_tag',
        env_image_tag,
    ]
    return command_runner(command, check=True, text=True)


async def _run_agent_prediction(
    instance: SweBenchInstance,
    dataset_name: str,
    split: str,
    temporal_api_url: str,
    run_id: str,
    model_name_or_path: str,
    workflow_timeout_seconds: int,
    client_factory: ClientFactory,
) -> PredictionRecord:
    started_at_monotonic = time.monotonic()
    started_at = _utc_now()
    workflow_run_id = _workflow_run_id(run_id, instance.instance_id)
    docker_image = _docker_image_for_instance(instance.instance_id)
    client = client_factory(temporal_api_url)
    handle = await client.start(
        'main_workflow',
        request={
            'raw_request': instance.problem_statement,
            'origin': {
                'kind': 'docker',
                'docker_image': docker_image,
                'container_repo_path': DEFAULT_CONTAINER_REPO_PATH,
            },
            'run_id': workflow_run_id,
        },
    )
    workflow_result: dict[str, object] | None = None
    status = 'completed'
    reason: str | None = None
    try:
        raw_result = await handle.result(timeout=workflow_timeout_seconds)
        if isinstance(raw_result, dict):
            workflow_result = raw_result
        else:
            status = 'failed'
            reason = 'workflow_result_invalid'
    except TimeoutError:
        status = 'timeout'
        reason = 'workflow_timeout'
    except WorkflowFailedError:
        status = 'failed'
        reason = 'workflow_failed'

    patch = ''
    usage = WorkflowUsageSummary()
    if workflow_result is not None:
        patch = _extract_patch(workflow_result)
        usage = _usage_from_workflow_result(workflow_result)
        workflow_status = _optional_workflow_string(workflow_result, 'workflow_status')
        agent_verdict = _optional_workflow_string(workflow_result, 'agent_verdict')
        reproduction_passed = _optional_workflow_bool(workflow_result, 'reproduction_passed')
        official_prediction_emitted = bool(
            workflow_result.get('official_prediction_emitted', bool(patch))
        )
        if not patch:
            status = 'failed'
            reason = 'workflow_patch_missing'
    else:
        workflow_status = None
        agent_verdict = None
        reproduction_passed = None
        official_prediction_emitted = False

    return PredictionRecord(
        instance_id=instance.instance_id,
        model_name_or_path=model_name_or_path,
        model_patch=patch,
        status=status,
        dataset_name=dataset_name,
        split=split,
        run_id=run_id,
        workflow_run_id=workflow_run_id,
        docker_image=docker_image,
        container_repo_path=DEFAULT_CONTAINER_REPO_PATH,
        cost=usage.total_cost_usd,
        llm_calls=usage.call_count,
        wall_clock_seconds=time.monotonic() - started_at_monotonic,
        reason=reason,
        workflow_status=workflow_status,
        agent_verdict=agent_verdict,
        reproduction_passed=reproduction_passed,
        official_prediction_emitted=official_prediction_emitted,
        started_at=started_at,
        completed_at=_utc_now(),
        instance=instance.model_dump(mode='json'),
        gold_patch=instance.gold_patch,
    )


def _extract_patch(workflow_result: dict[str, object]) -> str:
    patch = workflow_result.get('patch')
    if isinstance(patch, str):
        return patch
    return ''


def _usage_from_workflow_result(workflow_result: dict[str, object]) -> WorkflowUsageSummary:
    raw_usage = workflow_result.get('llm_usage')
    if isinstance(raw_usage, dict):
        return WorkflowUsageSummary.model_validate(raw_usage)
    return WorkflowUsageSummary()


def _optional_workflow_string(workflow_result: dict[str, object], key: str) -> str | None:
    value = workflow_result.get(key)
    if isinstance(value, str):
        return value
    return None


def _optional_workflow_bool(workflow_result: dict[str, object], key: str) -> bool | None:
    value = workflow_result.get(key)
    if isinstance(value, bool):
        return value
    return None


def write_predictions_jsonl(
    run_predictions_dir: Path,
    predictions: Sequence[PredictionRecord],
) -> Path:
    predictions_path = run_predictions_dir / 'all_preds.jsonl'
    with predictions_path.open('w', encoding='utf-8') as predictions_file:
        for prediction in predictions:
            predictions_file.write(json.dumps(_official_prediction(prediction)) + '\n')
    return predictions_path


def _official_prediction(prediction: PredictionRecord) -> dict[str, str]:
    return {
        'instance_id': prediction.instance_id,
        'model_name_or_path': prediction.model_name_or_path,
        'model_patch': prediction.model_patch,
    }


def run_official_evaluation(
    dataset_name: str,
    predictions_path: Path,
    run_id: str,
    max_workers: int,
    command_runner: CommandRunner = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        '-m',
        'swebench.harness.run_evaluation',
        '--dataset_name',
        dataset_name,
        '--predictions_path',
        str(predictions_path),
        '--max_workers',
        str(max_workers),
        '--run_id',
        run_id,
    ]
    return command_runner(command, check=True, text=True)


def _prediction_path(run_predictions_dir: Path, instance_id: str) -> Path:
    return run_predictions_dir / f'{instance_id}.json'


def _read_prediction(path: Path) -> PredictionRecord:
    return PredictionRecord.model_validate_json(path.read_text(encoding='utf-8'))


def _write_prediction(path: Path, prediction: PredictionRecord) -> None:
    path.write_text(prediction.model_dump_json(indent=2), encoding='utf-8')


def _docker_image_for_instance(instance_id: str) -> str:
    return f'sweb.eval.x86_64.{instance_id}:latest'


def _instance_id_from_docker_image(docker_image: str) -> str:
    image_name = docker_image.removeprefix('sweb.eval.x86_64.')
    return image_name.removesuffix(':latest')


def _workflow_run_id(run_id: str, instance_id: str) -> str:
    return f'{run_id}-{instance_id}'


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_id(dataset_name: str) -> str:
    dataset_slug = dataset_name.rsplit('/', maxsplit=1)[-1].lower().replace('_', '-')
    return f'agentic-{dataset_slug}'


async def _main_async(arguments: argparse.Namespace) -> None:
    predictions_path = arguments.predictions_dir / arguments.run_id / 'all_preds.jsonl'
    worker_process: subprocess.Popen[str] | None = None
    if not arguments.evaluate_only:
        if arguments.prepare_images:
            instances = load_swe_bench_instances(
                dataset_name=arguments.dataset_name,
                split=arguments.split,
                limit=arguments.subset,
                dataset_loader=getattr(arguments, 'dataset_loader', None),
            )
            prepare_swe_bench_images(
                dataset_name=arguments.dataset_name,
                split=arguments.split,
                instance_ids=[instance.instance_id for instance in instances],
                max_workers=arguments.max_workers,
                tag=arguments.image_tag,
                env_image_tag=arguments.env_image_tag,
            )
        if arguments.prepare_only:
            print(json.dumps({'prepared': True}, indent=2))
            return
        if arguments.start_worker:
            worker_process = _start_worker(arguments.temporal_database_url)
        try:
            predictions_path = await generate_predictions(
                dataset_name=arguments.dataset_name,
                split=arguments.split,
                subset=arguments.subset,
                temporal_api_url=arguments.temporal_api_url,
                predictions_dir=arguments.predictions_dir,
                run_id=arguments.run_id,
                model_name_or_path=arguments.model_name_or_path,
                workflow_timeout_seconds=arguments.workflow_timeout_seconds,
                force=arguments.force,
            )
        finally:
            if worker_process is not None:
                _stop_worker_process(worker_process)
    if not arguments.generate_only:
        run_official_evaluation(
            dataset_name=arguments.dataset_name,
            predictions_path=predictions_path,
            run_id=arguments.run_id,
            max_workers=arguments.max_workers,
        )
    print(json.dumps({'predictions_path': str(predictions_path)}, indent=2))


def _start_worker(temporal_database_url: str) -> subprocess.Popen[str]:
    environment = {
        **os.environ,
        'TEMPORAL_DATABASE_URL': temporal_database_url,
    }
    return subprocess.Popen(
        [sys.executable, '-m', 'src.worker'],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _stop_worker_process(worker_process: subprocess.Popen[str]) -> None:
    if worker_process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(
            ['taskkill', '/PID', str(worker_process.pid), '/T', '/F'],
            check=False,
            capture_output=True,
            text=True,
        )
    else:
        worker_process.terminate()
    try:
        worker_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        worker_process.kill()
        worker_process.wait(timeout=10)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default=DEFAULT_DATASET_NAME)
    parser.add_argument('--split', default=DEFAULT_SPLIT)
    parser.add_argument('--subset', type=int, default=1)
    parser.add_argument('--temporal-api-url', default=DEFAULT_TEMPORAL_API_URL)
    parser.add_argument('--temporal-database-url', default=DEFAULT_TEMPORAL_DATABASE_URL)
    parser.add_argument('--predictions-dir', type=Path, default=DEFAULT_PREDICTIONS_ROOT)
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--model-name-or-path', default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        '--workflow-timeout-seconds',
        type=int,
        default=DEFAULT_WORKFLOW_TIMEOUT_SECONDS,
    )
    parser.add_argument('--max-workers', type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument('--image-tag', default=DEFAULT_IMAGE_TAG)
    parser.add_argument('--env-image-tag', default=DEFAULT_ENV_IMAGE_TAG)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument(
        '--prepare-images',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument('--start-worker', action=argparse.BooleanOptionalAction, default=True)
    arguments = parser.parse_args()
    if arguments.run_id is None:
        arguments.run_id = _default_run_id(arguments.dataset_name)
    return arguments


def main() -> None:
    asyncio.run(_main_async(_parse_arguments()))


if __name__ == '__main__':
    main()
