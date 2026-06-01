from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import Field
from temporal_light import Client, WorkflowFailedError

from src.models.frozen_base_model import FrozenBaseModel

DEFAULT_DATASET_NAME = 'princeton-nlp/SWE-bench_Lite'
DEFAULT_SPLIT = 'test'
DEFAULT_MODEL_NAME = 'agentic-coding-runtime'
DEFAULT_PREDICTIONS_ROOT = Path('predictions')
DEFAULT_TEMPORAL_API_URL = 'http://localhost:8080'
DEFAULT_CONTAINER_REPO_PATH = '/testbed'
DEFAULT_WORKFLOW_TIMEOUT_SECONDS = 7200
DEFAULT_MAX_WORKERS = 1
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
    test_patch: str | None = None
    fail_to_pass: list[str] = Field(default_factory=list)
    pass_to_pass: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    language: str = PYTHON_LANGUAGE


class WorkflowUsageSummary(FrozenBaseModel):
    total_cost_usd: float = 0.0
    call_count: int = 0


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
    started_at: str | None = None
    completed_at: str | None = None
    instance: dict[str, object] = Field(default_factory=dict)


DatasetLoader = Callable[[str, str], Iterable[dict[str, object]]]
ClientFactory = Callable[[str], Client]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


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
) -> Path:
    instances = load_swe_bench_instances(
        dataset_name=dataset_name,
        split=split,
        limit=subset,
        dataset_loader=dataset_loader,
    )
    run_predictions_dir = predictions_dir / run_id
    run_predictions_dir.mkdir(parents=True, exist_ok=True)
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
        _write_prediction(prediction_path, prediction)
        prediction_records.append(prediction)
    return write_predictions_jsonl(run_predictions_dir, prediction_records)


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
        if not patch:
            status = 'failed'
            reason = 'workflow_patch_missing'

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
        started_at=started_at,
        completed_at=_utc_now(),
        instance=instance.model_dump(mode='json'),
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


def _workflow_run_id(run_id: str, instance_id: str) -> str:
    return f'{run_id}-{instance_id}'


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_id(dataset_name: str) -> str:
    dataset_slug = dataset_name.rsplit('/', maxsplit=1)[-1].lower().replace('_', '-')
    return f'agentic-{dataset_slug}'


async def _main_async(arguments: argparse.Namespace) -> None:
    predictions_path = arguments.predictions_dir / arguments.run_id / 'all_preds.jsonl'
    if not arguments.evaluate_only:
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
    if not arguments.generate_only:
        run_official_evaluation(
            dataset_name=arguments.dataset_name,
            predictions_path=predictions_path,
            run_id=arguments.run_id,
            max_workers=arguments.max_workers,
        )
    print(json.dumps({'predictions_path': str(predictions_path)}, indent=2))


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default=DEFAULT_DATASET_NAME)
    parser.add_argument('--split', default=DEFAULT_SPLIT)
    parser.add_argument('--subset', type=int, default=1)
    parser.add_argument('--temporal-api-url', default=DEFAULT_TEMPORAL_API_URL)
    parser.add_argument('--predictions-dir', type=Path, default=DEFAULT_PREDICTIONS_ROOT)
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--model-name-or-path', default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        '--workflow-timeout-seconds',
        type=int,
        default=DEFAULT_WORKFLOW_TIMEOUT_SECONDS,
    )
    parser.add_argument('--max-workers', type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    arguments = parser.parse_args()
    if arguments.run_id is None:
        arguments.run_id = _default_run_id(arguments.dataset_name)
    return arguments


def main() -> None:
    asyncio.run(_main_async(_parse_arguments()))


if __name__ == '__main__':
    main()
