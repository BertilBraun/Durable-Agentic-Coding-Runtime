from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest
from src.eval.swe_bench import (
    PatchComparison,
    PredictionRecord,
    _docker_image_for_instance,
    _main_async,
    generate_predictions,
    load_swe_bench_instances,
    prepare_swe_bench_images,
    run_official_evaluation,
    write_predictions_jsonl,
)


def test_load_swe_bench_instances_normalizes_official_dataset_rows() -> None:
    instances = load_swe_bench_instances(
        dataset_name='princeton-nlp/SWE-bench_Lite',
        split='test',
        limit=1,
        dataset_loader=lambda dataset_name, split: [
            {
                'instance_id': 'astropy__astropy-12907',
                'repo': 'astropy/astropy',
                'issue_id': 12907,
                'base_commit': 'abc123',
                'problem_statement': 'Fix the bug',
                'version': '5.0',
                'issue_url': 'https://github.com/astropy/astropy/issues/12907',
                'pr_url': 'https://github.com/astropy/astropy/pull/12908',
                'patch': 'gold patch is ignored',
                'test_patch': 'diff --git a/tests/test_bug.py b/tests/test_bug.py\n',
                'FAIL_TO_PASS': '["tests/test_bug.py::test_fixed"]',
                'PASS_TO_PASS': ['tests/test_existing.py::test_existing'],
            }
        ],
    )

    assert len(instances) == 1
    assert instances[0].instance_id == 'astropy__astropy-12907'
    assert instances[0].fail_to_pass == ['tests/test_bug.py::test_fixed']
    assert instances[0].pass_to_pass == ['tests/test_existing.py::test_existing']
    assert instances[0].test_patch == 'diff --git a/tests/test_bug.py b/tests/test_bug.py\n'
    assert instances[0].gold_patch == 'gold patch is ignored'
    assert instances[0].difficulty is None
    assert instances[0].language == 'python'


def test_load_swe_bench_instances_filters_to_python_before_limit() -> None:
    instances = load_swe_bench_instances(
        dataset_name='mixed-benchmark',
        split='test',
        limit=2,
        dataset_loader=lambda dataset_name, split: [
            _dataset_row('typescript__repo-1', language='typescript'),
            _dataset_row('python__repo-1', language='python'),
            _dataset_row('javascript__repo-1', language='javascript'),
            _dataset_row('python__repo-2', language='python'),
            _dataset_row('python__repo-3', language='python'),
        ],
    )

    assert [instance.instance_id for instance in instances] == [
        'python__repo-1',
        'python__repo-2',
    ]


def test_load_swe_bench_instances_treats_missing_language_as_unknown_for_unknown_dataset() -> None:
    with pytest.raises(ValueError, match='Requested 1 Python SWE-bench instances'):
        load_swe_bench_instances(
            dataset_name='custom/mixed-benchmark',
            split='test',
            limit=1,
            dataset_loader=lambda dataset_name, split: [
                _dataset_row('unknown__repo-1'),
            ],
        )


def test_load_swe_bench_instances_requires_subset_count_after_python_filter() -> None:
    with pytest.raises(ValueError, match='Requested 2 Python SWE-bench instances'):
        load_swe_bench_instances(
            dataset_name='mixed-benchmark',
            split='test',
            limit=2,
            dataset_loader=lambda dataset_name, split: [
                _dataset_row('typescript__repo-1', language='typescript'),
                _dataset_row('python__repo-1', language='python'),
                _dataset_row('javascript__repo-1', language='javascript'),
            ],
        )


@pytest.mark.asyncio
async def test_generate_predictions_writes_sidecar_and_official_jsonl(tmp_path: Path) -> None:
    async def fake_compare_patch_to_gold(**keyword_arguments: object) -> PatchComparison:
        assert keyword_arguments['instance_id'] == 'python__repo-1'
        assert keyword_arguments['model_patch'] == 'diff --git a/app.py b/app.py\n'
        assert keyword_arguments['gold_patch'] == 'diff --git a/gold.py b/gold.py\n'
        return PatchComparison(
            summary='Model patch changes the same file.',
            likely_equivalent=False,
            missing_from_model_patch=['gold behavior'],
            extra_in_model_patch=[],
            risk_notes=['needs review'],
        )

    predictions_path = await generate_predictions(
        dataset_name='princeton-nlp/SWE-bench_Lite',
        split='test',
        subset=1,
        temporal_api_url='http://temporal',
        predictions_dir=tmp_path,
        run_id='run-1',
        model_name_or_path='agentic-runtime',
        workflow_timeout_seconds=30,
        dataset_loader=_fake_dataset_loader,
        client_factory=FakeClient,
        docker_image_checker=lambda docker_images: None,
        patch_comparator=fake_compare_patch_to_gold,
    )

    sidecar = tmp_path / 'run-1' / 'python__repo-1.json'
    sidecar_payload = json.loads(sidecar.read_text(encoding='utf-8'))
    official_predictions = [
        json.loads(line) for line in predictions_path.read_text(encoding='utf-8').splitlines()
    ]

    assert predictions_path == tmp_path / 'run-1' / 'all_preds.jsonl'
    assert sidecar_payload['instance_id'] == 'python__repo-1'
    assert sidecar_payload['model_patch'] == 'diff --git a/app.py b/app.py\n'
    assert sidecar_payload['docker_image'] == 'sweb.eval.x86_64.python__repo-1:latest'
    assert sidecar_payload['cost'] == 1.25
    assert sidecar_payload['llm_calls'] == 7
    assert sidecar_payload['gold_patch'] == 'diff --git a/gold.py b/gold.py\n'
    assert sidecar_payload['patch_comparison'] == {
        'summary': 'Model patch changes the same file.',
        'likely_equivalent': False,
        'missing_from_model_patch': ['gold behavior'],
        'extra_in_model_patch': [],
        'risk_notes': ['needs review'],
    }
    assert official_predictions == [
        {
            'instance_id': 'python__repo-1',
            'model_name_or_path': 'agentic-runtime',
            'model_patch': 'diff --git a/app.py b/app.py\n',
        }
    ]
    assert FakeClient.started_requests[0]['request']['origin'] == {
        'kind': 'docker',
        'docker_image': 'sweb.eval.x86_64.python__repo-1:latest',
        'container_repo_path': '/testbed',
    }


@pytest.mark.asyncio
async def test_generate_predictions_reuses_existing_sidecar_without_force(tmp_path: Path) -> None:
    run_dir = tmp_path / 'run-1'
    run_dir.mkdir()
    existing_prediction = PredictionRecord(
        instance_id='python__repo-1',
        model_name_or_path='agentic-runtime',
        model_patch='diff --git a/cached.py b/cached.py\n',
        status='completed',
        dataset_name='princeton-nlp/SWE-bench_Lite',
        split='test',
        run_id='run-1',
        workflow_run_id='run-1-python__repo-1',
        docker_image='sweb.eval.x86_64.python__repo-1:latest',
        container_repo_path='/testbed',
    )
    (run_dir / 'python__repo-1.json').write_text(
        existing_prediction.model_dump_json(indent=2),
        encoding='utf-8',
    )
    FakeClient.started_requests = []

    predictions_path = await generate_predictions(
        dataset_name='princeton-nlp/SWE-bench_Lite',
        split='test',
        subset=1,
        temporal_api_url='http://temporal',
        predictions_dir=tmp_path,
        run_id='run-1',
        model_name_or_path='agentic-runtime',
        workflow_timeout_seconds=30,
        dataset_loader=_fake_dataset_loader,
        client_factory=FakeClient,
    )

    assert FakeClient.started_requests == []
    assert json.loads(predictions_path.read_text(encoding='utf-8'))['model_patch'] == (
        'diff --git a/cached.py b/cached.py\n'
    )


@pytest.mark.asyncio
async def test_generate_predictions_fails_before_workflow_when_image_is_missing(
    tmp_path: Path,
) -> None:
    FakeClient.started_requests = []

    def fake_image_checker(docker_images: list[str]) -> None:
        assert docker_images == ['sweb.eval.x86_64.python__repo-1:latest']
        raise RuntimeError(
            'Missing required SWE-bench Docker image. '
            '--tag latest --env_image_tag latest'
        )

    with pytest.raises(RuntimeError, match='--tag latest --env_image_tag latest'):
        await generate_predictions(
            dataset_name='princeton-nlp/SWE-bench_Lite',
            split='test',
            subset=1,
            temporal_api_url='http://temporal',
            predictions_dir=tmp_path,
            run_id='run-1',
            model_name_or_path='agentic-runtime',
            workflow_timeout_seconds=30,
            dataset_loader=_fake_dataset_loader,
            client_factory=FakeClient,
            docker_image_checker=fake_image_checker,
        )

    assert FakeClient.started_requests == []


def test_write_predictions_jsonl_keeps_only_official_eval_fields(tmp_path: Path) -> None:
    predictions_path = write_predictions_jsonl(
        tmp_path,
        [
            PredictionRecord(
                instance_id='python__repo-1',
                model_name_or_path='agentic-runtime',
                model_patch='diff --git a/app.py b/app.py\n',
                status='completed',
                dataset_name='princeton-nlp/SWE-bench_Lite',
                split='test',
                run_id='run-1',
                workflow_run_id='run-1-python__repo-1',
                docker_image='sweb.eval.x86_64.python__repo-1:latest',
                container_repo_path='/testbed',
                cost=1.25,
                llm_calls=7,
            )
        ],
    )

    assert json.loads(predictions_path.read_text(encoding='utf-8')) == {
        'instance_id': 'python__repo-1',
        'model_name_or_path': 'agentic-runtime',
        'model_patch': 'diff --git a/app.py b/app.py\n',
    }


def test_run_official_evaluation_invokes_swe_bench_module(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **keyword_arguments: object) -> FakeCompletedProcess:
        calls.append(command)
        assert keyword_arguments == {'check': True, 'text': True}
        return FakeCompletedProcess(returncode=0)

    result = run_official_evaluation(
        dataset_name='princeton-nlp/SWE-bench_Lite',
        predictions_path=tmp_path / 'all_preds.jsonl',
        run_id='run-1',
        max_workers=1,
        command_runner=fake_runner,
    )

    assert result.returncode == 0
    assert calls[0][1:3] == ['-m', 'swebench.harness.run_evaluation']
    assert '--dataset_name' in calls[0]
    assert 'princeton-nlp/SWE-bench_Lite' in calls[0]
    assert '--predictions_path' in calls[0]
    assert str(tmp_path / 'all_preds.jsonl') in calls[0]
    assert '--run_id' in calls[0]
    assert 'run-1' in calls[0]


def test_prepare_swe_bench_images_invokes_prepare_images_module() -> None:
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **keyword_arguments: object) -> FakeCompletedProcess:
        calls.append(command)
        assert keyword_arguments == {'check': True, 'text': True}
        return FakeCompletedProcess(returncode=0)

    result = prepare_swe_bench_images(
        dataset_name='princeton-nlp/SWE-bench_Lite',
        split='test',
        instance_ids=['astropy__astropy-12907'],
        max_workers=1,
        tag='latest',
        env_image_tag='latest',
        command_runner=fake_runner,
    )

    assert result.returncode == 0
    assert calls[0] == [
        calls[0][0],
        '-m',
        'swebench.harness.prepare_images',
        '--dataset_name',
        'princeton-nlp/SWE-bench_Lite',
        '--split',
        'test',
        '--instance_ids',
        'astropy__astropy-12907',
        '--max_workers',
        '1',
        '--tag',
        'latest',
        '--env_image_tag',
        'latest',
    ]


def test_docker_image_for_instance_uses_swe_bench_eval_image_name() -> None:
    assert (
        _docker_image_for_instance('astropy__astropy-12907')
        == 'sweb.eval.x86_64.astropy__astropy-12907:latest'
    )


@pytest.mark.asyncio
async def test_main_async_starts_worker_for_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, object]] = []

    def fake_prepare_swe_bench_images(**keyword_arguments: object) -> FakeCompletedProcess:
        calls.append(('prepare', keyword_arguments['instance_ids']))
        return FakeCompletedProcess(returncode=0)

    async def fake_generate_predictions(**keyword_arguments: object) -> Path:
        calls.append(('generate', keyword_arguments['temporal_api_url']))
        predictions_path = tmp_path / 'run-1' / 'all_preds.jsonl'
        predictions_path.parent.mkdir()
        predictions_path.write_text('', encoding='utf-8')
        return predictions_path

    def fake_start_worker(temporal_database_url: str) -> object:
        calls.append(('start_worker', temporal_database_url))
        return object()

    def fake_stop_worker(worker_process: object) -> None:
        calls.append(('stop_worker', worker_process))

    monkeypatch.setattr(
        'src.eval.swe_bench.prepare_swe_bench_images',
        fake_prepare_swe_bench_images,
    )
    monkeypatch.setattr('src.eval.swe_bench.generate_predictions', fake_generate_predictions)
    monkeypatch.setattr('src.eval.swe_bench._start_worker', fake_start_worker)
    monkeypatch.setattr('src.eval.swe_bench._stop_worker_process', fake_stop_worker)

    await _main_async(
        Namespace(
            dataset_name='princeton-nlp/SWE-bench_Lite',
            split='test',
            subset=1,
            temporal_api_url='http://temporal',
            temporal_database_url='postgresql://db',
            predictions_dir=tmp_path,
            run_id='run-1',
            model_name_or_path='model',
            workflow_timeout_seconds=30,
            force=False,
            evaluate_only=False,
            prepare_only=False,
            prepare_images=True,
            generate_only=True,
            max_workers=1,
            image_tag='latest',
            env_image_tag='latest',
            start_worker=True,
            dataset_loader=_fake_dataset_loader,
        )
    )

    assert calls[0] == ('prepare', ['python__repo-1'])
    assert calls[1] == ('start_worker', 'postgresql://db')
    assert calls[2] == ('generate', 'http://temporal')
    assert calls[3][0] == 'stop_worker'


@pytest.mark.asyncio
async def test_main_async_prepare_only_stops_before_worker_and_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, object]] = []

    def fake_prepare_swe_bench_images(**keyword_arguments: object) -> FakeCompletedProcess:
        calls.append(('prepare', keyword_arguments['instance_ids']))
        return FakeCompletedProcess(returncode=0)

    async def fake_generate_predictions(**keyword_arguments: object) -> Path:
        raise AssertionError('prepare-only should not generate predictions')

    def fake_start_worker(temporal_database_url: str) -> object:
        raise AssertionError('prepare-only should not start a worker')

    monkeypatch.setattr(
        'src.eval.swe_bench.prepare_swe_bench_images',
        fake_prepare_swe_bench_images,
    )
    monkeypatch.setattr('src.eval.swe_bench.generate_predictions', fake_generate_predictions)
    monkeypatch.setattr('src.eval.swe_bench._start_worker', fake_start_worker)

    await _main_async(
        Namespace(
            dataset_name='princeton-nlp/SWE-bench_Lite',
            split='test',
            subset=1,
            temporal_api_url='http://temporal',
            temporal_database_url='postgresql://db',
            predictions_dir=tmp_path,
            run_id='run-1',
            model_name_or_path='model',
            workflow_timeout_seconds=30,
            force=False,
            evaluate_only=False,
            prepare_only=True,
            prepare_images=True,
            generate_only=False,
            max_workers=1,
            image_tag='latest',
            env_image_tag='latest',
            start_worker=True,
            dataset_loader=_fake_dataset_loader,
        )
    )

    assert calls == [('prepare', ['python__repo-1'])]


def _fake_dataset_loader(dataset_name: str, split: str) -> list[dict[str, object]]:
    assert dataset_name == 'princeton-nlp/SWE-bench_Lite'
    assert split == 'test'
    return [_dataset_row('python__repo-1')]


def _dataset_row(instance_id: str, language: str | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        'instance_id': instance_id,
        'repo': 'python/repo',
        'base_commit': 'abc123',
        'problem_statement': 'Fix the bug',
        'patch': 'diff --git a/gold.py b/gold.py\n',
        'version': '1.0',
        'test_patch': '',
        'FAIL_TO_PASS': '[]',
        'PASS_TO_PASS': '[]',
    }
    if language is not None:
        row['language'] = language
    return row


class FakeHandle:
    async def result(self, timeout: int) -> dict[str, object]:
        assert timeout == 30
        return {
            'patch': 'diff --git a/app.py b/app.py\n',
            'llm_usage': {'total_cost_usd': 1.25, 'call_count': 7},
        }


class FakeClient:
    started_requests: ClassVar[list[dict[str, object]]] = []

    def __init__(self, base_url: str) -> None:
        assert base_url == 'http://temporal'

    async def start(self, workflow_name: str, **workflow_input: object) -> FakeHandle:
        assert workflow_name == 'main_workflow'
        self.started_requests.append(workflow_input)
        return FakeHandle()


@dataclass
class FakeCompletedProcess:
    returncode: int
