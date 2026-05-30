from pathlib import Path

from pytest import MonkeyPatch
from src.eval.smoke_workflow import (
    SmokeWorkflowFinalResult,
    SmokeWorkflowInput,
    SmokeWorkflowRequest,
    _smoke_result_from_workflow_result,
    _start_and_wait_for_workflow,
)
from src.models.task import TaskRequest
from src.models.worker import Confidence, WorkerResult, WorkerStatus
from src.models.worker import TestResult as WorkerTestResult


async def test_start_and_wait_for_workflow_uses_temporal_client(monkeypatch: MonkeyPatch) -> None:
    repository_path = Path('C:/repo')

    class FakeHandle:
        workflow_id = 'wf-smoke'

        async def result(self, timeout: int) -> dict[str, object]:
            assert timeout == 1
            return {
                'status': 'accept',
                'worker_results': [
                    {
                        'status': 'success',
                        'patch_id': 'smoke',
                        'diff_summary': 'Added smoke subtract function and test.',
                        'tests_run': ['pytest -q'],
                        'test_results': [
                            {
                                'command': 'pytest -q',
                                'exit_code': 0,
                                'stdout_summary': '1 passed',
                                'stderr_summary': '',
                                'passed': True,
                            }
                        ],
                        'discovered_issues': [],
                        'confidence': 'high',
                        'replan_suggestion': None,
                    }
                ],
            }

    class FakeClient:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        async def start(self, workflow_name: str, **workflow_input: object) -> FakeHandle:
            assert self.base_url == 'http://temporal.local'
            assert workflow_name == 'main_workflow'
            request = workflow_input['request']
            assert isinstance(request, dict)
            assert request['repo_path'] == str(repository_path)
            return FakeHandle()

    monkeypatch.setattr('src.eval.smoke_workflow.Client', FakeClient, raising=False)

    result = await _start_and_wait_for_workflow(
        temporal_api_url='http://temporal.local',
        repository_path=repository_path,
        timeout_seconds=1,
    )

    assert result.workflow_id == 'wf-smoke'
    assert result.status == 'completed'
    assert result.changed_diff is True
    assert result.test_result_passed is True


def test_smoke_workflow_request_serializes_temporal_payload() -> None:
    request = SmokeWorkflowRequest(
        workflow_name='main_workflow',
        workflow_input=SmokeWorkflowInput(
            request=TaskRequest(raw_request='Smoke test', repo_path='C:/repo', run_id='smoke')
        ),
    )

    assert request.model_dump(mode='json')['workflow_input']['request']['run_id'] == 'smoke'


def test_smoke_result_requires_changed_diff_and_passing_test_result() -> None:
    result = _smoke_result_from_workflow_result(
        SmokeWorkflowFinalResult(
            worker_results=[
                WorkerResult(
                    status=WorkerStatus.SUCCESS,
                    patch_id='smoke',
                    diff_summary='Added smoke subtract function and test.',
                    tests_run=['pytest -q'],
                    test_results=[
                        WorkerTestResult(
                            command='pytest -q',
                            exit_code=0,
                            stdout_summary='1 passed',
                            stderr_summary='',
                            passed=True,
                        )
                    ],
                    discovered_issues=[],
                    confidence=Confidence.HIGH,
                    replan_suggestion=None,
                )
            ],
        ),
        workflow_id='wf-smoke',
    )

    assert result.status == 'completed'
    assert result.changed_diff is True
    assert result.test_result_passed is True
