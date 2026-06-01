from pathlib import Path

import pytest
from pytest import MonkeyPatch
from src.activities.report_builder import FinalReport
from src.activities.reviewer import ReviewDecision, ReviewVerdict
from src.activities.workspace_manager import HostWorkspace
from src.eval.smoke_workflow import (
    _start_and_wait_for_workflow,
    _stop_worker_process,
)
from src.llm.client import LLMUsage
from src.models.plan import Plan
from src.models.task import TaskContract, TaskType


def _final_report() -> FinalReport:
    return FinalReport(
        status='accept',
        patch='diff --git a/app.py b/app.py',
        contract=TaskContract(task_type=TaskType.FEATURE, goal='Add subtract'),
        plan=Plan(
            summary='Add subtract',
            steps=[],
            integration_tests=[],
            definition_of_done=['diff reviewed'],
        ),
        worker_results=[],
        final_verdict=ReviewVerdict(
            verdict=ReviewDecision.ACCEPT,
            minimality_assessment='minimal',
            recommended_next_action='accept',
        ),
        workspace_info=HostWorkspace(
            run_id='smoke',
            base_sha='basesha',
            base_branch='main',
            current_branch='main',
            repo_path='C:/repo',
        ),
        llm_usage=LLMUsage(),
    )


async def test_start_and_wait_for_workflow_passes_origin_and_parses_report(
    monkeypatch: MonkeyPatch,
) -> None:
    repository_path = Path('C:/repo')
    final_report = _final_report()

    class FakeHandle:
        workflow_id = 'wf-smoke'

        async def result(self, timeout: int) -> dict[str, object]:
            assert timeout == 1
            return final_report.model_dump(mode='json')

    class FakeClient:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        async def start(self, workflow_name: str, **workflow_input: object) -> FakeHandle:
            assert self.base_url == 'http://temporal.local'
            assert workflow_name == 'main_workflow'
            request = workflow_input['request']
            assert isinstance(request, dict)
            assert request['origin']['kind'] == 'host'
            assert request['origin']['repo_path'] == str(repository_path)
            return FakeHandle()

    monkeypatch.setattr('src.eval.smoke_workflow.Client', FakeClient, raising=False)

    result = await _start_and_wait_for_workflow(
        temporal_api_url='http://temporal.local',
        repository_path=repository_path,
        timeout_seconds=1,
    )

    assert result.workflow_id == 'wf-smoke'
    assert result.status == 'success'
    assert result.report is not None
    assert result.report.patch == 'diff --git a/app.py b/app.py'


@pytest.mark.parametrize(
    ('raised_error', 'expected_status'),
    [(TimeoutError(), 'timeout')],
)
async def test_start_and_wait_for_workflow_reports_timeout(
    monkeypatch: MonkeyPatch,
    raised_error: Exception,
    expected_status: str,
) -> None:
    class FakeHandle:
        workflow_id = 'wf-smoke'

        async def result(self, timeout: int) -> dict[str, object]:
            raise raised_error

    class FakeClient:
        def __init__(self, base_url: str) -> None:
            pass

        async def start(self, workflow_name: str, **workflow_input: object) -> FakeHandle:
            return FakeHandle()

    monkeypatch.setattr('src.eval.smoke_workflow.Client', FakeClient, raising=False)

    result = await _start_and_wait_for_workflow(
        temporal_api_url='http://temporal.local',
        repository_path=Path('C:/repo'),
        timeout_seconds=1,
    )

    assert result.status == expected_status
    assert result.report is None


def test_stop_worker_process_kills_process_tree_on_windows(monkeypatch: MonkeyPatch) -> None:
    commands: list[list[str]] = []

    class FakeProcess:
        pid = 1234

        def poll(self) -> None:
            return None

        def wait(self, timeout: int) -> int:
            assert timeout == 10
            return 0

        def kill(self) -> None:
            raise AssertionError('kill should not be needed when taskkill succeeds')

    def fake_run(command: list[str], **keyword_arguments: object) -> None:
        commands.append(command)
        assert keyword_arguments['check'] is False
        assert keyword_arguments['capture_output'] is True
        assert keyword_arguments['text'] is True

    monkeypatch.setattr('src.eval.smoke_workflow.os.name', 'nt')
    monkeypatch.setattr('src.eval.smoke_workflow.subprocess.run', fake_run)

    _stop_worker_process(FakeProcess())  # type: ignore[arg-type]

    assert commands == [['taskkill', '/PID', '1234', '/T', '/F']]
