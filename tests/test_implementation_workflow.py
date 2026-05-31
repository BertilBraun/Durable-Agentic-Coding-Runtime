import pytest
from src.activities.reviewer import ReviewDecision, ReviewVerdict
from src.llm.client import LLMUsage
from src.models.context import ContextPack
from src.models.plan import PlanStep, Risk
from src.models.repo import RepoIndex
from src.models.task import TaskContract, TaskType
from src.models.worker import Confidence, WorkerResult, WorkerStatus
from src.workflows.implementation_workflow import (
    _worker_result_from_review,
    implementation_workflow,
)


def _zero_usage() -> LLMUsage:
    return LLMUsage()


def _unit_usage() -> LLMUsage:
    return LLMUsage(call_count=1, total_input_tokens=10, total_cost_usd=0.01)


@pytest.mark.asyncio
async def test_implementation_workflow_reviews_successful_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review_requests: list[object] = []

    async def fake_gather_context(request: object) -> tuple[ContextPack, LLMUsage]:
        return (
            ContextPack(
                task_summary='Implement behavior',
                snippets=[],
                artifact_references=[],
                budget_remaining=1,
            ),
            _unit_usage(),
        )

    async def fake_run_implementation_turn(request: object) -> tuple[WorkerResult, LLMUsage]:
        return (
            WorkerResult(
                status=WorkerStatus.SUCCESS,
                patch_id='patch-1',
                diff_summary='Changed behavior.',
                tests_run=['pytest -q'],
                test_results=[],
                discovered_issues=[],
                confidence=Confidence.HIGH,
                replan_suggestion=None,
            ),
            _unit_usage(),
        )

    async def fake_get_full_diff(workspace_info: object) -> str:
        return 'diff --git a/app.py b/app.py'

    async def fake_review_patch(request: object) -> tuple[ReviewVerdict, LLMUsage]:
        review_requests.append(request)
        return (
            ReviewVerdict(
                verdict=ReviewDecision.ACCEPT,
                blocking_issues=[],
                non_blocking_issues=[],
                evidence=['diff reviewed'],
                missing_tests=[],
                regression_risks=[],
                minimality_assessment='minimal',
                recommended_next_action='accept',
            ),
            _unit_usage(),
        )

    monkeypatch.setattr('src.workflows.implementation_workflow.gather_context', fake_gather_context)
    monkeypatch.setattr(
        'src.workflows.implementation_workflow.run_implementation_turn',
        fake_run_implementation_turn,
    )
    monkeypatch.setattr('src.workflows.implementation_workflow.get_full_diff', fake_get_full_diff)
    monkeypatch.setattr('src.workflows.implementation_workflow.review_patch', fake_review_patch)

    result = await implementation_workflow(
        step=_plan_step().model_dump(mode='json'),
        workspace=_workspace(),
        contract=_task_contract().model_dump(mode='json'),
        repo_index=RepoIndex().model_dump(mode='json'),
    )

    assert result['worker_result']['status'] == WorkerStatus.SUCCESS
    assert result['llm_usage']['call_count'] == 3
    assert len(review_requests) == 1


@pytest.mark.asyncio
async def test_implementation_workflow_fails_successful_step_rejected_by_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_gather_context(request: object) -> tuple[ContextPack, LLMUsage]:
        return (
            ContextPack(
                task_summary='Implement behavior',
                snippets=[],
                artifact_references=[],
                budget_remaining=1,
            ),
            _zero_usage(),
        )

    async def fake_run_implementation_turn(request: object) -> tuple[WorkerResult, LLMUsage]:
        return (
            WorkerResult(
                status=WorkerStatus.SUCCESS,
                patch_id='patch-1',
                diff_summary='Changed behavior.',
                tests_run=['pytest -q'],
                test_results=[],
                discovered_issues=[],
                confidence=Confidence.HIGH,
                replan_suggestion=None,
            ),
            _zero_usage(),
        )

    async def fake_get_full_diff(workspace_info: object) -> str:
        return 'diff --git a/app.py b/app.py'

    async def fake_review_patch(request: object) -> tuple[ReviewVerdict, LLMUsage]:
        return (
            ReviewVerdict(
                verdict=ReviewDecision.REJECT,
                blocking_issues=['missing regression test'],
                non_blocking_issues=[],
                evidence=[],
                missing_tests=['regression test'],
                regression_risks=[],
                minimality_assessment='unclear',
                recommended_next_action='revise',
            ),
            _zero_usage(),
        )

    monkeypatch.setattr('src.workflows.implementation_workflow.gather_context', fake_gather_context)
    monkeypatch.setattr(
        'src.workflows.implementation_workflow.run_implementation_turn',
        fake_run_implementation_turn,
    )
    monkeypatch.setattr('src.workflows.implementation_workflow.get_full_diff', fake_get_full_diff)
    monkeypatch.setattr('src.workflows.implementation_workflow.review_patch', fake_review_patch)

    result = await implementation_workflow(
        step=_plan_step().model_dump(mode='json'),
        workspace=_workspace(),
        contract=_task_contract().model_dump(mode='json'),
        repo_index=RepoIndex().model_dump(mode='json'),
    )

    assert result['worker_result']['status'] == WorkerStatus.FAILED
    assert result['worker_result']['discovered_issues'] == ['missing regression test']


@pytest.mark.asyncio
async def test_implementation_workflow_returns_blocked_result_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn_calls = 0

    async def fake_gather_context(request: object) -> tuple[ContextPack, LLMUsage]:
        return (
            ContextPack(
                task_summary='Implement behavior',
                snippets=[],
                artifact_references=[],
                budget_remaining=1,
            ),
            _zero_usage(),
        )

    async def fake_run_implementation_turn(request: object) -> tuple[WorkerResult, LLMUsage]:
        nonlocal turn_calls
        turn_calls += 1
        return (
            WorkerResult(
                status=WorkerStatus.BLOCKED,
                patch_id=None,
                diff_summary='Context budget exceeded.',
                tests_run=[],
                test_results=[],
                discovered_issues=['context utilization exceeded 80 percent'],
                confidence=Confidence.LOW,
                replan_suggestion='Split the step.',
            ),
            _zero_usage(),
        )

    async def fake_review_patch(request: object) -> tuple[ReviewVerdict, LLMUsage]:
        raise AssertionError('blocked worker results should not be reviewed as successes')

    monkeypatch.setattr('src.workflows.implementation_workflow.gather_context', fake_gather_context)
    monkeypatch.setattr(
        'src.workflows.implementation_workflow.run_implementation_turn',
        fake_run_implementation_turn,
    )
    monkeypatch.setattr('src.workflows.implementation_workflow.review_patch', fake_review_patch)

    result = await implementation_workflow(
        step=_plan_step().model_dump(mode='json'),
        workspace=_workspace(),
        contract=_task_contract().model_dump(mode='json'),
        repo_index=RepoIndex().model_dump(mode='json'),
    )

    assert turn_calls == 1
    assert result['worker_result']['status'] == WorkerStatus.BLOCKED
    assert result['worker_result']['replan_suggestion'] == 'Split the step.'


@pytest.mark.asyncio
async def test_implementation_workflow_returns_failed_result_without_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn_calls = 0

    async def fake_gather_context(request: object) -> tuple[ContextPack, LLMUsage]:
        return (
            ContextPack(
                task_summary='Implement behavior',
                snippets=[],
                artifact_references=[],
                budget_remaining=1,
            ),
            _zero_usage(),
        )

    async def fake_run_implementation_turn(request: object) -> tuple[WorkerResult, LLMUsage]:
        nonlocal turn_calls
        turn_calls += 1
        return (
            WorkerResult(
                status=WorkerStatus.FAILED,
                patch_id=None,
                diff_summary='Attempt failed.',
                tests_run=[],
                test_results=[],
                discovered_issues=['transient tool failure'],
                confidence=Confidence.LOW,
                replan_suggestion=None,
            ),
            _zero_usage(),
        )

    async def fake_review_patch(request: object) -> tuple[ReviewVerdict, LLMUsage]:
        raise AssertionError('failed worker results should not be reviewed')

    monkeypatch.setattr('src.workflows.implementation_workflow.gather_context', fake_gather_context)
    monkeypatch.setattr(
        'src.workflows.implementation_workflow.run_implementation_turn',
        fake_run_implementation_turn,
    )
    monkeypatch.setattr('src.workflows.implementation_workflow.review_patch', fake_review_patch)

    result = await implementation_workflow(
        step=_plan_step().model_dump(mode='json'),
        workspace=_workspace(),
        contract=_task_contract().model_dump(mode='json'),
        repo_index=RepoIndex().model_dump(mode='json'),
    )

    assert turn_calls == 1
    assert result['worker_result']['status'] == WorkerStatus.FAILED
    assert result['worker_result']['discovered_issues'] == ['transient tool failure']


def test_revise_review_verdict_requests_replan_with_blocking_feedback() -> None:
    worker_result = WorkerResult(
        status=WorkerStatus.SUCCESS,
        patch_id='patch-1',
        diff_summary='Changed behavior.',
        tests_run=[],
        test_results=[],
        discovered_issues=[],
        confidence=Confidence.HIGH,
        replan_suggestion=None,
    )
    review_verdict = ReviewVerdict(
        verdict=ReviewDecision.REVISE,
        blocking_issues=['missing regression test', 'update generated fixture'],
        non_blocking_issues=[],
        evidence=[],
        missing_tests=['regression test'],
        regression_risks=[],
        minimality_assessment='incomplete',
        recommended_next_action='revise patch',
    )

    result = _worker_result_from_review(worker_result, review_verdict)

    assert result.status == WorkerStatus.NEEDS_REPLAN
    assert result.replan_suggestion == 'missing regression test; update generated fixture'


def _plan_step() -> PlanStep:
    return PlanStep(
        id='step_1',
        goal='Implement behavior',
        target_files=['app.py'],
        tests_to_run=['pytest -q'],
        expected_result='Behavior works',
        risk=Risk.LOW,
        requires_human_approval=False,
    )


def _task_contract() -> TaskContract:
    return TaskContract(
        task_type=TaskType.BUGFIX,
        goal='Fix behavior',
        acceptance_criteria=[],
        non_goals=[],
        affected_areas=[],
        risk_areas=[],
        tests_expected=[],
        open_questions=[],
    )


def _workspace() -> dict[str, str]:
    return {
        'kind': 'host',
        'run_id': 'run-1',
        'base_sha': 'basesha',
        'base_branch': 'main',
        'current_branch': 'main',
        'repo_path': 'workspace',
    }
