import pytest
from src.models.context import ContextPack
from src.models.plan import PlanStep, Risk
from src.models.repo import RepoIndex
from src.models.review import ReviewDecision, ReviewVerdict
from src.models.task import TaskContract, TaskType
from src.models.worker import Confidence, WorkerResult, WorkerStatus
from src.workflows.implementation_workflow import implementation_workflow


@pytest.mark.asyncio
async def test_implementation_workflow_reviews_successful_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review_requests: list[object] = []

    async def fake_build_repo_index(workspace_info: object) -> RepoIndex:
        return RepoIndex()

    async def fake_gather_context(request: object) -> ContextPack:
        return ContextPack(
            task_summary="Implement behavior",
            relevant_snippets=[],
            recent_observations=[],
            failed_attempt_summaries=[],
            available_tools=[],
            budget_remaining=1,
        )

    async def fake_run_implementation_turn(request: object) -> WorkerResult:
        return WorkerResult(
            status=WorkerStatus.SUCCESS,
            patch_id="patch-1",
            diff_summary="Changed behavior.",
            tests_run=["pytest -q"],
            test_results=[],
            discovered_issues=[],
            confidence=Confidence.HIGH,
            replan_suggestion=None,
        )

    async def fake_get_full_diff(workspace_info: object) -> str:
        return "diff --git a/app.py b/app.py"

    async def fake_review_patch(request: object) -> ReviewVerdict:
        review_requests.append(request)
        return ReviewVerdict(
            verdict=ReviewDecision.ACCEPT,
            blocking_issues=[],
            non_blocking_issues=[],
            evidence=["diff reviewed"],
            missing_tests=[],
            regression_risks=[],
            minimality_assessment="minimal",
            recommended_next_action="accept",
        )

    monkeypatch.setattr(
        "src.workflows.implementation_workflow.build_repo_index", fake_build_repo_index
    )
    monkeypatch.setattr("src.workflows.implementation_workflow.gather_context", fake_gather_context)
    monkeypatch.setattr(
        "src.workflows.implementation_workflow.run_implementation_turn",
        fake_run_implementation_turn,
    )
    monkeypatch.setattr("src.workflows.implementation_workflow.get_full_diff", fake_get_full_diff)
    monkeypatch.setattr("src.workflows.implementation_workflow.review_patch", fake_review_patch)

    result = await implementation_workflow(
        step=_plan_step().model_dump(mode="json"),
        workspace=_workspace(),
        contract=_task_contract().model_dump(mode="json"),
    )

    assert result["status"] == WorkerStatus.SUCCESS
    assert len(review_requests) == 1


@pytest.mark.asyncio
async def test_implementation_workflow_fails_successful_step_rejected_by_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_build_repo_index(workspace_info: object) -> RepoIndex:
        return RepoIndex()

    async def fake_gather_context(request: object) -> ContextPack:
        return ContextPack(
            task_summary="Implement behavior",
            relevant_snippets=[],
            recent_observations=[],
            failed_attempt_summaries=[],
            available_tools=[],
            budget_remaining=1,
        )

    async def fake_run_implementation_turn(request: object) -> WorkerResult:
        return WorkerResult(
            status=WorkerStatus.SUCCESS,
            patch_id="patch-1",
            diff_summary="Changed behavior.",
            tests_run=["pytest -q"],
            test_results=[],
            discovered_issues=[],
            confidence=Confidence.HIGH,
            replan_suggestion=None,
        )

    async def fake_get_full_diff(workspace_info: object) -> str:
        return "diff --git a/app.py b/app.py"

    async def fake_review_patch(request: object) -> ReviewVerdict:
        return ReviewVerdict(
            verdict=ReviewDecision.REJECT,
            blocking_issues=["missing regression test"],
            non_blocking_issues=[],
            evidence=[],
            missing_tests=["regression test"],
            regression_risks=[],
            minimality_assessment="unclear",
            recommended_next_action="revise",
        )

    monkeypatch.setattr(
        "src.workflows.implementation_workflow.build_repo_index", fake_build_repo_index
    )
    monkeypatch.setattr("src.workflows.implementation_workflow.gather_context", fake_gather_context)
    monkeypatch.setattr(
        "src.workflows.implementation_workflow.run_implementation_turn",
        fake_run_implementation_turn,
    )
    monkeypatch.setattr("src.workflows.implementation_workflow.get_full_diff", fake_get_full_diff)
    monkeypatch.setattr("src.workflows.implementation_workflow.review_patch", fake_review_patch)

    result = await implementation_workflow(
        step=_plan_step().model_dump(mode="json"),
        workspace=_workspace(),
        contract=_task_contract().model_dump(mode="json"),
    )

    assert result["status"] == WorkerStatus.FAILED
    assert result["discovered_issues"] == ["missing regression test"]


def _plan_step() -> PlanStep:
    return PlanStep(
        id="step_1",
        goal="Implement behavior",
        target_files=["app.py"],
        allowed_files=["app.py"],
        tests_to_run=["pytest -q"],
        expected_result="Behavior works",
        risk=Risk.LOW,
        requires_human_approval=False,
    )


def _task_contract() -> TaskContract:
    return TaskContract(
        task_type=TaskType.BUGFIX,
        goal="Fix behavior",
        acceptance_criteria=[],
        non_goals=[],
        affected_areas=[],
        risk_areas=[],
        tests_expected=[],
        open_questions=[],
    )


def _workspace() -> dict[str, str]:
    return {
        "run_id": "run-1",
        "volume_name": "volume",
        "worktree_path": "workspace",
        "branch_name": "branch",
    }
