import pytest
from src.activities.complexity_assessor import ComplexityVerdict
from src.activities.planner import PlanRequest
from src.activities.reviewer import ReviewDecision, ReviewRequest, ReviewVerdict
from src.activities.workspace_manager import WorkspaceInfo
from src.llm.client import LLMUsage
from src.models.plan import Plan, PlanStep, Risk
from src.models.repo import RepoIndex
from src.models.task import TaskContract, TaskRequest, TaskType
from src.models.worker import Confidence, WorkerResult, WorkerStatus
from src.workflows.main_workflow import main_workflow
from temporal_light.exceptions import WorkflowSuspended


def _unit_usage() -> LLMUsage:
    return LLMUsage(call_count=1, total_input_tokens=10, total_cost_usd=0.01)


@pytest.mark.asyncio
async def test_main_workflow_replans_after_needs_replan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_requests: list[PlanRequest] = []
    spawned_step_ids: list[str] = []
    child_payloads = [
        {
            'worker_result': WorkerResult(
                status=WorkerStatus.NEEDS_REPLAN,
                patch_id=None,
                diff_summary='The original plan missed generated files.',
                tests_run=[],
                test_results=[],
                discovered_issues=['missing generated file'],
                confidence=Confidence.LOW,
                replan_suggestion='Add the generated file update.',
            ).model_dump(mode='json'),
            'llm_usage': _unit_usage().model_dump(mode='json'),
        },
        {
            'worker_result': WorkerResult(
                status=WorkerStatus.SUCCESS,
                patch_id='patch-2',
                diff_summary='Updated generated file.',
                tests_run=[],
                test_results=[],
                discovered_issues=[],
                confidence=Confidence.HIGH,
                replan_suggestion=None,
            ).model_dump(mode='json'),
            'llm_usage': _unit_usage().model_dump(mode='json'),
        },
    ]
    workspace_info = WorkspaceInfo(
        run_id='run-1',
        volume_name='volume',
        worktree_path='workspace',
        branch_name='branch',
    )
    contract = TaskContract(
        task_type=TaskType.BUGFIX,
        goal='Fix generated output',
        acceptance_criteria=[],
        non_goals=[],
        affected_areas=[],
        risk_areas=[],
        tests_expected=[],
        open_questions=[],
    )

    async def fake_build_contract(task_request: TaskRequest) -> tuple[TaskContract, LLMUsage]:
        return contract, _unit_usage()

    async def fake_create_workspace(
        run_id: str, repo_path: str, docker_image: str | None = None
    ) -> WorkspaceInfo:
        return workspace_info

    async def fake_build_repo_index(workspace: WorkspaceInfo) -> RepoIndex:
        return RepoIndex()

    async def fake_build_plan(request: PlanRequest) -> tuple[Plan, LLMUsage]:
        plan_requests.append(request)
        if len(plan_requests) == 1:
            return _plan_with_step('old-step'), _unit_usage()
        return _plan_with_step('new-step'), _unit_usage()

    async def fake_assess_complexity(
        task_contract: TaskContract,
    ) -> tuple[ComplexityVerdict, LLMUsage]:
        return (
            ComplexityVerdict(requires_human_approval=False, reasoning='Narrow bugfix.'),
            _unit_usage(),
        )

    async def fake_spawn_child(
        workflow_name: str,
        step: dict[str, object],
        workspace: dict[str, object],
        contract: dict[str, object],
        repo_index: dict[str, object],
    ) -> str:
        assert repo_index == RepoIndex().model_dump(mode='json')
        spawned_step_ids.append(str(step['id']))
        return f'child-{len(spawned_step_ids)}'

    async def fake_wait_for_child(child_id: str) -> dict[str, object]:
        return child_payloads.pop(0)

    async def fake_get_full_diff(workspace: WorkspaceInfo) -> str:
        return 'diff --git a/generated.py b/generated.py'

    async def fake_review_patch(request: ReviewRequest) -> tuple[ReviewVerdict, LLMUsage]:
        return (
            ReviewVerdict(
                verdict=ReviewDecision.ACCEPT,
                blocking_issues=[],
                non_blocking_issues=[],
                evidence=['workflow completed'],
                missing_tests=[],
                regression_risks=[],
                minimality_assessment='minimal',
                recommended_next_action='accept',
            ),
            _unit_usage(),
        )

    async def fake_destroy_workspace(workspace: WorkspaceInfo) -> None:
        return None

    monkeypatch.setattr('src.workflows.main_workflow.build_contract', fake_build_contract)
    monkeypatch.setattr('src.workflows.main_workflow.create_workspace', fake_create_workspace)
    monkeypatch.setattr('src.workflows.main_workflow.build_repo_index', fake_build_repo_index)
    monkeypatch.setattr('src.workflows.main_workflow.build_plan', fake_build_plan)
    monkeypatch.setattr('src.workflows.main_workflow.assess_complexity', fake_assess_complexity)
    monkeypatch.setattr('src.workflows.main_workflow.spawn_child', fake_spawn_child)
    monkeypatch.setattr('src.workflows.main_workflow.wait_for_child', fake_wait_for_child)
    monkeypatch.setattr('src.workflows.main_workflow.get_full_diff', fake_get_full_diff)
    monkeypatch.setattr('src.workflows.main_workflow.review_patch', fake_review_patch)
    monkeypatch.setattr('src.workflows.main_workflow.destroy_workspace', fake_destroy_workspace)

    report = await main_workflow(
        {'raw_request': 'fix generated output', 'repo_path': 'C:/repo', 'run_id': 'run-1'}
    )

    assert spawned_step_ids == ['old-step', 'new-step']
    assert len(plan_requests) == 2
    assert plan_requests[1].human_feedback == 'Add the generated file update.'
    assert len(plan_requests[1].worker_results) == 1
    assert report['worker_results'][-1]['status'] == WorkerStatus.SUCCESS
    # build_contract + build_plan(initial) + assess_complexity + 2 children + build_plan(replan) + review_patch
    assert report['llm_usage']['call_count'] == 7
    assert report['patch'] == 'diff --git a/generated.py b/generated.py'


@pytest.mark.asyncio
async def test_main_workflow_does_not_destroy_workspace_while_suspended_on_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destroyed_workspaces: list[WorkspaceInfo] = []
    spawned_step_ids: list[str] = []
    workspace_info = WorkspaceInfo(
        run_id='run-1',
        volume_name='volume',
        worktree_path='workspace',
        branch_name='branch',
    )
    contract = TaskContract(
        task_type=TaskType.BUGFIX,
        goal='Fix generated output',
        acceptance_criteria=[],
        non_goals=[],
        affected_areas=[],
        risk_areas=[],
        tests_expected=[],
        open_questions=[],
    )

    async def fake_build_contract(task_request: TaskRequest) -> tuple[TaskContract, LLMUsage]:
        return contract, _unit_usage()

    async def fake_create_workspace(
        run_id: str, repo_path: str, docker_image: str | None = None
    ) -> WorkspaceInfo:
        return workspace_info

    async def fake_build_repo_index(workspace: WorkspaceInfo) -> RepoIndex:
        return RepoIndex()

    async def fake_build_plan(request: PlanRequest) -> tuple[Plan, LLMUsage]:
        return _plan_with_step('step-1'), _unit_usage()

    async def fake_assess_complexity(
        task_contract: TaskContract,
    ) -> tuple[ComplexityVerdict, LLMUsage]:
        return (
            ComplexityVerdict(requires_human_approval=False, reasoning='Narrow bugfix.'),
            _unit_usage(),
        )

    async def fake_spawn_child(
        workflow_name: str,
        step: dict[str, object],
        workspace: dict[str, object],
        contract: dict[str, object],
        repo_index: dict[str, object],
    ) -> str:
        assert repo_index == RepoIndex().model_dump(mode='json')
        spawned_step_ids.append(str(step['id']))
        return 'child-1'

    async def fake_wait_for_child(child_id: str) -> dict[str, object]:
        raise WorkflowSuspended('Workflow waiting for child.')

    async def fake_destroy_workspace(workspace: WorkspaceInfo) -> None:
        destroyed_workspaces.append(workspace)

    monkeypatch.setattr('src.workflows.main_workflow.build_contract', fake_build_contract)
    monkeypatch.setattr('src.workflows.main_workflow.create_workspace', fake_create_workspace)
    monkeypatch.setattr('src.workflows.main_workflow.build_repo_index', fake_build_repo_index)
    monkeypatch.setattr('src.workflows.main_workflow.build_plan', fake_build_plan)
    monkeypatch.setattr('src.workflows.main_workflow.assess_complexity', fake_assess_complexity)
    monkeypatch.setattr('src.workflows.main_workflow.spawn_child', fake_spawn_child)
    monkeypatch.setattr('src.workflows.main_workflow.wait_for_child', fake_wait_for_child)
    monkeypatch.setattr('src.workflows.main_workflow.destroy_workspace', fake_destroy_workspace)

    with pytest.raises(WorkflowSuspended, match=r'Workflow waiting for child\.'):
        await main_workflow(
            {'raw_request': 'fix generated output', 'repo_path': 'C:/repo', 'run_id': 'run-1'}
        )

    assert spawned_step_ids == ['step-1']
    assert destroyed_workspaces == []


def _plan_with_step(step_id: str) -> Plan:
    return Plan(
        summary=f'Plan {step_id}',
        steps=[
            PlanStep(
                id=step_id,
                goal='Update generated output',
                target_files=['generated.py'],
                tests_to_run=[],
                expected_result='Generated output updated',
                risk=Risk.LOW,
                requires_human_approval=False,
            )
        ],
        integration_tests=[],
        rollback_strategy='git checkout',
        definition_of_done=['diff reviewed'],
    )
