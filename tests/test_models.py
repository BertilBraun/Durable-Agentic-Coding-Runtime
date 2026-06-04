from pydantic import ValidationError
from src.models.plan import PlannerTurn, PlanStep, Risk
from src.models.task import HostOrigin, TaskContract, TaskRequest, TaskType
from src.tools.definitions import RunShell, WriteFile


def test_task_request_is_frozen() -> None:
    task_request = TaskRequest(raw_request='Fix the parser', origin=HostOrigin(repo_path='C:/repo'))

    try:
        task_request.raw_request = 'Change the parser'
    except ValidationError:
        return

    raise AssertionError('TaskRequest should be immutable')


def test_plan_step_uses_serializable_enums() -> None:
    plan_step = PlanStep(
        id='step_1',
        goal='Patch parser edge case',
        target_files=['src/parser.py'],
        tests_to_run=['pytest tests/test_parser.py'],
        expected_result='Regression test passes',
        risk=Risk.LOW,
    )

    assert plan_step.model_dump(mode='json')['risk'] == 'low'


def test_planner_turn_accepts_only_read_only_tool_calls() -> None:
    turn = PlannerTurn(
        tool_calls=[
            RunShell(command='Get-Content src/app.py', timeout_seconds=10),
        ]
    )

    assert turn.model_dump(mode='json')['tool_calls'][0]['tool_name'] == 'run_shell'

    try:
        PlannerTurn(
            tool_calls=[
                WriteFile(file_path='src/app.py', content='mutating writes are not allowed'),
            ]
        )
    except ValidationError:
        return

    raise AssertionError('PlannerTurn should reject mutating tool calls')


def test_task_contract_accepts_complete_contract() -> None:
    task_contract = TaskContract(
        task_type=TaskType.BUGFIX,
        goal='Fix parser None handling',
        acceptance_criteria=['None input raises ValueError'],
        non_goals=['Rewrite parser'],
        affected_areas=['parser'],
        risk_areas=['input validation'],
        tests_expected=['pytest tests/test_parser.py'],
        open_questions=[],
    )

    assert task_contract.task_type == TaskType.BUGFIX
