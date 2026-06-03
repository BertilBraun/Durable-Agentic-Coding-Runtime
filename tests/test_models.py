from pydantic import ValidationError
from src.models.plan import PlanStep, Risk
from src.models.task import HostOrigin, TaskContract, TaskRequest, TaskType


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
