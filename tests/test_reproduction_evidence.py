import pytest
from src.models.reproduction import (
    NO_AFTER_EXIT_CODE,
    build_reproduction_evidence,
)


@pytest.mark.parametrize(
    ('before_exit_code', 'after_exit_code', 'expected_reproduced'),
    [
        (1, 0, True),
        (0, 0, False),
        (1, 1, False),
        (1, None, False),
    ],
)
def test_build_reproduction_evidence_reproduced_only_on_fail_then_pass(
    before_exit_code: int,
    after_exit_code: int | None,
    expected_reproduced: bool,
) -> None:
    evidence = build_reproduction_evidence(
        repro_command='pytest tests/test_bug.py',
        before_exit_code=before_exit_code,
        after_exit_code=after_exit_code,
    )
    assert evidence.reproduced is expected_reproduced


def test_build_reproduction_evidence_records_sentinel_when_no_after_observation() -> None:
    evidence = build_reproduction_evidence(
        repro_command='pytest tests/test_bug.py',
        before_exit_code=1,
        after_exit_code=None,
    )
    assert evidence.after_exit_code == NO_AFTER_EXIT_CODE
    assert evidence.passed_after is False
