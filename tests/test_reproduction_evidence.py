import pytest
from src.models.reproduction import (
    build_reproduction_evidence,
)


@pytest.mark.parametrize(
    ('passed_after',),
    [
        (True,),
        (False,),
    ],
)
def test_build_reproduction_evidence_records_after_status(
    passed_after: bool,
) -> None:
    evidence = build_reproduction_evidence(
        repro_command='pytest tests/test_bug.py',
        passed_after=passed_after,
    )
    assert evidence.repro_command == 'pytest tests/test_bug.py'
    assert evidence.passed_after is passed_after
