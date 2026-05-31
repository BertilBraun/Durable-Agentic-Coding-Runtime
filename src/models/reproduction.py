from __future__ import annotations

from pydantic import Field

from src.models.frozen_base_model import FrozenBaseModel
from src.runtime_enums import StrEnum

NO_AFTER_EXIT_CODE = -1


class ReproductionStatus(StrEnum):
    REPRODUCED = 'reproduced'
    COULD_NOT_REPRODUCE = 'could_not_reproduce'


class ReproductionResult(FrozenBaseModel):
    status: ReproductionStatus
    repro_command: str
    test_files: list[str] = Field(default_factory=list)
    failure_evidence: str


class ReproductionEvidence(FrozenBaseModel):
    repro_command: str
    failed_before: bool
    passed_after: bool
    before_exit_code: int
    after_exit_code: int

    @property
    def reproduced(self) -> bool:
        return self.failed_before and self.passed_after


def build_reproduction_evidence(
    repro_command: str,
    before_exit_code: int,
    after_exit_code: int | None,
) -> ReproductionEvidence:
    return ReproductionEvidence(
        repro_command=repro_command,
        failed_before=before_exit_code != 0,
        passed_after=after_exit_code == 0,
        before_exit_code=before_exit_code,
        after_exit_code=after_exit_code if after_exit_code is not None else NO_AFTER_EXIT_CODE,
    )
