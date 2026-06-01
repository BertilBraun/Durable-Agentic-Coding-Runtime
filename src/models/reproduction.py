from __future__ import annotations

from pydantic import Field

from src.models.frozen_base_model import FrozenBaseModel
from src.runtime_enums import StrEnum


class ReproductionStatus(StrEnum):
    REPRODUCED = 'reproduced'
    COULD_NOT_REPRODUCE = 'could_not_reproduce'


class ReproductionContext(FrozenBaseModel):
    repro_command: str
    failure_evidence: str


class ReproductionResult(FrozenBaseModel):
    status: ReproductionStatus
    repro_command: str
    test_files: list[str] = Field(default_factory=list)
    failure_evidence: str


class ReproductionEvidence(FrozenBaseModel):
    repro_command: str
    passed_after: bool


def build_reproduction_evidence(
    repro_command: str,
    passed_after: bool,
) -> ReproductionEvidence:
    return ReproductionEvidence(
        repro_command=repro_command,
        passed_after=passed_after,
    )
