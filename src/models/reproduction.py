from __future__ import annotations

from pydantic import Field

from src.models.frozen_base_model import FrozenBaseModel
from src.runtime_enums import StrEnum


class ReproductionStatus(StrEnum):
    REPRODUCED = 'reproduced'
    COULD_NOT_REPRODUCE = 'could_not_reproduce'


class ReproductionContext(FrozenBaseModel):
    repro_target: str
    failure_evidence: str


class ReproductionResult(FrozenBaseModel):
    status: ReproductionStatus
    repro_target: str
    test_files: list[str] = Field(default_factory=list)
    failure_evidence: str


class ReproductionEvidence(FrozenBaseModel):
    repro_target: str
    passed_after: bool


def build_reproduction_evidence(
    repro_target: str,
    passed_after: bool,
) -> ReproductionEvidence:
    return ReproductionEvidence(
        repro_target=repro_target,
        passed_after=passed_after,
    )
