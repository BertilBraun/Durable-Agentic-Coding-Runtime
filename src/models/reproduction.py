from __future__ import annotations

from pydantic import Field

from src.models.frozen_base_model import FrozenBaseModel
from src.runtime_enums import StrEnum


class ReproductionStatus(StrEnum):
    REPRODUCED = 'reproduced'
    COULD_NOT_REPRODUCE = 'could_not_reproduce'


class ReproductionBrief(FrozenBaseModel):
    summary: str = Field(description='What behavior the reproduction test must demonstrate.')
    is_round_trip: bool = Field(
        description=(
            'Whether the requested behavior is a symmetric operation (read/write, '
            'encode/decode, serialize/parse) whose correctness must be asserted as a round trip.'
        )
    )
    assertion_guidance: str = Field(
        description=(
            'Exactly what the reproduction test must assert: a round-trip invariant '
            '(read(write(x)) == x) when symmetric, otherwise the concrete behavior to check.'
        )
    )
    candidate_target_files: list[str] = Field(
        default_factory=list,
        description='Production files likely involved in implementing the behavior.',
    )
    regression_test_files: list[str] = Field(
        default_factory=list,
        description='Existing repository test files that form the regression set.',
    )


class ReproductionContext(FrozenBaseModel):
    repro_target: str
    failure_evidence: str
    regression_test_files: list[str] = Field(default_factory=list)


class ReproductionResult(FrozenBaseModel):
    status: ReproductionStatus
    repro_target: str
    test_files: list[str] = Field(default_factory=list)
    regression_test_files: list[str] = Field(
        default_factory=list,
        description='Existing repo test files (whole files, not node ids) to run for regressions.',
    )
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
