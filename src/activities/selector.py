from __future__ import annotations

from temporal_light import activity

from src.activities.reviewer import ReviewDecision, ReviewVerdict
from src.config import CONFIG
from src.models.frozen_base_model import FrozenBaseModel
from src.models.plan import Plan
from src.models.reproduction import ReproductionEvidence
from src.models.worker import Confidence, TestResult, WorkerResult, WorkerStatus

_CONFIDENCE_RANK: dict[Confidence, int] = {
    Confidence.LOW: 0,
    Confidence.MEDIUM: 1,
    Confidence.HIGH: 2,
}


class CandidateResult(FrozenBaseModel):
    index: int
    branch: str
    diff: str
    plan: Plan
    worker_results: list[WorkerResult]
    review_verdict: ReviewVerdict
    confidence: Confidence
    test_results: list[TestResult]
    reproduction_evidence: ReproductionEvidence | None = None


class SelectionRequest(FrozenBaseModel):
    candidates: list[CandidateResult]


def aggregate_candidate_confidence(worker_results: list[WorkerResult]) -> Confidence:
    successful_results = [
        worker_result
        for worker_result in worker_results
        if worker_result.status == WorkerStatus.SUCCESS
    ]
    return min(
        (worker_result.confidence for worker_result in successful_results or worker_results),
        key=lambda confidence: _CONFIDENCE_RANK[confidence],
        default=Confidence.LOW,
    )


def derive_candidate_confidence(candidate: CandidateResult) -> Confidence:
    if candidate.review_verdict.verdict != ReviewDecision.ACCEPT:
        return Confidence.LOW
    return candidate.confidence


def candidate_count_for_confidence(confidence: Confidence) -> int:
    match confidence:
        case Confidence.HIGH:
            return 1
        case Confidence.MEDIUM:
            return CONFIG.candidate_count_medium_confidence
        case Confidence.LOW:
            return CONFIG.candidate_count_low_confidence


def _passing_test_count(candidate: CandidateResult) -> int:
    return sum(1 for test_result in candidate.test_results if test_result.passed)


def _preference_key(candidate: CandidateResult) -> tuple[int, int, int, int, int]:
    return (
        1 if candidate.review_verdict.verdict == ReviewDecision.ACCEPT else 0,
        _passing_test_count(candidate),
        -len(candidate.review_verdict.blocking_issues),
        -len(candidate.diff),
        -candidate.index,
    )


def select_best_candidate(candidates: list[CandidateResult]) -> CandidateResult:
    if not candidates:
        raise ValueError('cannot select a winner from an empty candidate list')
    return max(candidates, key=_preference_key)


@activity(retries=0, timeout=120)
async def select_candidate(request: SelectionRequest) -> CandidateResult:
    # Selection, not combination: a combiner must re-validate its merged diff (see PLAN.md).
    return select_best_candidate(request.candidates)
