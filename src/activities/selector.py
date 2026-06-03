from __future__ import annotations

from src.activities.reviewer import ReviewVerdict
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


def candidate_count_for_confidence(confidence: Confidence) -> int:
    match confidence:
        case Confidence.HIGH:
            return 1
        case Confidence.MEDIUM:
            return CONFIG.candidate_count_medium_confidence
        case Confidence.LOW:
            return CONFIG.candidate_count_low_confidence
