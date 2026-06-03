import pytest
from src.activities.selector import (
    aggregate_candidate_confidence,
    candidate_count_for_confidence,
)
from src.config import CONFIG
from src.models.worker import Confidence, WorkerResult, WorkerStatus


@pytest.mark.parametrize(
    ('confidence', 'expected_count'),
    [
        (Confidence.HIGH, 1),
        (Confidence.MEDIUM, CONFIG.candidate_count_medium_confidence),
        (Confidence.LOW, CONFIG.candidate_count_low_confidence),
    ],
)
def test_candidate_count_for_confidence_maps_each_level(
    confidence: Confidence, expected_count: int
) -> None:
    assert candidate_count_for_confidence(confidence) == expected_count


def test_high_confidence_default_count_is_one() -> None:
    assert candidate_count_for_confidence(Confidence.HIGH) == 1


def test_default_candidate_counts_are_conservative() -> None:
    assert candidate_count_for_confidence(Confidence.MEDIUM) == 1
    assert candidate_count_for_confidence(Confidence.LOW) == 2


def test_corrected_replan_attempt_does_not_lower_candidate_confidence() -> None:
    worker_results = [
        WorkerResult(
            diff_summary='first attempt needed revision',
            confidence=Confidence.LOW,
            status=WorkerStatus.NEEDS_REPLAN,
        ),
        WorkerResult(
            diff_summary='revision passed',
            confidence=Confidence.HIGH,
            status=WorkerStatus.SUCCESS,
        ),
    ]

    assert aggregate_candidate_confidence(worker_results) == Confidence.HIGH
