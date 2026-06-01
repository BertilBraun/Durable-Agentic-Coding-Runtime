import pytest
from src.activities.reviewer import ReviewDecision, ReviewVerdict
from src.activities.selector import (
    CandidateResult,
    aggregate_candidate_confidence,
    candidate_count_for_confidence,
    derive_candidate_confidence,
    select_best_candidate,
)
from src.config import CONFIG
from src.models.plan import Plan
from src.models.worker import Confidence, WorkerResult, WorkerStatus
from src.models.worker import TestResult as ExecutedTestResult


def _plan() -> Plan:
    return Plan(summary='p')


def _verdict(decision: ReviewDecision, blocking_issues: list[str] | None = None) -> ReviewVerdict:
    return ReviewVerdict(
        blocking_issues=blocking_issues or [],
        minimality_assessment='minimal',
        recommended_next_action='ship',
        verdict=decision,
    )


def _test_result(passed: bool, sequence: int = 1) -> ExecutedTestResult:
    return ExecutedTestResult(
        sequence=sequence,
        command='pytest',
        exit_code=0 if passed else 1,
        stdout_summary='',
        stderr_summary='',
        passed=passed,
    )


def _candidate(
    index: int,
    decision: ReviewDecision = ReviewDecision.ACCEPT,
    confidence: Confidence = Confidence.HIGH,
    diff: str = 'diff',
    test_results: list[ExecutedTestResult] | None = None,
    blocking_issues: list[str] | None = None,
) -> CandidateResult:
    return CandidateResult(
        index=index,
        branch=f'cand-{index}',
        diff=diff,
        plan=_plan(),
        worker_results=[
            WorkerResult(
                diff_summary='did work', confidence=confidence, status=WorkerStatus.SUCCESS
            )
        ],
        review_verdict=_verdict(decision, blocking_issues),
        confidence=confidence,
        test_results=test_results or [],
    )


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


def test_accepted_high_confidence_candidate_stays_high() -> None:
    candidate = _candidate(0, ReviewDecision.ACCEPT, Confidence.HIGH)
    assert derive_candidate_confidence(candidate) == Confidence.HIGH


def test_revise_verdict_escalates_to_low_confidence() -> None:
    candidate = _candidate(0, ReviewDecision.REVISE, Confidence.HIGH)
    assert derive_candidate_confidence(candidate) == Confidence.LOW


def test_low_confidence_worker_escalates_even_when_accepted() -> None:
    candidate = _candidate(0, ReviewDecision.ACCEPT, Confidence.LOW)
    assert derive_candidate_confidence(candidate) == Confidence.LOW


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


def test_selector_prefers_accept_over_revise() -> None:
    revised = _candidate(0, ReviewDecision.REVISE)
    accepted = _candidate(1, ReviewDecision.ACCEPT)
    assert select_best_candidate([revised, accepted]).index == 1


def test_selector_prefers_more_passing_tests() -> None:
    fewer = _candidate(0, test_results=[_test_result(True)])
    more = _candidate(1, test_results=[_test_result(True, 1), _test_result(True, 2)])
    assert select_best_candidate([fewer, more]).index == 1


def test_selector_prefers_fewer_blocking_issues() -> None:
    noisy = _candidate(0, blocking_issues=['a', 'b'])
    clean = _candidate(1, blocking_issues=[])
    assert select_best_candidate([noisy, clean]).index == 1


def test_selector_breaks_ties_on_smaller_diff() -> None:
    large = _candidate(0, diff='x' * 100)
    small = _candidate(1, diff='x' * 10)
    assert select_best_candidate([large, small]).index == 1


def test_selector_single_candidate_is_trivial() -> None:
    only = _candidate(0)
    assert select_best_candidate([only]) is only
