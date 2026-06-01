from collections.abc import Awaitable, Callable

import pytest
from pydantic import BaseModel
from src.activities import plan_reviewer as plan_reviewer_module
from src.activities.plan_reviewer import (
    PlanReviewDecision,
    PlanReviewRequest,
    PlanReviewVerdict,
    review_plan,
)
from src.config import ModelRole
from src.llm.client import LLMUsage, Message, StructuredCompletion
from src.models.plan import Plan, PlanStep, Risk
from src.models.repo import RepoIndex
from src.models.task import TaskContract, TaskType


def _plan() -> Plan:
    return Plan(
        summary='Fix the bug',
        steps=[
            PlanStep(
                id='step-1',
                goal='Fix it',
                target_files=['app.py'],
                tests_to_run=[],
                expected_result='Bug fixed',
                risk=Risk.LOW,
                requires_human_approval=False,
            )
        ],
        integration_tests=[],
        definition_of_done=['diff reviewed'],
    )


def _request() -> PlanReviewRequest:
    return PlanReviewRequest(
        contract=TaskContract(task_type=TaskType.BUGFIX, goal='Fix the bug'),
        plan=_plan(),
        repo_index=RepoIndex(),
    )


def _patch_generate_structured(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[list[Message], type[BaseModel]], Awaitable[StructuredCompletion]],
) -> None:
    async def fake_generate_structured(
        role: ModelRole,
        messages: list[Message],
        output_type: type[BaseModel],
    ) -> StructuredCompletion:
        return await handler(messages, output_type)

    monkeypatch.setattr(plan_reviewer_module, 'generate_structured', fake_generate_structured)


def _completion(verdict: PlanReviewVerdict) -> StructuredCompletion:
    return StructuredCompletion(
        output=verdict,
        content=verdict.model_dump_json(),
        model='fake-model',
        context_limit_tokens=100,
        usage=LLMUsage(call_count=1),
    )


@pytest.mark.asyncio
async def test_accept_passes_through_with_empty_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(
        messages: list[Message], output_type: type[BaseModel]
    ) -> StructuredCompletion:
        return _completion(PlanReviewVerdict(decision=PlanReviewDecision.ACCEPT, feedback=''))

    _patch_generate_structured(monkeypatch, handler)

    verdict, _ = await review_plan(_request())

    assert verdict.decision == PlanReviewDecision.ACCEPT
    assert verdict.feedback == ''


@pytest.mark.asyncio
async def test_revise_returns_the_feedback_string(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(
        messages: list[Message], output_type: type[BaseModel]
    ) -> StructuredCompletion:
        return _completion(
            PlanReviewVerdict(
                decision=PlanReviewDecision.REVISE,
                blocking_issues=['missing the auth callers'],
                feedback='Inspect the auth callers and add a step covering them.',
            )
        )

    _patch_generate_structured(monkeypatch, handler)

    verdict, _ = await review_plan(_request())

    assert verdict.decision == PlanReviewDecision.REVISE
    assert verdict.feedback == 'Inspect the auth callers and add a step covering them.'
