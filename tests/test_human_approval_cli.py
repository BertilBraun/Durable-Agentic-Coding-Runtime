import pytest
from pydantic import ValidationError
from src.cli.human_approval import (
    HumanApprovalSignalRequest,
    build_signal_request,
    send_human_approval_signal,
)
from src.models.approval import ApprovalDecision, HumanApprovalSignal


def test_build_signal_request_uses_human_approval_signal_type() -> None:
    signal_request = build_signal_request(
        HumanApprovalSignal(decision=ApprovalDecision.APPROVE, feedback=None)
    )

    assert signal_request.signal_type == 'human_approval'
    assert signal_request.payload.decision == ApprovalDecision.APPROVE
    assert signal_request.payload.feedback is None


@pytest.mark.asyncio
async def test_send_human_approval_signal_posts_to_temporal() -> None:
    http_client = FakeHttpClient()

    await send_human_approval_signal(
        temporal_api_url='http://temporal',
        workflow_id='workflow-1',
        approval=HumanApprovalSignal(decision=ApprovalDecision.REVISE, feedback='Narrow scope'),
        http_client=http_client,
    )

    assert http_client.posted_url == 'http://temporal/workflows/workflow-1/signals'
    assert http_client.posted_json == {
        'signal_type': 'human_approval',
        'payload': {'decision': 'revise', 'feedback': 'Narrow scope'},
    }


class FakeHttpResponse:
    def raise_for_status(self) -> None:
        return None


class FakeHttpClient:
    def __init__(self) -> None:
        self.posted_url: str | None = None
        self.posted_json: dict[str, object] | None = None

    async def post(
        self,
        url: str,
        json: dict[str, object],
    ) -> FakeHttpResponse:
        self.posted_url = url
        self.posted_json = json
        return FakeHttpResponse()


def test_signal_request_type_is_frozen() -> None:
    signal_request = HumanApprovalSignalRequest(
        signal_type='human_approval',
        payload=HumanApprovalSignal(decision=ApprovalDecision.APPROVE, feedback=None),
    )

    with pytest.raises(ValidationError):
        signal_request.signal_type = 'other'
