from __future__ import annotations

import argparse
import asyncio
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict

from src.models.approval import ApprovalDecision, HumanApprovalSignal


class HumanApprovalSignalRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    signal_type: str
    payload: HumanApprovalSignal


class HttpResponse(Protocol):
    def raise_for_status(self) -> None: ...


class HttpClient(Protocol):
    async def post(self, url: str, json: dict[str, object]) -> HttpResponse: ...


def build_signal_request(approval: HumanApprovalSignal) -> HumanApprovalSignalRequest:
    return HumanApprovalSignalRequest(
        signal_type='human_approval',
        payload=approval,
    )


async def send_human_approval_signal(
    temporal_api_url: str,
    workflow_id: str,
    approval: HumanApprovalSignal,
    http_client: httpx.AsyncClient,
) -> None:
    signal_request = build_signal_request(approval)
    response = await http_client.post(
        url=f'{temporal_api_url.rstrip("/")}/workflows/{workflow_id}/signals',
        json=signal_request.model_dump(mode='json'),
    )
    response.raise_for_status()


async def run_from_arguments(arguments: argparse.Namespace) -> None:
    approval = HumanApprovalSignal(
        decision=ApprovalDecision(arguments.decision),
        feedback=arguments.feedback,
    )
    async with httpx.AsyncClient(timeout=30) as http_client:
        await send_human_approval_signal(
            temporal_api_url=arguments.temporal_api_url,
            workflow_id=arguments.workflow_id,
            approval=approval,
            http_client=http_client,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--temporal-api-url', required=True)
    parser.add_argument('--workflow-id', required=True)
    parser.add_argument(
        '--decision',
        choices=[decision.value for decision in ApprovalDecision],
        required=True,
    )
    parser.add_argument('--feedback', required=False)
    arguments = parser.parse_args()
    asyncio.run(run_from_arguments(arguments))


if __name__ == '__main__':
    main()
