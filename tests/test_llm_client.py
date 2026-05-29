import json
from typing import Any

import pytest
from openai.types.chat import ChatCompletion, ParsedChatCompletion
from src.llm.client import LLMClient, LLMUsageLedger, Message
from src.llm.config import ModelConfiguration, ModelContextLimit, ModelRole
from src.models.approval import ComplexityVerdict


class FakeCompletions:
    def __init__(self) -> None:
        self.parse_calls: list[dict[str, object]] = []

    async def create(self, **keyword_arguments: object) -> ChatCompletion:
        return ChatCompletion.model_validate(
            {
                'id': 'chatcmpl-test',
                'object': 'chat.completion',
                'created': 0,
                'model': 'completion-model',
                'choices': [
                    {
                        'index': 0,
                        'finish_reason': 'stop',
                        'message': {
                            'role': 'assistant',
                            'content': json.dumps({'requires_human_approval': False}),
                        },
                    }
                ],
                'usage': {
                    'prompt_tokens': 10,
                    'completion_tokens': 4,
                    'total_tokens': 14,
                },
            }
        )

    async def parse(
        self,
        response_format: type[ComplexityVerdict],
        **keyword_arguments: object,
    ) -> ParsedChatCompletion[ComplexityVerdict]:
        self.parse_calls.append({'response_format': response_format, **keyword_arguments})
        return ParsedChatCompletion[ComplexityVerdict].model_validate(
            {
                'id': 'chatcmpl-test',
                'object': 'chat.completion',
                'created': 0,
                'model': 'structured-model',
                'choices': [
                    {
                        'index': 0,
                        'finish_reason': 'stop',
                        'message': {
                            'role': 'assistant',
                            'content': json.dumps(
                                {
                                    'requires_human_approval': False,
                                    'reasoning': 'Narrow bugfix.',
                                }
                            ),
                            'parsed': {
                                'requires_human_approval': False,
                                'reasoning': 'Narrow bugfix.',
                            },
                        },
                    }
                ],
                'usage': {
                    'prompt_tokens': 10,
                    'completion_tokens': 4,
                    'total_tokens': 14,
                },
            }
        )


class FakeChat:
    def __init__(self, completions: FakeCompletions) -> None:
        self.completions = completions


class FakeBeta:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = FakeChat(completions)


class FakeAsyncOpenAI:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = FakeChat(self.completions)
        self.beta = FakeBeta(self.completions)


@pytest.mark.asyncio
async def test_structured_generation_parses_model_and_records_usage() -> None:
    LLMClient.reset_global_usage()
    async_openai_client = FakeAsyncOpenAI()
    llm_client = LLMClient(
        model_configuration=_model_configuration(),
        usage_ledger=LLMUsageLedger(),
        async_openai_client=async_openai_client,
    )

    verdict = await llm_client.generate_structured(
        role=ModelRole.COMPLEXITY_ASSESSOR,
        messages=[Message(role='user', content='Assess this task')],
        output_type=ComplexityVerdict,
    )

    assert verdict.requires_human_approval is False
    assert async_openai_client.completions.parse_calls[0]['response_format'] == ComplexityVerdict
    assert llm_client.usage_ledger.total_input_tokens == 10
    assert llm_client.usage_ledger.total_output_tokens == 4
    assert llm_client.last_input_token_count == 10
    assert llm_client.context_utilization() == 0.1
    usage_summary = LLMClient.global_usage_summary()
    assert usage_summary.call_count == 1
    assert usage_summary.total_input_tokens == 10
    assert usage_summary.total_output_tokens == 4


@pytest.mark.asyncio
async def test_completion_extracts_string_content_and_records_usage() -> None:
    llm_client = LLMClient(
        model_configuration=_model_configuration(),
        usage_ledger=LLMUsageLedger(),
        async_openai_client=FakeAsyncOpenAI(),
    )

    content = await llm_client.complete(
        role=ModelRole.COMPLEXITY_ASSESSOR,
        messages=[Message(role='user', content='Assess this task')],
    )

    assert content == '{"requires_human_approval": false}'
    assert llm_client.usage_ledger.total_input_tokens == 10
    assert llm_client.usage_ledger.total_output_tokens == 4


def _model_configuration() -> ModelConfiguration:
    model_names: dict[str, Any] = {
        'contract_builder_model': 'contract',
        'planner_model': 'planner',
        'complexity_assessor_model': 'complexity',
        'context_gatherer_model': 'context',
        'implementation_model': 'implementation',
        'reviewer_model': 'review',
        'summarizer_model': 'summary',
    }
    return ModelConfiguration(
        **model_names,
        model_context_limits=[
            ModelContextLimit(model='contract', context_limit_tokens=100),
            ModelContextLimit(model='planner', context_limit_tokens=100),
            ModelContextLimit(model='complexity', context_limit_tokens=100),
            ModelContextLimit(model='context', context_limit_tokens=100),
            ModelContextLimit(model='implementation', context_limit_tokens=100),
            ModelContextLimit(model='review', context_limit_tokens=100),
            ModelContextLimit(model='summary', context_limit_tokens=100),
        ],
    )
