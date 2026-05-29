import json

import pytest
from openai.types.chat import ChatCompletion, ParsedChatCompletion
from src.llm.client import LLMClient, LLMUsageLedger, Message
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
async def test_structured_generation_returns_llm_result_and_records_usage() -> None:
    LLMClient.reset_global_usage()
    async_openai_client = FakeAsyncOpenAI()
    llm_client = LLMClient(
        usage_ledger=LLMUsageLedger(),
        async_openai_client=async_openai_client,
    )

    result = await llm_client.generate_structured(
        messages=[Message(role='user', content='Assess this task')],
        output_type=ComplexityVerdict,
        model='complexity-model',
        context_limit_tokens=100,
    )

    assert async_openai_client.completions.parse_calls[0]['response_format'] == ComplexityVerdict
    assert result.model == 'complexity-model'
    assert result.input_tokens == 10
    assert result.output_tokens == 4
    assert result.context_utilization() == 0.1
    assert ComplexityVerdict.model_validate_json(result.content).requires_human_approval is False
    assert llm_client.usage_ledger.total_input_tokens == 10
    assert llm_client.usage_ledger.total_output_tokens == 4
    usage_summary = LLMClient.global_usage_summary()
    assert usage_summary.call_count == 1
    assert usage_summary.total_input_tokens == 10
    assert usage_summary.total_output_tokens == 4


@pytest.mark.asyncio
async def test_completion_returns_llm_result_and_records_usage() -> None:
    llm_client = LLMClient(
        usage_ledger=LLMUsageLedger(),
        async_openai_client=FakeAsyncOpenAI(),
    )

    result = await llm_client.complete(
        messages=[Message(role='user', content='Assess this task')],
        model='completion-model',
        context_limit_tokens=100,
    )

    assert result.content == '{"requires_human_approval": false}'
    assert result.model == 'completion-model'
    assert result.input_tokens == 10
    assert result.context_utilization() == 0.1
    assert llm_client.usage_ledger.total_input_tokens == 10
    assert llm_client.usage_ledger.total_output_tokens == 4
