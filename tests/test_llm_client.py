import json

import pytest
from openai.types.chat import ChatCompletion, ParsedChatCompletion
from src.activities.complexity_assessor import ComplexityVerdict
from src.config import ModelRole
from src.llm.client import LLMClient, LLMResult, LLMUsage, Message, _structured_output_type
from src.models.task import TaskContract


def test_structured_output_type_resolves_from_importable_path() -> None:
    assert _structured_output_type('src.models.task', 'TaskContract') is TaskContract


@pytest.mark.asyncio
async def test_generate_structured_passes_importable_output_type_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    async def fake_generate_structured_completion(**keyword_arguments: object) -> LLMResult:
        calls.append(keyword_arguments)
        _structured_output_type(
            str(keyword_arguments['output_type_module']),
            str(keyword_arguments['output_type_name']),
        )
        return LLMResult(
            content=json.dumps({'task_type': 'bugfix', 'goal': 'Fix the test'}),
            model=str(keyword_arguments['model']),
            context_limit_tokens=int(keyword_arguments['context_limit_tokens']),
            usage=LLMUsage(call_count=1),
        )

    import src.llm.client as llm_client_module

    monkeypatch.setattr(
        llm_client_module,
        'generate_structured_completion',
        fake_generate_structured_completion,
    )

    await llm_client_module.generate_structured(
        role=ModelRole.CONTRACT_BUILDER,
        messages=[Message(role='user', content='Fix the test')],
        output_type=TaskContract,
    )

    assert calls[0]['output_type_module'] == 'src.models.task'
    assert calls[0]['output_type_name'] == 'TaskContract'


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
async def test_structured_generation_returns_llm_result_with_usage() -> None:
    async_openai_client = FakeAsyncOpenAI()
    llm_client = LLMClient(async_openai_client=async_openai_client)

    result = await llm_client.generate_structured(
        messages=[Message(role='user', content='Assess this task')],
        output_type=ComplexityVerdict,
        model='complexity-model',
        context_limit_tokens=100,
    )

    assert async_openai_client.completions.parse_calls[0]['response_format'] == ComplexityVerdict
    assert result.model == 'complexity-model'
    assert result.usage.call_count == 1
    assert result.usage.total_input_tokens == 10
    assert result.usage.total_output_tokens == 4
    assert result.context_utilization() == 0.1
    assert ComplexityVerdict.model_validate_json(result.content).requires_human_approval is False


@pytest.mark.asyncio
async def test_reasoning_effort_attaches_to_structured_request() -> None:
    async_openai_client = FakeAsyncOpenAI()
    llm_client = LLMClient(async_openai_client=async_openai_client)

    await llm_client.generate_structured(
        messages=[Message(role='user', content='Plan this task')],
        output_type=ComplexityVerdict,
        model='claude-opus-4-7',
        context_limit_tokens=200_000,
        reasoning_effort='high',
    )

    parse_call = async_openai_client.completions.parse_calls[0]
    assert parse_call['reasoning_effort'] == 'high'


@pytest.mark.asyncio
async def test_reasoning_effort_off_omits_the_param() -> None:
    async_openai_client = FakeAsyncOpenAI()
    llm_client = LLMClient(async_openai_client=async_openai_client)

    await llm_client.generate_structured(
        messages=[Message(role='user', content='Review this patch')],
        output_type=ComplexityVerdict,
        model='claude-opus-4-7',
        context_limit_tokens=200_000,
        reasoning_effort='',
    )

    parse_call = async_openai_client.completions.parse_calls[0]
    assert 'reasoning_effort' not in parse_call


@pytest.mark.asyncio
async def test_completion_returns_llm_result_with_usage() -> None:
    llm_client = LLMClient(async_openai_client=FakeAsyncOpenAI())

    result = await llm_client.complete(
        messages=[Message(role='user', content='Assess this task')],
        model='completion-model',
        context_limit_tokens=100,
    )

    assert result.content == '{"requires_human_approval": false}'
    assert result.model == 'completion-model'
    assert result.usage.call_count == 1
    assert result.usage.total_input_tokens == 10
    assert result.usage.total_output_tokens == 4
    assert result.context_utilization() == 0.1
