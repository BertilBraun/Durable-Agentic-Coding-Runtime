from __future__ import annotations

from typing import Generic, Literal, Protocol, TypeVar

from openai import AsyncOpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ParsedChatCompletion
from pydantic import BaseModel, ConfigDict
from temporal_light import activity

from src.config import CONFIG, ModelRole

StructuredOutput = TypeVar('StructuredOutput', bound=BaseModel)


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal['system', 'user', 'assistant']
    content: str


class LLMUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cost_usd: float = 0.0

    def __add__(self, other: LLMUsage) -> LLMUsage:
        return LLMUsage(
            call_count=self.call_count + other.call_count,
            total_input_tokens=self.total_input_tokens + other.total_input_tokens,
            total_output_tokens=self.total_output_tokens + other.total_output_tokens,
            total_cache_read_tokens=(self.total_cache_read_tokens + other.total_cache_read_tokens),
            total_cost_usd=self.total_cost_usd + other.total_cost_usd,
        )


class LLMResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str
    model: str
    context_limit_tokens: int
    usage: LLMUsage

    def context_utilization(self) -> float:
        if self.context_limit_tokens <= 0:
            return 0.0
        return self.usage.total_input_tokens / self.context_limit_tokens


class StructuredCompletion(LLMResult, Generic[StructuredOutput]):
    output: StructuredOutput


class ChatCompletionCreator(Protocol):
    async def create(self, **keyword_arguments: object) -> ChatCompletion: ...
    async def parse(
        self, response_format: type[StructuredOutput], **keyword_arguments: object
    ) -> ParsedChatCompletion[StructuredOutput]: ...


class ChatCompletionNamespace(Protocol):
    completions: ChatCompletionCreator


class BetaNamespace(Protocol):
    chat: ChatCompletionNamespace


class AsyncOpenAIClient(Protocol):
    chat: ChatCompletionNamespace
    beta: BetaNamespace


_structured_output_registry: dict[str, type[BaseModel]] = {}


def register_structured_output(output_type: type[BaseModel]) -> type[BaseModel]:
    _structured_output_registry[output_type.__name__] = output_type
    return output_type


def _structured_output_type(name: str) -> type[BaseModel]:
    if name not in _structured_output_registry:
        raise ValueError(
            f'Structured output type {name!r} is not registered. '
            f'Apply @register_structured_output to its class definition.'
        )
    return _structured_output_registry[name]


class LLMClient:
    def __init__(self, async_openai_client: AsyncOpenAIClient | None = None) -> None:
        if async_openai_client is None:
            self.async_openai_client = AsyncOpenAI(
                api_key=CONFIG.llm_api_key,
                base_url=CONFIG.llm_base_url,
            )
        else:
            self.async_openai_client = async_openai_client

    async def complete(
        self,
        messages: list[Message],
        model: str,
        context_limit_tokens: int,
    ) -> LLMResult:
        response = await self.async_openai_client.chat.completions.create(
            model=model,
            messages=_format_messages_for_api(messages),
        )
        return _llm_result_from_response(
            model=model,
            context_limit_tokens=context_limit_tokens,
            response=response,
        )

    async def generate_structured(
        self,
        messages: list[Message],
        output_type: type[StructuredOutput],
        model: str,
        context_limit_tokens: int,
    ) -> LLMResult:
        response = await self.async_openai_client.beta.chat.completions.parse(
            model=model,
            messages=_format_messages_for_api(messages),
            response_format=output_type,
        )
        return _llm_result_from_response(
            model=model,
            context_limit_tokens=context_limit_tokens,
            response=response,
        )


@activity(retries=2, timeout=120, backoff_seconds=10)
async def generate_completion(
    messages: list[Message],
    model: str,
    context_limit_tokens: int,
) -> LLMResult:
    return await LLMClient().complete(
        messages=messages,
        model=model,
        context_limit_tokens=context_limit_tokens,
    )


@activity(retries=2, timeout=180, backoff_seconds=10)
async def generate_structured_completion(
    messages: list[Message],
    output_type_name: str,
    model: str,
    context_limit_tokens: int,
) -> LLMResult:
    return await LLMClient().generate_structured(
        messages=messages,
        output_type=_structured_output_type(output_type_name),
        model=model,
        context_limit_tokens=context_limit_tokens,
    )


async def generate(role: ModelRole, messages: list[Message]) -> LLMResult:
    return await generate_completion(
        messages=messages,
        model=CONFIG.model_for_role(role),
        context_limit_tokens=CONFIG.context_limit_for_role(role),
    )


async def generate_structured(
    role: ModelRole,
    messages: list[Message],
    output_type: type[StructuredOutput],
) -> StructuredCompletion[StructuredOutput]:
    register_structured_output(output_type)
    result = await generate_structured_completion(
        messages=messages,
        output_type_name=output_type.__name__,
        model=CONFIG.model_for_role(role),
        context_limit_tokens=CONFIG.context_limit_for_role(role),
    )
    return StructuredCompletion[output_type](
        output=output_type.model_validate_json(result.content),
        content=result.content,
        model=result.model,
        context_limit_tokens=result.context_limit_tokens,
        usage=result.usage,
    )


def _format_messages_for_api(messages: list[Message]) -> list[ChatCompletionMessageParam]:
    return [message.model_dump(mode='json') for message in messages]  # type: ignore[list-item]


def _extract_content(response: ChatCompletion) -> str:
    content = response.choices[0].message.content
    if content is None:
        raise ValueError('LLM response did not include content')
    return content


def _llm_result_from_response(
    model: str,
    context_limit_tokens: int,
    response: ChatCompletion,
) -> LLMResult:
    content = _extract_content(response)
    response_usage = response.usage
    if response_usage is None:
        usage = LLMUsage(call_count=1)
    else:
        input_tokens = response_usage.prompt_tokens
        output_tokens = response_usage.completion_tokens
        cache_read_tokens = _cache_read_tokens(response_usage)
        usage = LLMUsage(
            call_count=1,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_cache_read_tokens=cache_read_tokens,
            total_cost_usd=_estimate_cost_usd(
                model, input_tokens, output_tokens, cache_read_tokens
            ),
        )
    return LLMResult(
        content=content,
        model=model,
        context_limit_tokens=context_limit_tokens,
        usage=usage,
    )


def _cache_read_tokens(usage: CompletionUsage) -> int:
    prompt_tokens_details = usage.prompt_tokens_details
    if prompt_tokens_details is None:
        return 0
    return prompt_tokens_details.cached_tokens or 0


def _estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
) -> float:
    entry = CONFIG.models_by_id.get(model)
    if entry is None:
        return 0.0
    uncached_input_tokens = max(input_tokens - cache_read_tokens, 0)
    input_price = entry.input_price_usd_per_mtok
    output_price = entry.output_price_usd_per_mtok
    cache_read_price = entry.cache_read_price_usd_per_mtok
    return (
        (uncached_input_tokens * input_price)
        + (cache_read_tokens * cache_read_price)
        + (output_tokens * output_price)
    ) / 1_000_000
