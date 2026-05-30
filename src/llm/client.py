from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Generic, Literal, Protocol, TypeVar

from openai import AsyncOpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ParsedChatCompletion
from pydantic import BaseModel, ConfigDict, Field
from temporal_light import activity

from src.config import CONFIG, ModelRole

StructuredOutput = TypeVar('StructuredOutput', bound=BaseModel)


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal['system', 'user', 'assistant']
    content: str


class LLMResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str
    model: str
    context_limit_tokens: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0

    def context_utilization(self) -> float:
        if self.context_limit_tokens <= 0:
            return 0.0
        return self.input_tokens / self.context_limit_tokens


@dataclass(frozen=True)
class StructuredCompletion(Generic[StructuredOutput]):
    output: StructuredOutput
    result: LLMResult


class LLMUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0


class LLMUsageLedger(BaseModel):
    model_config = ConfigDict(frozen=False)

    calls: list[LLMUsage] = Field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cost_usd: float = 0.0

    def record(self, usage: LLMUsage) -> None:
        self.calls.append(usage)
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_cache_read_tokens += usage.cache_read_tokens
        self.total_cost_usd += usage.cost_usd


class LLMUsageSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int
    total_cost_usd: float


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
    # TODO again - note that global varaibles will not survive across activity invocations in temporal, so this global usage ledger will not work as intended, need to store in durable storage and pass around
    _global_usage_ledger: ClassVar[LLMUsageLedger] = LLMUsageLedger()

    def __init__(
        self,
        usage_ledger: LLMUsageLedger | None = None,
        async_openai_client: AsyncOpenAIClient | None = None,
    ) -> None:
        self.usage_ledger = usage_ledger or LLMUsageLedger()
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
        result = _llm_result_from_response(
            model=model,
            context_limit_tokens=context_limit_tokens,
            response=response,
        )
        self._record_usage(result)
        return result

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
        result = _llm_result_from_response(
            model=model,
            context_limit_tokens=context_limit_tokens,
            response=response,
        )
        self._record_usage(result)
        return result

    @classmethod
    def reset_global_usage(cls) -> None:
        cls._global_usage_ledger = LLMUsageLedger()

    @classmethod
    def global_usage_summary(cls) -> LLMUsageSummary:
        return LLMUsageSummary(
            call_count=len(cls._global_usage_ledger.calls),
            total_input_tokens=cls._global_usage_ledger.total_input_tokens,
            total_output_tokens=cls._global_usage_ledger.total_output_tokens,
            total_cache_read_tokens=cls._global_usage_ledger.total_cache_read_tokens,
            total_cost_usd=cls._global_usage_ledger.total_cost_usd,
        )

    def _record_usage(self, result: LLMResult) -> None:
        usage = LLMUsage(
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cache_read_tokens=result.cache_read_tokens,
            cost_usd=result.cost_usd,
        )
        self.usage_ledger.record(usage)
        self._global_usage_ledger.record(usage)


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
    return StructuredCompletion(
        output=output_type.model_validate_json(result.content),
        result=result,
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
    usage = response.usage
    if usage is None:
        return LLMResult(
            content=content,
            model=model,
            context_limit_tokens=context_limit_tokens,
        )
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    cache_read_tokens = _cache_read_tokens(usage)
    return LLMResult(
        content=content,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cost_usd=_estimate_cost_usd(model, input_tokens, output_tokens, cache_read_tokens),
        context_limit_tokens=context_limit_tokens,
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
    # TODO: add cache_control breakpoints for Anthropic-compatible providers.
    entry = CONFIG.models_by_id.get(model)
    if entry is None:
        return 0.0
    uncached_input_tokens = max(input_tokens - cache_read_tokens, 0)
    input_price = entry.input_price_usd_per_mtok
    output_price = entry.output_price_usd_per_mtok
    cache_read_price = entry.cache_read_price_usd_per_mtok
    return (
        (uncached_input_tokens * input_price) + (cache_read_tokens * cache_read_price) + (output_tokens * output_price)
    ) / 1_000_000
