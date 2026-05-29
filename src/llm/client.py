from __future__ import annotations

import os
from typing import ClassVar, Literal, Protocol, TypeVar

from openai import AsyncOpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ParsedChatCompletion
from pydantic import BaseModel, ConfigDict, Field

from src.llm.config import ModelConfiguration, ModelRole, load_model_configuration

StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class LLMUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: ModelRole
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


class LLMClient:
    _global_usage_ledger: ClassVar[LLMUsageLedger] = LLMUsageLedger()

    def __init__(
        self,
        model_configuration: ModelConfiguration | None = None,
        usage_ledger: LLMUsageLedger | None = None,
        async_openai_client: AsyncOpenAIClient | None = None,
    ) -> None:
        self.model_configuration = model_configuration or load_model_configuration()
        self.usage_ledger = usage_ledger or LLMUsageLedger()
        self.last_input_token_count = 0
        self.last_context_limit = 1
        self.async_openai_client = async_openai_client or AsyncOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )

    async def complete(self, role: ModelRole, messages: list[Message]) -> str:
        model = self.model_configuration.model_for_role(role)
        response = await self.async_openai_client.chat.completions.create(
            model=model,
            messages=_format_messages_for_api(messages),
        )
        self._record_usage(
            role=role,
            usage=_usage_from_response(role=role, model=model, response=response),
        )
        return _extract_content(response)

    async def generate_structured(
        self,
        role: ModelRole,
        messages: list[Message],
        output_type: type[StructuredOutput],
    ) -> StructuredOutput:
        model = self.model_configuration.model_for_role(role)
        response = await self.async_openai_client.beta.chat.completions.parse(
            model=model,
            messages=_format_messages_for_api(messages),
            response_format=output_type,
        )
        self._record_usage(
            role=role,
            usage=_usage_from_response(role=role, model=model, response=response),
        )
        return _extract_parsed(response)

    def context_utilization(self) -> float:
        return self.last_input_token_count / self.last_context_limit

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

    def _record_usage(self, role: ModelRole, usage: LLMUsage) -> None:
        self.usage_ledger.record(usage)
        self._global_usage_ledger.record(usage)
        self.last_input_token_count = usage.input_tokens
        self.last_context_limit = self.model_configuration.context_limit_for_role(role)


def _format_messages_for_api(messages: list[Message]) -> list[ChatCompletionMessageParam]:
    return [message.model_dump(mode="json") for message in messages]  # type: ignore[list-item]


def _extract_content(response: ChatCompletion) -> str:
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM response did not include content")
    return content


def _extract_parsed(response: ParsedChatCompletion[StructuredOutput]) -> StructuredOutput:
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("LLM structured response did not include parsed content")
    return parsed


def _usage_from_response(role: ModelRole, model: str, response: ChatCompletion) -> LLMUsage:
    usage = response.usage
    if usage is None:
        return LLMUsage(role=role, model=model)
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    cache_read_tokens = _cache_read_tokens(usage)
    return LLMUsage(
        role=role,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cost_usd=_estimate_cost_usd(model, input_tokens, output_tokens, cache_read_tokens),
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
    uncached_input_tokens = max(input_tokens - cache_read_tokens, 0)
    match model:
        case "claude-opus-4-7":
            input_price = 15.0
            output_price = 75.0
        case "claude-sonnet-4-6":
            input_price = 3.0
            output_price = 15.0
        case "claude-haiku-4-5-20251001":
            input_price = 0.8
            output_price = 4.0
        case "gemini-3.1-flash-lite":
            input_price = 0.25
            output_price = 1.50
        case _:
            input_price = 0.0
            output_price = 0.0
    return ((uncached_input_tokens * input_price) + (output_tokens * output_price)) / 1_000_000
