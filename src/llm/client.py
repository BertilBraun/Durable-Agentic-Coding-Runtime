from __future__ import annotations

import os
from typing import ClassVar, Literal, Protocol, TypeVar

from openai import AsyncOpenAI
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
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost_usd: float


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
    async def create(self, **keyword_arguments: object) -> object: ...


class ChatCompletionNamespace(Protocol):
    completions: ChatCompletionCreator


class AsyncOpenAIClient(Protocol):
    chat: ChatCompletionNamespace


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
        self.async_openai_client = async_openai_client
        self.last_input_token_count = 0
        self.last_context_limit = 1
        if self.async_openai_client is None and not _fake_mode_enabled():
            self.async_openai_client = AsyncOpenAI(
                api_key=os.getenv("LLM_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL"),
            )

    async def complete(self, role: ModelRole, messages: list[Message]) -> str:
        model = self.model_configuration.model_for_role(role)
        if _fake_mode_enabled():
            content = _fake_completion_content(role)
            self._record_usage(
                role=role,
                usage=_fake_usage(role=role, model=model, messages=messages, content=content),
            )
            return content
        if self.async_openai_client is None:
            raise ValueError("OpenAI client is required when fake mode is disabled")
        response = await self.async_openai_client.chat.completions.create(
            model=model,
            messages=[message.model_dump(mode="json") for message in messages],
        )
        content = _extract_content(response)
        self._record_usage(
            role=role,
            usage=_usage_from_response(role=role, model=model, response=response),
        )
        return content

    async def generate_structured(
        self,
        role: ModelRole,
        messages: list[Message],
        output_type: type[StructuredOutput],
    ) -> StructuredOutput:
        if _fake_mode_enabled():
            model = self.model_configuration.model_for_role(role)
            content = _fake_structured_content(output_type, messages)
            self._record_usage(
                role=role,
                usage=_fake_usage(role=role, model=model, messages=messages, content=content),
            )
            return output_type.model_validate_json(content)
        structured_messages = [
            *messages,
            Message(
                role="system",
                content=(
                    "Return only valid JSON matching the requested schema. "
                    f"Schema: {output_type.model_json_schema()}"
                ),
            ),
        ]
        content = await self.complete(role=role, messages=structured_messages)
        return output_type.model_validate_json(content)

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


def _fake_mode_enabled() -> bool:
    return os.getenv("LLM_FAKE_MODE") == "1"


def _fake_completion_content(role: ModelRole) -> str:
    match role:
        case ModelRole.IMPLEMENTATION:
            return ""
        case _:
            return "{}"


def _fake_structured_content(output_type: type[BaseModel], messages: list[Message]) -> str:
    match output_type.__name__:
        case "TaskContract":
            return (
                '{"task_type":"bugfix","goal":"Run the smoke workflow",'
                '"acceptance_criteria":["Workflow completes"],"non_goals":[],'
                '"affected_areas":["smoke"],"risk_areas":[],"tests_expected":[],'
                '"open_questions":[]}'
            )
        case "ComplexityVerdict":
            return '{"requires_human_approval":false,"reasoning":"Narrow smoke task."}'
        case "Plan":
            return (
                '{"summary":"Smoke test plan","steps":[{"id":"step_1",'
                '"goal":"Add a deterministic smoke patch","target_files":["app.py","test_app.py"],'
                '"allowed_files":["app.py","test_app.py"],"tests_to_run":["pytest -q"],'
                '"expected_result":"Patch changes code and tests pass",'
                '"risk":"low","requires_human_approval":false}],'
                '"integration_tests":["pytest -q"],"rollback_strategy":"Discard workspace",'
                '"definition_of_done":["Diff is non-empty","Smoke test passes"]}'
            )
        case "ContextGathererTurn":
            return (
                '{"done":true,"context_pack":{"task_summary":"Smoke context",'
                '"relevant_snippets":[],"recent_observations":[],'
                '"failed_attempt_summaries":[],"available_tools":[],"budget_remaining":0},'
                '"tool_calls":[]}'
            )
        case "ImplementationAgentTurn":
            if _fake_implementation_has_observations(messages):
                return (
                    '{"done":true,"worker_result":{"status":"success","patch_id":"smoke",'
                    '"diff_summary":"Added smoke subtract function and test.",'
                    '"tests_run":[],"test_results":[],"discovered_issues":[],'
                    '"confidence":"high","replan_suggestion":null},"tool_calls":[]}'
                )
            return _fake_implementation_tool_turn()
        case "ReviewVerdict":
            return (
                '{"verdict":"accept","blocking_issues":[],"non_blocking_issues":[],'
                '"evidence":["Smoke workflow completed"],"missing_tests":[],'
                '"regression_risks":[],"minimality_assessment":"No-op smoke change",'
                '"recommended_next_action":"Accept smoke result"}'
            )
        case _:
            raise ValueError(f"No fake structured response for {output_type.__name__}")


def _fake_implementation_has_observations(messages: list[Message]) -> bool:
    return any("tool=run_tests" in message.content for message in messages)


def _fake_implementation_tool_turn() -> str:
    patch = (
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1,2 +1,5 @@\n"
        " def add(first_number: int, second_number: int) -> int:\n"
        "     return first_number + second_number\n"
        "+\n"
        "+def subtract(first_number: int, second_number: int) -> int:\n"
        "+    return first_number - second_number\n"
        "diff --git a/test_app.py b/test_app.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/test_app.py\n"
        "@@ -0,0 +1,5 @@\n"
        "+from app import subtract\n"
        "+\n"
        "+\n"
        "+def test_subtract() -> None:\n"
        "+    assert subtract(3, 1) == 2\n"
    )
    escaped_patch = patch.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return (
        '{"done":false,"worker_result":null,"tool_calls":['
        f'{{"tool_name":"apply_patch","patch":"{escaped_patch}"}},'
        '{"tool_name":"run_tests","command":"pytest -q","timeout_seconds":60},'
        '{"tool_name":"git_diff","path":"."}'
        "]}"
    )


def _fake_usage(role: ModelRole, model: str, messages: list[Message], content: str) -> LLMUsage:
    input_tokens = sum(len(message.content.split()) for message in messages)
    output_tokens = len(content.split())
    return LLMUsage(
        role=role,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=0,
        cost_usd=0.0,
    )


def _extract_content(response: object) -> str:
    choices = response.choices
    first_choice = choices[0]
    message = first_choice.message
    content = message.content
    if not isinstance(content, str):
        raise ValueError("LLM response content was not a string")
    return content


def _usage_from_response(role: ModelRole, model: str, response: object) -> LLMUsage:
    raw_usage = getattr(response, "usage", None)
    input_tokens = int(getattr(raw_usage, "prompt_tokens", 0))
    output_tokens = int(getattr(raw_usage, "completion_tokens", 0))
    cache_read_tokens = _cache_read_tokens(raw_usage)
    return LLMUsage(
        role=role,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cost_usd=_estimate_cost_usd(model, input_tokens, output_tokens, cache_read_tokens),
    )


def _cache_read_tokens(raw_usage: object | None) -> int:
    if raw_usage is None:
        return 0
    prompt_tokens_details = getattr(raw_usage, "prompt_tokens_details", None)
    return int(getattr(prompt_tokens_details, "cached_tokens", 0))


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
        case _:
            input_price = 0.0
            output_price = 0.0
    return ((uncached_input_tokens * input_price) + (output_tokens * output_price)) / 1_000_000
