from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.activities.workspace_manager import (
    ToolExecutionRequest,
    WorkspaceInfo,
    run_tool,
)
from src.config import CONFIG, ModelRole
from src.llm.client import LLMUsage, Message, generate_structured
from src.models.context import ContextPack
from src.models.repo import RepoIndex
from src.tools.definitions import ContextGathererToolCall

CONTEXT_GATHERER_SYSTEM_PROMPT = (
    'You are the read-only context gatherer. Use only read_file_range, '
    'search_text, find_symbol, and find_references; never request mutating '
    'tools. Gather compact evidence for the requested step: relevant code, '
    'tests, risks, and open questions. Avoid repeating observations. Return '
    'done=true with ContextPack when enough context exists, or continue with '
    'another allowed tool call when a specific gap remains. Do not invent '
    'files, behavior, or test results.'
)


class ContextGatherRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    workspace_info: WorkspaceInfo
    repo_index: RepoIndex
    gatherer_prompt: str


class ContextGathererTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    done: bool
    context_pack: ContextPack | None = None
    tool_calls: list[ContextGathererToolCall] = Field(default_factory=list)


async def gather_context(request: ContextGatherRequest) -> tuple[ContextPack, LLMUsage]:
    messages = [
        Message(
            role='system',
            content=CONTEXT_GATHERER_SYSTEM_PROMPT,
            cacheable=True,
        ),
        Message(
            role='user',
            content=f'Repository index: {request.repo_index.model_dump_json()}',
            cacheable=True,
        ),
        Message(
            role='user',
            content=f'Prompt: {request.gatherer_prompt}',
        ),
    ]
    max_tool_calls = CONFIG.context_gatherer_max_tool_calls
    stop_threshold = CONFIG.context_utilization_stop_threshold
    tool_call_count = 0
    budget_exceeded = False
    usage = LLMUsage()

    while tool_call_count < max_tool_calls:
        completion = await generate_structured(
            role=ModelRole.CONTEXT_GATHERER,
            messages=messages,
            output_type=ContextGathererTurn,
        )
        usage += completion.usage
        turn = completion.output
        if completion.context_utilization() > stop_threshold:
            budget_exceeded = True
            break
        if turn.done and turn.context_pack is not None:
            return turn.context_pack, usage

        turn_observations: list[str] = []
        for tool in turn.tool_calls:
            if tool_call_count >= max_tool_calls:
                break
            tool_result = await run_tool(
                ToolExecutionRequest(
                    workspace_info=request.workspace_info,
                    tool=tool,
                    repo_index=request.repo_index,
                )
            )
            observation = tool_result.stdout or tool_result.stderr
            turn_observations.append(observation)
            tool_call_count += 1
        messages.append(Message(role='assistant', content=turn.model_dump_json()))
        messages.append(Message(role='user', content='\n'.join(turn_observations)))

    context_pack, finalize_usage = await _summarize_observations_into_context_pack(
        messages, budget_exceeded
    )
    return context_pack, usage + finalize_usage


async def _summarize_observations_into_context_pack(
    messages: list[Message],
    budget_exceeded: bool,
) -> tuple[ContextPack, LLMUsage]:
    finalize_instruction = (
        'Stop gathering more evidence. Set done=true and return a ContextPack '
        'summarizing the observations so far. Do not propose more tool calls.'
    )
    if budget_exceeded:
        finalize_instruction = f'The context budget is exhausted. {finalize_instruction}'
    messages = [*messages, Message(role='user', content=finalize_instruction)]
    completion = await generate_structured(
        role=ModelRole.CONTEXT_GATHERER,
        messages=messages,
        output_type=ContextGathererTurn,
    )
    turn = completion.output
    if turn.context_pack is None:
        raise ValueError('Context gatherer failed to produce a ContextPack on finalization.')
    return turn.context_pack, completion.usage
