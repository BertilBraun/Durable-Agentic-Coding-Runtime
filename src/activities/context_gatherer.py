from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.activities.workspace_manager import (
    ToolExecutionRequest,
    WorkspaceInfo,
    run_tool,
)
from src.config import ModelRole, settings
from src.llm.client import Message, generate_structured
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


async def gather_context(request: ContextGatherRequest) -> ContextPack:
    messages = [
        Message(
            role='system',
            content=CONTEXT_GATHERER_SYSTEM_PROMPT,
        ),
        Message(
            role='user',
            content=(
                f'Prompt: {request.gatherer_prompt}\n\n'
                f'Repository index: {request.repo_index.model_dump_json()}'
            ),
        ),
    ]
    observations: list[str] = []
    max_tool_calls = settings.context_gatherer_max_tool_calls
    stop_threshold = settings.context_utilization_stop_threshold
    tool_call_count = 0

    while tool_call_count < max_tool_calls:
        completion = await generate_structured(
            role=ModelRole.CONTEXT_GATHERER,
            messages=messages,
            output_type=ContextGathererTurn,
        )
        turn = completion.output
        if completion.result.context_utilization() > stop_threshold:
            return _best_effort_context_pack(request, observations)
        if turn.done and turn.context_pack is not None:
            return turn.context_pack

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
            observations.append(observation)
            turn_observations.append(observation)
            tool_call_count += 1
        messages.append(Message(role='assistant', content=turn.model_dump_json()))
        messages.append(Message(role='user', content='\n'.join(turn_observations)))

    # TODO the best effort context pack is really just a fallback, we should be trying to get the LLM to summarize observations and build the context pack for us, rather than just returning all observations as relevant snippets which is not really what we want, we want the LLM to do the work of figuring out what is actually relevant and summarizing it for us, this is just a stop gap to prevent complete failure when we hit token limits or something else goes wrong with the LLM generation process
    return _best_effort_context_pack(request, observations)


def _best_effort_context_pack(
    request: ContextGatherRequest,
    observations: list[str],
) -> ContextPack:
    return ContextPack(
        task_summary=request.gatherer_prompt,
        relevant_snippets=observations,
        recent_observations=[],
        failed_attempt_summaries=[],
        available_tools=['read_file_range', 'search_text', 'find_symbol', 'find_references'],
        budget_remaining=0,
    )
