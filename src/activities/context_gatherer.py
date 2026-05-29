from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field

from src.activities.temporal import durable_activity
from src.activities.workspace_manager import (
    ToolExecutionRequest,
    WorkspaceInfo,
)
from src.activities.workspace_manager import (
    run_tool_in_workspace as run_tool,
)
from src.llm.client import LLMClient, Message
from src.llm.config import CONTEXT_UTILIZATION_STOP_THRESHOLD, ModelRole
from src.llm.prompts import system_prompt_for_role
from src.models.context import ContextPack
from src.models.repo import RepoIndex
from src.tools.definitions import (
    Tool,
)
from src.tools.llm_schema import (
    ContextGathererToolCall,
    context_gatherer_tool_from_llm_tool_call,
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


@durable_activity(retries=1, timeout=300, backoff_seconds=5)
async def gather_context(request: ContextGatherRequest) -> ContextPack:
    llm_client = LLMClient()
    messages = [
        Message(
            role="system",
            content=system_prompt_for_role(ModelRole.CONTEXT_GATHERER),
        ),
        Message(
            role="user",
            content=(
                f"Prompt: {request.gatherer_prompt}\n\n"
                f"Repository index: {request.repo_index.model_dump_json()}"
            ),
        ),
    ]
    observations: list[str] = []
    max_tool_calls = int(os.getenv("CONTEXT_GATHERER_MAX_TOOL_CALLS", "10"))
    tool_call_count = 0

    while tool_call_count < max_tool_calls:
        turn = await llm_client.generate_structured(
            role=ModelRole.CONTEXT_GATHERER,
            messages=messages,
            output_type=ContextGathererTurn,
        )
        if llm_client.context_utilization() > CONTEXT_UTILIZATION_STOP_THRESHOLD:
            return _best_effort_context_pack(request, observations)
        if turn.done and turn.context_pack is not None:
            return turn.context_pack

        turn_observations: list[str] = []
        for tool_call in turn.tool_calls:
            if tool_call_count >= max_tool_calls:
                break
            try:
                tool = _tool_from_call(tool_call)
            except AssertionError as error:
                observation = (
                    f"invalid_tool_call tool={tool_call.tool_name}: {error}. "
                    "This restriction applies only to context gathering; implementation "
                    "workers may use mutating tools when their phase allows it."
                )
                observations.append(observation)
                turn_observations.append(observation)
                continue
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
        messages.append(Message(role="assistant", content=turn.model_dump_json()))
        messages.append(Message(role="user", content="\n".join(turn_observations)))

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
        available_tools=["read_file_range", "search_text", "find_symbol", "find_references"],
        budget_remaining=0,
    )


def _tool_from_call(tool_call: ContextGathererToolCall) -> Tool:
    return context_gatherer_tool_from_llm_tool_call(tool_call)
