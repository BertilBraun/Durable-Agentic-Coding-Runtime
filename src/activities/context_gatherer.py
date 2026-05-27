from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field

from src.activities.temporal import durable_activity
from src.activities.workspace_manager import WorkspaceInfo, run_tool
from src.llm.client import LLMClient, Message
from src.llm.config import ModelRole
from src.models.context import ContextPack
from src.models.repo import RepoIndex
from src.tools.definitions import FindReferences, FindSymbol, ReadFileRange, SearchText, Tool


class ContextGatherRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    workspace_info: WorkspaceInfo
    repo_index: RepoIndex
    gatherer_prompt: str


class ContextGathererToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: str
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    pattern: str | None = None
    directory: str | None = None
    file_glob: str | None = None
    symbol_name: str | None = None
    language: str | None = None


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
            content=(
                "Gather compact repository context. Use only read_file_range, search_text, "
                "find_symbol, and find_references. Return done=true with ContextPack when "
                "you have enough evidence."
            ),
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
        if turn.done and turn.context_pack is not None:
            return turn.context_pack

        for tool_call in turn.tool_calls:
            if tool_call_count >= max_tool_calls:
                break
            tool = _tool_from_call(tool_call)
            tool_result = await run_tool(request.workspace_info, tool)
            observations.append(tool_result.stdout or tool_result.stderr)
            tool_call_count += 1
        messages.append(Message(role="assistant", content=turn.model_dump_json()))
        messages.append(Message(role="user", content="\n".join(observations[-3:])))

    return ContextPack(
        task_summary=request.gatherer_prompt,
        relevant_snippets=observations,
        recent_observations=[],
        failed_attempt_summaries=[],
        available_tools=["read_file_range", "search_text", "find_symbol", "find_references"],
        budget_remaining=0,
    )


def _tool_from_call(tool_call: ContextGathererToolCall) -> Tool:
    match tool_call.tool_name:
        case "read_file_range":
            return ReadFileRange(
                file_path=tool_call.file_path or ".",
                start_line=tool_call.start_line or 1,
                end_line=tool_call.end_line or 200,
            )
        case "search_text":
            return SearchText(
                pattern=tool_call.pattern or "",
                directory=tool_call.directory or ".",
                file_glob=tool_call.file_glob or "*",
            )
        case "find_symbol":
            return FindSymbol(name=tool_call.symbol_name or "", language=tool_call.language or "")
        case "find_references":
            return FindReferences(
                symbol_name=tool_call.symbol_name or "", file_path=tool_call.file_path or "."
            )
        case _:
            raise ValueError(f"Unknown context gatherer tool: {tool_call.tool_name}")
