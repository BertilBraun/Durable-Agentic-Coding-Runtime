from __future__ import annotations

from pydantic import Field

from src.activities.workspace_manager import (
    ContextPackRequest,
    ToolExecutionRequest,
    Workspace,
    pack_context,
    run_tool,
)
from src.config import CONFIG, ModelRole
from src.llm.client import LLMUsage, Message, generate_structured
from src.models.context import ContextPack, ContextSnippet
from src.models.frozen_base_model import FrozenBaseModel
from src.models.repo import RepoIndex
from src.tools.definitions import ContextGathererToolCall

CONTEXT_GATHERER_SYSTEM_PROMPT = (
    'You are the read-only context gatherer. Use run_shell for read-only '
    'inspection only (listing files, reading file ranges, searching text) and '
    'find_definition, find_callers, and find_callees for symbol cross-references; '
    'never mutate the workspace. Write run_shell commands for the environment '
    'described in the user payload. Explore the codebase as needed, then curate the '
    'few file ranges that actually answer the requested step. Return done=true with '
    'a short task_summary and a list of ContextSnippet references (file_path, '
    'start_line, end_line, and a one-line reason explaining why the range is '
    'relevant). Never copy code into reason and never paraphrase file contents; the '
    'packer reads the real lines. Continue with another allowed tool call when a '
    'specific gap remains. Do not invent files, behavior, or test results.'
)


class ContextGatherRequest(FrozenBaseModel):
    workspace_info: Workspace
    repo_index: RepoIndex
    gatherer_prompt: str


class ContextGathererTurn(FrozenBaseModel):
    done: bool
    task_summary: str | None = None
    snippets: list[ContextSnippet] = Field(default_factory=list)
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
            content=(
                f'Environment: {request.workspace_info.describe_environment()}\n\n'
                f'Repository tree:\n{request.repo_index.directory_tree_text()}'
            ),
            cacheable=True,
        ),
        Message(
            role='user',
            content=f'Prompt: {request.gatherer_prompt}',
        ),
    ]
    max_tool_calls = CONFIG.context_gatherer_max_tool_calls
    stop_threshold = CONFIG.context_utilization_stop_threshold
    hard_stop_threshold = CONFIG.context_utilization_hard_stop_threshold
    tool_call_count = 0
    budget_exceeded = False
    latest_utilization = 0.0
    all_observations: list[str] = []
    usage = LLMUsage()

    while tool_call_count < max_tool_calls:
        completion = await generate_structured(
            role=ModelRole.CONTEXT_GATHERER,
            messages=messages,
            output_type=ContextGathererTurn,
        )
        usage += completion.usage
        turn = completion.output
        latest_utilization = completion.context_utilization()
        if latest_utilization > stop_threshold:
            budget_exceeded = True
            break
        if turn.done:
            return await _pack_curated_turn(request, turn), usage

        turn_observations: list[str] = []
        for tool in turn.tool_calls:
            if tool_call_count >= max_tool_calls:
                break
            tool_result = await run_tool(
                ToolExecutionRequest(
                    workspace=request.workspace_info,
                    tool=tool,
                    repo_index=request.repo_index,
                )
            )
            observation = tool_result.stdout or tool_result.stderr
            turn_observations.append(observation)
            tool_call_count += 1
        all_observations.extend(turn_observations)
        messages.append(Message(role='assistant', content=turn.model_dump_json()))
        messages.append(Message(role='user', content='\n'.join(turn_observations)))

    # Above the hard ceiling another LLM call would itself overflow, so emit a pack
    # that spills the collected observations to an artifact instead of curating snippets.
    if budget_exceeded and latest_utilization >= hard_stop_threshold:
        pack = await pack_context(
            ContextPackRequest(
                workspace=request.workspace_info,
                task_summary=request.gatherer_prompt,
                snippets=[],
                overflow_text='\n'.join(all_observations),
            )
        )
        return pack, usage

    finalize_turn, finalize_usage = await _finalize_emit_snippets(messages, budget_exceeded)
    pack = await _pack_curated_turn(request, finalize_turn)
    return pack, usage + finalize_usage


async def _pack_curated_turn(
    request: ContextGatherRequest,
    turn: ContextGathererTurn,
) -> ContextPack:
    if turn.task_summary is None:
        raise ValueError('Context gatherer marked done without a task_summary.')
    return await pack_context(
        ContextPackRequest(
            workspace=request.workspace_info,
            task_summary=turn.task_summary,
            snippets=turn.snippets,
        )
    )


async def _finalize_emit_snippets(
    messages: list[Message],
    budget_exceeded: bool,
) -> tuple[ContextGathererTurn, LLMUsage]:
    finalize_instruction = (
        'Stop gathering more evidence. Set done=true and return a short task_summary '
        'plus the curated ContextSnippet references (file_path, line range, reason). '
        'Do not propose more tool calls and do not copy code into the reason.'
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
    if turn.task_summary is None:
        raise ValueError('Context gatherer failed to produce a task_summary on finalization.')
    return turn, completion.usage
