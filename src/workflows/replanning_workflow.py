from __future__ import annotations

from temporal_light import run_child, workflow

from src.activities.planner import plan_next_turn
from src.activities.workspace_manager import ToolExecutionRequest, ToolResult, WorkspaceAdapter, run_tool
from src.llm.client import LLMUsage
from src.models.context import ContextPack
from src.models.plan import ContextNote, PlannerState, PlannerToolObservation, PlannerTurn
from src.models.repo import RepoIndex
from src.tools.definitions import PlannerToolCall
from src.workflows.context_gathering_workflow import parse_context_gathering_result

_MAX_PLANNER_TOOL_CALLS_PER_TURN = 2


@workflow
async def replanning_workflow(
    workspace: dict[str, object],
    repo_index: dict[str, object],
    max_planner_turns: int,
    planner_state: dict[str, object],
) -> dict[str, object]:
    workspace_info = WorkspaceAdapter.validate_python(workspace)
    repository_index = RepoIndex.model_validate(repo_index)
    state = PlannerState.model_validate(planner_state)
    usage = LLMUsage()
    context_notes: list[ContextNote] = []
    context_packs: list[ContextPack] = []
    planner_turn = PlannerTurn()
    planner_turn_count = 0

    for _ in range(max(0, max_planner_turns)):
        planner_turn, turn_usage = await plan_next_turn(state)
        planner_turn_count += 1
        usage += turn_usage
        if planner_turn.tool_calls:
            observations: list[PlannerToolObservation] = []
            for tool_call in planner_turn.tool_calls[:_MAX_PLANNER_TOOL_CALLS_PER_TURN]:
                tool_result = await _run_planner_tool(
                    workspace_info=workspace_info,
                    repo_index=repository_index,
                    tool_call=tool_call,
                )
                observations.append(_observation_from_tool_result(tool_result))
            state = state.model_copy(
                update={
                    'tool_observations': [
                        *state.tool_observations,
                        *observations,
                    ]
                }
            )
            continue
        if not planner_turn.context_requests:
            break
        new_notes: list[ContextNote] = []
        for request in planner_turn.context_requests:
            child_result = await run_child(
                'context_gathering_workflow',
                workspace=workspace_info.model_dump(mode='json'),
                repo_index=repository_index.model_dump(mode='json'),
                request=request.model_dump(mode='json'),
            )
            note, context_pack, context_usage = parse_context_gathering_result(child_result)
            usage += context_usage
            context_notes.append(note)
            new_notes.append(note)
            context_packs.append(context_pack)
        state = state.model_copy(update={'context_notes': [*state.context_notes, *new_notes]})
    return {
        'planner_turn': planner_turn.model_dump(mode='json'),
        'context_notes': [note.model_dump(mode='json') for note in context_notes],
        'context_packs': [pack.model_dump(mode='json') for pack in context_packs],
        'planner_turn_count': planner_turn_count,
        'llm_usage': usage.model_dump(mode='json'),
    }


async def _run_planner_tool(
    workspace_info: object,
    repo_index: RepoIndex,
    tool_call: PlannerToolCall,
) -> ToolResult:
    return await run_tool(
        ToolExecutionRequest(
            workspace=workspace_info,
            tool=tool_call,
            repo_index=repo_index,
        )
    )


def _observation_from_tool_result(tool_result: ToolResult) -> PlannerToolObservation:
    if tool_result.tool_name is None:
        raise ValueError('Planner tool result did not include a tool name.')
    return PlannerToolObservation(
        tool_name=tool_result.tool_name,
        stdout=tool_result.stdout,
        stderr=tool_result.stderr,
        exit_code=tool_result.exit_code,
        truncated=tool_result.truncated,
    )
