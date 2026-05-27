from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field

from src.activities.temporal import durable_activity
from src.activities.workspace_manager import WorkspaceInfo, run_tool
from src.llm.client import LLMClient, Message
from src.llm.config import ModelRole
from src.models.context import ContextPack
from src.models.plan import PlanStep
from src.models.task import TaskContract
from src.models.worker import Confidence, WorkerResult, WorkerStatus
from src.tools.definitions import (
    ApplyPatch,
    FindReferences,
    FindSymbol,
    GatherContext,
    GitCommit,
    GitDiff,
    GitStatus,
    ReadFileRange,
    RunLint,
    RunTests,
    RunTypecheck,
    SearchText,
    Tool,
    WriteFile,
)


class ImplementationTurnRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_step: PlanStep
    context_pack: ContextPack
    task_contract: TaskContract
    workspace_info: WorkspaceInfo


class ImplementationToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: str
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    pattern: str | None = None
    directory: str | None = None
    file_glob: str | None = None
    content: str | None = None
    patch: str | None = None
    path: str | None = None
    message: str | None = None
    command: str | None = None
    timeout_seconds: int | None = None
    name: str | None = None
    language: str | None = None
    symbol_name: str | None = None
    prompt: str | None = None


class ImplementationAgentTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    done: bool
    worker_result: WorkerResult | None = None
    tool_calls: list[ImplementationToolCall] = Field(default_factory=list)


@durable_activity(retries=1, timeout=600, backoff_seconds=5)
async def run_implementation_turn(request: ImplementationTurnRequest) -> WorkerResult:
    llm_client = LLMClient()
    messages = [
        Message(
            role="system",
            content=(
                "You are the implementation worker. Emit tool calls to inspect, edit, "
                "diff, and test the workspace. Return done=true with WorkerResult only "
                "when the step is complete, blocked, failed, or needs replanning."
            ),
        ),
        Message(role="user", content=request.model_dump_json()),
    ]
    max_tool_rounds = int(os.getenv("IMPLEMENTATION_MAX_TOOL_ROUNDS", "12"))
    for _ in range(max_tool_rounds):
        agent_turn = await llm_client.generate_structured(
            role=ModelRole.IMPLEMENTATION,
            messages=messages,
            output_type=ImplementationAgentTurn,
        )
        if agent_turn.done:
            if agent_turn.worker_result is None:
                raise ValueError("worker_result is required when implementation turn is done")
            return agent_turn.worker_result

        observations: list[str] = []
        for tool_call in agent_turn.tool_calls:
            tool = _tool_from_call(tool_call)
            tool_result = await run_tool(request.workspace_info, tool)
            observations.append(
                f"tool={tool_call.tool_name} exit_code={tool_result.exit_code}\n"
                f"stdout:\n{tool_result.stdout}\n"
                f"stderr:\n{tool_result.stderr}"
            )
        messages.append(Message(role="assistant", content=agent_turn.model_dump_json()))
        messages.append(Message(role="user", content="\n\n".join(observations)))

    return failed_worker_result("maximum implementation tool rounds reached")


@durable_activity(retries=0, timeout=120)
async def get_full_diff(workspace_info: WorkspaceInfo) -> str:
    tool_result = await run_tool(workspace_info, GitDiff(path="."))
    return tool_result.stdout


def failed_worker_result(reason: str) -> WorkerResult:
    return WorkerResult(
        status=WorkerStatus.FAILED,
        patch_id=None,
        diff_summary="Implementation did not complete within the iteration budget.",
        tests_run=[],
        test_results=[],
        discovered_issues=[reason],
        confidence=Confidence.LOW,
        replan_suggestion=None,
    )


def _tool_from_call(tool_call: ImplementationToolCall) -> Tool:
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
        case "write_file":
            return WriteFile(file_path=tool_call.file_path or "", content=tool_call.content or "")
        case "apply_patch":
            return ApplyPatch(patch=tool_call.patch or "")
        case "git_diff":
            return GitDiff(path=tool_call.path or ".")
        case "git_status":
            return GitStatus(path=tool_call.path or ".")
        case "git_commit":
            return GitCommit(message=tool_call.message or "agentic coding change")
        case "run_tests":
            return RunTests(
                command=tool_call.command or "",
                timeout_seconds=tool_call.timeout_seconds or 300,
            )
        case "run_lint":
            return RunLint(path=tool_call.path or ".")
        case "run_typecheck":
            return RunTypecheck(path=tool_call.path or ".")
        case "find_symbol":
            return FindSymbol(
                name=tool_call.name or tool_call.symbol_name or "",
                language=tool_call.language or "",
            )
        case "find_references":
            return FindReferences(
                symbol_name=tool_call.symbol_name or tool_call.name or "",
                file_path=tool_call.file_path or ".",
            )
        case "gather_context":
            return GatherContext(prompt=tool_call.prompt or "")
        case _:
            raise ValueError(f"Unknown implementation tool: {tool_call.tool_name}")
