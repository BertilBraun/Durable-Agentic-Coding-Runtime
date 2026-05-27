from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.activities.temporal import durable_activity
from src.activities.workspace_manager import ToolExecutionRequest, WorkspaceInfo, run_tool
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
    ToolName,
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

    tool_name: ToolName
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

    @model_validator(mode="after")
    def validate_payload(self) -> ImplementationToolCall:
        match self.tool_name:
            case ToolName.READ_FILE_RANGE:
                _require_tool_fields(self, ("file_path", "start_line", "end_line"))
            case ToolName.SEARCH_TEXT:
                _require_tool_fields(self, ("pattern", "directory", "file_glob"))
            case ToolName.WRITE_FILE:
                _require_tool_fields(self, ("file_path", "content"))
            case ToolName.APPLY_PATCH:
                _require_tool_fields(self, ("patch",))
            case (
                ToolName.GIT_DIFF | ToolName.GIT_STATUS | ToolName.RUN_LINT | ToolName.RUN_TYPECHECK
            ):
                _require_tool_fields(self, ("path",))
            case ToolName.GIT_COMMIT:
                _require_tool_fields(self, ("message",))
            case ToolName.RUN_TESTS:
                _require_tool_fields(self, ("command", "timeout_seconds"))
            case ToolName.FIND_SYMBOL:
                _require_tool_fields(self, ("language",))
                if self.name is None and self.symbol_name is None:
                    raise ValueError("find_symbol requires name or symbol_name")
            case ToolName.FIND_REFERENCES:
                _require_tool_fields(self, ("file_path",))
                if self.symbol_name is None and self.name is None:
                    raise ValueError("find_references requires symbol_name or name")
            case ToolName.GATHER_CONTEXT:
                _require_tool_fields(self, ("prompt",))
        return self


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
            tool_result = await run_tool(
                ToolExecutionRequest(workspace_info=request.workspace_info, tool=tool)
            )
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
    tool_result = await run_tool(
        ToolExecutionRequest(workspace_info=workspace_info, tool=GitDiff(path="."))
    )
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


def _require_tool_fields(tool_call: ImplementationToolCall, field_names: tuple[str, ...]) -> None:
    missing_fields = [
        field_name for field_name in field_names if getattr(tool_call, field_name) is None
    ]
    if missing_fields:
        raise ValueError(f"{tool_call.tool_name} missing required fields: {missing_fields}")


def _tool_from_call(tool_call: ImplementationToolCall) -> Tool:
    match tool_call.tool_name:
        case ToolName.READ_FILE_RANGE:
            return ReadFileRange(
                file_path=tool_call.file_path or ".",
                start_line=tool_call.start_line or 1,
                end_line=tool_call.end_line or 200,
            )
        case ToolName.SEARCH_TEXT:
            return SearchText(
                pattern=tool_call.pattern or "",
                directory=tool_call.directory or ".",
                file_glob=tool_call.file_glob or "*",
            )
        case ToolName.WRITE_FILE:
            return WriteFile(file_path=tool_call.file_path or "", content=tool_call.content or "")
        case ToolName.APPLY_PATCH:
            return ApplyPatch(patch=tool_call.patch or "")
        case ToolName.GIT_DIFF:
            return GitDiff(path=tool_call.path or ".")
        case ToolName.GIT_STATUS:
            return GitStatus(path=tool_call.path or ".")
        case ToolName.GIT_COMMIT:
            return GitCommit(message=tool_call.message or "agentic coding change")
        case ToolName.RUN_TESTS:
            return RunTests(
                command=tool_call.command or "",
                timeout_seconds=tool_call.timeout_seconds or 300,
            )
        case ToolName.RUN_LINT:
            return RunLint(path=tool_call.path or ".")
        case ToolName.RUN_TYPECHECK:
            return RunTypecheck(path=tool_call.path or ".")
        case ToolName.FIND_SYMBOL:
            return FindSymbol(
                name=tool_call.name or tool_call.symbol_name or "",
                language=tool_call.language or "",
            )
        case ToolName.FIND_REFERENCES:
            return FindReferences(
                symbol_name=tool_call.symbol_name or tool_call.name or "",
                file_path=tool_call.file_path or ".",
            )
        case ToolName.GATHER_CONTEXT:
            return GatherContext(prompt=tool_call.prompt or "")
