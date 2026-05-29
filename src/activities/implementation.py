from __future__ import annotations

import json
import os
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.activities.context_gatherer import (
    ContextGatherRequest,
    gather_context,
)
from src.activities.temporal import durable_activity
from src.activities.workspace_manager import (
    ToolExecutionRequest,
    ToolResult,
    WorkspaceInfo,
    run_tool,
)
from src.llm.client import LLMClient, Message
from src.llm.config import CONTEXT_UTILIZATION_STOP_THRESHOLD, ModelRole
from src.llm.prompts import system_prompt_for_role
from src.models.context import ContextPack
from src.models.plan import PlanStep
from src.models.repo import RepoIndex
from src.models.task import TaskContract
from src.models.worker import Confidence, TestResult, WorkerResult, WorkerStatus
from src.tools.definitions import (
    ApplyPatch,
    FindReferences,
    FindSymbol,
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

DEFAULT_READ_FILE_END_LINE = 400
DEFAULT_RUN_TESTS_TIMEOUT_SECONDS = 120
IMPLEMENTATION_AVAILABLE_TOOLS = (
    "read_file_range",
    "search_text",
    "write_file",
    "apply_patch",
    "git_diff",
    "git_status",
    "run_tests",
    "run_lint",
    "run_typecheck",
    "find_symbol",
    "find_references",
    "gather_context",
)


class ImplementationTurnRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_step: PlanStep
    context_pack: ContextPack
    task_contract: TaskContract
    workspace_info: WorkspaceInfo
    repo_index: RepoIndex


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
                _require_tool_fields(self, ("file_path",))
            case ToolName.SEARCH_TEXT:
                _require_tool_fields(self, ("pattern", "directory", "file_glob"))
            case ToolName.WRITE_FILE:
                _require_tool_fields(self, ("file_path", "content"))
            case ToolName.APPLY_PATCH:
                _require_tool_fields(self, ("patch",))
            case (
                ToolName.GIT_DIFF | ToolName.GIT_STATUS | ToolName.RUN_LINT | ToolName.RUN_TYPECHECK
            ):
                return self
            case ToolName.GIT_COMMIT:
                _require_tool_fields(self, ("message",))
            case ToolName.RUN_TESTS:
                _require_tool_fields(self, ("command",))
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


class ImplementationGenerationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: list[Message]


class ImplementationGenerationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_turn: ImplementationAgentTurn
    context_utilization: float


@dataclass(frozen=True)
class ImplementationEvidence:
    tests_run: tuple[str, ...]
    test_results: tuple[TestResult, ...]
    saw_diff: bool


@durable_activity(retries=1, timeout=180, backoff_seconds=5)
async def generate_implementation_agent_turn(
    request: ImplementationGenerationRequest,
) -> ImplementationGenerationResult:
    llm_client = LLMClient()
    agent_turn = await llm_client.generate_structured(
        role=ModelRole.IMPLEMENTATION,
        messages=request.messages,
        output_type=ImplementationAgentTurn,
    )
    return ImplementationGenerationResult(
        agent_turn=agent_turn,
        context_utilization=llm_client.context_utilization(),
    )


async def run_implementation_turn(request: ImplementationTurnRequest) -> WorkerResult:
    messages = [
        Message(
            role="system",
            content=system_prompt_for_role(ModelRole.IMPLEMENTATION),
        ),
        Message(role="user", content=json.dumps(_llm_user_payload(request))),
    ]
    max_tool_rounds = int(os.getenv("IMPLEMENTATION_MAX_TOOL_ROUNDS", "12"))
    tests_run: list[str] = []
    test_results: list[TestResult] = []
    saw_diff = False
    completed_tool_calls: list[str] = []
    for _ in range(max_tool_rounds):
        generation_result = await generate_implementation_agent_turn(
            ImplementationGenerationRequest(messages=messages)
        )
        agent_turn = generation_result.agent_turn
        if generation_result.context_utilization > CONTEXT_UTILIZATION_STOP_THRESHOLD:
            return _context_budget_blocked_worker_result(
                completed_tool_calls=completed_tool_calls,
                pending_tool_calls=[
                    tool_call.tool_name.value for tool_call in agent_turn.tool_calls
                ],
            )
        if agent_turn.done:
            if agent_turn.worker_result is None:
                raise ValueError("worker_result is required when implementation turn is done")
            return _worker_result_with_evidence(
                worker_result=agent_turn.worker_result,
                evidence=ImplementationEvidence(
                    tests_run=tuple(tests_run),
                    test_results=tuple(test_results),
                    saw_diff=saw_diff,
                ),
            )

        observations: list[str] = []
        for tool_call in agent_turn.tool_calls:
            if tool_call.tool_name == ToolName.GATHER_CONTEXT:
                assert tool_call.prompt is not None, "gather_context prompt was not validated"
                gathered_context = await gather_context(
                    ContextGatherRequest(
                        workspace_info=request.workspace_info,
                        repo_index=request.repo_index,
                        gatherer_prompt=tool_call.prompt,
                    )
                )
                observations.append(
                    f"tool={tool_call.tool_name} context_pack:\n"
                    f"{gathered_context.model_dump_json()}"
                )
                completed_tool_calls.append(tool_call.tool_name.value)
                continue
            tool = _tool_from_call(tool_call)
            tool_result = await run_tool(
                ToolExecutionRequest(
                    workspace_info=request.workspace_info,
                    tool=tool,
                    repo_index=request.repo_index,
                )
            )
            completed_tool_calls.append(tool_call.tool_name.value)
            match tool:
                case RunTests(command=command):
                    tests_run.append(command)
                    test_results.append(_test_result_from_tool_result(command, tool_result))
                case GitDiff():
                    saw_diff = saw_diff or bool(tool_result.stdout.strip())
                case _:
                    pass
            observations.append(
                f"tool={tool_call.tool_name} exit_code={tool_result.exit_code}\n"
                f"stdout:\n{tool_result.stdout}\n"
                f"stderr:\n{tool_result.stderr}"
            )
        messages.append(Message(role="assistant", content=agent_turn.model_dump_json()))
        messages.append(Message(role="user", content="\n\n".join(observations)))

    return failed_worker_result("maximum implementation tool rounds reached")


def _llm_user_payload(request: ImplementationTurnRequest) -> dict[str, object]:
    return {
        "plan_step": request.plan_step.model_dump(mode="json"),
        "context_pack": request.context_pack.model_dump(mode="json"),
        "task_contract": request.task_contract.model_dump(mode="json"),
        "workspace_info": request.workspace_info.model_dump(mode="json"),
        "available_tools": list(IMPLEMENTATION_AVAILABLE_TOOLS),
    }


def _context_budget_blocked_worker_result(
    completed_tool_calls: list[str],
    pending_tool_calls: list[str],
) -> WorkerResult:
    completed_summary = ", ".join(completed_tool_calls) if completed_tool_calls else "none"
    pending_summary = ", ".join(pending_tool_calls) if pending_tool_calls else "none"
    return WorkerResult(
        status=WorkerStatus.BLOCKED,
        patch_id=None,
        diff_summary="Implementation stopped before exceeding the context window budget.",
        tests_run=[],
        test_results=[],
        discovered_issues=["context utilization exceeded 80 percent"],
        confidence=Confidence.LOW,
        replan_suggestion=(
            "Context budget exceeded. Completed tool calls: "
            f"{completed_summary}. Pending tool calls: {pending_summary}."
        ),
    )


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


def _worker_result_with_evidence(
    worker_result: WorkerResult,
    evidence: ImplementationEvidence,
) -> WorkerResult:
    if worker_result.status != WorkerStatus.SUCCESS:
        return worker_result
    tests_run = list(dict.fromkeys([*worker_result.tests_run, *evidence.tests_run]))
    test_results = [*worker_result.test_results, *evidence.test_results]
    has_diff_evidence = evidence.saw_diff
    if not has_diff_evidence and not test_results:
        return failed_worker_result("success result missing diff or test evidence")
    return worker_result.model_copy(
        update={
            "tests_run": tests_run,
            "test_results": test_results,
        }
    )


def _test_result_from_tool_result(command: str, tool_result: ToolResult) -> TestResult:
    return TestResult(
        command=command,
        exit_code=tool_result.exit_code,
        stdout_summary=tool_result.stdout,
        stderr_summary=tool_result.stderr,
        passed=tool_result.exit_code == 0,
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
            assert tool_call.file_path is not None, "read_file_range file_path was not validated"
            return ReadFileRange(
                file_path=tool_call.file_path,
                start_line=_read_file_start_line(tool_call),
                end_line=_read_file_end_line(tool_call),
            )
        case ToolName.SEARCH_TEXT:
            assert tool_call.pattern is not None, "search_text pattern was not validated"
            assert tool_call.directory is not None, "search_text directory was not validated"
            assert tool_call.file_glob is not None, "search_text file_glob was not validated"
            return SearchText(
                pattern=tool_call.pattern,
                directory=tool_call.directory,
                file_glob=tool_call.file_glob,
            )
        case ToolName.WRITE_FILE:
            assert tool_call.file_path is not None, "write_file file_path was not validated"
            assert tool_call.content is not None, "write_file content was not validated"
            return WriteFile(file_path=tool_call.file_path, content=tool_call.content)
        case ToolName.APPLY_PATCH:
            assert tool_call.patch is not None, "apply_patch patch was not validated"
            return ApplyPatch(patch=tool_call.patch)
        case ToolName.GIT_DIFF:
            return GitDiff(path=_tool_path(tool_call))
        case ToolName.GIT_STATUS:
            return GitStatus(path=_tool_path(tool_call))
        case ToolName.GIT_COMMIT:
            assert tool_call.message is not None, "git_commit message was not validated"
            return GitCommit(message=tool_call.message)
        case ToolName.RUN_TESTS:
            assert tool_call.command is not None, "run_tests command was not validated"
            return RunTests(
                command=tool_call.command,
                timeout_seconds=_run_tests_timeout_seconds(tool_call),
            )
        case ToolName.RUN_LINT:
            return RunLint(path=_tool_path(tool_call))
        case ToolName.RUN_TYPECHECK:
            return RunTypecheck(path=_tool_path(tool_call))
        case ToolName.FIND_SYMBOL:
            symbol_name = tool_call.name if tool_call.name is not None else tool_call.symbol_name
            assert symbol_name is not None, "find_symbol name was not validated"
            assert tool_call.language is not None, "find_symbol language was not validated"
            return FindSymbol(
                name=symbol_name,
                language=tool_call.language,
            )
        case ToolName.FIND_REFERENCES:
            symbol_name = (
                tool_call.symbol_name if tool_call.symbol_name is not None else tool_call.name
            )
            assert symbol_name is not None, "find_references symbol_name was not validated"
            assert tool_call.file_path is not None, "find_references file_path was not validated"
            return FindReferences(
                symbol_name=symbol_name,
                file_path=tool_call.file_path,
            )
        case ToolName.GATHER_CONTEXT:
            raise AssertionError("gather_context must be dispatched before tool conversion")
        case _:
            raise AssertionError(f"Unhandled tool in _tool_from_call: {tool_call.tool_name}")


def _read_file_start_line(tool_call: ImplementationToolCall) -> int:
    if tool_call.start_line is None:
        return 1
    return tool_call.start_line


def _read_file_end_line(tool_call: ImplementationToolCall) -> int:
    if tool_call.end_line is None:
        return DEFAULT_READ_FILE_END_LINE
    return tool_call.end_line


def _run_tests_timeout_seconds(tool_call: ImplementationToolCall) -> int:
    if tool_call.timeout_seconds is None:
        return DEFAULT_RUN_TESTS_TIMEOUT_SECONDS
    return tool_call.timeout_seconds


def _tool_path(tool_call: ImplementationToolCall) -> str:
    if tool_call.path is None:
        return "."
    return tool_call.path
