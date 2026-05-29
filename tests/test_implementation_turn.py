import json

import pytest
from pydantic import BaseModel, ValidationError
from src.activities.context_gatherer import ContextGatherRequest
from src.activities.implementation import (
    IMPLEMENTATION_AVAILABLE_TOOLS,
    ImplementationAgentTurn,
    ImplementationGenerationResult,
    ImplementationTurnRequest,
    _tool_from_call,
    run_implementation_turn,
)
from src.activities.workspace_manager import ToolExecutionRequest, ToolResult, WorkspaceInfo
from src.llm.client import Message
from src.llm.config import ModelRole
from src.models.context import ContextPack
from src.models.plan import PlanStep, Risk
from src.models.repo import RepoIndex
from src.models.task import TaskContract, TaskType
from src.models.worker import Confidence, WorkerStatus
from src.tools.definitions import GitDiff, GitStatus, ReadFileRange, RunTests, ToolName
from src.tools.llm_schema import ImplementationToolCallAdapter


class FakeImplementationClient:
    def __init__(self) -> None:
        self.call_count = 0

    async def generate_structured(
        self,
        role: ModelRole,
        messages: list[Message],
        output_type: type[BaseModel],
    ) -> BaseModel:
        self.call_count += 1
        if self.call_count == 1:
            assert output_type.__name__ == "ImplementationAgentTurn"
            return output_type.model_validate(
                {
                    "done": False,
                    "tool_calls": [
                        {
                            "tool_name": "read_file_range",
                            "file_path": "src/app.py",
                            "start_line": 1,
                            "end_line": 5,
                        }
                    ],
                }
            )
        assert output_type.__name__ == "ImplementationAgentTurn"
        return output_type.model_validate(
            {
                "done": True,
                "worker_result": {
                    "status": "blocked",
                    "patch_id": None,
                    "diff_summary": "Read target file.",
                    "tests_run": [],
                    "test_results": [],
                    "discovered_issues": [],
                    "confidence": "low",
                    "replan_suggestion": "Need an edit or test to complete the step.",
                },
            }
        )

    def context_utilization(self) -> float:
        return 0.0


@pytest.mark.asyncio
async def test_implementation_turn_executes_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_names: list[str] = []
    repo_indexes: list[RepoIndex | None] = []

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        tool_names.append(type(request.tool).__name__)
        repo_indexes.append(request.repo_index)
        return ToolResult(stdout="file content", stderr="", exit_code=0, truncated=False)

    fake_client = FakeImplementationClient()

    async def fake_generate_implementation_agent_turn(
        request: object,
    ) -> ImplementationGenerationResult:
        return ImplementationGenerationResult(
            agent_turn=await fake_client.generate_structured(
                role=ModelRole.IMPLEMENTATION,
                messages=request.messages,
                output_type=ImplementationAgentTurn,
            ),
            context_utilization=fake_client.context_utilization(),
        )

    monkeypatch.setattr(
        "src.activities.implementation.generate_implementation_agent_turn",
        fake_generate_implementation_agent_turn,
    )
    monkeypatch.setattr("src.activities.implementation.run_tool", fake_run_tool)

    worker_result = await run_implementation_turn(_implementation_request())

    assert worker_result.status == WorkerStatus.BLOCKED
    assert worker_result.confidence == Confidence.LOW
    assert tool_names == ["ReadFileRange"]
    assert repo_indexes == [RepoIndex()]


@pytest.mark.asyncio
async def test_implementation_turn_dispatches_gather_context_without_run_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages: list[list[Message]] = []
    captured_gather_prompts: list[str] = []

    class FakeGatherContextClient:
        def __init__(self) -> None:
            self.call_count = 0

        async def generate_structured(
            self,
            role: ModelRole,
            messages: list[Message],
            output_type: type[BaseModel],
        ) -> BaseModel:
            self.call_count += 1
            captured_messages.append(messages)
            if self.call_count == 1:
                return output_type.model_validate(
                    {
                        "done": False,
                        "tool_calls": [
                            {
                                "tool_name": "gather_context",
                                "prompt": "Find auth callers",
                            }
                        ],
                    }
                )
            return output_type.model_validate(
                {
                    "done": True,
                    "worker_result": {
                        "status": "blocked",
                        "patch_id": None,
                        "diff_summary": "Need implementation after context review.",
                        "tests_run": [],
                        "test_results": [],
                        "discovered_issues": [],
                        "confidence": "low",
                        "replan_suggestion": "Use gathered auth context.",
                    },
                }
            )

        def context_utilization(self) -> float:
            return 0.0

    async def fake_gather_context(request: ContextGatherRequest) -> ContextPack:
        captured_gather_prompts.append(request.gatherer_prompt)
        return ContextPack(
            task_summary="Auth context",
            relevant_snippets=["src/auth.py: token handler"],
            recent_observations=["found caller"],
            failed_attempt_summaries=[],
            available_tools=["read_file_range"],
            budget_remaining=4,
        )

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        raise AssertionError(f"run_tool should not handle gather_context: {request.tool}")

    fake_client = FakeGatherContextClient()

    async def fake_generate_implementation_agent_turn(
        request: object,
    ) -> ImplementationGenerationResult:
        return ImplementationGenerationResult(
            agent_turn=await fake_client.generate_structured(
                role=ModelRole.IMPLEMENTATION,
                messages=request.messages,
                output_type=ImplementationAgentTurn,
            ),
            context_utilization=fake_client.context_utilization(),
        )

    monkeypatch.setattr(
        "src.activities.implementation.generate_implementation_agent_turn",
        fake_generate_implementation_agent_turn,
    )
    monkeypatch.setattr("src.activities.implementation.gather_context", fake_gather_context)
    monkeypatch.setattr("src.activities.implementation.run_tool", fake_run_tool)

    worker_result = await run_implementation_turn(_implementation_request())

    assert worker_result.status == WorkerStatus.BLOCKED
    assert captured_gather_prompts == ["Find auth callers"]
    assert "Auth context" in captured_messages[1][-1].content
    assert "src/auth.py: token handler" in captured_messages[1][-1].content


def test_implementation_tool_call_uses_tool_name_enum() -> None:
    tool_call = ImplementationToolCallAdapter.validate_python(
        {
            "tool_name": "read_file_range",
            "file_path": "src/app.py",
            "start_line": 1,
            "end_line": 5,
        }
    )

    assert tool_call.tool_name == ToolName.READ_FILE_RANGE


def test_implementation_tool_call_rejects_unknown_tool() -> None:
    with pytest.raises(ValidationError):
        ImplementationAgentTurn.model_validate(
            {
                "done": False,
                "tool_calls": [
                    {
                        "tool_name": "delete_everything",
                    }
                ],
            }
        )


def test_implementation_tool_call_rejects_missing_required_payload_field() -> None:
    with pytest.raises(ValidationError):
        ImplementationAgentTurn.model_validate(
            {
                "done": False,
                "tool_calls": [
                    {
                        "tool_name": "read_file_range",
                        "start_line": 1,
                        "end_line": 5,
                    }
                ],
            }
        )


def test_implementation_read_file_range_preserves_explicit_file_window() -> None:
    agent_turn = ImplementationAgentTurn.model_validate(
        {
            "done": False,
            "tool_calls": [
                {
                    "tool_name": "read_file_range",
                    "file_path": "app/main.py",
                    "start_line": 1,
                    "end_line": 400,
                }
            ],
        }
    )

    tool = _tool_from_call(agent_turn.tool_calls[0])

    assert tool == ReadFileRange(
        file_path="app/main.py",
        start_line=1,
        end_line=400,
    )


def test_implementation_run_tests_preserves_explicit_timeout_and_directory() -> None:
    agent_turn = ImplementationAgentTurn.model_validate(
        {
            "done": False,
            "tool_calls": [
                {
                    "tool_name": "run_tests",
                    "command": "pytest -q",
                    "timeout_seconds": 120,
                    "directory": ".",
                }
            ],
        }
    )

    tool = _tool_from_call(agent_turn.tool_calls[0])

    assert tool == RunTests(
        command="pytest -q",
        timeout_seconds=120,
        directory=".",
    )


def test_implementation_run_tests_preserves_directory() -> None:
    agent_turn = ImplementationAgentTurn.model_validate(
        {
            "done": False,
            "tool_calls": [
                {
                    "tool_name": "run_tests",
                    "command": "pytest -q",
                    "timeout_seconds": 120,
                    "directory": "examples/agentic-fastapi-smoke",
                }
            ],
        }
    )

    tool = _tool_from_call(agent_turn.tool_calls[0])

    assert tool == RunTests(
        command="pytest -q",
        timeout_seconds=120,
        directory="examples/agentic-fastapi-smoke",
    )


def test_implementation_git_status_preserves_explicit_path() -> None:
    agent_turn = ImplementationAgentTurn.model_validate(
        {
            "done": False,
            "tool_calls": [
                {
                    "tool_name": "git_status",
                    "path": ".",
                }
            ],
        }
    )

    tool = _tool_from_call(agent_turn.tool_calls[0])

    assert tool == GitStatus(path=".")


@pytest.mark.asyncio
async def test_implementation_turn_preserves_run_tests_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_timeout_seconds: list[int] = []

    class FakeRunTestsClient:
        async def generate_structured(
            self,
            role: ModelRole,
            messages: list[Message],
            output_type: type[BaseModel],
        ) -> BaseModel:
            return output_type.model_validate(
                {
                    "done": False,
                    "tool_calls": [
                        {
                            "tool_name": "run_tests",
                            "command": "pytest -q",
                            "timeout_seconds": 19,
                            "directory": ".",
                        }
                    ],
                }
            )

        def context_utilization(self) -> float:
            return 0.0

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        match request.tool:
            case RunTests(timeout_seconds=timeout_seconds):
                captured_timeout_seconds.append(timeout_seconds)
            case _:
                raise AssertionError(f"Unexpected tool: {request.tool}")
        return ToolResult(stdout="tests failed", stderr="", exit_code=1, truncated=False)

    monkeypatch.setenv("IMPLEMENTATION_MAX_TOOL_ROUNDS", "1")
    monkeypatch.setattr("src.activities.implementation.LLMClient", FakeRunTestsClient)
    monkeypatch.setattr("src.activities.implementation.run_tool", fake_run_tool)

    worker_result = await run_implementation_turn(_implementation_request())

    assert worker_result.status == WorkerStatus.FAILED
    assert captured_timeout_seconds == [19]


@pytest.mark.asyncio
async def test_implementation_turn_adds_run_tests_result_to_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSuccessAfterTestsClient:
        def __init__(self) -> None:
            self.call_count = 0

        async def generate_structured(
            self,
            role: ModelRole,
            messages: list[Message],
            output_type: type[BaseModel],
        ) -> BaseModel:
            self.call_count += 1
            if self.call_count == 1:
                return output_type.model_validate(
                    {
                        "done": False,
                        "tool_calls": [
                            {
                                "tool_name": "run_tests",
                                "command": "pytest tests/test_app.py -q",
                                "timeout_seconds": 30,
                                "directory": ".",
                            },
                            {
                                "tool_name": "git_diff",
                                "path": ".",
                            },
                        ],
                    }
                )
            return output_type.model_validate(
                {
                    "done": True,
                    "worker_result": {
                        "status": "success",
                        "patch_id": "patch-1",
                        "diff_summary": "Updated app behavior.",
                        "tests_run": [],
                        "test_results": [],
                        "discovered_issues": [],
                        "confidence": "high",
                        "replan_suggestion": None,
                    },
                }
            )

        def context_utilization(self) -> float:
            return 0.0

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        match request.tool:
            case RunTests():
                return ToolResult(stdout="1 passed", stderr="", exit_code=0, truncated=False)
            case GitDiff():
                return ToolResult(
                    stdout="diff --git a/app.py b/app.py",
                    stderr="",
                    exit_code=0,
                    truncated=False,
                )
            case _:
                raise AssertionError(f"Unexpected tool: {request.tool}")

    fake_client = FakeSuccessAfterTestsClient()

    async def fake_generate_implementation_agent_turn(
        request: object,
    ) -> ImplementationGenerationResult:
        return ImplementationGenerationResult(
            agent_turn=await fake_client.generate_structured(
                role=ModelRole.IMPLEMENTATION,
                messages=request.messages,
                output_type=ImplementationAgentTurn,
            ),
            context_utilization=fake_client.context_utilization(),
        )

    monkeypatch.setattr(
        "src.activities.implementation.generate_implementation_agent_turn",
        fake_generate_implementation_agent_turn,
    )
    monkeypatch.setattr("src.activities.implementation.run_tool", fake_run_tool)

    worker_result = await run_implementation_turn(_implementation_request())

    assert worker_result.status == WorkerStatus.SUCCESS
    assert worker_result.tests_run == ["pytest tests/test_app.py -q"]
    assert len(worker_result.test_results) == 1
    assert worker_result.test_results[0].passed is True
    assert worker_result.test_results[0].stdout_summary == "1 passed"


@pytest.mark.asyncio
async def test_implementation_turn_rejects_success_without_diff_or_test_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeUnsupportedSuccessClient:
        async def generate_structured(
            self,
            role: ModelRole,
            messages: list[Message],
            output_type: type[BaseModel],
        ) -> BaseModel:
            return output_type.model_validate(
                {
                    "done": True,
                    "worker_result": {
                        "status": "success",
                        "patch_id": "patch-1",
                        "diff_summary": "",
                        "tests_run": [],
                        "test_results": [],
                        "discovered_issues": [],
                        "confidence": "high",
                        "replan_suggestion": None,
                    },
                }
            )

        def context_utilization(self) -> float:
            return 0.0

    monkeypatch.setattr("src.activities.implementation.LLMClient", FakeUnsupportedSuccessClient)

    worker_result = await run_implementation_turn(_implementation_request())

    assert worker_result.status == WorkerStatus.FAILED
    assert worker_result.discovered_issues == ["success result missing diff or test evidence"]


@pytest.mark.asyncio
async def test_implementation_turn_rejects_success_with_only_diff_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeNarrativeDiffClient:
        async def generate_structured(
            self,
            role: ModelRole,
            messages: list[Message],
            output_type: type[BaseModel],
        ) -> BaseModel:
            return output_type.model_validate(
                {
                    "done": True,
                    "worker_result": {
                        "status": "success",
                        "patch_id": "patch-1",
                        "diff_summary": "No changes were needed.",
                        "tests_run": [],
                        "test_results": [],
                        "discovered_issues": [],
                        "confidence": "high",
                        "replan_suggestion": None,
                    },
                }
            )

        def context_utilization(self) -> float:
            return 0.0

    monkeypatch.setattr("src.activities.implementation.LLMClient", FakeNarrativeDiffClient)

    worker_result = await run_implementation_turn(_implementation_request())

    assert worker_result.status == WorkerStatus.FAILED
    assert worker_result.discovered_issues == ["success result missing diff or test evidence"]


@pytest.mark.asyncio
async def test_implementation_turn_blocks_when_context_budget_is_high(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHighContextClient:
        async def generate_structured(
            self,
            role: ModelRole,
            messages: list[Message],
            output_type: type[BaseModel],
        ) -> BaseModel:
            return output_type.model_validate(
                {
                    "done": False,
                    "tool_calls": [
                        {
                            "tool_name": "read_file_range",
                            "file_path": "src/app.py",
                            "start_line": 1,
                            "end_line": 5,
                        }
                    ],
                }
            )

        def context_utilization(self) -> float:
            return 0.81

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        raise AssertionError(f"run_tool should not run after budget block: {request.tool}")

    monkeypatch.setattr("src.activities.implementation.LLMClient", FakeHighContextClient)
    monkeypatch.setattr("src.activities.implementation.run_tool", fake_run_tool)

    worker_result = await run_implementation_turn(_implementation_request())

    assert worker_result.status == WorkerStatus.BLOCKED
    assert worker_result.confidence == Confidence.LOW
    assert worker_result.replan_suggestion is not None
    assert "read_file_range" in worker_result.replan_suggestion


@pytest.mark.asyncio
async def test_implementation_turn_user_message_excludes_repo_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_user_payloads: list[dict[str, object]] = []

    class FakePromptClient:
        async def generate_structured(
            self,
            role: ModelRole,
            messages: list[Message],
            output_type: type[BaseModel],
        ) -> BaseModel:
            captured_user_payloads.append(json.loads(messages[1].content))
            return output_type.model_validate(
                {
                    "done": True,
                    "worker_result": {
                        "status": "blocked",
                        "patch_id": None,
                        "diff_summary": "Prompt inspected.",
                        "tests_run": [],
                        "test_results": [],
                        "discovered_issues": [],
                        "confidence": "low",
                        "replan_suggestion": "Continue.",
                    },
                }
            )

        def context_utilization(self) -> float:
            return 0.0

    monkeypatch.setattr("src.activities.implementation.LLMClient", FakePromptClient)

    await run_implementation_turn(_implementation_request())

    assert captured_user_payloads == [
        {
            "plan_step": _implementation_request().plan_step.model_dump(mode="json"),
            "context_pack": _implementation_request().context_pack.model_dump(mode="json"),
            "task_contract": _implementation_request().task_contract.model_dump(mode="json"),
            "workspace_info": _implementation_request().workspace_info.model_dump(mode="json"),
            "available_tools": list(IMPLEMENTATION_AVAILABLE_TOOLS),
        }
    ]


def _implementation_request() -> ImplementationTurnRequest:
    return ImplementationTurnRequest(
        plan_step=PlanStep(
            id="step_1",
            goal="Read target file",
            target_files=["src/app.py"],
            allowed_files=["src/app.py"],
            tests_to_run=[],
            expected_result="File inspected",
            risk=Risk.LOW,
            requires_human_approval=False,
        ),
        context_pack=ContextPack(
            task_summary="Inspect file",
            relevant_snippets=[],
            recent_observations=[],
            failed_attempt_summaries=[],
            available_tools=[],
            budget_remaining=3,
        ),
        task_contract=TaskContract(
            task_type=TaskType.BUGFIX,
            goal="Inspect file",
            acceptance_criteria=[],
            non_goals=[],
            affected_areas=[],
            risk_areas=[],
            tests_expected=[],
            open_questions=[],
        ),
        workspace_info=WorkspaceInfo(
            run_id="run-1",
            volume_name="volume",
            worktree_path="workspace",
            branch_name="branch",
        ),
        repo_index=RepoIndex(),
    )
