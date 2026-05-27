import pytest
from pydantic import BaseModel, ValidationError
from src.activities.implementation import (
    ImplementationAgentTurn,
    ImplementationToolCall,
    ImplementationTurnRequest,
    run_implementation_turn,
)
from src.activities.workspace_manager import ToolResult, WorkspaceInfo
from src.llm.client import Message
from src.llm.config import ModelRole
from src.models.context import ContextPack
from src.models.plan import PlanStep, Risk
from src.models.task import TaskContract, TaskType
from src.models.worker import Confidence, WorkerStatus
from src.tools.definitions import RunTests, Tool, ToolName


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
                    "status": "success",
                    "patch_id": "patch-1",
                    "diff_summary": "Read target file.",
                    "tests_run": [],
                    "test_results": [],
                    "discovered_issues": [],
                    "confidence": "high",
                    "replan_suggestion": None,
                },
            }
        )


@pytest.mark.asyncio
async def test_implementation_turn_executes_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_names: list[str] = []

    async def fake_run_tool(workspace_info: WorkspaceInfo, tool: Tool) -> ToolResult:
        tool_names.append(type(tool).__name__)
        return ToolResult(stdout="file content", stderr="", exit_code=0, truncated=False)

    monkeypatch.setattr("src.activities.implementation.LLMClient", FakeImplementationClient)
    monkeypatch.setattr("src.activities.implementation.run_tool", fake_run_tool)

    worker_result = await run_implementation_turn(_implementation_request())

    assert worker_result.status == WorkerStatus.SUCCESS
    assert worker_result.confidence == Confidence.HIGH
    assert tool_names == ["ReadFileRange"]


def test_implementation_tool_call_uses_tool_name_enum() -> None:
    tool_call = ImplementationToolCall.model_validate(
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
                        }
                    ],
                }
            )

    async def fake_run_tool(workspace_info: WorkspaceInfo, tool: Tool) -> ToolResult:
        match tool:
            case RunTests(timeout_seconds=timeout_seconds):
                captured_timeout_seconds.append(timeout_seconds)
            case _:
                raise AssertionError(f"Unexpected tool: {tool}")
        return ToolResult(stdout="tests failed", stderr="", exit_code=1, truncated=False)

    monkeypatch.setenv("IMPLEMENTATION_MAX_TOOL_ROUNDS", "1")
    monkeypatch.setattr("src.activities.implementation.LLMClient", FakeRunTestsClient)
    monkeypatch.setattr("src.activities.implementation.run_tool", fake_run_tool)

    worker_result = await run_implementation_turn(_implementation_request())

    assert worker_result.status == WorkerStatus.FAILED
    assert captured_timeout_seconds == [19]


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
    )
