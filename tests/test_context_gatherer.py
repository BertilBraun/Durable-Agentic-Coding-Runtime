import pytest
from pydantic import BaseModel, ValidationError
from src.activities.context_gatherer import (
    DEFAULT_READ_FILE_END_LINE,
    ContextGathererToolCall,
    ContextGathererTurn,
    ContextGatherRequest,
    _tool_from_call,
    gather_context,
)
from src.activities.workspace_manager import ToolExecutionRequest, ToolResult, WorkspaceInfo
from src.llm.client import Message
from src.llm.config import ModelRole
from src.models.repo import RepoIndex
from src.tools.definitions import ReadFileRange, ToolName


def test_context_gatherer_rejects_unknown_tool() -> None:
    with pytest.raises(ValidationError):
        ContextGathererTurn.model_validate(
            {
                "done": False,
                "tool_calls": [
                    {
                        "tool_name": "delete_everything",
                    }
                ],
            }
        )


def test_context_gatherer_rejects_mutating_tool() -> None:
    turn = ContextGathererTurn.model_validate(
        {
            "done": False,
            "tool_calls": [
                {
                    "tool_name": "write_file",
                    "file_path": "src/app.py",
                }
            ],
        }
    )

    with pytest.raises(AssertionError, match="Context gatherer cannot call tool: write_file"):
        _tool_from_call(turn.tool_calls[0])


def test_context_gatherer_tool_conversion_asserts_missing_required_payload_field() -> None:
    turn = ContextGathererTurn.model_validate(
        {
            "done": False,
            "tool_calls": [
                {
                    "tool_name": "find_references",
                }
            ],
        }
    )

    with pytest.raises(AssertionError, match="find_references symbol_name was not validated"):
        _tool_from_call(turn.tool_calls[0])


def test_context_gatherer_tool_conversion_asserts_post_validation_missing_field() -> None:
    tool_call = ContextGathererToolCall.model_construct(
        tool_name=ToolName.SEARCH_TEXT,
        pattern=None,
        directory=".",
        file_glob="*.py",
    )

    with pytest.raises(AssertionError, match="search_text pattern was not validated"):
        _tool_from_call(tool_call)


@pytest.mark.asyncio
async def test_context_gatherer_reports_invalid_tool_call_to_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages: list[list[Message]] = []

    class FakeInvalidToolClient:
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
                                "tool_name": "write_file",
                                "file_path": "app/main.py",
                                "content": "mutating call",
                            }
                        ],
                    }
                )
            return output_type.model_validate(
                {
                    "done": True,
                    "context_pack": {
                        "task_summary": "done",
                        "relevant_snippets": [],
                        "recent_observations": [],
                        "failed_attempt_summaries": [],
                        "available_tools": [],
                        "budget_remaining": 1,
                    },
                }
            )

        def context_utilization(self) -> float:
            return 0.0

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        raise AssertionError(f"run_tool should not execute invalid call: {request.tool}")

    monkeypatch.setattr("src.activities.context_gatherer.LLMClient", FakeInvalidToolClient)
    monkeypatch.setattr("src.activities.context_gatherer.run_tool", fake_run_tool)

    await gather_context(_context_gather_request())

    assert "invalid_tool_call" in captured_messages[1][-1].content
    assert "write_file" in captured_messages[1][-1].content


def test_context_gatherer_read_file_range_defaults_to_initial_file_window() -> None:
    turn = ContextGathererTurn.model_validate(
        {
            "done": False,
            "tool_calls": [
                {
                    "tool_name": "read_file_range",
                    "file_path": "app/main.py",
                }
            ],
        }
    )

    tool = _tool_from_call(turn.tool_calls[0])

    assert tool == ReadFileRange(
        file_path="app/main.py",
        start_line=1,
        end_line=DEFAULT_READ_FILE_END_LINE,
    )


@pytest.mark.asyncio
async def test_context_gatherer_sends_only_current_turn_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages: list[list[Message]] = []
    tool_outputs = ["turn one first", "turn one second", "turn two first", "turn two second"]

    class FakeContextGathererClient:
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
            if self.call_count < 3:
                return output_type.model_validate(
                    {
                        "done": False,
                        "tool_calls": [
                            {
                                "tool_name": "search_text",
                                "pattern": f"pattern-{self.call_count}-a",
                                "directory": ".",
                                "file_glob": "*.py",
                            },
                            {
                                "tool_name": "search_text",
                                "pattern": f"pattern-{self.call_count}-b",
                                "directory": ".",
                                "file_glob": "*.py",
                            },
                        ],
                    }
                )
            return output_type.model_validate(
                {
                    "done": True,
                    "context_pack": {
                        "task_summary": "done",
                        "relevant_snippets": [],
                        "recent_observations": [],
                        "failed_attempt_summaries": [],
                        "available_tools": [],
                        "budget_remaining": 1,
                    },
                }
            )

        def context_utilization(self) -> float:
            return 0.0

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        return ToolResult(stdout=tool_outputs.pop(0), stderr="", exit_code=0, truncated=False)

    monkeypatch.setattr("src.activities.context_gatherer.LLMClient", FakeContextGathererClient)
    monkeypatch.setattr("src.activities.context_gatherer.run_tool", fake_run_tool)

    await gather_context(_context_gather_request())

    third_call_last_user_message = captured_messages[2][-1].content
    assert "turn two first" in third_call_last_user_message
    assert "turn two second" in third_call_last_user_message
    assert "turn one first" not in third_call_last_user_message
    assert "turn one second" not in third_call_last_user_message


@pytest.mark.asyncio
async def test_context_gatherer_returns_best_effort_when_context_budget_is_high(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHighContextGathererClient:
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
                            "tool_name": "search_text",
                            "pattern": "handler",
                            "directory": ".",
                            "file_glob": "*.py",
                        }
                    ],
                }
            )

        def context_utilization(self) -> float:
            return 0.81

    async def fake_run_tool(request: ToolExecutionRequest) -> ToolResult:
        raise AssertionError(f"run_tool should not run after budget stop: {request.tool}")

    monkeypatch.setattr("src.activities.context_gatherer.LLMClient", FakeHighContextGathererClient)
    monkeypatch.setattr("src.activities.context_gatherer.run_tool", fake_run_tool)

    context_pack = await gather_context(_context_gather_request())

    assert context_pack.task_summary == "Find relevant code"
    assert context_pack.relevant_snippets == []
    assert context_pack.budget_remaining == 0


def _context_gather_request() -> ContextGatherRequest:
    return ContextGatherRequest(
        workspace_info=WorkspaceInfo(
            run_id="run-1",
            volume_name="volume",
            worktree_path="workspace",
            branch_name="branch",
        ),
        repo_index=RepoIndex(),
        gatherer_prompt="Find relevant code",
    )
