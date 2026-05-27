import pytest
from pydantic import ValidationError
from src.activities.context_gatherer import ContextGathererTurn


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
    with pytest.raises(ValidationError):
        ContextGathererTurn.model_validate(
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


def test_context_gatherer_rejects_missing_required_payload_field() -> None:
    with pytest.raises(ValidationError):
        ContextGathererTurn.model_validate(
            {
                "done": False,
                "tool_calls": [
                    {
                        "tool_name": "find_references",
                    }
                ],
            }
        )
