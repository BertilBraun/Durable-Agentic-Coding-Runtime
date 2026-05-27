from src.eval.smoke_workflow import (
    SmokeWorkflowInput,
    SmokeWorkflowRequest,
    _smoke_result_from_completed_event,
)
from src.models.task import TaskRequest


def test_smoke_workflow_request_serializes_temporal_payload() -> None:
    request = SmokeWorkflowRequest(
        workflow_name="main_workflow",
        workflow_input=SmokeWorkflowInput(
            request=TaskRequest(raw_request="Smoke test", repo_path="C:/repo", run_id="smoke")
        ),
    )

    assert request.model_dump(mode="json")["workflow_input"]["request"]["run_id"] == "smoke"


def test_smoke_result_requires_changed_diff_and_passing_test_result() -> None:
    result = _smoke_result_from_completed_event(
        {
            "type": "workflow_completed",
            "result": {
                "status": "accept",
                "worker_results": [
                    {
                        "status": "success",
                        "patch_id": "smoke",
                        "diff_summary": "Added smoke subtract function and test.",
                        "tests_run": ["pytest -q"],
                        "test_results": [
                            {
                                "command": "pytest -q",
                                "exit_code": 0,
                                "stdout_summary": "1 passed",
                                "stderr_summary": "",
                                "passed": True,
                            }
                        ],
                        "discovered_issues": [],
                        "confidence": "high",
                        "replan_suggestion": None,
                    }
                ],
            },
        }
    )

    assert result.status == "completed"
    assert result.changed_diff is True
    assert result.test_result_passed is True
