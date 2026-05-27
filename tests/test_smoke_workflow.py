from src.eval.smoke_workflow import SmokeWorkflowInput, SmokeWorkflowRequest
from src.models.task import TaskRequest


def test_smoke_workflow_request_serializes_temporal_payload() -> None:
    request = SmokeWorkflowRequest(
        workflow_name="main_workflow",
        workflow_input=SmokeWorkflowInput(
            request=TaskRequest(raw_request="Smoke test", repo_path="C:/repo", run_id="smoke")
        ),
    )

    assert request.model_dump(mode="json")["workflow_input"]["request"]["run_id"] == "smoke"
