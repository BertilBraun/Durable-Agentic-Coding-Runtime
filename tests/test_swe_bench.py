import base64
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import pytest
from src.eval.swe_bench import (
    EvaluationTaskResult,
    SweBenchInstance,
    _apply_patch_to_container,
    _build_report,
    _evaluate_patch_with_oracle,
    _extract_patch_from_llm_response,
    _extract_patch_from_workflow_result,
    _pull_official_image,
    _run_baseline_task,
    _run_framework_task,
    _run_oracle,
    _select_evaluation_instances,
    _start_official_container,
)
from src.llm.client import Message
from src.llm.config import ModelRole


def test_select_evaluation_instances_returns_five_supported_instances() -> None:
    instances = [
        _instance("unsupported-1", "rust"),
        _instance("python-1", "python"),
        _instance("typescript-1", "typescript"),
        _instance("javascript-1", "javascript"),
        _instance("python-2", "python"),
        _instance("typescript-2", "typescript"),
        _instance("python-3", "python"),
    ]

    selected_instances = _select_evaluation_instances(
        instances=instances,
        limit=5,
        supported_only=True,
    )

    assert [instance.instance_id for instance in selected_instances] == [
        "python-1",
        "typescript-1",
        "javascript-1",
        "python-2",
        "typescript-2",
    ]


def test_select_evaluation_instances_can_keep_unsupported_instances() -> None:
    instances = [_instance("unsupported-1", "rust"), _instance("python-1", "python")]

    selected_instances = _select_evaluation_instances(
        instances=instances,
        limit=2,
        supported_only=False,
    )

    assert [instance.instance_id for instance in selected_instances] == [
        "unsupported-1",
        "python-1",
    ]


def test_select_evaluation_instances_limit_counts_eligible_instances() -> None:
    instances = [
        _instance("unsupported-1", "rust"),
        _instance("python-1", "python"),
        _instance("unsupported-2", "go"),
        _instance("python-2", "python"),
        _instance("python-3", "python"),
    ]

    selected_instances = _select_evaluation_instances(
        instances=instances,
        limit=2,
        supported_only=False,
    )

    assert [instance.instance_id for instance in selected_instances] == [
        "unsupported-1",
        "python-1",
        "unsupported-2",
        "python-2",
    ]


def test_select_evaluation_instances_without_limit_keeps_all_instances() -> None:
    instances = [_instance("unsupported-1", "rust"), _instance("python-1", "python")]

    selected_instances = _select_evaluation_instances(
        instances=instances,
        limit=None,
        supported_only=False,
    )

    assert [instance.instance_id for instance in selected_instances] == [
        "unsupported-1",
        "python-1",
    ]


def test_extract_patch_from_llm_response_uses_diff_fence() -> None:
    patch = _extract_patch_from_llm_response(
        "Here is the patch:\n```diff\ndiff --git a/app.py b/app.py\n```\n"
    )

    assert patch == "diff --git a/app.py b/app.py\n"


def test_build_report_includes_required_summary_fields() -> None:
    report = _build_report(
        task_results=[
            _task_result("resolved-1", "completed", True, 2.0, 4, None),
            _task_result("failed-1", "failed", False, 1.0, 3, "oracle_failed"),
            _task_result("skipped-1", "skipped", False, 0.0, 0, "unsupported_language"),
        ],
        baseline_results=[
            _task_result("baseline-1", "baseline_patch_generated", False, 0.5, 1, None)
        ],
    )

    assert report.resolved == 1
    assert report.failed == 1
    assert report.skipped == 1
    assert report.cost_per_resolved_task == 2.0
    assert report.llm_calls_per_task.resolved == 4.0
    assert report.llm_calls_per_task.failed == 3.0
    assert report.skip_reasons[0].reason == "unsupported_language"
    assert report.skip_reasons[0].count == 1


def test_evaluate_patch_with_oracle_resolves_after_successful_apply() -> None:
    docker_client = FakeDockerClient()

    result = _evaluate_patch_with_oracle(
        instance=SweBenchInstance(
            instance_id="python-1",
            repo="owner/repo",
            problem_statement="Fix the bug",
            language="python",
            fail_to_pass=["tests/test_bug.py::test_fixed"],
            pass_to_pass=["tests/test_existing.py::test_existing"],
            docker_image="sweb.eval.x86_64.python-1:latest",
        ),
        patch="diff --git a/app.py b/app.py\n",
        docker_client=docker_client,
    )

    assert result.resolved is True
    assert result.reason is None
    assert docker_client.images.pulled_images == ["sweb.eval.x86_64.python-1:latest"]
    assert "git apply" in docker_client.containers.executions[0]["command"][-1]
    assert docker_client.containers.executions[1]["command"] == [
        "sh",
        "-lc",
        "pytest tests/test_bug.py::test_fixed",
    ]


def test_evaluate_patch_with_oracle_fails_when_patch_apply_fails() -> None:
    docker_client = FakeDockerClient()
    docker_client.containers.git_apply_exit_code = 1

    result = _evaluate_patch_with_oracle(
        instance=_instance("python-1", "python"),
        patch="diff --git a/app.py b/app.py\n",
        docker_client=docker_client,
    )

    assert result.resolved is False
    assert result.reason == "patch_apply_failed"


def test_evaluate_patch_with_oracle_reads_artifact_patch(
    tmp_path: Path,
) -> None:
    docker_client = FakeDockerClient()
    patch_content = "diff --git a/app.py b/app.py\n"
    patch_path = tmp_path / "patch.diff"
    patch_path.write_text(patch_content, encoding="utf-8")

    result = _evaluate_patch_with_oracle(
        instance=_instance("python-1", "python"),
        patch=f"@{patch_path}",
        docker_client=docker_client,
    )

    assert result.resolved is True
    apply_shell_command = docker_client.containers.executions[0]["command"][-1]
    encoded = apply_shell_command.split("echo ")[1].split(" |")[0]
    assert base64.b64decode(encoded).decode() == patch_content


@pytest.mark.asyncio
async def test_run_baseline_task_scores_generated_patch_with_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker_client = FakeDockerClient()

    class FakeBaselineClient:
        def __init__(self) -> None:
            self.usage_ledger = FakeUsageLedger()

        async def complete(self, role: ModelRole, messages: list[Message]) -> str:
            self.usage_ledger.calls.append("call")
            return "```diff\ndiff --git a/app.py b/app.py\n```\n"

    monkeypatch.setattr("src.eval.swe_bench.LLMClient", FakeBaselineClient)

    result = await _run_baseline_task(
        instance=_instance("python-1", "python"),
        docker_client=docker_client,
    )

    assert result.status == "resolved"
    assert result.resolved is True
    assert result.llm_calls == 1
    assert result.patch == "diff --git a/app.py b/app.py\n"


@pytest.mark.asyncio
async def test_run_framework_task_scores_workflow_patch_with_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker_client = FakeDockerClient()

    class FakeAsyncClient:
        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(
            self,
            exception_type: type[BaseException] | None,
            exception: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            return None

        async def post(self, url: str, json: dict[str, object]) -> "FakeHttpResponse":
            return FakeHttpResponse({"workflow_id": "workflow-1"})

        async def get(self, url: str) -> "FakeHttpResponse":
            return FakeHttpResponse(
                {
                    "status": "completed",
                    "result": {
                        "patch": "diff --git a/app.py b/app.py\n",
                        "llm_usage": {"total_cost_usd": 1.25, "call_count": 7},
                    },
                }
            )

    monkeypatch.setattr("src.eval.swe_bench.httpx.AsyncClient", FakeAsyncClient)

    result = await _run_framework_task(
        instance=_instance("python-1", "python"),
        temporal_api_url="http://temporal",
        docker_client=docker_client,
    )

    assert result.status == "resolved"
    assert result.resolved is True
    assert result.cost_usd == 1.25
    assert result.llm_calls == 7
    assert result.patch == "diff --git a/app.py b/app.py\n"


def test_pull_official_image_requires_instance_image() -> None:
    with pytest.raises(ValueError, match="docker_image"):
        _pull_official_image(
            instance=_instance_without_image(),
            docker_client=FakeDockerClient(),
        )


def test_pull_official_image_uses_instance_image() -> None:
    docker_client = FakeDockerClient()

    _pull_official_image(
        instance=_instance("python-1", "python"),
        docker_client=docker_client,
    )

    assert docker_client.images.pulled_images == ["sweb.eval.x86_64.python-1:latest"]


def test_start_official_container_uses_testbed_workdir() -> None:
    docker_client = FakeDockerClient()

    container_id = _start_official_container(
        instance=_instance("python-1", "python"),
        docker_client=docker_client,
    )

    assert container_id == "container-1"
    assert docker_client.containers.run_arguments == {
        "image": "sweb.eval.x86_64.python-1:latest",
        "command": "sleep infinity",
        "detach": True,
        "working_dir": "/testbed",
    }


def test_apply_patch_to_container_encodes_patch_for_git_apply() -> None:
    docker_client = FakeDockerClient()
    patch_content = "diff --git a/app.py b/app.py\n"

    result = _apply_patch_to_container(
        container_id="container-1",
        patch=patch_content,
        docker_client=docker_client,
    )

    assert result.exit_code == 0
    assert result.applied is True
    apply_shell_command = docker_client.containers.executions[0]["command"][-1]
    assert "git apply -" in apply_shell_command
    encoded = apply_shell_command.split("echo ")[1].split(" |")[0]
    assert base64.b64decode(encoded).decode() == patch_content


def test_apply_patch_to_container_rejects_empty_patch() -> None:
    with pytest.raises(ValueError, match="patch"):
        _apply_patch_to_container(
            container_id="container-1",
            patch="  \n",
            docker_client=FakeDockerClient(),
        )


def test_run_oracle_resolves_when_fail_to_pass_and_pass_to_pass_succeed() -> None:
    docker_client = FakeDockerClient()

    result = _run_oracle(
        container_id="container-1",
        instance=SweBenchInstance(
            instance_id="python-1",
            repo="owner/repo",
            problem_statement="Fix the bug",
            language="python",
            fail_to_pass=["tests/test_bug.py::test_fixed"],
            pass_to_pass=["tests/test_existing.py::test_existing"],
            docker_image="sweb.eval.x86_64.python-1:latest",
        ),
        docker_client=docker_client,
    )

    assert result.resolved is True
    assert [command_result.command for command_result in result.command_results] == [
        "pytest tests/test_bug.py::test_fixed",
        "pytest tests/test_existing.py::test_existing",
    ]


def test_run_oracle_fails_when_pass_to_pass_regresses() -> None:
    docker_client = FakeDockerClient()
    docker_client.containers.command_exit_codes = {
        "pytest tests/test_bug.py::test_fixed": 0,
        "pytest tests/test_existing.py::test_existing": 1,
    }

    result = _run_oracle(
        container_id="container-1",
        instance=SweBenchInstance(
            instance_id="python-1",
            repo="owner/repo",
            problem_statement="Fix the bug",
            language="python",
            fail_to_pass=["tests/test_bug.py::test_fixed"],
            pass_to_pass=["tests/test_existing.py::test_existing"],
            docker_image="sweb.eval.x86_64.python-1:latest",
        ),
        docker_client=docker_client,
    )

    assert result.resolved is False
    assert result.reason == "pass_to_pass_failed"


def test_extract_patch_from_workflow_result_uses_patch_artifact_path() -> None:
    patch = _extract_patch_from_workflow_result(
        {
            "patch_artifact": {
                "path": "/artifacts/run-1/patch.diff",
                "summary": "Patch artifact",
                "kind": "diff",
            }
        }
    )

    assert patch == "@/artifacts/run-1/patch.diff"


def test_extract_patch_from_workflow_result_uses_inline_patch() -> None:
    patch = _extract_patch_from_workflow_result(
        {
            "patch": "diff --git a/app.py b/app.py\n",
        }
    )

    assert patch == "diff --git a/app.py b/app.py\n"


def test_extract_patch_from_workflow_result_rejects_missing_patch() -> None:
    with pytest.raises(ValueError, match="patch"):
        _extract_patch_from_workflow_result({"status": "accept"})


def _instance(instance_id: str, language: str) -> SweBenchInstance:
    return SweBenchInstance(
        instance_id=instance_id,
        repo="owner/repo",
        problem_statement="Fix the bug",
        language=language,
        docker_image=f"sweb.eval.x86_64.{instance_id}:latest",
    )


def _instance_without_image() -> SweBenchInstance:
    return SweBenchInstance(
        instance_id="python-no-image",
        repo="owner/repo",
        problem_statement="Fix the bug",
        language="python",
        docker_image=None,
    )


def _task_result(
    instance_id: str,
    status: str,
    resolved: bool,
    cost_usd: float,
    llm_calls: int,
    reason: str | None,
) -> EvaluationTaskResult:
    return EvaluationTaskResult(
        instance_id=instance_id,
        status=status,
        resolved=resolved,
        cost_usd=cost_usd,
        llm_calls=llm_calls,
        wall_clock_seconds=1.0,
        reason=reason,
    )


class FakeUsageLedger:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.total_cost_usd = 0.5


class FakeHttpResponse:
    def __init__(self, response_json: dict[str, object]) -> None:
        self.response_json = response_json

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.response_json


class FakeImages:
    def __init__(self) -> None:
        self.pulled_images: list[str] = []

    def pull(self, image: str) -> None:
        self.pulled_images.append(image)


@dataclass
class FakeExecResult:
    exit_code: int
    output: tuple[bytes | None, bytes | None]


class FakeContainer:
    id = "container-1"

    def __init__(self, containers: "FakeContainers") -> None:
        self._containers = containers

    def exec_run(
        self,
        command: list[str],
        workdir: str | None = None,
        demux: bool = False,
    ) -> FakeExecResult:
        self._containers.executions.append({"command": command, "workdir": workdir})
        shell_command = command[-1] if isinstance(command, list) else command
        if "git apply" in shell_command:
            exit_code = self._containers.git_apply_exit_code
        else:
            exit_code = self._containers.command_exit_codes.get(shell_command, 0)
        return FakeExecResult(exit_code=exit_code, output=(b"ok", b""))

    def stop(self, timeout: int = 10) -> None:
        pass

    def remove(self, force: bool = False) -> None:
        pass


class FakeContainers:
    def __init__(self) -> None:
        self.run_arguments: dict[str, object] | None = None
        self.executions: list[dict[str, object]] = []
        self.command_exit_codes: dict[str, int] = {}
        self.git_apply_exit_code = 0

    def run(self, **keyword_arguments: object) -> FakeContainer:
        self.run_arguments = keyword_arguments
        return FakeContainer(self)

    def get(self, container_id: str) -> FakeContainer:
        return FakeContainer(self)


class FakeDockerClient:
    def __init__(self) -> None:
        self.images = FakeImages()
        self.containers = FakeContainers()
