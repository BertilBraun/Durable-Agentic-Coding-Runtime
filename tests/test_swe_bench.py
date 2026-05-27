import pytest
from src.eval.swe_bench import (
    SweBenchInstance,
    _pull_official_image,
    _select_evaluation_instances,
    _start_official_container,
)


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


class FakeImages:
    def __init__(self) -> None:
        self.pulled_images: list[str] = []

    def pull(self, image: str) -> None:
        self.pulled_images.append(image)


class FakeContainer:
    id = "container-1"


class FakeContainers:
    def __init__(self) -> None:
        self.run_arguments: dict[str, object] | None = None

    def run(self, **keyword_arguments: object) -> FakeContainer:
        self.run_arguments = keyword_arguments
        return FakeContainer()


class FakeDockerClient:
    def __init__(self) -> None:
        self.images = FakeImages()
        self.containers = FakeContainers()
