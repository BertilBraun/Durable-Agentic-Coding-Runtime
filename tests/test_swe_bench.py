from src.eval.swe_bench import SweBenchInstance, _select_evaluation_instances


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


def _instance(instance_id: str, language: str) -> SweBenchInstance:
    return SweBenchInstance(
        instance_id=instance_id,
        repo="owner/repo",
        problem_statement="Fix the bug",
        language=language,
        docker_image=f"sweb.eval.x86_64.{instance_id}:latest",
    )
