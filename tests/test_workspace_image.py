from pathlib import Path


def test_workspace_image_contains_fastapi_smoke_dependencies() -> None:
    dockerfile = Path('workspace.Dockerfile').read_text(encoding='utf-8')

    assert 'fastapi' in dockerfile
    assert 'httpx' in dockerfile
