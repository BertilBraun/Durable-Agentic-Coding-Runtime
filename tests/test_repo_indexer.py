import pytest
from src.activities.repo_indexer import build_repo_index
from src.activities.workspace_manager import CommandResult, HostWorkspace
from src.models.repo import FileEntry, Language, RepoIndex


class RecordingWorkspace(HostWorkspace):
    commands: list[list[str]]
    tracked_files: list[str]

    def run_command(self, command: list[str], timeout: int | None = None) -> CommandResult:
        self.commands.append(command)
        if command == ['git', 'ls-files', '-z']:
            return CommandResult(
                stdout='\0'.join(self.tracked_files) + '\0',
                stderr='',
                exit_code=0,
            )
        raise AssertionError(f'unexpected command: {command}')


def _recording_workspace() -> RecordingWorkspace:
    return RecordingWorkspace(
        run_id='run-1',
        base_sha='basesha',
        base_branch='main',
        current_branch='main',
        repo_path='workspace',
        commands=[],
        tracked_files=['module.py', 'package/service.py', 'README.md'],
    )


@pytest.mark.asyncio
async def test_repo_indexer_builds_file_overview_without_reading_files() -> None:
    workspace = _recording_workspace()

    repository_index = await build_repo_index(workspace)

    assert workspace.commands == [['git', 'ls-files', '-z']]
    assert repository_index.symbols == []
    assert repository_index.references == []
    assert [
        (entry.path, entry.language, entry.size_bytes) for entry in repository_index.file_tree
    ] == [
        ('module.py', Language.PYTHON, 0),
        ('package/service.py', Language.PYTHON, 0),
        ('README.md', Language.UNKNOWN, 0),
    ]


def test_directory_tree_text_shows_files_for_small_repositories() -> None:
    repository_index = RepoIndex(
        file_tree=[
            FileEntry(path='app.py', language=Language.PYTHON, size_bytes=0),
            FileEntry(path='tests/test_app.py', language=Language.PYTHON, size_bytes=0),
        ]
    )

    assert repository_index.directory_tree_text() == 'app.py\ntests\n  test_app.py'


def test_directory_tree_text_expands_by_level_until_entry_budget() -> None:
    repository_index = RepoIndex(
        file_tree=[
            FileEntry(path='alpha/a.py', language=Language.PYTHON, size_bytes=0),
            FileEntry(path='alpha/b.py', language=Language.PYTHON, size_bytes=0),
            FileEntry(path='beta/c.py', language=Language.PYTHON, size_bytes=0),
            FileEntry(path='docs/readme.md', language=Language.UNKNOWN, size_bytes=0),
        ]
    )

    assert repository_index.directory_tree_text() == (
        'alpha\n  a.py\n  b.py\nbeta\n  c.py\ndocs\n  readme.md'
    )


def test_directory_tree_text_stops_expanding_before_entry_budget_is_exceeded() -> None:
    repository_index = RepoIndex(
        file_tree=[
            FileEntry(
                path=f'pkg_{package}/module_{index}.py',
                language=Language.PYTHON,
                size_bytes=0,
            )
            for package in range(10)
            for index in range(10)
        ]
    )

    lines = repository_index.directory_tree_text().splitlines()
    assert lines == [f'pkg_{index}' for index in range(10)]


@pytest.mark.asyncio
async def test_repo_indexer_serializes_compact_overview_for_large_repositories() -> None:
    workspace = _recording_workspace().model_copy(
        update={
            'tracked_files': [f'pkg/module_{index}.py' for index in range(31)]
            + ['tests/test_app.py']
        }
    )

    repository_index = await build_repo_index(workspace)
    payload = repository_index.model_dump(mode='json')

    assert repository_index.tracked_file_count == 32
    assert repository_index.file_tree == []
    assert repository_index.directory_tree_text() == 'pkg\ntests'
    assert 'module_0.py' not in str(payload)
    assert payload == {
        'file_tree': [],
        'symbols': [],
        'references': [],
        'overview_text': 'pkg\ntests',
        'tracked_file_count': 32,
    }


def test_build_repo_index_is_durable_activity() -> None:
    assert getattr(build_repo_index, '__is_activity__', False) is True
    activity_policy = build_repo_index.__activity_policy__
    assert activity_policy.max_retries == 1
    assert activity_policy.timeout_seconds == 120
