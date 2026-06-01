from __future__ import annotations

from temporal_light import activity

from src.activities.workspace_manager import Workspace
from src.models.repo import (
    FULL_FILE_TREE_LIMIT,
    FileEntry,
    Language,
    RepoIndex,
    directory_overview_for_paths,
)

SKIPPED_DIRECTORY_NAMES = frozenset(
    {
        '.git',
        '.venv',
        'venv',
        'node_modules',
        '__pycache__',
        'dist',
        'build',
        'vendor',
        'generated',
    }
)


@activity(retries=1, timeout=120)
async def build_repo_index(workspace: Workspace) -> RepoIndex:
    tracked_files = [
        relative_path
        for relative_path in _list_tracked_files(workspace)
        if not _is_skipped_path(relative_path)
    ]
    if len(tracked_files) > FULL_FILE_TREE_LIMIT:
        return RepoIndex(
            overview_text=directory_overview_for_paths(tracked_files),
            tracked_file_count=len(tracked_files),
        )
    file_entries: list[FileEntry] = []
    for relative_path in tracked_files:
        file_entries.append(
            FileEntry(
                path=relative_path,
                language=_language_for_path(relative_path),
                size_bytes=0,
            )
        )
    return RepoIndex(file_tree=file_entries, tracked_file_count=len(file_entries))


def _list_tracked_files(workspace: Workspace) -> list[str]:
    result = workspace.run_command(['git', 'ls-files', '-z'])
    if result.exit_code != 0:
        raise RuntimeError(f'git ls-files failed ({result.exit_code}): {result.stderr}')
    return [path for path in result.stdout.split('\0') if path]


def _is_skipped_path(relative_path: str) -> bool:
    return any(part in SKIPPED_DIRECTORY_NAMES for part in relative_path.split('/'))


def _language_for_path(relative_path: str) -> Language:
    suffix = relative_path.rsplit('.', 1)[-1] if '.' in relative_path else ''
    match suffix:
        case 'py':
            return Language.PYTHON
        case _:
            return Language.UNKNOWN
