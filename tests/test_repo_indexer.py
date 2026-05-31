import subprocess
from pathlib import Path

import pytest
from src.activities import repo_indexer
from src.activities.repo_indexer import build_repo_index
from src.activities.workspace_manager import HostWorkspace
from src.models.repo import Language, ReferenceKind, SymbolKind


def _committed_workspace(repository_path: Path) -> HostWorkspace:
    _run_git(repository_path, 'init')
    _run_git(repository_path, 'config', 'user.email', 'test@example.com')
    _run_git(repository_path, 'config', 'user.name', 'Test User')
    _run_git(repository_path, 'add', '.')
    _run_git(repository_path, 'commit', '-m', 'initial')
    head = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return HostWorkspace(
        run_id='run-1',
        base_sha=head,
        base_branch='main',
        current_branch='main',
        repo_path=str(repository_path),
    )


def _run_git(repository_path: Path, *arguments: str) -> None:
    subprocess.run(
        ['git', *arguments],
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.asyncio
async def test_repo_indexer_finds_python_top_level_symbols(tmp_path: Path) -> None:
    (tmp_path / 'module.py').write_text(
        'class Parser:\n'
        '    def parse(self):\n'
        '        return None\n'
        '\n'
        'def build_parser():\n'
        '    return Parser()\n',
        encoding='utf-8',
    )
    workspace = _committed_workspace(tmp_path)

    repository_index = await build_repo_index(workspace)

    symbol_pairs = {(symbol.name, symbol.kind) for symbol in repository_index.symbols}
    assert ('Parser', SymbolKind.CLASS) in symbol_pairs
    assert ('build_parser', SymbolKind.FUNCTION) in symbol_pairs
    assert ('parse', SymbolKind.METHOD) in symbol_pairs
    assert repository_index.file_tree[0].language == Language.PYTHON


@pytest.mark.asyncio
async def test_repo_indexer_indexes_javascript_class_methods(tmp_path: Path) -> None:
    (tmp_path / 'service.ts').write_text(
        'export class Service {\n  start() {\n    return true;\n  }\n}\n',
        encoding='utf-8',
    )
    workspace = _committed_workspace(tmp_path)

    repository_index = await build_repo_index(workspace)

    symbol_pairs = {(symbol.name, symbol.kind) for symbol in repository_index.symbols}
    assert ('Service', SymbolKind.CLASS) in symbol_pairs
    assert ('start', SymbolKind.METHOD) in symbol_pairs


@pytest.mark.asyncio
async def test_repo_indexer_finds_javascript_exported_symbols(tmp_path: Path) -> None:
    (tmp_path / 'component.tsx').write_text(
        'export function Widget() {\n'
        '  return null;\n'
        '}\n'
        '\n'
        'export const useWidget = () => {\n'
        '  return Widget;\n'
        '};\n',
        encoding='utf-8',
    )
    workspace = _committed_workspace(tmp_path)

    repository_index = await build_repo_index(workspace)

    symbol_pairs = {(symbol.name, symbol.kind) for symbol in repository_index.symbols}
    assert ('Widget', SymbolKind.FUNCTION) in symbol_pairs
    assert ('useWidget', SymbolKind.FUNCTION) in symbol_pairs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('file_name', 'source', 'comment_line'),
    [
        (
            'module.py',
            'def build_parser():\n'
            '    return None\n'
            '\n'
            'def main():\n'
            '    # build_parser is mentioned here\n'
            '    return build_parser()\n',
            5,
        ),
        (
            'service.ts',
            'function buildParser() {\n'
            '  return null;\n'
            '}\n'
            '\n'
            'function main() {\n'
            '  // buildParser is mentioned here\n'
            '  return buildParser();\n'
            '}\n',
            6,
        ),
    ],
)
async def test_repo_indexer_captures_call_and_excludes_comment_mention(
    tmp_path: Path,
    file_name: str,
    source: str,
    comment_line: int,
) -> None:
    callee_name = 'build_parser' if file_name.endswith('.py') else 'buildParser'
    (tmp_path / file_name).write_text(source, encoding='utf-8')
    workspace = _committed_workspace(tmp_path)

    repository_index = await build_repo_index(workspace)

    calls = [
        reference
        for reference in repository_index.references
        if reference.symbol_name == callee_name and reference.kind == ReferenceKind.CALL
    ]
    assert len(calls) == 1
    assert calls[0].kind == ReferenceKind.CALL
    references_on_comment_line = [
        reference for reference in repository_index.references if reference.line == comment_line
    ]
    assert references_on_comment_line == []


def test_repo_indexer_has_no_non_tree_sitter_symbol_fallbacks() -> None:
    assert not hasattr(repo_indexer, '_python_symbols')
    assert not hasattr(repo_indexer, '_javascript_family_symbols')


def test_build_repo_index_is_durable_activity() -> None:
    assert getattr(build_repo_index, '__is_activity__', False) is True
    activity_policy = build_repo_index.__activity_policy__
    assert activity_policy.max_retries == 1
    assert activity_policy.timeout_seconds == 120
