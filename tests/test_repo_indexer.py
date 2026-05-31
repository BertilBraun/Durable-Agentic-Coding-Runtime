from pathlib import Path

import pytest
from src.activities import repo_indexer
from src.activities.repo_indexer import build_repo_index
from src.activities.workspace_manager import HostWorkspace
from src.models.repo import Language, SymbolKind


def _host_workspace(repo_path: str) -> HostWorkspace:
    return HostWorkspace(
        run_id='run-1',
        base_sha='basesha',
        base_branch='main',
        current_branch='main',
        repo_path=repo_path,
    )


@pytest.mark.asyncio
async def test_repo_indexer_finds_python_top_level_symbols(tmp_path: Path) -> None:
    source_file = tmp_path / 'module.py'
    source_file.write_text(
        'class Parser:\n'
        '    def parse(self):\n'
        '        return None\n'
        '\n'
        'def build_parser():\n'
        '    return Parser()\n',
        encoding='utf-8',
    )
    workspace = _host_workspace(str(tmp_path))

    repository_index = await build_repo_index(workspace)

    symbol_pairs = {(symbol.name, symbol.kind) for symbol in repository_index.symbols}
    assert ('Parser', SymbolKind.CLASS) in symbol_pairs
    assert ('build_parser', SymbolKind.FUNCTION) in symbol_pairs
    assert ('parse', SymbolKind.METHOD) in symbol_pairs
    assert repository_index.file_tree[0].language == Language.PYTHON


@pytest.mark.asyncio
async def test_repo_indexer_indexes_javascript_class_methods(tmp_path: Path) -> None:
    source_file = tmp_path / 'service.ts'
    source_file.write_text(
        'export class Service {\n  start() {\n    return true;\n  }\n}\n',
        encoding='utf-8',
    )
    workspace = _host_workspace(str(tmp_path))

    repository_index = await build_repo_index(workspace)

    symbol_pairs = {(symbol.name, symbol.kind) for symbol in repository_index.symbols}
    assert ('Service', SymbolKind.CLASS) in symbol_pairs
    assert ('start', SymbolKind.METHOD) in symbol_pairs


@pytest.mark.asyncio
async def test_repo_indexer_finds_javascript_exported_symbols(tmp_path: Path) -> None:
    source_file = tmp_path / 'component.tsx'
    source_file.write_text(
        'export function Widget() {\n'
        '  return null;\n'
        '}\n'
        '\n'
        'export const useWidget = () => {\n'
        '  return Widget;\n'
        '};\n',
        encoding='utf-8',
    )
    workspace = _host_workspace(str(tmp_path))

    repository_index = await build_repo_index(workspace)

    symbol_pairs = {(symbol.name, symbol.kind) for symbol in repository_index.symbols}
    assert ('Widget', SymbolKind.FUNCTION) in symbol_pairs
    assert ('useWidget', SymbolKind.FUNCTION) in symbol_pairs


def test_repo_indexer_has_no_non_tree_sitter_symbol_fallbacks() -> None:
    assert not hasattr(repo_indexer, '_python_symbols')
    assert not hasattr(repo_indexer, '_javascript_family_symbols')


def test_build_repo_index_is_durable_activity() -> None:
    assert getattr(build_repo_index, '__is_activity__', False) is True
    activity_policy = build_repo_index.__activity_policy__
    assert activity_policy.max_retries == 1
    assert activity_policy.timeout_seconds == 120
