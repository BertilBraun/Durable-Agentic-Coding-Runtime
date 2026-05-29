from __future__ import annotations

from pathlib import Path

import tree_sitter_javascript
import tree_sitter_python
from tree_sitter import Language as TreeSitterLanguage
from tree_sitter import Node, Parser

from src.activities.temporal import durable_activity
from src.activities.workspace_manager import WorkspaceInfo
from src.models.repo import FileEntry, Language, RepoIndex, Symbol, SymbolKind

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


@durable_activity(retries=1, timeout=120)
async def build_repo_index(workspace_info: WorkspaceInfo) -> RepoIndex:
    workspace_path = Path(workspace_info.worktree_path)
    file_entries: list[FileEntry] = []
    symbols: list[Symbol] = []

    for file_path in sorted(workspace_path.rglob('*')):
        if not file_path.is_file() or _is_skipped_path(file_path):
            continue

        relative_path = file_path.relative_to(workspace_path).as_posix()
        language = _language_for_path(file_path)
        file_entries.append(
            FileEntry(
                path=relative_path,
                language=language,
                size_bytes=file_path.stat().st_size,
            )
        )
        symbols.extend(
            _symbols_for_file(file_path=file_path, relative_path=relative_path, language=language)
        )

    return RepoIndex(file_tree=file_entries, symbols=symbols)


def _is_skipped_path(file_path: Path) -> bool:
    return any(part in SKIPPED_DIRECTORY_NAMES for part in file_path.parts)


def _language_for_path(file_path: Path) -> Language:
    match file_path.suffix:
        case '.py':
            return Language.PYTHON
        case '.ts':
            return Language.TYPESCRIPT
        case '.tsx':
            return Language.TSX
        case '.js':
            return Language.JAVASCRIPT
        case '.jsx':
            return Language.JSX
        case _:
            return Language.UNKNOWN


def _symbols_for_file(file_path: Path, relative_path: str, language: Language) -> list[Symbol]:
    match language:
        case Language.UNKNOWN:
            return []
        case (
            Language.PYTHON
            | Language.TYPESCRIPT
            | Language.TSX
            | Language.JAVASCRIPT
            | Language.JSX
        ):
            return _tree_sitter_symbols_for_file(
                file_path=file_path,
                relative_path=relative_path,
                language=language,
            )
        case _:
            raise AssertionError(f'Unhandled language in _symbols_for_file: {language}')


def _tree_sitter_symbols_for_file(
    file_path: Path,
    relative_path: str,
    language: Language,
) -> list[Symbol]:
    parser = _tree_sitter_parser(language)
    syntax_tree = parser.parse(file_path.read_bytes())
    symbols: list[Symbol] = []
    for node in syntax_tree.root_node.children:
        match language:
            case Language.PYTHON:
                symbols.extend(_python_tree_sitter_symbols(node, relative_path))
            case Language.TYPESCRIPT | Language.TSX | Language.JAVASCRIPT | Language.JSX:
                symbols.extend(_javascript_tree_sitter_symbols(node, relative_path, language))
            case _:
                raise AssertionError(
                    f'Unhandled language in _tree_sitter_symbols_for_file: {language}'
                )
    return symbols


def _tree_sitter_parser(language: Language) -> Parser:
    parser = Parser()
    match language:
        case Language.PYTHON:
            parser.language = TreeSitterLanguage(tree_sitter_python.language())
            return parser
        case Language.TYPESCRIPT | Language.TSX | Language.JAVASCRIPT | Language.JSX:
            parser.language = TreeSitterLanguage(tree_sitter_javascript.language())
            return parser
        case Language.UNKNOWN:
            raise ValueError('Cannot create a tree-sitter parser for unknown language')


def _python_tree_sitter_symbols(node: Node, relative_path: str) -> list[Symbol]:
    if node.type not in {'function_definition', 'class_definition'}:
        return []
    name_node = node.child_by_field_name('name')
    if name_node is None:
        return []
    kind = SymbolKind.CLASS if node.type == 'class_definition' else SymbolKind.FUNCTION
    return [_node_symbol(name_node, node, kind, relative_path, Language.PYTHON)]


def _javascript_tree_sitter_symbols(
    node: Node,
    relative_path: str,
    language: Language,
) -> list[Symbol]:
    match node.type:
        case 'function_declaration' | 'class_declaration':
            return _javascript_named_declaration_symbols(node, relative_path, language)
        case 'export_statement':
            declaration_node = node.child_by_field_name('declaration')
            if declaration_node is None:
                return []
            return _javascript_tree_sitter_symbols(declaration_node, relative_path, language)
        case 'lexical_declaration' | 'variable_declaration':
            return _javascript_variable_symbols(node, relative_path, language)
        case _:
            return []


def _javascript_named_declaration_symbols(
    node: Node,
    relative_path: str,
    language: Language,
) -> list[Symbol]:
    name_node = node.child_by_field_name('name')
    if name_node is None:
        return []
    kind = SymbolKind.CLASS if node.type == 'class_declaration' else SymbolKind.FUNCTION
    return [_node_symbol(name_node, node, kind, relative_path, language)]


def _javascript_variable_symbols(
    node: Node,
    relative_path: str,
    language: Language,
) -> list[Symbol]:
    symbols: list[Symbol] = []
    for child in node.children:
        if child.type != 'variable_declarator':
            continue
        value_node = child.child_by_field_name('value')
        if value_node is None or value_node.type != 'arrow_function':
            continue
        name_node = child.child_by_field_name('name')
        if name_node is None:
            continue
        symbols.append(_node_symbol(name_node, child, SymbolKind.FUNCTION, relative_path, language))
    return symbols


def _node_symbol(
    name_node: Node,
    source_node: Node,
    kind: SymbolKind,
    relative_path: str,
    language: Language,
) -> Symbol:
    return Symbol(
        name=name_node.text.decode('utf-8', errors='replace'),
        kind=kind,
        file_path=relative_path,
        start_line=source_node.start_point[0] + 1,
        end_line=source_node.end_point[0] + 1,
        language=language,
    )
