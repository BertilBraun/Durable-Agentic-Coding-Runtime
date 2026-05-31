from __future__ import annotations

from collections.abc import Iterator

import tree_sitter_javascript
import tree_sitter_python
from temporal_light import activity
from tree_sitter import Language as TreeSitterLanguage
from tree_sitter import Node, Parser

from src.activities.workspace_manager import Workspace
from src.models.repo import (
    FileEntry,
    Language,
    Reference,
    ReferenceKind,
    RepoIndex,
    Symbol,
    SymbolKind,
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
    file_entries: list[FileEntry] = []
    symbols: list[Symbol] = []
    references: list[Reference] = []

    for relative_path in _list_tracked_files(workspace):
        if _is_skipped_path(relative_path):
            continue
        source_bytes = _read_tracked_file(workspace, relative_path)
        language = _language_for_path(relative_path)
        file_entries.append(
            FileEntry(
                path=relative_path,
                language=language,
                size_bytes=len(source_bytes),
            )
        )
        file_symbols, file_references = _index_file(
            source_bytes=source_bytes,
            relative_path=relative_path,
            language=language,
        )
        symbols.extend(file_symbols)
        references.extend(file_references)

    return RepoIndex(file_tree=file_entries, symbols=symbols, references=references)


def _list_tracked_files(workspace: Workspace) -> list[str]:
    result = workspace.run_command(['git', 'ls-files', '-z'])
    if result.exit_code != 0:
        raise RuntimeError(f'git ls-files failed ({result.exit_code}): {result.stderr}')
    return [path for path in result.stdout.split('\0') if path]


def _read_tracked_file(workspace: Workspace, relative_path: str) -> bytes:
    result = workspace.run_command(['git', 'show', f'{workspace.base_sha}:{relative_path}'])
    if result.exit_code != 0:
        return b''
    return result.stdout.encode('utf-8')


def _is_skipped_path(relative_path: str) -> bool:
    return any(part in SKIPPED_DIRECTORY_NAMES for part in relative_path.split('/'))


def _language_for_path(relative_path: str) -> Language:
    suffix = relative_path.rsplit('.', 1)[-1] if '.' in relative_path else ''
    match suffix:
        case 'py':
            return Language.PYTHON
        case 'ts':
            return Language.TYPESCRIPT
        case 'tsx':
            return Language.TSX
        case 'js':
            return Language.JAVASCRIPT
        case 'jsx':
            return Language.JSX
        case _:
            return Language.UNKNOWN


def _index_file(
    source_bytes: bytes,
    relative_path: str,
    language: Language,
) -> tuple[list[Symbol], list[Reference]]:
    match language:
        case Language.UNKNOWN:
            return [], []
        case (
            Language.PYTHON
            | Language.TYPESCRIPT
            | Language.TSX
            | Language.JAVASCRIPT
            | Language.JSX
        ):
            parser = _tree_sitter_parser(language)
            root_node = parser.parse(source_bytes).root_node
            symbols = _symbols_for_tree(root_node, relative_path, language)
            references = _references_for_tree(root_node, relative_path, language)
            return symbols, references
        case _:
            raise AssertionError(f'Unhandled language in _index_file: {language}')


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


def _symbols_for_tree(root_node: Node, relative_path: str, language: Language) -> list[Symbol]:
    match language:
        case Language.PYTHON:
            return _python_tree_sitter_symbols(root_node, relative_path, inside_class=False)
        case Language.TYPESCRIPT | Language.TSX | Language.JAVASCRIPT | Language.JSX:
            return _javascript_tree_sitter_symbols(root_node, relative_path, language)
        case _:
            raise AssertionError(f'Unhandled language in _symbols_for_tree: {language}')


def _python_tree_sitter_symbols(
    node: Node,
    relative_path: str,
    inside_class: bool,
) -> list[Symbol]:
    symbols: list[Symbol] = []
    for child in node.children:
        match child.type:
            case 'class_definition':
                name_node = child.child_by_field_name('name')
                if name_node is not None:
                    symbols.append(
                        _node_symbol(
                            name_node, child, SymbolKind.CLASS, relative_path, Language.PYTHON
                        )
                    )
                symbols.extend(_python_tree_sitter_symbols(child, relative_path, inside_class=True))
            case 'function_definition':
                name_node = child.child_by_field_name('name')
                if name_node is not None:
                    kind = SymbolKind.METHOD if inside_class else SymbolKind.FUNCTION
                    symbols.append(
                        _node_symbol(name_node, child, kind, relative_path, Language.PYTHON)
                    )
                symbols.extend(
                    _python_tree_sitter_symbols(child, relative_path, inside_class=False)
                )
            case _:
                symbols.extend(
                    _python_tree_sitter_symbols(child, relative_path, inside_class=inside_class)
                )
    return symbols


def _javascript_tree_sitter_symbols(
    node: Node,
    relative_path: str,
    language: Language,
) -> list[Symbol]:
    symbols: list[Symbol] = []
    for child in node.children:
        symbols.extend(_javascript_symbol_for_node(child, relative_path, language))
        symbols.extend(_javascript_tree_sitter_symbols(child, relative_path, language))
    return symbols


def _javascript_symbol_for_node(
    node: Node,
    relative_path: str,
    language: Language,
) -> list[Symbol]:
    match node.type:
        case 'function_declaration':
            return _named_field_symbol(node, node, SymbolKind.FUNCTION, relative_path, language)
        case 'class_declaration':
            return _named_field_symbol(node, node, SymbolKind.CLASS, relative_path, language)
        case 'method_definition':
            return _named_field_symbol(node, node, SymbolKind.METHOD, relative_path, language)
        case 'variable_declarator':
            value_node = node.child_by_field_name('value')
            if value_node is None or value_node.type != 'arrow_function':
                return []
            return _named_field_symbol(node, node, SymbolKind.FUNCTION, relative_path, language)
        case _:
            return []


def _named_field_symbol(
    name_source: Node,
    source_node: Node,
    kind: SymbolKind,
    relative_path: str,
    language: Language,
) -> list[Symbol]:
    name_node = name_source.child_by_field_name('name')
    if name_node is None:
        return []
    return [_node_symbol(name_node, source_node, kind, relative_path, language)]


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


def _references_for_tree(
    root_node: Node, relative_path: str, language: Language
) -> list[Reference]:
    callee_positions = _call_callee_positions(root_node, language)
    identifier_types = _identifier_node_types(language)
    references: list[Reference] = []
    for node in _walk_nodes(root_node):
        if node.type not in identifier_types:
            continue
        position = (node.start_byte, node.end_byte)
        kind = ReferenceKind.CALL if position in callee_positions else ReferenceKind.MENTION
        references.append(
            Reference(
                symbol_name=node.text.decode('utf-8', errors='replace'),
                file_path=relative_path,
                line=node.start_point[0] + 1,
                kind=kind,
            )
        )
    return references


def _call_callee_positions(root_node: Node, language: Language) -> set[tuple[int, int]]:
    positions: set[tuple[int, int]] = set()
    for node in _walk_nodes(root_node):
        name_node = _callee_name_node(node, language)
        if name_node is not None:
            positions.add((name_node.start_byte, name_node.end_byte))
    return positions


def _callee_name_node(node: Node, language: Language) -> Node | None:
    match language:
        case Language.PYTHON:
            if node.type != 'call':
                return None
            function_node = node.child_by_field_name('function')
            return _python_callee_name_node(function_node)
        case _:
            if node.type != 'call_expression':
                return None
            function_node = node.child_by_field_name('function')
            return _javascript_callee_name_node(function_node)


def _python_callee_name_node(function_node: Node | None) -> Node | None:
    if function_node is None:
        return None
    match function_node.type:
        case 'identifier':
            return function_node
        case 'attribute':
            return function_node.child_by_field_name('attribute')
        case _:
            return None


def _javascript_callee_name_node(function_node: Node | None) -> Node | None:
    if function_node is None:
        return None
    match function_node.type:
        case 'identifier':
            return function_node
        case 'member_expression':
            return function_node.child_by_field_name('property')
        case _:
            return None


def _identifier_node_types(language: Language) -> frozenset[str]:
    match language:
        case Language.PYTHON:
            return frozenset({'identifier'})
        case _:
            return frozenset({'identifier', 'property_identifier'})


def _walk_nodes(node: Node) -> Iterator[Node]:
    yield node
    for child in node.children:
        yield from _walk_nodes(child)
