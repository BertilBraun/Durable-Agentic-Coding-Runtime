from __future__ import annotations

import ast
import re
from pathlib import Path

from src.activities.workspace_manager import WorkspaceInfo
from src.models.repo import FileEntry, Language, RepoIndex, Symbol, SymbolKind

SKIPPED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        "dist",
        "build",
        "vendor",
        "generated",
    }
)


async def build_repo_index(workspace_info: WorkspaceInfo) -> RepoIndex:
    workspace_path = Path(workspace_info.worktree_path)
    file_entries: list[FileEntry] = []
    symbols: list[Symbol] = []

    for file_path in sorted(workspace_path.rglob("*")):
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
        case ".py":
            return Language.PYTHON
        case ".ts":
            return Language.TYPESCRIPT
        case ".tsx":
            return Language.TSX
        case ".js":
            return Language.JAVASCRIPT
        case ".jsx":
            return Language.JSX
        case _:
            return Language.UNKNOWN


def _symbols_for_file(file_path: Path, relative_path: str, language: Language) -> list[Symbol]:
    tree_sitter_symbols = _tree_sitter_symbols_for_file(
        file_path=file_path,
        relative_path=relative_path,
        language=language,
    )
    if tree_sitter_symbols is not None:
        return tree_sitter_symbols

    match language:
        case Language.PYTHON:
            return _python_symbols(file_path=file_path, relative_path=relative_path)
        case Language.TYPESCRIPT | Language.TSX | Language.JAVASCRIPT | Language.JSX:
            return _javascript_family_symbols(
                file_path=file_path, relative_path=relative_path, language=language
            )
        case Language.UNKNOWN:
            return []


def _tree_sitter_symbols_for_file(
    file_path: Path,
    relative_path: str,
    language: Language,
) -> list[Symbol] | None:
    parser = _tree_sitter_parser(language)
    if parser is None:
        return None
    source_bytes = file_path.read_bytes()
    syntax_tree = parser.parse(source_bytes)
    symbols: list[Symbol] = []
    for node in syntax_tree.root_node.children:
        match language:
            case Language.PYTHON:
                symbols.extend(_python_tree_sitter_symbols(node, relative_path))
            case Language.TYPESCRIPT | Language.TSX | Language.JAVASCRIPT | Language.JSX:
                symbols.extend(_javascript_tree_sitter_symbols(node, relative_path, language))
            case Language.UNKNOWN:
                return []
    return symbols


def _tree_sitter_parser(language: Language) -> object | None:
    try:
        from tree_sitter import Language as TreeSitterLanguage
        from tree_sitter import Parser
    except ImportError:
        return None

    match language:
        case Language.PYTHON:
            try:
                import tree_sitter_python
            except ImportError:
                return None
            parser = Parser()
            parser.language = TreeSitterLanguage(tree_sitter_python.language())
            return parser
        case Language.TYPESCRIPT | Language.TSX | Language.JAVASCRIPT | Language.JSX:
            try:
                import tree_sitter_javascript
            except ImportError:
                return None
            parser = Parser()
            parser.language = TreeSitterLanguage(tree_sitter_javascript.language())
            return parser
        case Language.UNKNOWN:
            return None


def _python_tree_sitter_symbols(node: object, relative_path: str) -> list[Symbol]:
    node_type = node.type
    if node_type not in {"function_definition", "class_definition"}:
        return []
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return []
    kind = SymbolKind.CLASS if node_type == "class_definition" else SymbolKind.FUNCTION
    return [_node_symbol(name_node, node, kind, relative_path, Language.PYTHON)]


def _javascript_tree_sitter_symbols(
    node: object,
    relative_path: str,
    language: Language,
) -> list[Symbol]:
    match node.type:
        case "function_declaration" | "class_declaration":
            return _javascript_named_declaration_symbols(node, relative_path, language)
        case "export_statement":
            declaration_node = node.child_by_field_name("declaration")
            if declaration_node is None:
                return []
            return _javascript_tree_sitter_symbols(declaration_node, relative_path, language)
        case "lexical_declaration" | "variable_declaration":
            return _javascript_variable_symbols(node, relative_path, language)
        case _:
            return []


def _javascript_named_declaration_symbols(
    node: object,
    relative_path: str,
    language: Language,
) -> list[Symbol]:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return []
    kind = SymbolKind.CLASS if node.type == "class_declaration" else SymbolKind.FUNCTION
    return [_node_symbol(name_node, node, kind, relative_path, language)]


def _javascript_variable_symbols(
    node: object,
    relative_path: str,
    language: Language,
) -> list[Symbol]:
    symbols: list[Symbol] = []
    for child in node.children:
        if child.type != "variable_declarator":
            continue
        value_node = child.child_by_field_name("value")
        if value_node is None or value_node.type != "arrow_function":
            continue
        name_node = child.child_by_field_name("name")
        if name_node is None:
            continue
        symbols.append(_node_symbol(name_node, child, SymbolKind.FUNCTION, relative_path, language))
    return symbols


def _node_symbol(
    name_node: object,
    source_node: object,
    kind: SymbolKind,
    relative_path: str,
    language: Language,
) -> Symbol:
    return Symbol(
        name=name_node.text.decode("utf-8", errors="replace"),
        kind=kind,
        file_path=relative_path,
        start_line=source_node.start_point[0] + 1,
        end_line=source_node.end_point[0] + 1,
        language=language,
    )


def _python_symbols(file_path: Path, relative_path: str) -> list[Symbol]:
    source = file_path.read_text(encoding="utf-8", errors="ignore")
    syntax_tree = ast.parse(source)
    symbols: list[Symbol] = []
    for node in syntax_tree.body:
        match node:
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                symbols.append(
                    Symbol(
                        name=node.name,
                        kind=SymbolKind.FUNCTION,
                        file_path=relative_path,
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        language=Language.PYTHON,
                    )
                )
            case ast.ClassDef():
                symbols.append(
                    Symbol(
                        name=node.name,
                        kind=SymbolKind.CLASS,
                        file_path=relative_path,
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        language=Language.PYTHON,
                    )
                )
    return symbols


def _javascript_family_symbols(
    file_path: Path, relative_path: str, language: Language
) -> list[Symbol]:
    source_lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    symbols: list[Symbol] = []
    for line_number, line in enumerate(source_lines, start=1):
        stripped_line = line.strip()
        function_match = re.match(
            r"^(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)",
            stripped_line,
        )
        arrow_match = re.match(
            r"^(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(",
            stripped_line,
        )
        if function_match is not None:
            name = function_match.group(1)
            symbols.append(
                _source_symbol(name, SymbolKind.FUNCTION, relative_path, line_number, language)
            )
        if arrow_match is not None:
            name = arrow_match.group(1)
            symbols.append(
                _source_symbol(name, SymbolKind.FUNCTION, relative_path, line_number, language)
            )
        class_match = re.match(r"^(?:export\s+)?class\s+([A-Za-z_$][\w$]*)", stripped_line)
        if class_match is not None:
            name = class_match.group(1)
            symbols.append(
                _source_symbol(name, SymbolKind.CLASS, relative_path, line_number, language)
            )
    return symbols


def _source_symbol(
    name: str,
    kind: SymbolKind,
    relative_path: str,
    line_number: int,
    language: Language,
) -> Symbol:
    return Symbol(
        name=name,
        kind=kind,
        file_path=relative_path,
        start_line=line_number,
        end_line=line_number,
        language=language,
    )
