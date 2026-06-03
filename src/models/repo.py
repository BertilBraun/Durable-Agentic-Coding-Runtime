from __future__ import annotations

from pydantic import Field

from src.models.frozen_base_model import FrozenBaseModel
from src.runtime_enums import StrEnum

FULL_FILE_TREE_LIMIT = 30


class Language(StrEnum):
    PYTHON = 'python'
    TYPESCRIPT = 'typescript'
    JAVASCRIPT = 'javascript'
    JSX = 'jsx'
    TSX = 'tsx'
    # Outside the parseable set above (Markdown, config, etc.): tracked but not symbol-indexed.
    UNKNOWN = 'unknown'


class SymbolKind(StrEnum):
    FUNCTION = 'function'
    CLASS = 'class'
    METHOD = 'method'


class ReferenceKind(StrEnum):
    CALL = 'call'
    MENTION = 'mention'


class FileEntry(FrozenBaseModel):
    path: str
    language: Language
    size_bytes: int


class Symbol(FrozenBaseModel):
    name: str
    kind: SymbolKind
    file_path: str
    start_line: int
    end_line: int
    language: Language


class Reference(FrozenBaseModel):
    symbol_name: str
    file_path: str
    line: int
    kind: ReferenceKind


class RepoIndex(FrozenBaseModel):
    file_tree: list[FileEntry] = Field(default_factory=list)
    symbols: list[Symbol] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    overview_text: str = ''
    tracked_file_count: int = 0

    def directory_tree_text(self) -> str:
        if self.overview_text:
            return self.overview_text
        root: dict[str, dict] = {}
        paths = sorted(entry.path for entry in self.file_tree)
        if len(paths) > FULL_FILE_TREE_LIMIT:
            return _bounded_directory_tree_text(paths, FULL_FILE_TREE_LIMIT)
        for path in paths:
            cursor = root
            for part in path.split('/'):
                cursor = cursor.setdefault(part, {})
        lines: list[str] = []
        _append_tree_lines(root, 0, lines)
        return '\n'.join(lines)


def directory_overview_for_paths(paths: list[str]) -> str:
    return _bounded_directory_tree_text(sorted(paths), FULL_FILE_TREE_LIMIT)


def _bounded_directory_tree_text(paths: list[str], max_entries: int) -> str:
    root: dict[str, dict] = {}
    for path in sorted(paths):
        cursor = root
        for part in path.split('/'):
            cursor = cursor.setdefault(part, {})
    expanded_depth = 0
    lines = _tree_lines_to_depth(root, expanded_depth)
    while True:
        next_lines = _tree_lines_to_depth(root, expanded_depth + 1)
        if len(next_lines) > max_entries or next_lines == lines:
            return '\n'.join(lines)
        lines = next_lines
        expanded_depth += 1


def _tree_lines_to_depth(node: dict[str, dict], max_depth: int) -> list[str]:
    lines: list[str] = []
    _append_tree_lines_to_depth(node, 0, max_depth, lines)
    return lines


def _append_tree_lines_to_depth(
    node: dict[str, dict],
    depth: int,
    max_depth: int,
    lines: list[str],
) -> None:
    for name in sorted(node):
        lines.append(f'{"  " * depth}{name}')
        if depth < max_depth:
            _append_tree_lines_to_depth(node[name], depth + 1, max_depth, lines)


def _append_tree_lines(node: dict[str, dict], depth: int, lines: list[str]) -> None:
    for name in sorted(node):
        lines.append(f'{"  " * depth}{name}')
        _append_tree_lines(node[name], depth + 1, lines)
