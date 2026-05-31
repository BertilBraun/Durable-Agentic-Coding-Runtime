from __future__ import annotations

from pydantic import Field

from src.models.frozen_base_model import FrozenBaseModel
from src.runtime_enums import StrEnum


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

    def directory_tree_text(self) -> str:
        root: dict[str, dict] = {}
        for path in sorted(entry.path for entry in self.file_tree):
            cursor = root
            for part in path.split('/'):
                cursor = cursor.setdefault(part, {})
        lines: list[str] = []
        _append_tree_lines(root, 0, lines)
        return '\n'.join(lines)


def _append_tree_lines(node: dict[str, dict], depth: int, lines: list[str]) -> None:
    for name in sorted(node):
        lines.append(f'{"  " * depth}{name}')
        _append_tree_lines(node[name], depth + 1, lines)
