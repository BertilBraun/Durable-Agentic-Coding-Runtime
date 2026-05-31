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


class RepoIndex(FrozenBaseModel):
    file_tree: list[FileEntry] = Field(default_factory=list)
    symbols: list[Symbol] = Field(default_factory=list)
