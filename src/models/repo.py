from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.runtime_enums import StrEnum


class Language(StrEnum):
    PYTHON = 'python'
    TYPESCRIPT = 'typescript'
    JAVASCRIPT = 'javascript'
    JSX = 'jsx'
    TSX = 'tsx'
    UNKNOWN = 'unknown'  # TODO why is that here? When is that used? Markdown files? Env files etc? Supported are only py/ts/js but obviously repos contain a lot of other file types as well, so we need a way to handle those as well without breaking - TODO document better


class SymbolKind(StrEnum):
    FUNCTION = 'function'
    CLASS = 'class'
    METHOD = 'method'


class FileEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    language: Language
    size_bytes: int


class Symbol(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    kind: SymbolKind
    file_path: str
    start_line: int
    end_line: int
    language: Language


class RepoIndex(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_tree: list[FileEntry] = Field(default_factory=list)
    symbols: list[Symbol] = Field(default_factory=list)
