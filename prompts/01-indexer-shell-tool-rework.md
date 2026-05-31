# Prompt 01 — Indexer + shell tool + tool-surface rework

This document is the single source of truth for this change. Read it fully, then implement in small
commits, keeping `ruff format`, `ruff check`, and `pytest -q` green before each (see the repo-wide
constraints in [README.md](README.md)).

---

## 1. Why

Two problems with today's tool surface and index:

1. **Too many narrow tools that are just shell wrappers.** `command_for_tool` in
   [src/tools/handlers.py](../src/tools/handlers.py) turns most tools into a one-line `sh -lc`
   command: `read_file_range` → `sed`, `search_text` → `rg`, `write_file` → `base64 -d`,
   `git_diff` / `git_status` / `git_commit`, `run_lint` → `ruff`, `run_typecheck` → `mypy`. Each is
   a predefined single shell command. A single general **shell tool** subsumes all of them, and the
   LLM is perfectly capable of composing those commands itself given a description of the
   environment.

2. **The index isn't an index, it's a flat symbol table + grep.** `build_repo_index`
   ([src/activities/repo_indexer.py](../src/activities/repo_indexer.py)) extracts symbols
   (name/kind/file/line) with tree-sitter. The only consumers that genuinely need it are:
   - `find_symbol` → `_find_indexed_symbol` (exact name+language lookup), and
   - `find_references` → `_find_indexed_references`, which **just word-greps the host files** — the
     same thing `rg -w` does, but host-only and slower (see `_indexed_tool_result` in
     [src/activities/workspace_manager.py](../src/activities/workspace_manager.py)).

   The whole point of an index is the stuff that is *hard in shell*: resolve a symbol to its
   definition, and find its **callers** (and callees). grep can't tell a call from a comment or a
   string. That capability is the index's reason to exist; the grep-flavored `find_references` is
   not.

This change makes the shell tool the catch-all for "run a command," deletes the redundant tools, and
turns the index into a real symbol + cross-reference index exposed through a small set of
genuinely-useful tools.

This also folds in the two cleanups deferred from the workspace redesign:
- the grep-based indexed `find_references` (deleted here, not patched), and
- the host/docker branch in `build_repo_index` (`extract_docker_repo_snapshot` tar path) — removed
  by building the index through `run_command` instead of the host filesystem.

---

## 2. What to build

### 2.1 The shell tool

Add a `RunShell` tool to [src/tools/definitions.py](../src/tools/definitions.py):

```python
class RunShell(ToolBase):
    tool_name: Literal[ToolName.RUN_SHELL] = Field(default=ToolName.RUN_SHELL, ...)
    command: str = Field(description='Shell command to run from the repository root.')
    timeout_seconds: int = Field(description='Maximum seconds to allow the command to run.')
```

- Add `RUN_SHELL = 'run_shell'` to `ToolName`.
- The handler must run the command through the **workspace's native shell**, which is the *one*
  place platform/mode matters. Put the shell selection on the `Workspace` subclasses (a small
  method like `shell_invocation(command: str) -> list[str]`), not a `match` in `handlers.py`:
  - `DockerWorkspace` → `['sh', '-lc', command]` (image is Linux).
  - `HostWorkspace` → **detect the host platform** and emit the right shell: POSIX →
    `['sh', '-lc', command]`; native Windows → `['powershell', '-NoProfile', '-Command', command]`.
    **Native Windows host is in scope** — this is exactly why the shell tool carries an environment
    descriptor. Detect via `sys.platform` / `os.name` at command-build time on the host (the
    `HostWorkspace` runs locally, so it can inspect the real platform).
- Provide the LLM an **environment descriptor** so it can write correct commands for the actual
  shell: a short generated string stating the **OS and shell** (PowerShell vs POSIX `sh`), **how to
  do multi-line commands** in that shell, and **quoting/escaping** rules. This is mandatory, not
  optional — it's how the model writes valid commands on Windows vs Linux. Surface it in the
  implementation + context-gatherer system prompts (or as a field in the tool's user payload).
  Derive it from the concrete `Workspace` (a `describe_environment() -> str` method).

`RunShell` is a `RunTests`-shaped tool: it carries `timeout_seconds`, so `_tool_timeout_seconds`
must return it for `RunShell` too.

### 2.2 Delete the redundant tools

Remove these tools entirely — `RunShell` subsumes them: `read_file_range`, `search_text`,
`git_diff`, `git_status`, `git_commit`, `run_typecheck`, and the grep-based `find_references`.
Remove their `ToolName` members, their classes, their `command_for_tool` cases, their
path-validation cases, and any references in the tool-call type aliases (`ImplementationToolCall`,
`ContextGathererToolCall`).

**Keep these** (they carry semantics a raw shell call would lose):
- `RunTests` — owns the timeout and feeds the `TestResult` evidence chain in
  [src/activities/implementation.py](../src/activities/implementation.py). Keep it distinct so test
  evidence stays structured.
- `RunLint` — keep it as a **dedicated** tool so the project's lint rules are always applied and
  cannot be skipped or quietly altered by a hand-written shell command. It is the canonical
  "lint with our rules" affordance, not just a convenience wrapper.
- `WriteFile` and `ApplyPatch` — structured editing is safer and more reviewable than shell
  heredocs/`base64`; keep them.
- `GatherContext` — intercepted before `run_tool`; unchanged here (its retrieval is reworked in
  Prompt 02).

> If you disagree on the keep-set, that's the one real judgment call — state your reasoning in the
> PR. Default to the set above.

`command_for_tool` shrinks to: `WriteFile`, `ApplyPatch`, `RunTests`, `RunLint`, `RunShell`, and the
index tools (which never reach `command_for_tool`). Re-confirm `_validate_tool_paths` still guards
the relative-path tools that remain (`WriteFile`, `RunLint`).

### 2.3 The reworked index

Replace the symbol-only `RepoIndex` with a real symbol + cross-reference index.

Model ([src/models/repo.py](../src/models/repo.py)) — extend, keep `FrozenBaseModel`:
- Keep `Symbol` (definition: name, kind, file, line range, language).
- Add a **reference / call-site** record: `Reference(symbol_name, file_path, line, kind)` where
  `kind ∈ {call, mention}` (a `ReferenceKind` enum). A *call* is a call expression whose callee
  resolves to `symbol_name`; a *mention* is any other identifier use. Callers of `X` = references
  to `X` with `kind=call`.
- `RepoIndex` gains `references: list[Reference]`. `file_tree` stays.

Extraction — extend the tree-sitter pass in `repo_indexer.py`:
- For Python: capture `call` nodes (`function` child → callee identifier / attribute), and
  identifier references; for JS/TS: `call_expression` (`function` field). This is **name-keyed,
  not scope-precise** — it ignores comments/strings and distinguishes calls from mentions, which is
  the big win over grep, but it does not resolve overloads/shadowing. That is an acceptable first
  cut.

> **Design fork (decide and record):** heuristic tree-sitter call-site capture (above — language-
> agnostic-ish, cheap, imprecise) **vs** a real per-language analyzer (Python `ast`+`symtable` or
> jedi; TS `tsserver`) for scope-accurate xref. Recommend shipping the tree-sitter version first
> behind the new tools, and leaving a note for a precise analyzer later. Do not build the precise
> analyzer in this prompt unless it's cheap.

### 2.4 New index-backed tools

Replace `find_symbol` / `find_references` with tools that reflect what the index is *for*:

- `FindDefinition(name, language)` → definition site(s) from `RepoIndex.symbols`. (This is today's
  `find_symbol`; rename for clarity or keep the name — your call, just be consistent.)
- `FindCallers(symbol_name)` → reference records with `kind=call` for that symbol, formatted
  `path:line: <call site>`.
- Optionally `FindCallees(file_path, symbol_name)` → calls made *inside* a given definition's line
  range. Add only if the extraction makes it easy.

These are served entirely from the in-memory `RepoIndex` (no filesystem, no `run_command`), so they
work identically in HOST and DOCKER — **no `match workspace` branch**. Serve them in
`_indexed_tool_result`; `command_for_tool` should `raise AssertionError` for them (as `find_symbol`
does today) since they must come from the index.

### 2.5 Build the index through `run_command` (removes the docker branch)

Rework `build_repo_index` so it does **not** read the host filesystem directly:
- Enumerate files via `workspace.run_command(['git', 'ls-files'])` (respects `.gitignore`, works in
  both modes), and read file contents via `run_command` (e.g. `git show :path` or `cat`), then run
  tree-sitter on the bytes in-process.
- Delete `extract_docker_repo_snapshot` and the `match workspace` in `build_repo_index`. The index
  build becomes mode-agnostic: one path, driven by `run_command`.
- Watch cost: this is N small commands. Batching (one command that emits a tar/stream to stdout, or
  `git ls-files -z` + a single archive read) is fine — keep it simple first, optimize if the smoke
  run is slow.

### 2.6 Stop dumping the whole index into prompts (here, the index side)

Today [planner.py](../src/activities/planner.py) and
[context_gatherer.py](../src/activities/context_gatherer.py) inject the entire
`repo_index.model_dump_json()` into the prompt — fine for a toy repo, useless and expensive for a
real one. In this prompt, **replace the full-index dump with at most a file tree — ideally just a
directory/folder tree**. Nothing more: no symbols, no references, no per-file detail in the prompt.
Everything beyond the tree is pulled on demand through the tools (`FindDefinition`, `FindCallers`,
`RunShell`/`rg`). The deeper "how context is assembled" rework is Prompt 02 — cross-reference it;
here, just collapse the blob to the tree.

---

## 3. Tasks (suggested commit order)

1. Add `RunShell` + `ToolName.RUN_SHELL`; add `Workspace.shell_invocation` / `describe_environment`;
   wire `RunShell` through `command_for_tool` + `_tool_timeout_seconds`. Tests: a `RunShell` round
   trips and is dispatched via `run_command` with the right timeout.
2. Delete the redundant tools (§2.2): remove classes, `ToolName` members, `command_for_tool` cases,
   path-validation cases, alias membership. Update prompts that name those tools. Fix/replace tests
   that referenced them.
3. Index model + extraction (§2.3): add `Reference`/`ReferenceKind`, extend the tree-sitter pass.
   Tests: a call site is captured as `kind=call`, a comment mention is not a call (parametrize
   Python + TS).
4. New tools (§2.4): `FindDefinition` / `FindCallers` (+ optional `FindCallees`), served from the
   index in `_indexed_tool_result`. Tests: callers found, comment/string occurrences excluded.
5. `build_repo_index` via `run_command` (§2.5); delete `extract_docker_repo_snapshot` and the
   `match workspace` branch. Update repo-indexer tests to drive a real `HostWorkspace` on a temp git
   repo (they already do something close).
6. Trim the index blob in planner/context-gatherer prompts (§2.6).

---

## 4. Footguns

- **`git ls-files` needs a git repo and a committed/at-least-tracked tree.** Setup already asserts a
  clean tree; new untracked files won't be listed until added. For base-`sha` indexing that's fine.
- **Tree-sitter call capture is name-keyed.** Don't claim scope accuracy. `FindCallers('foo')`
  returns call sites whose callee identifier is `foo`, across the repo — good enough, not a compiler.
- **Don't reintroduce a host/docker branch.** Everything new goes through `run_command` or is served
  from the in-memory index. If you're writing `match workspace`, stop and reconsider.
- **Windows host shell is in scope.** `shell_invocation` must detect the platform and emit
  PowerShell on Windows, POSIX `sh` elsewhere, and the environment descriptor must tell the model
  which one it's talking to (multi-line + quoting differ sharply between PowerShell and `sh`). Test
  both shells' invocation construction.
- **Test evidence depends on `RunTests` staying structured.** Don't fold `RunTests` into `RunShell`
  or the `TestResult` extraction in `implementation.py` loses its signal.

---

## 5. Out of scope

- A precise per-language semantic analyzer (note it for later).
- Incremental / on-demand reindex after edits — the index is a base-`sha` snapshot; revisit only if
  a consumer demonstrably needs post-edit freshness.
- The full context-assembly redesign (Prompt 02).

Note: native-Windows host support is **in scope** for the shell tool (see §2.1), not deferred.
