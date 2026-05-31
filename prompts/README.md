# Implementation Prompts

Each file here is a self-contained implementation prompt (single source of truth for that
change), written to be picked up in its own chat. They assume the workspace redesign is already
merged: one persistent environment per run, the `Workspace` polymorphic seam
(`HostWorkspace` / `DockerWorkspace`, single `run_command` boundary), sequential candidates, and
diffs computed against `base_sha`.

Suggested order (later prompts depend on earlier ones):

1. [01 — Indexer + shell tool + tool-surface rework](01-indexer-shell-tool-rework.md)
   Add a general shell tool, delete the narrow shell-wrapper tools and the grep-based
   `find_references`, and rework the indexer into a real symbol + xref (find-definition /
   find-callers) index. Folds in the two deferred cleanups (indexed `FindReferences`,
   `build_repo_index` host/docker branch).
2. [02 — Context packer rework](02-context-packer-rework.md)
   Replace the LLM-summarized `ContextPack` with typed, grounded context (file + line range +
   metadata) and stop dumping the whole repo index into prompts. Depends on (1)'s tools.
3. [03 — Reproduce → repair → regression loop](03-reproduce-repair-regression-loop.md)
   For bugfix tasks, gate success on a repro test that fails on base then passes after the fix.
4. [04 — Planner review → replan loop](04-planner-review-replan.md)
   LLM plan-review feedback loop before implementation; extended thinking for planner/contract.
5. [05 — Sequential candidates + combiner](05-sequential-candidates-combiner.md)
   Adaptive, confidence-driven candidate count (escalate only when the first attempt is shaky),
   run sequentially, then select into one final solution (combiner deferred).

## Constraints that apply to every prompt

- **Temporal-Light.** IO lives in `@activity` functions; `@workflow` code stays deterministic
  (no wall-clock, no RNG, no direct IO; bounded loops only). Only JSON-serializable values cross
  activity/child boundaries. Discriminated unions (`Field(discriminator='kind')`) restore to the
  right subclass; `get_type_hints` strips `Annotated`, so the bare union + the `kind` literal is
  what actually round-trips on replay. Activity args on the forward path are the real objects;
  results are restored via the return annotation on replay.
- **The `Workspace` seam.** `run_command(command: list[str], timeout) -> CommandResult` is the
  *only* place host vs docker may differ. New behavior must go through it; do not reach into the
  host filesystem or `docker exec` anywhere else, and do not add new `match workspace` branches.
- **Coding standards** (`CLAUDE.md` / `CODING_STANDARDS.md`): full descriptive names, type hints
  on every function, `FrozenBaseModel` for data, enums over literal strings, `match`/`case` over
  `isinstance` chains, no comments unless a non-obvious *why*, no lazy imports (import at module
  top), validate only at boundaries (`assert` for invariants, `raise ValueError` for bad input).
  Tests call production code directly; the LLM client is faked via injection
  (`tests/fakes/openai_client.py`), never a production fake-mode branch; one logical assertion per
  test; parametrize similar cases.
- **Verification before every commit:**
  ```powershell
  python -m ruff format src tests
  python -m ruff check src tests
  python -m pytest -q
  ```
  Work in small commits; keep all three green at each one.
