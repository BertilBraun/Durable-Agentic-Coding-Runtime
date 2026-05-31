# Prompt 02 — Context packer rework

This document is the single source of truth for this change. It is more **investigate-then-design**
than the others: produce a concrete design and a small implementation plan, surface the one open
fork for sign-off, then implement. Keep `ruff format` / `ruff check` / `pytest -q` green per commit
(see [README.md](README.md)). **Depends on Prompt 01** (the shell + index tools context is pulled
with).

---

## 1. Why

Context is assembled in two questionable ways today:

1. **The whole repo index is dumped into prompts.** `planner.py` and `context_gatherer.py` inject
   `repo_index.model_dump_json()` verbatim. Fine for a toy repo, useless and expensive at scale.
   (Prompt 01 trims this to a compact overview; this prompt decides the real assembly strategy.)

2. **`ContextPack` is LLM-summarized free text.** `gather_context`
   ([src/activities/context_gatherer.py](../src/activities/context_gatherer.py)) runs a cheap-model
   multi-turn loop and returns a `ContextPack`
   ([src/models/context.py](../src/models/context.py)) whose fields are ungrounded prose:
   `relevant_snippets`, `recent_observations`, `failed_attempt_summaries`. A model rewriting code
   into prose is exactly where hallucination enters — the implementer then trusts snippets that may
   not match the file. There's also a deterministic fallback path
   (`_deterministic_context_pack`) that already hints the prose is unreliable under budget pressure.

PLAN.md §22.5 deferred item 1 already states the intended direction: *split retrieval (typed,
grounded snippets from the cheap model) from packing (a deterministic assembler with a bounded
observation window and artifact references); redefine `ContextPack` to drop ungrounded fields.* This
prompt executes that.

---

## 2. Investigate first

Before changing anything, write down (in the PR description or a scratch note) the answers to:

- Every consumer of `ContextPack` and which fields each actually reads. Grep `ContextPack`,
  `context_pack`, `relevant_snippets`, `recent_observations`, `failed_attempt_summaries`,
  `budget_remaining`. (Today: built in `gather_context`, consumed in `run_implementation_turn`'s
  user payload, and the deterministic fallback.)
- Where the cheap-model retrieval loop adds value vs where it just paraphrases. The retrieval *act*
  (deciding which files/symbols to read via the tools) is valuable; the *packing* (turning reads
  into prose) is the suspect part.
- How big the dumped index actually gets on a mid-size repo (sanity-check the cost claim).

---

## 3. Proposed design (validate, then build)

**Split retrieval from packing.**

- **Retrieval** stays a cheap-model agentic loop, but it emits **typed, grounded references**, not
  prose. Define a grounded snippet type (`FrozenBaseModel`):
  ```python
  class ContextSnippet(FrozenBaseModel):
      file_path: str
      start_line: int
      end_line: int
      reason: str          # why this is relevant — the only free-text field, and it's about, not a copy of, the code
  ```
  The cheap model chooses *what* to include (file + line range + one-line reason) using the Prompt
  01 tools (`FindDefinition`, `FindCallers`, search via `RunShell`/`rg`, `RunTests`); it does not
  paraphrase code.

- **Packing** is deterministic: an assembler that, given the snippet references, reads the exact
  lines from the workspace (via `run_command`) and assembles a bounded context block (head/tail or
  budgeted line count, with artifact references for anything over budget — reuse the
  `ArtifactReference` mechanism). No model call, no rewriting; the code in the pack is the code in
  the file.

- **Redefine `ContextPack`** to drop the ungrounded fields. Likely shape: `task_summary` (short,
  model-written, *about* the task — acceptable), `snippets: list[ContextSnippet]`,
  `artifact_references: list[ArtifactReference]`, `budget_remaining`. Drop `relevant_snippets`
  (prose), `recent_observations`, `failed_attempt_summaries` unless a consumer truly needs them
  (carry failures as structured records if so).

**Pull, don't dump.** With grounded retrieval + the Prompt 01 tools, the planner and implementer
should *pull* context on demand rather than receiving a giant up-front blob. The planner gets the
compact overview (from Prompt 01) plus the ability to call `gather_context`; the implementer already
calls `gather_context` per step. Make sure nothing re-introduces an unbounded dump.

---

## 4. The open fork (sign-off before building)

> **Cheap-LLM retrieval loop vs no-LLM deterministic retrieval.**
> - **Keep the cheap LLM** to *select* snippets (it's good at "which 6 places matter for this
>   task"), but make its output strictly typed/grounded (above). Recommended.
> - **Drop the LLM entirely**: deterministic retrieval from the index + task keywords (callers of
>   named symbols, files touched by the contract's `affected_areas`, etc.). Cheaper and fully
>   reproducible, but blunter selection.
>
> Recommendation: keep the cheap LLM for selection, deterministic for packing. Flag for the user;
> this is the one decision that changes the build.

---

## 5. Tasks (after sign-off)

1. Add `ContextSnippet`; redefine `ContextPack` (drop ungrounded fields). Update the model + every
   consumer; fix tests.
2. Change the gatherer's structured output to emit `ContextSnippet`s (typed) instead of prose.
3. Add the deterministic packer: read exact lines via `run_command`, assemble within budget, spill
   overflow to `ArtifactReference`s. No LLM call in the packer.
4. Repoint planner/implementer to the compact overview + on-demand `gather_context`; remove any
   remaining unbounded index/context dumps.
5. Update `context_gatherer` budget/stop logic to the new shape; keep the bounded-loop + hard-stop
   behavior, but the "summarize into prose" finalize step becomes "emit collected snippet refs."

---

## 6. Footguns

- **Grounded means grounded.** The packer must read the real current lines from the workspace at
  pack time, not echo what the retrieval model said was there. If they disagree, the file wins.
- **Line ranges drift after edits.** Snippets captured pre-edit may point at shifted lines later.
  Either re-resolve at pack time or accept staleness within a single gather call (don't cache
  snippets across edits).
- **Don't lose the budget discipline.** The existing context-utilization stop/hard-stop thresholds
  exist for a reason; preserve bounded behavior under the new shape.
- Keep the LLM client faked via injection in tests; assert on the *typed* snippet structure, not on
  prose.
