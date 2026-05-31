# Prompt 03 — Reproduce → repair → regression loop (bugfix gating)

Single source of truth for this change. Keep `ruff format` / `ruff check` / `pytest -q` green per
commit (see [README.md](README.md)). **Depends on Prompts 01 and 02** (the shell + index tools and
the grounded context pack), both already merged into master.

---

## 1. Why

For bugfix tasks, "I fixed it" is only trustworthy if a test **failed before** the change and
**passes after** it. Today the planner *prompt* nudges toward reproduction
([planner.py](../src/activities/planner.py)) and the implementation turn collects `TestResult`s
([implementation.py](../src/activities/implementation.py)), but nothing **gates** success on a real
fail→pass transition, and those `TestResult`s are an unreliable basis for gating: they are run
against a half-mutated working tree, are pruned by `_select_reported_test_results`, and are
model-steered. A model can claim a fix with a test that never reproduced the bug.

The fix is to make reproduction a **structured phase the orchestrator owns**, and to gate on
**exit codes the orchestrator observes itself** — not on the implementation worker's self-reported
evidence. Implementation `TestResult`s stay purely informational (a steering signal during the tool
loop); the gate gets its own measurement and its own typed storage.

---

## 2. The model

A dedicated reproduction phase runs up front for `TaskType.BUGFIX`, then a deterministic gate runs
the reproduced command after every plan step and as a mandatory final check.

```text
contract  →  setup_environment  →  build_repo_index  →  begin_candidate
          →  (BUGFIX only) reproduce_bug                 ← writes a failing regression test
          →  build_plan (grounded in the repro traceback)
          →  run plan steps, re-running the repro command after each (deterministic gate)
          →  mandatory final gate (repro command green + pre-existing suite green)
          →  review  →  finalize
```

Key insight: **"failed before" is observed directly at reproduction time.** When the reproduction
agent has written the test but no fix exists yet, the tree is `base + test`; the orchestrator runs
the command and *sees it fail* — that failure **is** the clean reproduction. We never reconstruct a
before-state by reverting source. The ongoing gate then only has to confirm the command goes green
on the fixed tree and that pre-existing tests still pass.

### 2.1 The reproduction agent (new role)

Add `ModelRole.REPRODUCER` and a bounded agentic activity `reproduce_bug` (mirror
`run_implementation_turn`: a turn loop over `generate_structured` + `run_tool`, bounded by a config
cap). The reproducer **explores and edits** — it gets the full implementation tool surface
(`run_shell`, `find_definition` / `find_callers` / `find_callees`, `gather_context`, `write_file` /
`apply_patch`, `run_tests`); reuse `ImplementationToolCall` rather than inventing a narrower union.
It runs on the candidate workspace (so `begin_candidate` moves above it), with the contract +
directory tree as context.

Its job, in its own system prompt: locate the bug, write a **single focused regression test that
fails on the current (unfixed) code**, and return the exact command that runs it. It must **not**
fix the bug. Output a typed result:

```python
class ReproductionResult(FrozenBaseModel):
    status: ReproductionStatus        # reproduced | could_not_reproduce
    repro_command: str                # exact RunTests-style command for the new test
    test_files: list[str]             # files the agent added/modified to hold the test
    failure_evidence: str             # the observed failing output (for planner context + the report)
```

**The orchestrator verifies, the model does not assert.** After the turn, `reproduce_bug` runs
`repro_command` itself (via `run_tool` with `RunTests`) and requires `exit_code != 0`. If the
command does not actually fail on the current tree, reproduction has failed regardless of what the
model said — return `could_not_reproduce`. A genuine "can't reproduce" (vague contract, flaky /
environmental bug) is an honest terminal/blocked outcome, **not** a reason to proceed to a fix.

### 2.2 Separate, typed gating evidence

The gate owns its own record — not `WorkerResult.test_results`:

```python
class ReproductionEvidence(FrozenBaseModel):
    repro_command: str
    failed_before: bool      # observed in reproduce_bug (test present, fix absent)
    passed_after: bool       # observed by the final gate on the fixed tree
    before_exit_code: int
    after_exit_code: int
```

`reproduced = failed_before and passed_after`. `failed_before` / `before_exit_code` come from the
reproduction phase; `passed_after` / `after_exit_code` come from the gate.

### 2.3 The gate in `main_workflow`

[main_workflow.py](../src/workflows/main_workflow.py), `TaskType.BUGFIX` only (no-op otherwise):

- Move `begin_candidate` above the bugfix reproduction so the agent has a writable tree.
- Run `reproduce_bug`; if `could_not_reproduce`, block with a clear reason and stop (don't plan a
  fix for a bug you can't demonstrate).
- Thread `repro_command` + `failure_evidence` into `PlanRequest` so the plan is grounded in the real
  traceback, and flip the planner nudge into a fact: *a failing regression test already exists
  (command X); plan the fix that makes it pass without weakening or deleting it.*
- **After each plan step**, run `repro_command` through a deterministic `run_tool(RunTests(...))`
  call and record the exit code. This gives early-stop the moment it is green and attributes which
  step fixed (or regressed) it.
- **Mandatory final gate** before review: `repro_command` must be green **and** the pre-existing
  test suite must still pass (regression). If the repro command is not green, the candidate is not
  acceptable — feed `'bug not reproduced: regression test still failing'` (with the latest output)
  back through the existing `NEEDS_REPLAN` plumbing so the next attempt fixes it.
- Build `ReproductionEvidence` and thread it into `FinalReport`
  ([report_builder.py](../src/activities/report_builder.py)); update result consumers.

### 2.4 Bound the loop

The `NEEDS_REPLAN` path in `_run_plan_steps` has **no iteration cap today** — it repopulates
`pending_plan_steps` from a fresh plan and can loop indefinitely. Since the gate now feeds that path,
add a hard replan cap (config-driven) and surface a clear blocked result when it is hit.

---

## 3. Tasks

1. Add `ReproductionStatus`, `ReproductionResult`, `ReproductionEvidence` (`FrozenBaseModel`s), and a
   pure helper that builds `ReproductionEvidence` from the two observed exit codes. Unit-test the
   helper directly (parametrize fail→pass / pass-only / fail-only / no-after).
2. Add `ModelRole.REPRODUCER` (config binding + a capable default model, e.g. the implementation
   model) and the `reproduce_bug` agentic activity with its own system prompt, reusing the
   implementation tool surface. It must self-verify the command fails before returning `reproduced`.
   Consider mirroring `implementation_workflow` as a `reproduction_workflow` child for symmetry with
   the existing spawn-child pattern; inline is acceptable if it keeps determinism.
3. Wire `main_workflow` for `TaskType.BUGFIX`: move `begin_candidate` up, run reproduction, thread
   evidence into planning, run the per-step + final gate, block+replan when not green, no-op for
   other task types. Add the replan cap.
4. Thread `ReproductionEvidence` into `FinalReport` and update consumers.
5. Tighten the planner + implementation prompts: a failing regression test already exists; make it
   pass, keep it, do not weaken or delete it.

---

## 4. Footguns

- **Gate on observed exit codes, not model claims.** Use exit codes from `RunTests` runs the
  *orchestrator* triggers — never the worker's `diff_summary`, self-reported success, or the
  implementation turn's pruned `test_results`.
- **"Failed before" must mean failed on the bug.** The reproduction agent's command must fail on
  test-present/fix-absent code. A failure from a syntax/import/collection error is not a
  reproduction; at minimum require the *same* command later passing, and best-effort prefer an
  assertion failure over a collection error (don't over-engineer).
- **Reproduction can legitimately fail.** `could_not_reproduce` is an honest terminal outcome, not a
  bug to route around.
- **Don't weaken the test.** The implementation worker must not delete or neuter the regression test
  to make the gate pass; the prompt forbids it and the final suite run is a backstop.
- **Keep it bounded.** Add the missing hard cap on the `NEEDS_REPLAN` loop now that the gate drives
  it.
- **Only bugfix.** No repro gating on feature / refactor / test / docs / frontend tasks.
- **Multi-candidate (forward-looking).** Under Prompt 05 the reproduced test is the shared oracle
  across candidates; design `reproduce_bug` so its test + command could be seeded into each
  candidate later, but don't build that here.
