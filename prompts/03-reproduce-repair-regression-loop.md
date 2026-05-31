# Prompt 03 — Reproduce → repair → regression loop (bugfix gating)

Single source of truth for this change. Small, focused task. Keep `ruff format` / `ruff check` /
`pytest -q` green per commit (see [README.md](README.md)).

---

## 1. Why

For bugfix tasks, "I fixed it" is only trustworthy if there was a test that **failed before** the
change and **passes after** it. Today the planner *prompt* nudges toward reproduction
([planner.py](../src/activities/planner.py)) and the implementation turn collects `TestResult`s
([implementation.py](../src/activities/implementation.py)), but nothing **gates** success on a real
fail→pass transition. A model can claim a fix with a test that never actually reproduced the bug, or
with no failing-first evidence at all.

This adds a structured gate for `TaskType.BUGFIX`: reproduce (test must fail on base) → repair →
regression (same test must now pass, and pre-existing tests still pass).

---

## 2. What to build

A deterministic gate, driven by evidence already flowing through the system, applied when
`contract.task_type == TaskType.BUGFIX`.

### 2.1 Evidence: capture the fail→pass transition

The repro test must be run **once before the fix** (expected: fail) and **once after** (expected:
pass). Two viable shapes — pick the simpler that fits the current flow:

- **(Recommended) Order-based within the candidate.** Require, among the candidate's collected
  `TestResult`s, a test command `C` such that an earlier run of `C` failed (`exit_code != 0`) and a
  later run of `C` passed. That *is* the reproduce→repair→regression signal, and `TestResult`
  already carries `command`, `exit_code`, `passed`, `sequence`. Add a check that derives the
  transition from the ordered results.
- **Explicit repro step.** Run the repro test against `base_sha` first (before any edits) via a
  dedicated activity, assert failure, then proceed. More explicit, more orchestration.

Either way, surface the result as a typed record, e.g. `FrozenBaseModel`:
```python
class ReproductionEvidence(FrozenBaseModel):
    repro_command: str
    failed_before: bool
    passed_after: bool
```
`reproduced = failed_before and passed_after`.

### 2.2 The gate in the workflow

In [main_workflow.py](../src/workflows/main_workflow.py), after `_run_plan_steps` and before/at the
final review, for bugfix contracts:
- Compute `ReproductionEvidence` from the aggregated `worker_results` test results.
- If not `reproduced`, the candidate is **not acceptable**: mark the final status failed/blocked
  with a clear reason (`'bug not reproduced: no failing-then-passing test'`), and feed that as
  replan feedback (mirror the existing `NEEDS_REPLAN` path) so the next attempt writes a real repro
  test. Keep iteration bounded.
- If `reproduced`, proceed to review/finalize as normal.

For non-bugfix task types, the gate is a no-op.

### 2.3 Make the requirement explicit to the model

Reinforce in the implementation system prompt (and keep the planner nudge) that bugfix work must:
land a test that **fails on the unmodified code**, then make it pass, and keep it as a regression
test. The gate enforces it; the prompt makes the model aim for it.

---

## 3. Tasks

1. Add `ReproductionEvidence` + a pure function that derives it from an ordered `list[TestResult]`
   (fail of `C` followed by later pass of `C`). Unit-test it directly: parametrize
   no-test / pass-only / fail-then-pass / fail-only.
2. Wire the gate into `main_workflow` for `TaskType.BUGFIX`: block + replan feedback when not
   reproduced; pass through otherwise. Keep replan bounded (reuse existing replan plumbing).
3. Thread `ReproductionEvidence` into `FinalReport` (so the eval/oracle and humans can see the
   transition). Update `report_builder` + any result consumers.
4. Tighten the implementation/planner prompt wording on fail-first repro.

---

## 4. Footguns

- **Gate on observed exit codes, not model claims.** Use `TestResult.exit_code`/`passed` from real
  `RunTests` runs — never the model's `diff_summary` or self-reported success.
- **"Failed before" must mean failed on the bug, not failed to collect.** A test that errored
  because of a syntax error or missing import is not a reproduction. At minimum require the *same*
  command later passing; ideally check the early failure was a test assertion failure, not a
  collection error (best-effort — don't over-engineer).
- **Keep it bounded.** The block→replan path must have a hard iteration cap (it already does via the
  plan-step loop); don't add an unbounded retry.
- **Only bugfix.** Don't impose repro gating on feature/docs/refactor tasks.
