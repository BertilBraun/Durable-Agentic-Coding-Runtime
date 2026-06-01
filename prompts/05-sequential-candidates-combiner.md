# Prompt 05 — Sequential candidates + combiner

Single source of truth for this change. Keep `ruff format` / `ruff check` / `pytest -q` green per
commit (see [README.md](README.md)).

---

## 1. Why

The workspace redesign built the candidate lifecycle but the workflow runs exactly **one** candidate
(`begin_candidate(workspace, 0)` in [main_workflow.py](../src/workflows/main_workflow.py)). The
`Workspace` base already supports the full shape: `begin_candidate(k)` (branch `agentic/{run_id}/
cand-{k}` off `base_sha`), `reset_to_base()` (between candidates), `diff_against_base()`, and
`finalize_to_base(winner_branch, ...)`. This prompt runs **K candidates sequentially** and then
**selects/combines** them into one final solution.

Sequential, not parallel: two candidates cannot share one in-place HOST repo or one DOCKER
container, and the redesign deliberately kept a single environment per run. Each candidate runs to
completion, then we reset to base before the next.

**Candidate base = post-reproduction snapshot, not raw `base_sha`.** The plan (and, for bugfix
tasks, the reproduction) is per-run: setup → reproduce → plan → review happen once, then we branch a
candidate from the resulting repository state. This matters because the reproduction phase *writes a
regression test into the working tree* (`reproduce_bug` adds it with `write_file` / `apply_patch` —
see [reproduction.py](../src/activities/reproduction.py)), and `reset_to_base()`'s `git clean -fd`
would wipe that test before candidate 1. So after reproduction, commit the working tree to a per-run
**candidate base** ref `C`; `begin_candidate(k)` branches off `C` and `reset_to_base(...)` resets to
`C` between candidates (for non-bugfix runs nothing was written, so `C == base_sha` and behavior is
identical to today). `reproduction_context` is immutable metadata — pass it into each candidate as a
value. Keep `diff_against_base()` / `finalize_to_base()` measuring against the original `base_sha`,
**not** `C`: the candidate patch deliberately includes the regression test, which is a valid
regression test worth shipping alongside the fix.

---

## 2. What to build

### 2.1 Adaptive, confidence-driven candidate count

Candidates are **not** a fixed count — the number is decided adaptively by how confident we are in
the first accepted implementation. Run one deep candidate first; only fan out if it's shaky. This
keeps the common case cheap and reserves extra candidates for genuinely uncertain tasks.

Logic in `main_workflow` (sequential throughout):

```text
candidates = [run_candidate(0)]            # always run the first candidate to completion + review
target_count = candidate_count_for_confidence(candidates[0])
while len(candidates) < target_count:
    reset_to_base(...)                     # between candidates only
    candidates.append(run_candidate(len(candidates)))
final = select(candidates)                 # trivial when len == 1
finalize_winner(workspace, final.branch)
```

`candidate_count_for_confidence(first_candidate)` maps the first candidate's confidence to a target:
- **high** → `1` (keep it; we're done — no extra candidates)
- **medium** → `2`
- **low** → `4`

(Defaults — make them config, see below.) Derive the confidence from the signal we already have: the
candidate's overall `Confidence` (`src/models/worker.py`) combined with the reviewer verdict (an
accepted, high-confidence worker = high; a `REVISE`/low-confidence worker escalates). Define one
small pure function `candidate_count_for_confidence(...)` so the mapping is testable in isolation;
don't scatter the thresholds.

Where a candidate run is: `begin_candidate(k)` → `_run_plan_steps(...)` → `get_full_diff` →
`review_patch`, captured as a typed `CandidateResult`.

- Add a `reset_to_base(workspace)` activity wrapping `workspace.reset_to_base()` (mirrors
  `begin_candidate` / `finalize_winner`); register it in [worker.py](../src/worker.py).
- Capture the **candidate base** once, after reproduction and before the candidate loop: commit the
  post-reproduction working tree to a per-run ref and key `begin_candidate(k)` / `reset_to_base(...)`
  to that ref instead of raw `base_sha` (a small `Workspace` seam tweak — e.g. a
  `snapshot_candidate_base(workspace)` activity that returns the ref, or a base-ref argument threaded
  through). With no reproduction the snapshot equals `base_sha`. `diff_against_base` /
  `finalize_to_base` stay against the original `base_sha` so the regression test ships in the patch.
- Add confidence→count config to [config.py](../src/config.py), e.g.
  `CANDIDATE_COUNT_MEDIUM_CONFIDENCE` (default `2`) and `CANDIDATE_COUNT_LOW_CONFIDENCE` (default
  `4`); high is always `1`. A global hard cap (the low value) bounds the loop.
- Define a typed `CandidateResult` (`FrozenBaseModel`): index, branch, diff, worker_results,
  review verdict, confidence, and the test/repro evidence already aggregated.
- **High confidence on the first candidate must behave exactly as today** (one candidate, trivial
  single-element selection, then finalize) — this is the overwhelmingly common path; don't regress
  its cost.

Note on "complex tasks prefer one candidate": we deliberately start with a single deep attempt
rather than fanning out up front, and only escalate on low confidence. So a complex-but-confidently-
solved task stays at one candidate; a shaky task escalates regardless of complexity. If you also
want the complexity assessor to influence the initial attempt, keep that a separate later knob —
the confidence-driven escalation above is the mechanism for this prompt.

### 2.2 Selector / combiner

After the loop, choose the final solution from `candidates`:

- **Selector (recommended first cut):** an activity that picks the best candidate by a deterministic
  preference over evidence — reviewer `ACCEPT` over `REVISE`, more passing tests, fewer blocking
  issues, smaller diff as a tiebreak — optionally with an LLM judge for ties. The winner's branch
  goes to `finalize_winner(workspace, winner_branch)`. Branch reset between candidates means only
  the winner's branch needs to survive to finalize; candidate branches are kept (cleanup flag
  default off) as an audit trail.
- **Combiner (later):** an LLM that synthesizes one solution from multiple candidate diffs. This is
  materially harder (overlapping/conflicting diffs, needs its own branch built off `base_sha`, then
  validated like a candidate) and is **out of scope for this prompt** — ship the selector now and
  record the combiner as explicit follow-up. Per the coding standards (no scattered TODO/PR-ref
  comments), record it as a deferred milestone entry in `PLAN.md` (and, if anywhere in code, a
  single one-line note at the selector explaining *why* it's selection-not-combination for now —
  not a bare `TODO`). If a combiner is built later, the combined solution must be re-validated
  (tests, and the Prompt 03 bugfix gate) before finalize — never finalize an unverified merged diff.

### 2.3 Finalize

`finalize_winner(workspace, winner_branch)` already returns to the base branch and applies the
winner as uncommitted edits. For the selector, `winner_branch` is the chosen candidate's branch. For
a combiner, finalize the combined branch instead.

---

## 3. Tasks

1. Add `reset_to_base` activity + register it; add the candidate-base snapshot after reproduction
   (key `begin_candidate` / `reset_to_base` to it, `== base_sha` when no repro ran); add the
   confidence→count config; add `CandidateResult`; add the pure `candidate_count_for_confidence(...)`.
   Test the mapping directly (parametrize high→1, medium→2, low→4).
2. Convert the single-candidate section of `main_workflow` into the adaptive escalation loop, with
   `reset_to_base` between candidates. Test (fakes injected): a high-confidence first candidate runs
   exactly one candidate and reproduces current behavior; a low-confidence first candidate escalates
   to the configured count, resets between candidates, and finalizes the selected branch.
3. Add the selector activity (deterministic preference; optional LLM tiebreak). Test the preference
   ordering directly (parametrize: accept-vs-revise, more-passing-tests, smaller-diff tiebreak).
4. Thread the chosen `CandidateResult` into `FinalReport`; keep `llm_usage` summed across all
   candidates + selection. Record the combiner follow-up in `PLAN.md`.

---

## 4. Footguns

- **System-state bleed between sequential candidates on HOST** (footgun #2 from the redesign):
  `reset_to_base` does `git reset --hard` + `git clean -fd`, which restores the *tree* but not
  global state — a `pip install` from candidate 0 persists into candidate 1. Accept on HOST; on
  DOCKER, a clean slate means recreating the container between candidates (a `reset_to_base`
  override on `DockerWorkspace`, or recreate-on-reset). Decide and document; don't pretend HOST
  candidates are isolated.
- **Reset to the post-reproduction candidate base, not raw `base_sha`.** The reproducer writes a
  regression test into the working tree; a `reset_to_base` keyed to `base_sha` runs `git clean -fd`
  and deletes that test for every candidate after the first. Capture the candidate base once after
  reproduction (§2.1) and branch/reset relative to it. Keep `diff_against_base` against the original
  `base_sha` so the test ships in the patch.
- **Never finalize an unvalidated combined diff.** A merged solution must pass the same gates a
  candidate does.
- **Determinism + usage.** The loop is bounded and deterministic; sum `LLMUsage` across every
  candidate and the selector. Test assertions must account for K candidates.
- **`reset_to_base` only between candidates**, never after the last one (the last winner's branch
  feeds finalize). Off-by-one here silently discards the winning work.
- **Detached HEAD / base branch** handling already lives in `finalize_to_base`; don't duplicate it.
