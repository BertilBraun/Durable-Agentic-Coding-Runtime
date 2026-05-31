# Prompt 04 — Planner review → replan loop (+ extended thinking)

Single source of truth for this change. Small task. Keep `ruff format` / `ruff check` / `pytest -q`
green per commit (see [README.md](README.md)).

---

## 1. Why

A strong plan up front is the cheapest lever on final quality. Today the only feedback loop around
the plan is **human** (`approve_plan_or_replan` in
[src/activities/human_approval.py](../src/activities/human_approval.py)), gated by the complexity
assessor. There is no automated "review the plan, then replan with that feedback" step, and the
planner/contract-builder run without an extended thinking budget. We want the model to critique its
own plan and revise before any implementation starts.

---

## 2. What to build

### 2.1 An LLM plan reviewer

Add a `review_plan` activity, shaped like the existing patch reviewer
([src/activities/reviewer.py](../src/activities/reviewer.py)): structured verdict, not prose.

```python
class PlanReviewDecision(StrEnum):
    ACCEPT = 'accept'
    REVISE = 'revise'

class PlanReviewVerdict(FrozenBaseModel):
    decision: PlanReviewDecision
    blocking_issues: list[str] = []
    suggestions: list[str] = []
    feedback: str                 # the revision guidance fed back into build_plan
```

It judges the plan against the contract + repo overview: coverage of acceptance criteria, missing
steps, oversized/undersized steps, risky/unjustified changes, missing reproduction step for bugfix
tasks (cross-ref Prompt 03).

### 2.2 A bounded review→replan loop

In [main_workflow.py](../src/workflows/main_workflow.py), between `build_plan` and the candidate
lifecycle:

```
plan = build_plan(...)
for _ in range(max_plan_review_rounds):       # config; small, e.g. 2
    verdict = review_plan(contract, repo_overview, plan)
    if verdict.decision == ACCEPT: break
    extra_context = gather_context(... gatherer_prompt = verdict.feedback ...)   # see below
    plan = build_plan(PlanRequest(..., revision_feedback=verdict.feedback, context=extra_context))
```

- **The planner must be able to gather context between rounds.** A plan reviewer that says "you
  missed the auth callers" is useless if the planner can't then go look at them. Between review and
  replan, run `gather_context` (the same retrieval activity the implementer uses; grounded snippets
  after Prompt 02) seeded by the reviewer's feedback, and feed the result into the next `build_plan`.
  `build_plan` therefore needs to accept gathered context (it currently only gets the contract +
  repo overview + worker results). This is what makes the loop actually improve the plan rather
  than just re-prompt it.
- Add `MAX_PLAN_REVIEW_ROUNDS` to [config.py](../src/config.py) (parsed like the existing ints;
  default 1–2). Keep the loop bounded and deterministic (workflow code).
- This composes with the existing **human** approval: human-or-LLM feedback both flow into
  `build_plan`'s revision-guidance field. Consider renaming `PlanRequest.human_feedback` →
  `revision_feedback` (it's no longer human-only); update callers (`human_approval.py`, the replan
  path in `_run_plan_steps`). Optional but cleaner.
- Order vs human approval: run the LLM review→replan loop first (cheap iteration), then the human
  approval gate if `complexity_verdict.requires_human_approval`. The human sees an already
  self-reviewed plan.

### 2.3 Extended thinking for planning roles

Give `ModelRole.PLANNER` and `ModelRole.CONTRACT_BUILDER` (and `review_plan`'s role) an extended
thinking budget. Wire it through the LLM client — check how
[src/llm/client.py](../src/llm/client.py) builds requests and add a per-role thinking-budget knob
(config + plumbed into `generate_structured`). Follow the Anthropic extended-thinking conventions
(the `claude-api` skill covers the request shape and the interaction with tool use / structured
output). Add `THINKING_BUDGET_TOKENS_<ROLE>` style config with sane defaults; default others to off.

---

## 3. Tasks

1. Add `review_plan` activity + `PlanReviewVerdict` / `PlanReviewDecision`. Register it in
   [worker.py](../src/worker.py). Test: ACCEPT passes through; REVISE returns feedback.
2. Let `build_plan` accept gathered context, and run `gather_context` between review and replan
   seeded by the reviewer feedback. Test: the gathered context reaches the next `build_plan` call.
3. Add the bounded review→replan loop to `main_workflow` + `MAX_PLAN_REVIEW_ROUNDS` config. Test
   (fakes injected): a REVISE then ACCEPT yields exactly two `build_plan` calls (with a
   `gather_context` between them) and proceeds with the revised plan; usage accumulates.
4. (Optional) Rename `human_feedback` → `revision_feedback` across `PlanRequest` and callers.
5. Extended thinking: per-role budget config + client plumbing for planner / contract-builder /
   plan-reviewer. Test the request carries the budget for those roles and not others.

---

## 4. Footguns

- **Bounded, deterministic loop.** Hard cap the rounds; no convergence-until-happy. Workflow code
  stays deterministic — all LLM work is in activities.
- **Don't double-count usage.** Each `build_plan` / `review_plan` call adds to `LLMUsage`; the
  test's call-count assertions must reflect the loop.
- **Thinking budget interacts with structured output and tool use** — verify the structured-output
  path still parses with thinking enabled (the `claude-api` skill notes the gotchas); keep budgets
  modest.
- **Human approval still works.** The LLM loop augments, doesn't replace, the human gate; preserve
  the `approve_plan_or_replan` path and its signal handling.
