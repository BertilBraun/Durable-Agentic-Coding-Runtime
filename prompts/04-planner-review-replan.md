# Prompt 04 — Planner review → replan loop (+ extended thinking)

Single source of truth for this change. Keep `ruff format` / `ruff check` / `pytest -q` green per
commit (see [README.md](README.md)). **Depends on Prompts 01–03**, all merged into master — the
shell + index tools, the grounded context pack (`gather_context` → `ContextPack` of
`PackedSnippet`s), and the reproduction phase + `ReproductionContext` that `build_plan` already
threads for bugfix tasks.

---

## 1. Why

A strong plan up front is the cheapest lever on final quality: every implementation step, the bugfix
gate, and the final review all inherit the plan's blind spots. Today the only feedback loop around
the plan is **human** — `approve_plan_or_replan`
([src/activities/human_approval.py](../src/activities/human_approval.py)), reached only when the
complexity assessor flags the task. There is no automated "critique the plan, then replan with that
critique" step, so an ordinary task whose plan misses an acceptance criterion or a whole caller set
goes straight into implementation with that gap baked in. Separately, the planner and
contract-builder run with **no extended-thinking budget**, even though they are exactly the roles
where more deliberation pays off.

This change adds a bounded, model-driven **review → replan** loop before any implementation starts,
and gives the planning roles a thinking budget.

---

## 2. The model

The new loop sits between `build_plan` and the complexity/approval gate, and — critically — lets the
planner **gather new context** in response to the review before it replans.

```text
contract  →  setup_environment  →  build_repo_index  →  begin_candidate
          →  (BUGFIX only) reproduce_bug
          →  build_plan
          →  ┌─► review_plan ──ACCEPT──────────────────────────┐     (new, ≤ MAX_PLAN_REVIEW_ROUNDS)
             │      │ REVISE                                    │
             │      ▼                                           │
             │   gather_context(gatherer_prompt = feedback)     │
             │      ▼                                           │
             └─── build_plan(revision_feedback + context) ◄─────┘
          →  assess_complexity  →  (if flagged) approve_plan_or_replan
          →  run plan steps (+ bugfix gate)  →  review  →  finalize
```

Key insight: **a plan critique is only useful if the planner can act on it.** A reviewer that says
"you missed the auth callers" changes nothing unless the planner then *looks at* those callers
before replanning. So each REVISE round runs `gather_context` (the same grounded retrieval the
implementer uses, post-Prompt 02) seeded by the reviewer's feedback, and feeds the resulting
`ContextPack` into the next `build_plan`. That is what makes the loop *improve* the plan instead of
just re-prompting it.

The loop is **advisory and bounded**: it hard-caps at `MAX_PLAN_REVIEW_ROUNDS`, and on reaching the
cap it proceeds with the latest plan (the downstream complexity/human gate and the implementation +
bugfix gates remain the real backstops — a plan review never *blocks*).

### 2.1 The plan reviewer (new role)

Add a `review_plan` function shaped exactly like the patch reviewer
([src/activities/reviewer.py](../src/activities/reviewer.py)): a plain `async def` that calls
`generate_structured` and returns `(verdict, usage)`. **It is not an `@activity` and is not
registered in `worker.py`** — `review_patch` isn't either; the only registered LLM activity is
`generate_structured_completion`, which `generate_structured` calls under the hood and which
auto-registers the output type.

```python
class PlanReviewDecision(StrEnum):
    ACCEPT = 'accept'
    REVISE = 'revise'

class PlanReviewVerdict(FrozenBaseModel):
    decision: PlanReviewDecision
    blocking_issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    feedback: str                 # the revision guidance fed verbatim into the next build_plan

class PlanReviewRequest(FrozenBaseModel):
    contract: TaskContract
    plan: Plan
    repo_index: RepoIndex                       # use repo_index.directory_tree_text() for the overview
    reproduction: ReproductionContext | None = None
```

Add `ModelRole.PLAN_REVIEWER` with a config binding (a capable default, e.g. the planner's model).
Its system prompt judges the plan against the contract + repo overview:

- coverage of every acceptance criterion; no missing or duplicated steps;
- step sizing (the planner targets ~5–10 min steps — flag oversized/undersized);
- risky or unjustified changes, and changes that stray outside `affected_areas`;
- **for bugfix tasks** (when `reproduction` is present): the plan must make the existing failing
  regression test (`reproduction.repro_command`) pass *without weakening, skipping, or deleting it*,
  and must not plan a separate reproduction step (cross-ref Prompt 03).

ACCEPT means "good enough to implement," not "perfect" — keep `feedback` empty on ACCEPT.

### 2.2 The bounded review → replan loop

In [main_workflow.py](../src/workflows/main_workflow.py), insert the loop **after** `build_plan` and
**before** `assess_complexity` (so the human, if reached, reviews an already self-revised plan). The
loop is deterministic workflow code; all LLM work stays in the called functions.

```python
plan, plan_usage = await build_plan(PlanRequest(..., reproduction=reproduction_context))
usage += plan_usage
for _ in range(CONFIG.max_plan_review_rounds):
    verdict, review_usage = await review_plan(
        PlanReviewRequest(contract=contract, plan=plan, repo_index=repo_index,
                          reproduction=reproduction_context),
    )
    usage += review_usage
    if verdict.decision == PlanReviewDecision.ACCEPT:
        break
    context_pack, gather_usage = await gather_context(
        ContextGatherRequest(workspace_info=candidate_workspace, repo_index=repo_index,
                             gatherer_prompt=verdict.feedback),
    )
    usage += gather_usage
    plan, replan_usage = await build_plan(
        PlanRequest(contract=contract, repo_index=repo_index, worker_results=[],
                    revision_feedback=verdict.feedback, reproduction=reproduction_context,
                    context=context_pack),
    )
    usage += replan_usage
```

`build_plan` must learn to accept gathered context. Today `PlanRequest` carries `contract`,
`repo_index`, `worker_results`, `human_feedback`, and `reproduction`; the user message stitches the
contract, repo tree, worker results, and reproduction guidance. Add an optional
`context: ContextPack | None = None` and render its `PackedSnippet`s into the message (reuse the
snippet formatting style the implementation payload already uses), so the revised plan is grounded
in the lines the reviewer pointed at.

Add `MAX_PLAN_REVIEW_ROUNDS` to [config.py](../src/config.py) (parsed like the other ints; default
`2`). Composes with human approval: the LLM loop runs first (cheap iteration), then
`assess_complexity` → `approve_plan_or_replan` if flagged — unchanged.

### 2.3 Rename `human_feedback` → `revision_feedback`

The revision-guidance field on `PlanRequest` is no longer human-only — it now also carries the plan
reviewer's `feedback`, and (post-Prompt 03) the worker's `replan_suggestion` and the bugfix gate's
failure text already flow through it. Rename `PlanRequest.human_feedback` → `revision_feedback` and
update the local variable in `build_plan` plus all call sites: `approve_plan_or_replan`
([human_approval.py](../src/activities/human_approval.py)), and the replan paths in `_run_plan_steps`
/ `_replan` ([main_workflow.py](../src/workflows/main_workflow.py)). Mechanical but do it in its own
commit so the diff is reviewable.

### 2.4 Extended thinking for the planning roles

Give `ModelRole.CONTRACT_BUILDER`, `ModelRole.PLANNER`, and `ModelRole.PLAN_REVIEWER` an
extended-thinking budget; default every other role to **off**.

Wire a per-role budget through the LLM client ([src/llm/client.py](../src/llm/client.py)). The path
today is `generate_structured(role, …)` → `@activity generate_structured_completion(messages,
output_type_name, model, context_limit_tokens)` → `LLMClient().generate_structured(… model,
context_limit_tokens)` → `async_openai_client.beta.chat.completions.parse(...)`. Add a
`thinking_budget_tokens: int` that:

- is resolved in `generate` / `generate_structured` from `CONFIG.thinking_budget_for_role(role)`
  (a new `dict[ModelRole, int]` on `Settings`, loaded from `THINKING_BUDGET_TOKENS_<ROLE>` env vars,
  `0` = off), and passed as a plain int across the activity boundary (JSON-serializable — fine);
- is threaded into both `generate_completion` and `generate_structured_completion` (new keyword,
  default `0`) and into the `LLMClient` methods;
- when `> 0` **and** the model id starts with `claude-`, attaches Anthropic extended thinking to the
  request. **Use the `claude-api` skill** for the exact request shape and the gotchas: thinking is
  `{"type": "enabled", "budget_tokens": N}`, it requires `max_tokens > budget_tokens`, temperature
  must be left at default, and it interacts with the structured-output/tool-use path used by
  `.parse(...)` — verify parsing still returns valid JSON with thinking on. Non-`claude-` models
  ignore the budget (leave off).

Keep budgets modest (e.g. 4–8k tokens). Thinking tokens are billed as output — the existing cost
estimator already prices output tokens, so usage accounting needs no special case beyond whatever
the endpoint reports.

---

## 3. Tasks

1. Add `PlanReviewDecision` / `PlanReviewVerdict` / `PlanReviewRequest` and the `review_plan`
   function (mirror `review_patch`; **not** an `@activity`, **not** registered in `worker.py`), plus
   `ModelRole.PLAN_REVIEWER` + its config binding. Test: ACCEPT passes through with empty feedback;
   REVISE returns the feedback string.
2. Extend `PlanRequest`/`build_plan` to accept an optional `context: ContextPack` and render its
   snippets into the planner message. Test: snippets from a supplied `ContextPack` appear in the
   generated prompt.
3. Add the bounded loop + `MAX_PLAN_REVIEW_ROUNDS` to `main_workflow`. Test (fakes injected): a
   REVISE-then-ACCEPT run makes exactly two `build_plan` calls with one `gather_context` between
   them, proceeds with the revised plan, and accumulates usage across review + gather + replan; a
   run that REVISEs every round stops at the cap and proceeds with the latest plan.
4. Rename `human_feedback` → `revision_feedback` across `PlanRequest` and all callers (own commit).
5. Extended thinking: per-role budget config + client plumbing for contract-builder, planner, and
   plan-reviewer. Test: the request built for those roles carries the thinking budget and a request
   for an off-by-default role (e.g. reviewer/summarizer) does not.

---

## 4. Footguns

- **Bounded, deterministic loop.** Hard-cap the rounds; never "loop until the reviewer is happy."
  Workflow code stays deterministic — all LLM work lives in the called async functions/activities.
- **A plan review never blocks.** On cap-out, proceed with the latest plan; do not emit a blocked
  result. The complexity/human gate and the implementation + bugfix gates are the real backstops.
- **The loop must change the inputs, not just retry.** If you skip `gather_context` (or feed an
  empty pack), `build_plan` sees the same inputs and tends to emit the same plan. The grounded
  context seeded by the reviewer's feedback is the point.
- **Don't double-count usage.** Each `review_plan`, `gather_context`, and `build_plan` call adds to
  `LLMUsage`; the loop's call-count assertions must reflect every iteration.
- **Thinking interacts with structured output.** The structured path goes through `.parse(...)`
  (tool-use under the hood). Verify it still parses with thinking enabled (see the `claude-api`
  skill), respect `max_tokens > budget_tokens`, and leave temperature default. Keep budgets modest.
- **Human approval still works.** The LLM loop augments, never replaces, `approve_plan_or_replan`;
  preserve its signal handling and its own replan path.
- **Bugfix grounding carries through.** Keep threading `ReproductionContext` into both `build_plan`
  and `review_plan` so the reviewer can enforce "make the repro test pass without weakening it."
