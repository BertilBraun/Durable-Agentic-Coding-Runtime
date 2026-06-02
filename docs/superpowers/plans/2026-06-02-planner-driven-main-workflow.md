# Planner-Driven Main Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework `main_workflow` into a planner-driven loop where context gathering is explicit, future steps are regenerated from compact state after each completed step, and implementers receive concrete, relevant files and step-specific context.

**Architecture:** Keep the LLM planner flexible, but stop treating the workflow as a static plan executor. The main workflow will maintain normalized planning state: contract, reproduction evidence, gathered context notes, completed step history, current diff/test/reproduction facts, and the latest future steps. Each iteration asks the planner either for ephemeral context requests or for future implementation steps, runs one step, records one compact history entry, refreshes evidence, then replans.

**Tech Stack:** Python, Pydantic models, Temporal-Light workflows/activities, existing LLM structured completion client, existing workspace/tool/context activities.

---

## Why This Change

The current workflow can churn on broad replans and repeated worker prose. In the SWE-bench Astropy runs, one localized issue produced dozens of worker results, many `needs_replan` outcomes, repeated "success" claims, and a final reproduction gate failure. The root problem is not only model quality: the workflow gives the planner too much stale unstructured history and gives implementers too little concrete context.

This rework keeps the planner as the flexible coordinator, but changes what the planner sees and produces:

- Context requests are ephemeral. The next planner call receives fetched context notes, not the previous context-request message.
- Completed implementation steps are compressed into one history entry each.
- The planner outputs only future steps. Completed steps are immutable history and are not repeated in the prompt.
- Each implementation step carries its own relevant files, context summary, required changes, tests, and out-of-scope constraints.
- Test evidence in history is compact: keep only the last passing test run, or all failing test runs since the last passing run.
- Main workflow code is split into named subfunctions so the loop remains readable.

---

## Target File Map

### Modify `src/models/plan.py`

Responsibility: structured planner output and planner-visible state models.

Add or replace these models:

- `ContextRequest`: what the planner wants inspected, with reason, queries, and optional relevant files.
- `ContextNote`: compact evidence produced from context gathering.
- `StepHistoryEntry`: compact result of one accepted implementation step attempt.
- `PlanningEvidence`: current objective facts such as diff summary, selected test history, reproduction result.
- `PlannerState`: normalized state passed into the planner.
- `PlannerTurn`: planner output containing context requests, future steps, and done signal.
- Expanded `PlanStep`: per-step relevant files, context summary, required changes, out-of-scope constraints, and tests.

Keep `Plan` temporarily for compatibility if needed, but new code should consume `PlannerTurn` and future `PlanStep` values.

### Modify `src/activities/planner.py`

Responsibility: planning one turn from normalized state.

Add `plan_next_turn(PlannerState) -> tuple[PlannerTurn, LLMUsage]`.

Keep `build_plan` temporarily for tests or older workflows, but `main_workflow` should move to `plan_next_turn`.

Update the planner prompt to:

- coordinate the workflow, not implement code;
- request more context before planning if relevant files/functions are not known;
- output context requests or future steps, not both unless the steps are already fully supported by existing context;
- output only future steps, never completed steps;
- make every step concrete enough for an implementer to execute without rediscovering global context;
- avoid splitting into artificial inspect/create-test/run-test steps;
- permit very concrete test-first steps when useful;
- preserve existing tests and avoid broad rewrites;
- use reproduction evidence as a central fact for bugfixes.

### Modify `src/workflows/main_workflow.py`

Responsibility: orchestration loop and final report assembly.

Replace `_run_plan_steps` static-plan execution with a planner-driven loop:

1. Bootstrap contract, workspace, repo index, reproduction context, and initial context.
2. Build `PlannerState`.
3. Call `plan_next_turn`.
4. If context requests exist, gather context, append `ContextNote`, and call planner again.
5. If no future steps and `done=True`, exit loop.
6. Otherwise run exactly the first future step with step-local candidates.
7. Convert the selected step result into one `StepHistoryEntry`.
8. Refresh objective evidence: full diff summary, selected test history, reproduction command result.
9. Repeat until done or caps are reached.
10. Run final review and final reproduction/test checks.

Extract named helpers so `main_workflow` is readable:

- `_bootstrap_workflow_state`
- `_run_planner_loop`
- `_run_context_requests`
- `_context_note_from_pack`
- `_run_next_step`
- `_record_step_history`
- `_refresh_planning_evidence`
- `_select_relevant_test_history`
- `_build_final_report`
- `_finalize_or_block`

Reuse existing candidate helpers where possible:

- `_run_step_with_candidates`
- `_run_step_candidate`
- `_select_step_candidate`
- `_step_candidate_target_count`

Remove or retire static-plan helpers after tests migrate:

- `_run_plan_steps`
- `_replan`
- broad `_gate_failure_feedback` replanning behavior

### Modify `src/activities/implementation.py`

Responsibility: execute one concrete step using provided context.

Update `ImplementationRequest` and prompt handling so each worker receives:

- `PlanStep.context_summary`
- `PlanStep.relevant_files`
- `PlanStep.required_changes`
- `PlanStep.out_of_scope`
- `PlanStep.tests_to_run`
- prior accepted step summaries only, not full planner transcript

Prompt changes:

- The worker executes exactly this one step.
- It must preserve prior accepted changes.
- It should inspect provided relevant files first if needed.
- It must not broaden scope or delete existing tests unless the step explicitly says so.
- It returns observations useful to the planner, not a new plan.

### Modify `src/models/worker.py`

Responsibility: step result structure.

Add fields to `WorkerResult`:

- `observations: list[str]`
- `planner_notes: list[str]`
- `files_changed: list[str]`

Keep existing fields for compatibility:

- `diff_summary`
- `tests_run`
- `test_results`
- `discovered_issues`
- `replan_suggestion`
- `confidence`
- `status`

The planner loop should prefer `observations` and `planner_notes` over free-form `replan_suggestion`.

### Modify `src/activities/context_gatherer.py`

Responsibility: gather context from planner context requests.

Add a small wrapper activity or helper:

- input: `ContextRequest`
- output: `ContextNote`

Implementation should call existing `gather_context`, then summarize the packed snippets into a `ContextNote`.

The planner should never receive the original `ContextRequest` again after it has been fulfilled. It receives only the `ContextNote`.

### Modify `src/activities/reviewer.py`

Responsibility: final review only, not main routing.

Keep review flexible, but final acceptance must receive current objective evidence:

- final diff
- compact step history
- selected test history
- reproduction evidence
- final integration test result

Reviewer prompt should explicitly distinguish:

- workflow produced a patch;
- reproduction passed;
- tests passed;
- patch is acceptable.

### Modify `src/activities/report_builder.py` and `src/eval/swe_bench.py`

Responsibility: sidecar clarity.

Expose separate fields:

- `workflow_status`
- `agent_verdict`
- `reproduction_passed`
- `official_prediction_emitted`

This prevents a sidecar from saying only `"completed"` when the internal final verdict is `revise` or reproduction is still failing.

### Modify Tests

Add or update:

- `tests/test_planner.py`
- `tests/test_main_workflow.py`
- `tests/test_implementation_turn.py`
- `tests/test_implementation_workflow.py`
- `tests/test_context_gatherer.py`
- `tests/test_swe_bench.py`
- `tests/test_prompts.py`

---

## Task 1: Add Planner State Models

**Files:**

- Modify: `src/models/plan.py`
- Test: `tests/test_planner.py`
- Test: `tests/test_main_workflow.py`

- [ ] **Step 1: Add new model classes in `src/models/plan.py`**

Add these models while keeping existing `Plan` and `PlanContext` during migration:

```python
class ContextRequest(FrozenBaseModel):
    id: str = Field(description='Stable id for this context request.')
    reason: str = Field(description='Why this context is needed before planning implementation work.')
    queries: list[str] = Field(
        default_factory=list,
        description='Concrete read-only questions or searches the context gatherer should answer.',
    )
    relevant_files: list[str] = Field(
        default_factory=list,
        description='Files the planner already suspects are relevant.',
    )


class ContextNote(FrozenBaseModel):
    id: str = Field(description='Stable id tying this note to gathered evidence.')
    summary: str = Field(description='Concise evidence summary produced after gathering context.')
    relevant_files: list[str] = Field(
        default_factory=list,
        description='Files supported by the gathered evidence.',
    )
    snippets: list[PackedSnippet] = Field(
        default_factory=list,
        description='Packed code snippets that support this note.',
    )


class StepHistoryEntry(FrozenBaseModel):
    step_id: str
    outcome: WorkerStatus
    confidence: Confidence
    summary: str
    files_changed: list[str] = Field(default_factory=list)
    tests: list[TestResult] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    planner_notes: list[str] = Field(default_factory=list)


class PlanningEvidence(FrozenBaseModel):
    diff_summary: str | None = None
    selected_test_results: list[TestResult] = Field(default_factory=list)
    reproduction_command: str | None = None
    reproduction_passed: bool | None = None
    reproduction_stdout_summary: str | None = None
    reproduction_stderr_summary: str | None = None


class PlannerState(FrozenBaseModel):
    contract: TaskContract
    repo_index: RepoIndex
    reproduction: ReproductionContext | None = None
    context_notes: list[ContextNote] = Field(default_factory=list)
    completed_steps: list[StepHistoryEntry] = Field(default_factory=list)
    previous_future_steps: list[PlanStep] = Field(default_factory=list)
    evidence: PlanningEvidence = Field(default_factory=PlanningEvidence)


class PlannerTurn(FrozenBaseModel):
    context_requests: list[ContextRequest] = Field(default_factory=list)
    future_steps: list[PlanStep] = Field(default_factory=list)
    done: bool = False
    done_reason: str | None = None
```

Resolve imports carefully to avoid circular imports. If `TaskContract`, `RepoIndex`, `WorkerStatus`, `Confidence`, or `TestResult` create a circular dependency, move these new models into `src/models/planning.py` and import them from there.

- [ ] **Step 2: Expand `PlanStep`**

Add fields:

```python
context_summary: str = Field(
    default='',
    description='Step-specific code and domain context the implementer should rely on.',
)
required_changes: list[str] = Field(
    default_factory=list,
    description='Concrete changes this step must make or verify.',
)
out_of_scope: list[str] = Field(
    default_factory=list,
    description='Explicit work the implementer must not do in this step.',
)
```

Keep `target_files` for this refactor and update its description to mean the files the implementer should inspect or modify for the step.

- [ ] **Step 3: Add model serialization tests**

Add a test that constructs a `PlannerState` with one `ContextNote`, one completed step, and one future step, then calls `model_dump(mode='json')`.

Run:

```powershell
uv run pytest tests/test_planner.py -q
```

Expected: planner model tests pass.

---

## Task 2: Add Planner Turn Activity

**Files:**

- Modify: `src/activities/planner.py`
- Test: `tests/test_planner.py`
- Test: `tests/test_prompts.py`

- [ ] **Step 1: Add `plan_next_turn`**

Add:

```python
async def plan_next_turn(state: PlannerState) -> tuple[PlannerTurn, LLMUsage]:
    completion = await generate_structured(
        role=ModelRole.PLANNER,
        messages=[
            Message(role='system', content=PLANNER_TURN_SYSTEM_PROMPT, cacheable=True),
            Message(role='user', content=_render_planner_state(state)),
        ],
        output_type=PlannerTurn,
    )
    return completion.output, completion.usage
```

Keep `build_plan` intact until `main_workflow` tests are migrated.

- [ ] **Step 2: Add `_render_planner_state`**

Render only normalized state:

- contract JSON;
- reproduction command and observed failure;
- repository tree;
- context notes;
- completed step history;
- previous future step summary;
- current evidence.

Do not render:

- prior planner raw output;
- context request text after it has been fulfilled;
- implementation child prompts;
- full worker result lists.

- [ ] **Step 3: Replace planner prompt**

Add `PLANNER_TURN_SYSTEM_PROMPT` with these rules:

```text
You are the workflow coordinator. You do not edit code.
You receive normalized state, not a chat transcript.
If relevant files/functions are not known well enough to create concrete implementation steps, request context instead of guessing.
If you request context, make each request concrete and do not output implementation steps that depend on missing context.
When enough context exists, output only future steps. Never repeat completed steps.
Each future step must be independently executable and must include target files, step-specific context, required changes, tests to run, expected result, and out-of-scope constraints.
Prefer one substantial concrete step over artificial inspect/create-test/run-test splits.
Very concrete test-first steps are allowed when useful, but the step must say exactly what behavior, file, and failure mode is expected.
For bugfixes, use reproduction evidence as the central source of truth.
Do not ask workers to weaken, skip, delete, or rewrite existing tests to get green.
Set done=true only when no implementation work remains and current evidence supports final verification.
```

- [ ] **Step 4: Add prompt tests**

Assert the prompt contains:

- `normalized state`;
- `request context instead of guessing`;
- `output only future steps`;
- `Never repeat completed steps`;
- `target files`;
- `out-of-scope`.

Run:

```powershell
uv run pytest tests/test_planner.py tests/test_prompts.py -q
```

Expected: planner tests pass.

---

## Task 3: Add Context Request Fulfillment

**Files:**

- Modify: `src/workflows/main_workflow.py`
- Modify: `src/activities/context_gatherer.py` if a helper belongs there
- Test: `tests/test_main_workflow.py`
- Test: `tests/test_context_gatherer.py`

- [ ] **Step 1: Add `_run_context_requests` helper**

In `main_workflow.py`, add:

```python
async def _run_context_requests(
    workspace_info: Workspace,
    repo_index: RepoIndex,
    requests: list[ContextRequest],
) -> tuple[list[ContextNote], LLMUsage]:
    usage = LLMUsage()
    notes: list[ContextNote] = []
    for request in requests:
        prompt = _context_prompt_from_request(request)
        context_pack, gather_usage = await gather_context(
            ContextGatherRequest(
                workspace_info=workspace_info,
                repo_index=repo_index,
                gatherer_prompt=prompt,
            )
        )
        usage += gather_usage
        notes.append(_context_note_from_pack(request, context_pack))
    return notes, usage
```

- [ ] **Step 2: Add `_context_prompt_from_request`**

Prompt should include the request reason, queries, and suspected files. It should not include prior planner output.

- [ ] **Step 3: Add `_context_note_from_pack`**

Convert a `ContextPack` into `ContextNote`:

- `id` from request id;
- `summary` from `context_pack.task_summary`;
- `relevant_files` from unique snippet file paths plus request relevant files;
- `snippets` from `context_pack.snippets`.

- [ ] **Step 4: Add tests**

Test that a planner context request:

- calls `gather_context`;
- appends a `ContextNote`;
- next planner call receives the note, not the original context request payload.

Run:

```powershell
uv run pytest tests/test_main_workflow.py tests/test_context_gatherer.py -q
```

Expected: context request tests pass.

---

## Task 4: Implement Planner Loop State Machine

**Files:**

- Modify: `src/workflows/main_workflow.py`
- Test: `tests/test_main_workflow.py`

- [ ] **Step 1: Introduce loop state dataclass**

Add:

```python
@dataclass
class PlannerLoopState:
    workspace_info: Workspace
    planner_state: PlannerState
    latest_future_steps: list[PlanStep]
    worker_results: list[WorkerResult]
    usage: LLMUsage
```

- [ ] **Step 2: Extract bootstrap helper**

Create:

```python
async def _bootstrap_workflow_state(
    task_request: TaskRequest,
    run_id: str,
) -> tuple[TaskContract, Workspace, RepoIndex, ReproductionContext | None, LLMUsage]:
    ...
```

Move contract, workspace setup, repo index, and reproduction setup into this helper.

- [ ] **Step 3: Add `_run_planner_loop`**

Shape:

```python
async def _run_planner_loop(
    workspace_info: Workspace,
    contract: TaskContract,
    repo_index: RepoIndex,
    reproduction: ReproductionContext | None,
) -> PlanExecutionResult:
    state = _initial_planner_loop_state(...)
    for _ in range(CONFIG.max_planner_turns):
        turn, turn_usage = await plan_next_turn(state.planner_state)
        state.usage += turn_usage
        if turn.context_requests:
            notes, context_usage = await _run_context_requests(...)
            state.usage += context_usage
            state.planner_state = _with_context_notes(state.planner_state, notes)
            continue
        if turn.done or not turn.future_steps:
            break
        state.latest_future_steps = turn.future_steps
        state = await _run_next_step(state)
        state = await _refresh_planning_evidence(state)
    return _plan_execution_result_from_state(state)
```

- [ ] **Step 4: Add planner turn cap**

Add config field if absent:

```python
max_planner_turns: int = 25
```

When cap is hit, append a blocked `WorkerResult` with `diff_summary='Planner turn cap reached before done.'`.

- [ ] **Step 5: Replace call site**

In `main_workflow`, replace static `build_plan`, `_review_and_revise_plan`, complexity approval for full plan, and `_run_plan_steps` with `_run_planner_loop`.

Keep human approval disabled or scoped to high-risk steps during this migration. If preserving approval is required, apply it to each `PlanStep.requires_human_approval` before `_run_next_step`.

- [ ] **Step 6: Add tests**

Add tests for:

- context request turn followed by implementation turn;
- planner outputs two future steps, workflow runs only first, then replans;
- completed step is represented as history in the next planner state;
- previous planner raw output is not included in next planner call;
- planner turn cap produces blocked result.

Run:

```powershell
uv run pytest tests/test_main_workflow.py -q
```

Expected: main workflow loop tests pass.

---

## Task 5: Update Step Candidate Execution

**Files:**

- Modify: `src/workflows/main_workflow.py`
- Modify: `src/activities/implementation.py`
- Test: `tests/test_main_workflow.py`
- Test: `tests/test_implementation_turn.py`
- Test: `tests/test_implementation_workflow.py`

- [ ] **Step 1: Keep step-local candidates**

Retain:

- `_run_step_with_candidates`
- `_run_step_candidate`
- `_select_step_candidate`

Ensure they run candidates only for the current first future step.

- [ ] **Step 2: Pass richer step context to implementer**

Update `_run_implementation_child` payload to include expanded `PlanStep` fields:

- `context_summary`;
- `required_changes`;
- `out_of_scope`;
- `target_files`;
- `tests_to_run`;
- prior accepted step summaries.

- [ ] **Step 3: Update implementation prompt**

Add prompt requirements:

```text
You execute exactly one planner-selected step.
The planner already gathered relevant context; use the provided target files and context summary first.
Preserve accepted prior steps.
Do not broaden scope.
Do not delete or rewrite existing tests unless the step explicitly requires it.
Return observations and planner_notes that help the outer planner decide the next future step.
```

- [ ] **Step 4: Add implementation request tests**

Test that `generate_structured` receives a user payload containing:

- `context_summary`;
- `required_changes`;
- `out_of_scope`;
- `target_files`;
- `completed_step_summaries`.

Run:

```powershell
uv run pytest tests/test_implementation_turn.py tests/test_implementation_workflow.py -q
```

Expected: implementation tests pass.

---

## Task 6: Compact Step History and Test Evidence

**Files:**

- Modify: `src/workflows/main_workflow.py`
- Modify: `src/models/worker.py`
- Test: `tests/test_main_workflow.py`
- Test: `tests/test_selector.py` if confidence aggregation changes

- [ ] **Step 1: Add `_record_step_history`**

Convert selected `StepCandidateRun` into exactly one `StepHistoryEntry`.

Fields:

- selected step id;
- final worker status;
- confidence;
- final diff summary;
- changed files if available;
- selected tests from `_select_relevant_test_history`;
- observations;
- planner notes.

- [ ] **Step 2: Add `_select_relevant_test_history`**

Rule requested by user:

- If there is a passing test result, keep only the last passing test result.
- Also keep failing test results that happened after that last passing result.
- If there is no passing test result, keep all failing test results.

Implementation:

```python
def _select_relevant_test_history(results: list[TestResult]) -> list[TestResult]:
    last_pass_index = None
    for index, result in enumerate(results):
        if result.passed:
            last_pass_index = index
    if last_pass_index is None:
        return [result for result in results if not result.passed]
    return [
        result
        for index, result in enumerate(results)
        if index == last_pass_index or index > last_pass_index
    ]
```

- [ ] **Step 3: Add tests for test-history selection**

Cases:

- all failing -> all failures kept;
- fail, pass -> only pass kept;
- fail, pass, fail -> pass and trailing fail kept;
- pass, pass -> only last pass kept.

Run:

```powershell
uv run pytest tests/test_main_workflow.py -q
```

Expected: history compaction tests pass.

---

## Task 7: Refresh Objective Evidence After Each Step

**Files:**

- Modify: `src/workflows/main_workflow.py`
- Test: `tests/test_main_workflow.py`

- [ ] **Step 1: Add `_refresh_planning_evidence`**

Gather:

- current full diff or a short diff summary;
- selected test history from current step;
- reproduction command result if reproduction exists.

Use existing:

- `get_full_diff`;
- `_run_repro_command`.

Do not run full integration tests after every step unless the planner step explicitly requests them. Full integration tests remain final evidence or step-specific evidence.

- [ ] **Step 2: Add reproduction state to planner evidence**

If reproduction exists, store:

- command;
- `passed`;
- stdout summary;
- stderr summary.

The planner receives this every turn and decides future work.

- [ ] **Step 3: Add tests**

Test that after a step:

- reproduction result is run once;
- planner state includes `reproduction_passed`;
- failed reproduction does not directly route to a hard-coded corrective branch;
- planner receives the failed evidence and emits the next step.

Run:

```powershell
uv run pytest tests/test_main_workflow.py -q
```

Expected: evidence refresh tests pass.

---

## Task 8: Final Acceptance and Report Semantics

**Files:**

- Modify: `src/workflows/main_workflow.py`
- Modify: `src/activities/report_builder.py`
- Modify: `src/eval/swe_bench.py`
- Test: `tests/test_main_workflow.py`
- Test: `tests/test_swe_bench.py`

- [ ] **Step 1: Add final evidence check helper**

Add:

```python
async def _finalize_or_block(
    workspace: Workspace,
    repo_index: RepoIndex,
    reproduction: ReproductionContext | None,
    integration_tests: list[str],
) -> tuple[ReproductionEvidence | None, list[TestResult], WorkerResult | None]:
    ...
```

Behavior:

- If reproduction exists and final reproduction command fails, return a blocked/revise worker result.
- If integration tests fail, return a blocked/revise worker result.
- Otherwise return passing evidence.

This is a final invariant, not deterministic mid-loop routing.

- [ ] **Step 2: Separate status fields**

Update final report model and SWE sidecar to distinguish:

- `workflow_status`: did orchestration finish;
- `agent_verdict`: final reviewer verdict;
- `reproduction_passed`: final reproduction boolean;
- `official_prediction_emitted`: whether a patch was written for SWE-bench output.

- [ ] **Step 3: Keep official prediction behavior unchanged**

`all_preds.jsonl` should still contain only the model patch prediction. The extra status fields belong in sidecar JSON.

- [ ] **Step 4: Add tests**

Cases:

- workflow completes but final reproduction fails -> sidecar says workflow completed, agent verdict revise/block, reproduction false;
- final reproduction passes -> sidecar says reproduction true;
- official prediction emitted remains true when patch exists.

Run:

```powershell
uv run pytest tests/test_main_workflow.py tests/test_swe_bench.py -q
```

Expected: final status tests pass.

---

## Task 9: Retire Static Plan/Replan Logic

**Files:**

- Modify: `src/workflows/main_workflow.py`
- Modify: `tests/test_main_workflow.py`

- [ ] **Step 1: Remove unused `_run_candidate` if obsolete**

The current workflow no longer runs whole-plan candidates. Remove `_run_candidate` if no tests or call sites need it.

- [ ] **Step 2: Remove broad `_replan` path**

Remove static-plan replan code that sends accumulated `worker_results` back to `build_plan`.

- [ ] **Step 3: Keep corrective step helper only if still needed**

If `_corrective_plan_step` is no longer used, remove it. The planner loop now emits future steps directly.

- [ ] **Step 4: Update tests to assert no whole-plan candidate reruns**

Test that:

- a two-step planner output runs step 1;
- after step 1, planner is called again;
- step 2 is not run from stale original plan if planner changes it.

Run:

```powershell
uv run pytest tests/test_main_workflow.py -q
```

Expected: no old static-plan tests remain.

---

## Task 10: Full Verification

**Files:**

- No production file changes beyond previous tasks.

- [ ] **Step 1: Run focused tests**

```powershell
uv run pytest tests/test_main_workflow.py tests/test_planner.py tests/test_implementation_turn.py tests/test_implementation_workflow.py tests/test_context_gatherer.py tests/test_swe_bench.py tests/test_prompts.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run full suite**

```powershell
uv run pytest -q
```

Expected: full suite passes.

- [ ] **Step 3: Run lint on changed files**

```powershell
uv run ruff check src/models/plan.py src/models/worker.py src/activities/planner.py src/activities/implementation.py src/activities/context_gatherer.py src/workflows/main_workflow.py src/eval/swe_bench.py tests
```

Expected: no ruff errors.

- [ ] **Step 4: Manual smoke run**

Run the existing smoke workflow:

```powershell
uv run python -m src.eval.smoke_workflow
```

Expected:

- planner may request context before first implementation step;
- implementer receives concrete target files and context summary;
- only one step executes before planner is called again;
- final report has clear workflow and agent verdict semantics.

- [ ] **Step 5: SWE-bench single instance run**

From the Linux/WSL environment where SWE-bench works:

```bash
python -m src.eval.swe_bench --subset 1
```

Expected:

- image preparation still skips existing images;
- sidecar includes planner history, final reproduction status, and patch comparison;
- official prediction output remains compatible with SWE-bench harness.

---

## Migration Notes

- Do not commit `Temporal-Light` changes as part of this plan unless explicitly requested.
- Keep official SWE-bench prediction JSONL format unchanged.
- Avoid introducing deterministic mid-loop routing beyond evidence gathering and final invariants.
- The planner owns future step selection.
- Context requests are fulfilled and discarded; only context notes persist.
- Completed steps are immutable history; planner outputs future steps only.
- Test evidence is compacted using the "last passing or failures since last passing" rule.
- Preserve existing candidate-per-step behavior, but do not run whole-plan candidates.

---

## Self-Review

- Spec coverage: The plan covers planner-driven loop, ephemeral context requests, concrete implementer step context, compact history, test-history selection, subfunction extraction, final status semantics, and candidate handling.
- Placeholder scan: No implementation step uses TBD/TODO/fill-in language. Where exact implementation may vary because of circular imports, the plan gives the concrete fallback file `src/models/planning.py`.
- Type consistency: New model names are consistent across tasks: `ContextRequest`, `ContextNote`, `StepHistoryEntry`, `PlanningEvidence`, `PlannerState`, `PlannerTurn`.
- Scope check: The plan is one subsystem refactor: main workflow planning/execution. SWE-bench changes are limited to reporting semantics required by this refactor.
