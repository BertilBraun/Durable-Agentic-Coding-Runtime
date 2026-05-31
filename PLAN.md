# Project Definition: Durable Agentic Coding Runtime

A durable, evidence-driven runtime that turns a natural-language coding task into a reviewed code
change. A workflow engine (Temporal-Light) is the control plane; LLMs are policy; tools and tests
are ground truth. The deliverable is always a diff against the repository's starting commit.

This document is the **as-built map plus roadmap**, kept deliberately short. It summarizes how the
system works and points at the code rather than re-specifying it; the concrete next changes live as
standalone prompts under [prompts/](prompts/). It is not an exhaustive pre-implementation spec — the
implementation is the source of truth for detail.

---

## 1. Design principles

- **Orchestrate evidence, not just agents.** Progress is gated on observed facts (a non-empty diff,
  a test that ran and passed), never on the model's narrative claim.
- **Workflow engine as control plane.** All durable state and step sequencing live in the engine;
  LLM calls and IO are activities. Workflow code is deterministic.
- **LLM as policy, tools as ground truth.** The model decides *what* to do; tools/tests decide
  *whether it worked*.
- **Compact context, rich external state.** Keep the model's context small; push bulk
  (logs, large outputs) to the artifact store and pass references.

## 2. Non-goals

- Not a chat assistant or an IDE plugin; it runs tasks to completion under a budget.
- **Front-end / visual / browser-driven work is dropped.** Earlier drafts had a UI milestone
  (screenshot diffing, browser tools); that is scrapped and not planned.
- No multi-repo orchestration, no long-lived service management.

---

## 3. Architecture as built

Pipeline for one task (`main_workflow` in
[src/workflows/main_workflow.py](src/workflows/main_workflow.py)):

```
TaskRequest(origin) ── build_contract ── setup_environment ── build_repo_index
   ── build_plan ── assess_complexity ── [optional human approval]
   ── begin_candidate ── run plan steps (child implementation_workflow per step)
   ── get_full_diff ── review_patch ── FinalReport ── finalize_winner ── teardown_environment
```

Module map (everything under `src/`):

| Concern | Code |
| --- | --- |
| Task input (discriminated `Origin`: host/docker) | [models/task.py](src/models/task.py) |
| Environment: persistent `Workspace` seam + activities | [activities/workspace_manager.py](src/activities/workspace_manager.py) |
| Tool schema (`Tool` union, `ToolName`) / command building | [tools/definitions.py](src/tools/definitions.py), [tools/handlers.py](src/tools/handlers.py) |
| Repo index (tree-sitter symbols + call/mention xref) | [activities/repo_indexer.py](src/activities/repo_indexer.py), [models/repo.py](src/models/repo.py) |
| Contract / complexity / planner | [activities/contract_builder.py](src/activities/contract_builder.py), [activities/complexity_assessor.py](src/activities/complexity_assessor.py), [activities/planner.py](src/activities/planner.py) |
| Context gatherer (cheap-model retrieval → deterministic grounded packing) | [activities/context_gatherer.py](src/activities/context_gatherer.py), [models/context.py](src/models/context.py) |
| Implementation worker (bounded tool loop) | [activities/implementation.py](src/activities/implementation.py), [workflows/implementation_workflow.py](src/workflows/implementation_workflow.py) |
| Reviewer / final report | [activities/reviewer.py](src/activities/reviewer.py), [activities/report_builder.py](src/activities/report_builder.py) |
| Human approval (signal-driven) | [activities/human_approval.py](src/activities/human_approval.py), [cli/human_approval.py](src/cli/human_approval.py) |
| LLM client + model routing/config | [llm/client.py](src/llm/client.py), [config.py](src/config.py) |
| Worker registration | [worker.py](src/worker.py) |
| Evaluation (SWE-bench, smoke) | [eval/swe_bench.py](src/eval/swe_bench.py), [eval/smoke_workflow.py](src/eval/smoke_workflow.py) |

### Environment model

One **persistent environment per run**, worked **in place** (no repo copy), behind a single
polymorphic seam `Workspace.run_command(command, timeout) -> CommandResult`:

- **HOST** (`HostWorkspace`): subprocesses in the user's existing local repo — the quick dev mode.
- **DOCKER** (`DockerWorkspace`): `docker exec` in one long-lived container started from an image
  that already contains the repo (e.g. SWE-bench's `/testbed`) — the eval mode.

Setup asserts a clean tree and records `base_sha` (and the base branch, which may be detached).
Candidates run **sequentially**: each gets a branch `agentic/{run_id}/cand-{k}` off `base_sha`, the
agent commits its work there, and the result is `git diff base_sha`. Finalize returns to the base
branch and applies the winner as **uncommitted** edits (the Claude-Code feel); candidate branches
are kept unless `CLEANUP_CANDIDATE_BRANCHES` is set. `run_command` is the *only* place host and
docker differ — nothing else branches on workspace kind.

### Determinism & serialization (Temporal-Light)

- IO lives in `@activity` functions; `@workflow` code is deterministic (no wall-clock, RNG, or IO;
  bounded loops only). Only JSON-serializable values cross activity/child boundaries.
- Discriminated unions (`Field(discriminator='kind')`, e.g. `Origin`, `Workspace`) restore to the
  right subclass on replay; activity args on the forward path are the real objects, results are
  restored via the return annotation.
- Implementation steps run as child workflows via `spawn_child` / `wait_for_child`.

### Evidence & safety

- A `SUCCESS` `WorkerResult` requires real evidence: a `GitDiff` that returned output, or a
  `TestResult` from an actual `RunTests`. Narrative `diff_summary` text is never sufficient.
- Subagents stop cleanly (`BLOCKED` + replan suggestion) when context utilization crosses a
  threshold, instead of silently degrading. Tool path inputs are validated against absolute paths
  and parent traversal.
- Large tool outputs spill to the artifact store and return an `ArtifactReference`.

---

## 4. Evaluation

- **SWE-bench** ([eval/swe_bench.py](src/eval/swe_bench.py)): pulls the official per-instance image,
  runs the framework (and a single-call baseline) as DOCKER-origin tasks, applies the resulting
  patch in a fresh official container, and scores it with the `FAIL_TO_PASS` / `PASS_TO_PASS`
  oracle. Reports `resolved` / `failed` / `skipped`, cost-per-resolved, and llm-calls-per-task.
  Non-Python/TS tasks are skipped.
- **Smoke** ([eval/smoke_workflow.py](src/eval/smoke_workflow.py)): spins up a worker against a
  temp git repo via a HOST origin and asserts a real change + passing test end-to-end.

## 5. Configuration

Settings load from env in [config.py](src/config.py) (`CONFIG`), with model routing per
`ModelRole` resolved against `src/llm/models.csv`. Notable knobs: `LLM_API_KEY` / `LLM_BASE_URL`,
`MODEL_<ROLE>`, `HUMAN_APPROVAL_ENABLED`, `CLEANUP_CANDIDATE_BRANCHES`,
`IMPLEMENTATION_MAX_TOOL_ROUNDS`, the context-utilization thresholds, and the tool-output
size/compaction limits. See [.env.example](.env.example).

Data models are frozen Pydantic (`FrozenBaseModel`). Coding standards live in
[CODING_STANDARDS.md](CODING_STANDARDS.md): full descriptive names, type hints everywhere, enums
over literal strings, `match`/`case` over `isinstance`, comments only for a non-obvious *why*,
validate at boundaries, tests call production code directly with the LLM client faked via injection
([tests/fakes/openai_client.py](tests/fakes/openai_client.py)).

---

## 6. Current state & known gaps

The pipeline is implemented and green under unit/fake-mode tests. Outstanding:

- **No real-LLM run yet.** End-to-end verification is still fake-mode/unit-test based. The highest-
  value next step is a live smoke run (real model, small local repo), then a small SWE-bench subset.
- **DOCKER failure cleanup.** `teardown_environment` only runs on normal completion; container
  removal has no recovery path on failure (revisit when Temporal-Light supports
  failure/cancellation compensation).
- **Precise xref.** Call-site capture is tree-sitter heuristic (name-keyed, ignores comments and
  strings, distinguishes calls from mentions) — not scope-accurate. A precise per-language analyzer
  (Python `ast`/`symtable`, TS `tsserver`) is deferred.

## 7. Roadmap

Concrete next changes are specified as standalone prompts in [prompts/](prompts/) (each
self-contained, picked up in its own chat):

1. ~~**Indexer + shell tool + tool-surface rework**~~ — done: `run_shell` with a per-workspace
   environment descriptor (Windows/PowerShell and POSIX), narrow shell-wrapper tools deleted, and a
   real symbol + cross-reference (`find_definition` / `find_callers` / `find_callees`) index built
   through `run_command`; prompts now carry only the directory tree. —
   [prompts/01](prompts/01-indexer-shell-tool-rework.md)
2. ~~**Context packer rework**~~ — done: the cheap-model gatherer loop now curates typed
   `ContextSnippet` references (file + line range + reason) instead of paraphrasing code, and a
   deterministic `pack_context` activity reads the real workspace lines into `PackedSnippet`s within
   a character budget, spilling overflow to `CONTEXT_OVERFLOW` artifacts. The ungrounded prose fields
   are gone; planner and implementer pull context on demand with no unbounded index/context dump. —
   [prompts/02](prompts/02-context-packer-rework.md)
3. **Reproduce → repair → regression loop** — gate bugfix success on a test that fails on base then
   passes after the fix. — [prompts/03](prompts/03-reproduce-repair-regression-loop.md)
4. **Planner review → replan loop** — LLM plan review with context-gathering between rounds, plus
   extended thinking for planning roles. — [prompts/04](prompts/04-planner-review-replan.md)
5. **Adaptive sequential candidates + selector** — escalate candidate count by confidence
   (high→1, medium→2, low→4), select the best; combiner deferred. —
   [prompts/05](prompts/05-sequential-candidates-combiner.md)

Then: live smoke run → SWE-bench subset → fix what the real runs expose before scaling the benchmark.

---

## 8. Local commands

Verification (green before every commit):

```powershell
python -m ruff format src tests
python -m ruff check src tests
python -m pytest -q
$env:RUN_HOST_TESTS='1'; python -m pytest tests/test_workspace_manager_integration.py -q
```

Start Temporal-Light, then run the live HOST smoke:

```powershell
$env:POSTGRES_USER='tl'; $env:POSTGRES_PASSWORD='changeme'; $env:API_PORT='8080'
docker compose -f Temporal-Light\docker-compose.yml -f Temporal-Light\docker-compose.override.yml up -d postgres migrate api
python -m src.eval.smoke_workflow --temporal-api-url http://localhost:8080 `
  --temporal-database-url postgresql://tl:changeme@localhost:5432/temporal_light --timeout-seconds 120
```

SWE-bench subset:

```powershell
python -m src.eval.swe_bench --instances <instances.json> --temporal-api-url http://localhost:8080 --five-task-subset
```
