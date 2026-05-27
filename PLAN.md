# Project Definition: Durable Agentic Coding Runtime

## 1. Working Title

**Durable Agentic Coding Runtime**

A workflow-engine-backed coding agent system that converts software engineering requests into verified, minimal, reviewable code changes using structured planning, controlled context, deterministic tooling, execution-grounded validation, optional human approval, and candidate search.

---

## 2. Core Objective

Build a practical agentic coding system that can reliably perform non-trivial software engineering tasks in real repositories while remaining inspectable, resumable, auditable, and controllable.

The system should not simply expose a shell to an LLM. It should orchestrate a structured process:

1. Convert vague user intent into a concrete implementation contract.
2. Localize relevant code using deterministic repository analysis.
3. Build compact context packs for each model call.
4. Execute implementation steps in isolated workspaces.
5. Test, review, and validate each patch against explicit acceptance criteria.
6. Escalate to parallel candidate generation when uncertainty is high.
7. Select, refine, and commit the best patch.
8. Produce a final report with evidence, risks, and tests run.

The system should optimize for **correctness, debuggability, minimality, and trust**, not just raw task completion.

---

## 3. Non-Goals

The project should avoid becoming a vague multi-agent roleplay framework.

Out of scope initially:

* Generic chat-agent framework.
* Arbitrary business-process automation.
* Large persona-based agent teams such as “manager agent”, “developer agent”, “architect agent” without measurable purpose.
* Full IDE replacement.
* Fully autonomous production deployment.
* Training custom models or reward models in the first version.
* Solving large, ambiguous product tasks without human approval.

The first version should focus on repository-level coding tasks with strong execution feedback.

---

## 4. Design Philosophy

### 4.1 Orchestrate Evidence, Not Just Agents

Every important decision should be tied to evidence:

* Which requirement does this change address?
* Which files/functions are relevant?
* Which tests prove the behavior?
* Which previous attempt failed and why?
* Which risks remain?

The system should avoid blindly trusting model reasoning.

### 4.2 Workflow Engine as Control Plane

The workflow engine provides:

* durable state
* retries
* persistence
* checkpoints
* human approval signals
* parallel child workflows
* execution history
* failure recovery
* auditability

The workflow engine does **not** make the agent smarter by itself. It enables higher-quality strategies such as parallel search, rollback, validation gates, and structured review.

### 4.3 LLM as Policy, Tools as Ground Truth

The LLM proposes actions. Deterministic tools execute, validate, and constrain them.

The agent should not be responsible for manually searching an entire repository through raw shell commands when AST, symbol, test, and diff tools can provide structured information.

### 4.4 Compact Context, Rich External State

The active LLM context should remain small and relevant.

Large artifacts such as logs, screenshots, diffs, prompts, completions, test reports, and file snapshots should be stored externally and referenced through IDs plus compact summaries.

---

## 5. Target Use Cases

### 5.1 Bug Fixing

Input:

* issue description
* failing test
* stack trace
* user-reported behavior

Expected behavior:

* reproduce bug when possible
* identify relevant code
* patch minimally
* add or update regression test
* run targeted and broader tests
* produce final patch and evidence

### 5.2 Feature Implementation

Input:

* feature request
* rough acceptance criteria
* optional design constraints

Expected behavior:

* create implementation contract
* plan subtasks
* request human approval for non-trivial changes
* implement stepwise
* run tests per step
* replan when discoveries invalidate assumptions
* produce reviewable commit

### 5.3 Refactoring

Input:

* desired refactor
* constraints
* affected modules

Expected behavior:

* preserve behavior
* avoid unrelated changes
* run regression tests
* validate public API compatibility
* generate a minimal diff

### 5.4 Front-End / UI Work

Input:

* UI bug, design change, or feature request
* optional screenshot, mockup, component description, or visual acceptance criteria

Expected behavior:

* run local dev server
* inspect page through browser automation
* capture screenshots
* compare screenshots before/after
* optionally inspect DOM/accessibility tree
* validate visual behavior against task contract
* run unit/component/e2e tests where available

Visual inspection should be treated as first-class execution feedback for UI tasks, analogous to test output for backend tasks.

---

## 6. System Architecture

```text
User Request
    |
    v
Requirement / Contract Builder
    |
    v
Repository Indexer + Context Discovery
    |
    v
Planner
    |
    v
Human Approval Gate [optional]
    |
    v
Implementation Workflow
    |
    +--> Single Worker Attempt
    |       |
    |       +--> edit / test / inspect / debug loop
    |
    +--> Parallel Candidate Attempts [on escalation]
            |
            +--> candidate patches
    |
    v
Validation + Review
    |
    v
Candidate Selection / Refinement
    |
    v
Final Commit / PR Report
```

---

## 7. Main Components

## 7.1 Workflow Orchestrator

The durable control plane.

Responsibilities:

* start and resume agent runs
* maintain run state
* call LLM activities
* call tool activities
* enforce budgets
* manage retries/timeouts
* spawn child workflows
* handle human approval signals
* store artifact references
* ensure deterministic workflow execution

Important rule:

LLM calls, shell commands, filesystem access, browser automation, and other nondeterministic or side-effecting operations must run as activities, not directly inside deterministic workflow logic.

---

## 7.2 Task Contract Builder

Converts raw user intent into a concrete task contract.

Output fields:

```json
{
  "task_type": "bugfix | feature | refactor | frontend | test | docs | unknown",
  "goal": "...",
  "acceptance_criteria": ["..."],
  "non_goals": ["..."],
  "expected_behavior": ["..."],
  "affected_areas": ["..."],
  "risk_areas": ["..."],
  "tests_expected": ["..."],
  "requires_human_approval": true,
  "open_questions": ["..."]
}
```

For small bug fixes, this may be lightweight. For feature work, it should be human-reviewable before implementation.

---

## 7.3 Repository Indexer

Builds deterministic repository understanding.

Responsibilities:

* file tree indexing
* language detection
* dependency metadata extraction
* AST parsing
* symbol indexing
* import graph construction
* call/reference graph where feasible
* test discovery
* mapping tests to modules/symbols where feasible
* detecting generated/vendor/lock files

Potential tools:

```text
repo_summary
list_files
find_symbol
read_symbol
find_references
find_callers
find_callees
show_import_graph
list_tests
list_tests_for_symbol
read_file_range
search_text
```

This should reduce reliance on brute-force LLM-driven repository search.

---

## 7.4 Context Packer

Central component that decides what each model call sees.

Inputs:

* task contract
* current phase
* active subtask
* relevant code snippets
* current diff summary
* recent observations
* failed-attempt lessons
* test results
* available actions
* budget state

Outputs:

* compact prompt context
* artifact references
* selected code snippets
* tool availability list

The context packer should aggressively avoid:

* full repository dumps
* full logs
* repeated old observations
* irrelevant tools
* unbounded chat history
* stale reasoning traces

---

## 7.5 Planner

Creates an implementation plan from the task contract and repository context.

Plan schema:

```json
{
  "summary": "...",
  "steps": [
    {
      "id": "step_1",
      "goal": "...",
      "target_files": ["..."],
      "allowed_files": ["..."],
      "tests_to_run": ["..."],
      "expected_result": "...",
      "risk": "low | medium | high",
      "requires_human_approval": false
    }
  ],
  "integration_tests": ["..."],
  "rollback_strategy": "...",
  "definition_of_done": ["..."]
}
```

The plan should be revisable. Discoveries during implementation may trigger replanning.

---

## 7.6 Implementation Worker

A bounded worker that executes one subtask or one full small task.

Responsibilities:

* inspect relevant context
* request additional snippets/tools if needed
* edit code
* run tests
* inspect failures
* update failure memory
* stop when done, blocked, or over budget

The worker should operate under a local contract:

```json
{
  "subtask_goal": "...",
  "allowed_files": ["..."],
  "relevant_context": ["..."],
  "acceptance_tests": ["..."],
  "constraints": ["..."],
  "budget": {...}
}
```

Outputs:

```json
{
  "status": "success | failed | blocked | needs_replan",
  "patch_id": "...",
  "diff_summary": "...",
  "tests_run": ["..."],
  "test_results": ["..."],
  "discovered_issues": ["..."],
  "confidence": "low | medium | high",
  "replan_suggestion": "..."
}
```

---

## 7.7 Tool Execution Layer

Provides typed, validated tools.

Tools should be powerful enough to support real work, but not so numerous that tool selection becomes noisy.

### Essential Tools

```text
read_file_range
write_file
apply_patch
search_text
run_command
git_status
git_diff
git_apply
git_revert_file
git_revert_hunk
run_tests
run_lint
run_typecheck
```

### Structured Code Tools

```text
find_symbol
read_symbol
find_references
find_callers
find_callees
show_import_graph
list_tests
list_tests_for_symbol
```

### Test/Failure Tools

```text
run_targeted_tests
run_related_tests
run_full_tests
summarize_test_failure
classify_failure
extract_relevant_traceback
```

### Front-End / Visual Tools

```text
start_dev_server
stop_dev_server
open_browser
navigate_to_url
capture_screenshot
capture_element_screenshot
get_dom_snapshot
get_accessibility_tree
inspect_console_errors
inspect_network_errors
compare_screenshots
measure_layout_shift
run_e2e_test
```

Visual tools should support:

* full-page screenshots
* viewport-specific screenshots
* element-level screenshots
* before/after comparison
* console error extraction
* network failure extraction
* DOM and accessibility-tree inspection
* responsive viewport checks

The agent should be able to reason over visual observations, especially for UI implementation and debugging.

---

## 7.8 Workspace Manager

Manages isolated execution environments.

Responsibilities:

* create workspace
* clone repository
* install dependencies
* start persistent shell/session
* run commands with timeouts
* manage environment variables
* snapshot workspace
* restore workspace
* create candidate branches
* isolate parallel attempts
* collect artifacts
* destroy workspace

Important behaviors:

* each candidate patch should run in its own branch or workspace snapshot
* destructive commands should be constrained
* large outputs should be truncated and stored externally
* commands should have clear shell semantics

Shell documentation exposed to agent should include:

```text
- shell type
- current working directory
- whether state persists between commands
- quoting rules
- newline behavior
- timeout behavior
- stdout/stderr truncation
- environment variables
- forbidden commands
```

---

## 7.9 Artifact Store

Stores large artifacts outside the workflow history.

Artifacts:

* prompts
* completions
* tool inputs/outputs
* stdout/stderr logs
* test reports
* diffs
* screenshots
* DOM snapshots
* accessibility snapshots
* browser console logs
* network logs
* workspace snapshots
* final reports

Workflow state should contain compact summaries and artifact IDs, not large blobs.

Example:

```json
{
  "artifact_id": "test_log_123",
  "kind": "test_output",
  "summary": "3 parser tests failed due to None handling",
  "uri": "artifact://runs/42/test_log_123"
}
```

---

## 7.10 Verifier / Reviewer

Grounded review component.

Inputs:

* task contract
* implementation plan
* current diff
* relevant code snippets
* test results
* failed attempt summaries
* risk checklist

Outputs:

```json
{
  "verdict": "accept | revise | reject | needs_human",
  "blocking_issues": ["..."],
  "non_blocking_issues": ["..."],
  "evidence": ["..."],
  "missing_tests": ["..."],
  "regression_risks": ["..."],
  "minimality_assessment": "...",
  "recommended_next_action": "..."
}
```

Review types:

* contract compliance review
* patch minimality review
* regression risk review
* test adequacy review
* security/side-effect review
* UI visual review for front-end tasks

---

## 7.11 Candidate Search and Selection

Escalation mechanism for hard or flaky tasks.

Initial policy:

1. Start with one worker attempt.
2. If the worker fails, loops, has low confidence, or produces a risky patch, spawn multiple independent candidates.
3. Validate each candidate independently.
4. Select the best candidate using tests, diff minimality, contract compliance, and reviewer judgment.
5. Optionally refine the winning patch.

Candidate diversity sources:

* different random seeds
* different model variants
* different localized context packs
* different implementation plans
* different generated tests

Selection criteria:

```text
- tests pass
- patch is minimal
- acceptance criteria satisfied
- generated regression test passes
- no unrelated files changed
- reviewer accepts
- lower regression risk
- simpler implementation
```

---

## 7.12 Human Interaction Layer

Human review should be inserted where it has high leverage:

* approving non-trivial implementation contracts
* resolving ambiguous requirements
* approving public API changes
* approving risky migrations
* approving final patch before commit/PR
* choosing between semantically different candidate patches

Human interaction should use structured requests:

```json
{
  "question": "Should the API remain backward compatible with v1?",
  "options": ["yes", "no", "unsure"],
  "context_summary": "...",
  "impact": "This determines whether adapter code is needed."
}
```

Avoid asking humans vague questions unless genuinely necessary.

---

## 8. Core Workflow

## 8.1 Full Feature Workflow

```text
1. Receive user request
2. Build task contract
3. Index repository or load existing index
4. Localize relevant modules/files/tests
5. Create implementation plan
6. Ask human to approve/modify plan if task is non-trivial
7. Create workspace snapshot
8. For each plan step:
    a. create local subtask contract
    b. pack context
    c. run implementation worker
    d. run targeted tests
    e. review local patch
    f. merge, rollback, or replan
9. Run integration validation
10. Run final review
11. If low confidence, spawn parallel candidates or ask human
12. Produce commit and final report
```

## 8.2 Bug Fix Workflow

```text
1. Receive issue / failing behavior
2. Build lightweight contract
3. Attempt reproduction
4. Localize relevant code/tests
5. Generate or identify regression test
6. Patch minimally
7. Run targeted tests
8. Run related tests
9. Review diff
10. Commit or report blocker
```

## 8.3 Front-End Visual Workflow

```text
1. Receive UI task or visual bug
2. Build UI-specific task contract
3. Start dev server
4. Open browser to target route
5. Capture baseline screenshot
6. Inspect DOM, console, network, accessibility tree as needed
7. Implement UI changes
8. Capture after screenshot
9. Compare before/after against acceptance criteria
10. Run component/e2e tests where available
11. Review visual diff and code diff
12. Commit or request human visual approval
```

---

## 9. Agent Action Schema

Agents should not emit arbitrary free-form control instructions.

Example action schema:

```json
{
  "action": "run_tests",
  "args": {
    "command": "pytest tests/test_parser.py -q",
    "timeout_seconds": 120
  },
  "rationale": "Validate the parser regression before broader testing.",
  "expected_observation": "The new regression test should fail before the fix and pass after.",
  "risk": "low"
}
```

The workflow validates:

* whether the action is allowed in this phase
* whether paths are inside the workspace
* whether command is safe
* whether mutation is allowed
* whether human approval is required
* whether budget remains

---

## 10. State Model

Example run state:

```json
{
  "run_id": "...",
  "task_contract": {},
  "repo_index_id": "...",
  "plan": {},
  "phase": "implementation",
  "active_step": {},
  "workspace_id": "...",
  "current_diff_summary": "...",
  "test_results": [],
  "failed_attempts": [],
  "candidate_patches": [],
  "review_verdicts": [],
  "visual_artifacts": [],
  "budget": {
    "model_calls_remaining": 20,
    "tool_calls_remaining": 80,
    "parallel_candidates_remaining": 4,
    "wall_clock_deadline": "..."
  },
  "blockers": [],
  "final_status": null
}
```

---

## 11. Budget and Escalation Policy

The system should not blindly maximize token use.

Default policy:

```text
Start with one attempt.
Escalate only when uncertainty justifies it.
```

Escalation triggers:

* repeated same failure
* low-confidence review
* flaky or contradictory test results
* task classified as high complexity
* multiple plausible implementation strategies
* visual ambiguity
* risky public API changes

Possible escalations:

* ask human
* replan
* spawn 2-4 independent candidates
* run broader tests
* request specialized review
* reduce scope

---

## 12. Safety and Side-Effect Controls

The system should enforce:

* sandboxed workspaces
* no uncontrolled access to host filesystem
* no secret exfiltration
* network restrictions configurable by project
* no destructive commands outside workspace
* command timeouts
* output size limits
* path validation
* explicit approval for deployment, publishing, migration, or external side effects

---

## 13. Evaluation

## 13.1 Metrics

Primary metrics:

```text
- task success rate
- test pass rate
- regression rate
- patch minimality
- human approval rate
- number of required human interventions
- cost per successful task
- wall-clock time
- number of model/tool calls
- rollback frequency
- flaky result frequency
```

Secondary metrics:

```text
- localization accuracy
- context size per call
- average failed attempts before success
- reviewer disagreement rate
- generated test usefulness
- visual validation usefulness for UI tasks
```

## 13.2 Benchmark Sets

Initial internal benchmark:

```text
20 small bug fixes
20 medium bug fixes
20 feature additions
20 refactors
20 front-end/UI tasks
20 ambiguous tasks requiring human approval
```

Each task should track:

* expected behavior
* ground-truth tests where possible
* allowed files
* hidden regression tests where possible
* human judgment where necessary

## 13.3 Ablations

Evaluate:

```text
single loop vs workflow-guided loop
no AST tools vs AST tools
no reviewer vs reviewer
single candidate vs parallel candidates
no visual tools vs screenshot/browser tools for UI tasks
full context vs compact context packing
human-approved plan vs no plan approval
```

---

## 14. MVP Scope

The first MVP should support backend-oriented coding tasks before full UI automation.

### MVP Features

```text
- durable run workflow
- task contract builder
- repo file/search tools
- basic AST symbol lookup for one language
- shell command activity
- read/write/apply patch tools
- git diff/status tools
- targeted test runner
- artifact store
- compact context packer
- implementation worker loop
- final diff reviewer
- final report
```

### MVP Exclusions

```text
- full browser automation
- multi-language AST support
- learned reward model
- sophisticated MCTS
- production deployment
- complex multi-agent debate
```

---

## 15. Second Milestone

Add stronger correctness mechanisms:

```text
- reproduction-before-repair flow
- generated regression tests
- failure memory
- patch minimality checker
- workspace snapshots and rollback
- candidate branch isolation
- 2-4 parallel candidate attempts
- candidate selector
- human approval gate for plans
```

---

## 16. Third Milestone: Front-End / Visual Agent Support

Add browser-backed visual inspection.

Features:

```text
- start dev server
- browser navigation
- screenshot capture
- element screenshot capture
- console log inspection
- network error inspection
- DOM snapshot
- accessibility tree snapshot
- responsive viewport checks
- before/after screenshot comparison
- optional human visual approval
```

This milestone makes the system useful for UI tasks where textual tests are insufficient.

---

## 17. Fourth Milestone: Advanced Search and Verification

Add test-time compute strategies:

```text
- parallel candidate generation
- beam-style trajectory search
- trajectory quality scoring
- specialized reviewers
- patch ensemble selection
- cost-aware escalation
```

Optional later research:

```text
- process reward model
- learned trajectory ranker
- MCTS over implementation trajectories
- historical run memory across repositories
```

---

## 18. Key Risks

### 18.1 Context Pollution

The agent may receive too much irrelevant information.

Mitigation:

* central context packer
* compact summaries
* artifact references
* AST/symbolic retrieval

### 18.2 Plausible but Wrong Patches

The agent may produce changes that look reasonable but do not solve the task.

Mitigation:

* reproduction-before-repair
* acceptance criteria
* regression tests
* verifier grounded in diff/tests/context

### 18.3 Overbroad Changes

The agent may refactor unrelated code.

Mitigation:

* allowed files
* patch minimality review
* diff size warnings
* generated/vendor file protection

### 18.4 Tool Misuse

The agent may issue invalid or unsafe commands.

Mitigation:

* typed tools
* command validation
* shell documentation
* safe defaults
* timeouts

### 18.5 Expensive Search

Parallel candidates may burn tokens quickly.

Mitigation:

* escalation policy
* budgets
* early stopping
* cheap model routing for summarization

### 18.6 UI Ambiguity

Visual tasks may lack clear pass/fail tests.

Mitigation:

* screenshots
* before/after comparison
* accessibility tree
* human visual approval
* explicit visual acceptance criteria

---

## 19. Recommended First Implementation Path

### Step 1: Build the core loop

```text
Workflow run
→ context pack
→ LLM action
→ validate action
→ execute tool
→ reduce state
→ repeat
```

### Step 2: Add repository tools

```text
read_file_range
search_text
apply_patch
git_diff
run_tests
```

### Step 3: Add task contracts and planning

```text
contract builder
plan builder
human approval optional
```

### Step 4: Add verifier

```text
diff review
test adequacy review
minimality review
```

### Step 5: Add isolation and rollback

```text
workspace snapshots
candidate branches
revert tools
```

### Step 6: Add parallel candidates

```text
spawn K child workflows
validate patches
select winner
```

### Step 7: Add visual/browser tooling

```text
start dev server
capture screenshot
inspect console/DOM/network
compare screenshots
```

---

## 20. Success Criteria

The project is successful if it can:

1. Reliably solve small-to-medium repository tasks with minimal human intervention.
2. Produce patches that are test-backed and reviewable.
3. Keep token use bounded through compact context management.
4. Recover from worker crashes or failed attempts.
5. Explain why a patch was made and what evidence supports it.
6. Escalate intelligently to parallel search or human approval.
7. Support front-end tasks through visual inspection, not just code/test output.
8. Provide enough traceability to debug failed agent runs.

The long-term success criterion is not “the agent always succeeds.” It is:

> When the agent succeeds, the result is trustworthy. When it fails, the failure is diagnosable.

---

## 21. Concrete Architecture Decisions

This section records all binding implementation decisions. Anything here overrides the more abstract descriptions above.

---

### 21.1 Workflow Orchestrator: Temporal-Light

The workflow orchestrator is [Temporal-Light](https://github.com/BertilBraun/Temporal-Light), a lightweight self-hosted durable workflow engine backed by PostgreSQL.

Temporal-Light runs as a **completely separate Docker Compose stack** from the agentic coding system. The agent code interacts with it exclusively via the Temporal-Light HTTP API (`TEMPORAL_API_URL` env var). No shared imports, no shared compose file.

#### Child Workflows (implemented)

Temporal-Light ships `spawn_child` / `wait_for_child` in `temporal_light.children`:

```python
child_a = await spawn_child(“implementation_workflow”, subtask=subtask_a)
child_b = await spawn_child(“implementation_workflow”, subtask=subtask_b)

result_a = await wait_for_child(child_a)
result_b = await wait_for_child(child_b)
```

- `spawn_child` writes a `child_started` event (idempotent on replay), creates a new workflow row, returns `child_id`.
- `wait_for_child` suspends the parent until the child signals completion via a reserved `__child_completed__` signal. Raises `ChildWorkflowFailedError` if the child failed.
- Parent workflow `parent_info` is stored in the child's STARTED event payload and used to wake the parent on completion or failure.

#### Divergence Handling

No special handling. Divergence errors crash the workflow. User deletes the workflow row and restarts the task. This is acceptable for development and SWE-bench evaluation.

---

### 21.2 Implementation Worker as Nested Workflow

The Implementation Worker (Section 7.6) is a **nested `@workflow`**, not an activity. This gives full durability across the subtask loop: each edit/test/inspect step is an activity inside the child workflow, so crashes mid-subtask are recoverable.

```
main_workflow
└── implementation_workflow (child, one per plan step)
    ├── context_gather_activity
    ├── tool_executor_activity (edit / apply_patch / run_tests / ...)
    ├── tool_executor_activity
    └── reviewer_activity
```

Parallel candidate escalation spawns multiple independent `implementation_workflow` children and waits for all via `wait_for_child`.

---

### 21.3 Docker Architecture

```
agentic-coding/docker-compose.yml
├── agent-worker          # Temporal-Light worker: hosts all @workflow + @activity defs
│                         # connects to Temporal-Light via TEMPORAL_API_URL
└── artifacts-volume      # Named Docker volume for large blobs (logs, diffs, screenshots)

[separate] temporal-light/docker-compose.yml
├── postgres
├── temporal-api
└── temporal-worker
```

#### Workspace Containers: Per-Step

Each tool execution step runs inside a **fresh Docker container** spawned by the `WorkspaceManager` activity via the Docker Python SDK. The container is created from the workspace image, receives the current repo state via a volume, executes the command, and is destroyed after the step.

Per-step containers prevent state leakage between tool calls and reduce the surface area for subtle environment bugs. The repo state is persisted via git commits between steps — the container is stateless, the git working tree is the state.

#### Workspace Image

Base image includes: `git`, `python 3.12`, `node 20`, `tree-sitter`, `ruff`, `pytest`, `npm`. No IDE tooling.

#### Isolation

Each workflow run gets an isolated git worktree. Each candidate in a parallel search gets its own branch off that worktree. Network egress from workspace containers is configurable per project.

---

### 21.4 Supported Languages

Tree-sitter AST tools support **Python** and **TypeScript/JavaScript/React** only. Tasks in other languages are marked `skipped` in evaluation output.

---

### 21.5 Context Gatherer

The context gatherer is a cheap-model agent available to the main implementation agent as a tool:

```
main agent calls: gather_context(prompt=”I need the contract for POST /auth/token and its callers”)
    └── context_gather_activity:
        ├── cheap model receives: task goal + repo file tree + symbol index summary + caller prompt
        ├── runs multi-turn loop: find_symbol / read_file_range / find_references / ...
        ├── terminates when model sets done=true or N tool calls exceeded
        └── returns compiled context block (structured, no exploration trace)

main agent receives: compiled context block (tool call itself collapsed from history)
```

The tool call + result pair is **collapsed** in the main agent's running context: only the compiled context block is injected, the `gather_context(...)` tool call is dropped. The main agent sees what was gathered without the mechanical exploration trace.

Budget: configurable `CONTEXT_GATHERER_MAX_TOOL_CALLS` (default: 10).

---

### 21.6 LLM Client

All LLM calls go through a single `LLMClient` class. No direct SDK calls outside this class.

```python
T = TypeVar(“T”, bound=BaseModel)

class LLMClient:
    async def complete(
        self,
        role: ModelRole,
        messages: list[Message],
    ) -> str: ...

    async def generate_structured(
        self,
        role: ModelRole,
        messages: list[Message],
        output_type: type[T],
    ) -> T: ...
```

- Uses the OpenAI Python SDK with `base_url` / `api_key` from config for provider flexibility.
- Tracks `input_tokens`, `output_tokens`, `cache_read_tokens`, and `cost_usd` for every call.
- Accumulates totals into the current run's cost ledger (stored as part of the workflow's final report).
- Model routing via `ModelRole` enum → model string, loaded from env vars (see below).
- Prompt caching: deferred post-MVP. Marked as `# TODO: add cache_control breakpoints for Anthropic`.

#### Model Configuration

```python
class ModelRole(StrEnum):
    CONTRACT_BUILDER = “contract_builder”
    PLANNER         = “planner”
    CONTEXT_GATHERER = “context_gatherer”
    IMPLEMENTATION  = “implementation”
    REVIEWER        = “reviewer”
    SUMMARIZER      = “summarizer”

# env vars (with defaults)
MODEL_CONTRACT  = claude-opus-4-7
MODEL_PLANNER   = claude-opus-4-7
MODEL_CONTEXT   = claude-haiku-4-5-20251001
MODEL_IMPL      = claude-sonnet-4-6
MODEL_REVIEWER  = claude-sonnet-4-6
MODEL_SUMMARY   = claude-haiku-4-5-20251001
```

---

### 21.7 Artifact Storage

No separate artifact store service or database table. Large blobs (test logs, diffs, screenshots) are written to the `artifacts-volume` Docker volume by activities. The activity returns a compact result containing the file path and a short summary. That result lives in Temporal-Light's event log (JSONB payload). The path is sufficient to retrieve the blob later.

```python
@dataclass(frozen=True)
class ArtifactReference:
    path: str          # path on artifacts-volume, e.g. /artifacts/run-abc/test_log.txt
    summary: str       # e.g. “3 parser tests failed: test_none_handling, ...”
    kind: ArtifactKind
```

---

### 21.8 Evaluation: SWE-bench Verified

- Dataset: **SWE-bench Verified** only (not full SWE-bench).
- Language filter: skip tasks whose repository is not Python or TypeScript/JavaScript. Log `status=skipped, reason=unsupported_language`.
- Metrics reported:
  - `resolved`: % of eligible tasks where `FAIL_TO_PASS` tests pass and `PASS_TO_PASS` tests don't regress.
  - `skipped`: % of total tasks skipped (with breakdown by reason).
  - `cost_per_resolved_task`: average USD cost for tasks that resolved.
  - `llm_calls_per_task`: average LLM calls for resolved vs. failed tasks.
- **Baseline comparison**: same model, no framework — direct “here is the problem statement, produce a patch” single LLM call. Run both on the same filtered task set and report the delta.
- **SWE-bench Docker images**: use the official per-instance images. The `WorkspaceManager` pulls the image for each task; no custom environment setup needed.
- Eval harness: a standalone Python script (`eval/swe_bench.py`) that iterates task instances, starts a workflow via the Temporal-Light HTTP client, polls for completion, extracts the patch from the artifact volume, applies it in the SWE-bench environment, and runs the test oracle.

---

### 21.8b Human Approval Gate

#### Complexity Assessment

Before presenting a plan to the human, an LLM activity classifies the task's complexity. The output is a structured verdict:

```python
class ComplexityVerdict(BaseModel):
    requires_human_approval: bool
    reasoning: str  # one sentence
```

Factors that push `requires_human_approval = True`:

- Estimated diff touches more than ~3 files.
- Task involves public API changes, database migrations, or authentication logic.
- Task type is `feature` or `refactor` (not `bugfix` or `docs`).
- Acceptance criteria are ambiguous or contradictory.
- Risk areas include security, data integrity, or breaking changes.

For `bugfix` tasks with a clear failing test and a narrow scope, `requires_human_approval` is typically `False` and the system proceeds directly to implementation.

#### Approval Loop

When `requires_human_approval = True`, the workflow enters this loop before implementation begins:

```text
present_plan_to_human(plan, task_contract)
    └── human responds:
        ├── approve  → exit loop, proceed to implementation
        └── revise(feedback: str)
                └── planner_activity(task_contract, prior_plan, human_feedback)
                        └── returns revised_plan
                                └── loop back to present_plan_to_human
```

The human signal payload:

```python
class HumanApprovalSignal(BaseModel):
    decision: Literal["approve", "revise"]
    feedback: str | None  # required when decision == "revise"
```

The plan presented to the human includes: goal, step list with risk levels, affected files, rollback strategy, and definition of done. The human sees enough to make a meaningful judgement and propose targeted changes.

There is no timeout on the approval wait — the workflow suspends indefinitely until the human responds.

The revision loop has no hard limit on iterations. If the human keeps revising, the planner keeps replanning. This is intentional: human approval is a trust gate, not a rubber stamp.

---

### 21.9 Coding Standards

See `CODING_STANDARDS.md`. Summary:

- Full type annotations everywhere.
- No raw `dict` or string keys for structured data — always a `dataclass`, Pydantic `BaseModel`, or `NamedTuple`.
- Enums over literal strings for fixed value sets.
- `match/case` over `isinstance` chains.
- No abbreviations in names.
- `ruff format` + `ruff check` clean before every commit.
- Python 3.10+.

---

## 22. Current Implementation Status and Next Plan

This section is the current execution checkpoint for continuing implementation. It complements the architecture in Section 21.

### 22.1 Current Git State

Latest commits:

```text
2dc8e86 Complete SWE-bench evaluation report
15d5b81 Stop agents near context limit
ea10d90 Replan after worker replan signal
8a54970 Assert validated tool call fields
7cdd6af Use per-turn context observations
bfe99d7 Require observed diff evidence
1b2917d Dispatch gather context activity
8d27b12 Extract SWE-bench workflow patches
```

The `Temporal-Light` dependency is tracked as a Git submodule via `.gitmodules`.

### 22.2 Implemented

Project scaffold and packaging:

- `pyproject.toml` with Python 3.10+ dependencies.
- `Dockerfile`, `workspace.Dockerfile`, `docker-compose.yml`, and `docker-compose.override.yml`.
- `.gitignore`, `.dockerignore`, `.env.example`.
- Agent worker code auto-reloads in development through `docker-compose.override.yml`.

Data contracts:

- Frozen Pydantic models under `src/models/` for tasks, plans, repository indexes, context packs, worker results, reviews, and approval signals.
- Python 3.10-compatible `StrEnum` base in `src/runtime_enums.py`.

LLM layer:

- `src/llm/config.py` defines `ModelRole` and env-based model routing.
- `src/llm/client.py` centralizes all OpenAI-compatible calls with structured output parsing and usage ledger.
- `LLMClient` tracks `last_input_token_count` and exposes `context_utilization()` using configured model context limits.
- `LLM_FAKE_MODE=1` provides deterministic structured responses for smoke workflows.

Tools and workspace:

- Typed tool dataclasses in `src/tools/definitions.py` with `ToolName` enum.
- Path validation in `src/tools/handlers.py` blocks absolute paths and parent traversal.
- Docker-backed workspace manager in `src/activities/workspace_manager.py`; each tool invocation runs in a fresh container.
- Large tool outputs (>20 KB) are written to the artifacts volume and returned as `ArtifactReference`.

Repository indexing:

- `src/activities/repo_indexer.py` uses tree-sitter (required dependency) for Python and TypeScript/JavaScript/TSX/JSX.
- `FindSymbol` and `FindReferences` use the in-memory `RepoIndex` built at workflow start.

Activities:

- Contract builder, complexity assessor, planner, context gatherer, implementation turn, reviewer, report builder, human plan presentation, tool executor, and workspace manager activities.
- Activity wrapper serializes Pydantic/dataclass inputs and outputs to JSON-safe Temporal-Light event payloads.
- `run_implementation_turn` dispatches `gather_context` as an activity and injects the returned `ContextPack` as a structured observation.
- `gather_context` feeds only the current turn's tool observations back to the cheap-model agent while retaining all observations for fallback context packs.
- Implementation and context gatherer tool conversion assert validated fields instead of silently substituting defaults.

Workflows:

- `main_workflow`: contract → workspace → repo index → plan → optional human approval loop → child implementation workflows → final diff → review → final report → destroy workspace.
- `implementation_workflow`: gather context → bounded implementation turns (max 12 rounds) → step-level review on success → return `WorkerResult`.
- Child workflow integration uses Temporal-Light `spawn_child` and `wait_for_child`.
- `main_workflow` handles `WorkerStatus.NEEDS_REPLAN` by calling `build_plan` again with accumulated worker results and then running the revised plan from the start.
- `main_workflow` stops executing further steps when a child returns `FAILED` or `BLOCKED`; the final report includes the terminal worker result.
- Implementation and context gatherer agents stop explicitly when context utilization exceeds 80 percent instead of attempting silent compression.

Implementation evidence:

- Successful `WorkerResult` requires either `saw_diff=True` (a `GitDiff` call returned non-empty output) or at least one `TestResult`; bare LLM claims and narrative `diff_summary` text without evidence are rejected.
- `diff_summary` in `WorkerResult` is LLM-generated free text describing what changed — it is not a raw git diff.

Smoke test:

- `src/eval/smoke_workflow.py` creates a real repo, runs a fake-mode workflow, and asserts a non-empty diff and passing test result.

SWE-bench harness (`src/eval/swe_bench.py`):

- Typed models: `SweBenchInstance`, `EvaluationTaskResult`, `EvaluationReport`.
- Language filter: skips non-Python/TypeScript tasks.
- `--subset N` selects the first N eligible tasks; the default run processes all selected instances.
- Baseline runner makes a single implementation-model call and extracts a unified diff patch from the response.
- Official SWE-bench Docker image pull and container start per instance.
- Patch application inside containers.
- Oracle test execution (`FAIL_TO_PASS` / `PASS_TO_PASS`).
- Patch extraction from completed workflow artifact volume.
- JSON evaluation reports include `resolved`, `failed`, `skipped`, skip-reason breakdown, `cost_per_resolved_task`, and resolved/failed `llm_calls_per_task`.

### 22.3 Current Gaps

Real-LLM validation:

- The system has still not run against a real LLM on a real repository. All end-to-end verification remains fake-mode/unit-test based.
- The next high-leverage milestone is a live smoke run with non-fake model calls and a small local repository.

Evaluation gaps:

- SWE-bench baseline currently generates and stores a patch, but baseline oracle execution is not yet wired into `run_evaluation`.
- Framework task evaluation still treats workflow completion as resolved; it does not yet extract the workflow patch, apply it in the official SWE-bench image, and run the oracle in the main evaluation loop.
- Cost accounting in SWE-bench framework results is still placeholder until workflow reports expose the LLM usage ledger.

Lower-priority gaps:

- Human approval CLI helper for sending signals to a waiting workflow is missing.
- `destroy_workspace` runs only on normal completion; failure cleanup has no recovery path.
- Workspace uses a cloned copy, not a true git worktree from the source repository.

### 22.3a Completed This Checkpoint

All high-priority bugs, design changes, and SWE-bench report work listed below were completed in commits `1b2917d` through `2dc8e86`.

High-priority bugs:

- **`GatherContext` dispatch is broken.** When the implementation agent emits a `GATHER_CONTEXT` tool call, `_tool_from_call` returns a `GatherContext` object which routes through `command_for_tool` in `handlers.py`. That handler returns `["sh", "-lc", "printf %s <prompt>"]` — it echoes the prompt text back as stdout. The result is useless. `GATHER_CONTEXT` must be intercepted in `run_implementation_turn` before `run_tool` and dispatched to the `gather_context()` activity instead. The `GatherContext` case in `handlers.py` should be removed entirely.

- **`diff_summary` evidence check accepts any non-empty string.** `_worker_result_with_evidence` checks `bool(worker_result.diff_summary.strip())` as evidence of a real diff. Since `diff_summary` is LLM-generated text, a model can write "no changes were needed" and pass the check. The only trustworthy diff evidence is `saw_diff=True` from an actual `GitDiff` tool call that returned non-empty output.

- **Context gatherer observation accumulation is wrong.** In `gather_context`, `observations` accumulates across all turns, but `messages.append(Message(role="user", content="\n".join(observations[-3:])))` feeds only the last 3 entries from the full accumulated list, not the results from the most recent turn. The model sees a stale and incomplete picture of what just happened.

- **Silent `or` defaults in `_tool_from_call` mask post-validation bugs.** Both `implementation.py` and `context_gatherer.py` use `or ""` / `or "."` fallbacks when constructing tool objects from validated calls. Since the validator has already required these fields, a `None` value at this point is a bug — it should raise, not silently substitute a default.

Design gaps:

- **`main_workflow` ignores `NEEDS_REPLAN` status.** After each `wait_for_child`, the returned `WorkerResult.status` is never checked. If a step returns `NEEDS_REPLAN` with a `replan_suggestion`, the workflow continues with the original plan unchanged. The correct behavior: call `build_plan` again with the accumulated results and `replan_suggestion` as feedback, then continue with the remaining steps of the revised plan.

- **No context window management.** There is no mechanism to detect when an agent is approaching its context limit. When the context fills up, the LLM degrades silently — truncated history, confused reasoning. Instead: after each LLM call, check `input_tokens / model_context_limit` against a threshold (0.80). Subagents (context gatherer, implementation worker) should stop at the threshold and return a valid partial result with `status=BLOCKED` and a `replan_suggestion` describing what was done and what remains. No magic compression; explicit clean stop.

SWE-bench work:

- Baseline patch generation and JSON evaluation report fields were implemented.
- Full oracle execution for baseline and framework patches remains in the current gaps list above.

### 22.4 Completed Checkpoint Plan

Work in small commits. Keep `ruff format`, `ruff check`, and `pytest -q` green before each commit.

1. Fix `GatherContext` dispatch.

   - In `run_implementation_turn`, intercept `ToolName.GATHER_CONTEXT` before `run_tool` and call `gather_context()` activity directly.
   - Inject the returned `ContextPack` as a structured observation into the message history.
   - Remove the `GatherContext` case from `handlers.py`.
   - Add a test that a `GATHER_CONTEXT` tool call in a fake implementation turn produces a `ContextPack` observation and does not invoke `run_tool`.

2. Fix implementation evidence check.

   - Replace the `bool(worker_result.diff_summary.strip())` check in `_worker_result_with_evidence` with `evidence.saw_diff`.
   - `diff_summary` is narrative text; only `saw_diff` reflects an actual observed file change.
   - Update affected tests.

3. Fix context gatherer observation accumulation.

   - Collect observations per turn into a local list; append only that turn's observations to messages.
   - The `observations` list across the whole session can be kept for the fallback `ContextPack.relevant_snippets`.
   - Add a test that observations from turn N do not appear in the message sent at turn N-1.

4. Replace silent `or` defaults with assertions.

   - In `_tool_from_call` in both `implementation.py` and `context_gatherer.py`, replace `field or default` fallbacks with `assert field is not None` (these values were validated earlier; `None` here is a bug).
   - Add tests that confirm the validator catches missing fields before `_tool_from_call` is reached.

5. Handle `NEEDS_REPLAN` in `main_workflow`.

   - After each `wait_for_child`, check `worker_result.status`.
   - If `NEEDS_REPLAN`, call `build_plan` with the current `contract`, `repo_index`, accumulated `worker_results`, and `worker_result.replan_suggestion` as `human_feedback`.
   - Continue with the revised plan's steps from the beginning (the replanner decides which steps remain).
   - Add a fake-mode test that a `NEEDS_REPLAN` result triggers a second `build_plan` call.

6. Implement context window budget tracking.

   - Add a `context_utilization() -> float` method to `LLMClient` that returns `last_input_tokens / model_context_limit` after each call. Model context limits are defined alongside model role routing in `src/llm/config.py`.
   - In `run_implementation_turn`: after each LLM call, if utilization > 0.80, return immediately with `WorkerStatus.BLOCKED` and a `replan_suggestion` summarizing completed tool calls and what was still pending.
   - In `gather_context`: after each LLM call, if utilization > 0.80, return the best-effort `ContextPack` from observations gathered so far.
   - Add tests for the early-exit path.

7. Complete SWE-bench evaluation harness.

   - Implement baseline runner: for each eligible task, make a single LLM call with the problem statement and extract a patch.
   - Write JSON evaluation report with `resolved`, `failed`, `skipped`, `cost_per_resolved_task`, and `llm_calls_per_task`.
   - Add a five-task smoke run mode (`--subset 5`) for fast local validation.

### 22.5 Next Implementation Plan

Work in small commits. Keep `ruff format`, `ruff check`, and `pytest -q` green before each commit.

1. Wire SWE-bench oracle execution into the main evaluation loop for both framework and baseline patches.
2. Expose LLM usage ledger totals in workflow final reports so evaluation cost metrics are real instead of placeholders.
3. Run the first non-fake live smoke workflow against a small local repository and document the result.
4. Add cleanup/recovery handling so `destroy_workspace` runs after workflow failures.
5. Build a small CLI helper for sending human approval signals to waiting workflows.

### 22.6 Commands for Next Session

Start or verify Temporal-Light:

```powershell
$env:POSTGRES_USER='tl'
$env:POSTGRES_PASSWORD='changeme'
$env:API_PORT='8080'
docker compose -f Temporal-Light\docker-compose.yml -f Temporal-Light\docker-compose.override.yml up -d postgres migrate api
Invoke-WebRequest -UseBasicParsing http://localhost:8080/workflows
```

Build workspace and worker images:

```powershell
docker build -f workspace.Dockerfile -t durable-agentic-workspace:latest .
docker build -t durable-agentic-worker:latest .
```

Run local verification:

```powershell
python -m ruff format src tests
python -m ruff check src tests
python -m pytest -q
$env:RUN_DOCKER_TESTS='1'; python -m pytest tests/test_workspace_manager_integration.py -q
python -m src.eval.smoke_workflow --temporal-api-url http://localhost:8080 --temporal-database-url postgresql://tl:changeme@localhost:5432/temporal_light --workspace-image durable-agentic-workspace:latest --timeout-seconds 120
```

Run agent worker via compose with auto-reload:

```powershell
$env:TEMPORAL_DATABASE_URL='postgresql://tl:changeme@host.docker.internal:5432/temporal_light'
$env:TEMPORAL_API_URL='http://host.docker.internal:8080'
$env:WORKSPACE_IMAGE='durable-agentic-workspace:latest'
docker compose up --build agent-worker
```
