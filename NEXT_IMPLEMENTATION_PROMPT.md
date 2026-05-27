# Prompt for Next Chat: Continue Durable Agentic Coding Runtime

You are continuing implementation of the **Durable Agentic Coding Runtime** in:

```text
c:\Projects\Agentic-Coding
```

Planning and approval are complete. Read these files first:

1. `PLAN.md`, especially Sections 21 and 22.
2. `CODING_STANDARDS.md`.
3. Current implementation files under `src/`.
4. Temporal-Light public behavior in `Temporal-Light/README.md`; do not import Temporal-Light internals except through the installed public package (`temporal_light`).

Current latest commit at handoff:

```text
01dc851 Add auto-reloading for agent worker in development
438368a Add Temporal workflow smoke runner
944372d Add deterministic LLM smoke mode
07c742b Require tree-sitter for repo indexing
9f2b2dc Initial durable agent runtime scaffold
```

Important current facts:

- `Temporal-Light` is a Git submodule.
- Agent worker auto-reloads in development via `docker-compose.override.yml`.
- Tree-sitter is a required dependency. Do not add AST/regex fallback parsing.
- All structured data crossing boundaries must be Pydantic models/dataclasses/NamedTuples, not raw dicts.
- Full type annotations are mandatory.
- Use `StrEnum`/enums for fixed value sets.
- Use `match/case` where dispatching by type/value.
- Every LLM call must go through `src/llm/client.py`.
- Every command/tool execution must go through `WorkspaceManager.run_tool`.
- Keep changes small and commit after each completed part.

Already implemented:

- Project scaffold, models, LLM client/config, tool definitions/handlers.
- Docker workspace lifecycle and fresh-container tool execution.
- Tree-sitter repository indexing for Python and JS/TS/TSX/JSX.
- Contract, planner, complexity, context gatherer, implementation, review, report, human approval activities.
- `main_workflow` and `implementation_workflow`.
- Deterministic `LLM_FAKE_MODE=1` for smoke workflow testing.
- `src/eval/smoke_workflow.py`, which runs a live Temporal-Light smoke workflow.
- Initial `swe_bench.py` scaffold, not complete.

Verification that has passed:

```powershell
python -m ruff check src tests
python -m pytest -q
$env:RUN_DOCKER_TESTS='1'; python -m pytest tests/test_workspace_manager_integration.py -q
docker build -f workspace.Dockerfile -t durable-agentic-workspace:latest .
docker build -t durable-agentic-worker:latest .
docker run --rm durable-agentic-worker:latest python -m ruff check /app/src
python -m src.eval.smoke_workflow --temporal-api-url http://localhost:8080 --temporal-database-url postgresql://tl:changeme@localhost:5432/temporal_light --workspace-image durable-agentic-workspace:latest --timeout-seconds 120
```

Temporal-Light startup commands:

```powershell
$env:POSTGRES_USER='tl'
$env:POSTGRES_PASSWORD='changeme'
$env:API_PORT='8080'
docker compose -f Temporal-Light\docker-compose.yml -f Temporal-Light\docker-compose.override.yml up -d postgres migrate api
Invoke-WebRequest -UseBasicParsing http://localhost:8080/workflows
```

Agent compose startup with auto-reload:

```powershell
$env:TEMPORAL_DATABASE_URL='postgresql://tl:changeme@host.docker.internal:5432/temporal_light'
$env:TEMPORAL_API_URL='http://host.docker.internal:8080'
$env:WORKSPACE_IMAGE='durable-agentic-workspace:latest'
docker compose up --build agent-worker
```

Next implementation priorities, in order:

1. Harden tool schemas and dispatch.

   - Replace implementation/context gatherer `tool_name: str` with `ToolName`.
   - Add typed Pydantic tool-call models for each tool-call payload crossing LLM/activity boundaries.
   - Validate workspace-relative paths and reject absolute paths or `..` traversal.
   - Add tests first for invalid paths, unknown tools, and timeout propagation.
   - Commit this slice.

2. Make symbol tools use `RepoIndex`.

   - Add typed `ToolExecutionRequest` with `WorkspaceInfo`, optional `RepoIndex`, and tool.
   - Implement `FindSymbol` and `FindReferences` from indexed data where possible.
   - Use text scanning only where reference lookup requires reading files.
   - Add Python and TSX tests.
   - Commit this slice.

3. Add artifact storage for large outputs.

   - Write large stdout/stderr/diff/test logs to `/artifacts/<run_id>/...`.
   - Return compact `ArtifactReference` values.
   - Keep Temporal-Light event payloads compact.
   - Add tests for large-output artifact creation.
   - Commit this slice.

4. Improve implementation evidence.

   - Convert `RunTests` outputs into `TestResult`.
   - Require success to include diff/test evidence unless the step is explicitly no-op.
   - Run step review inside `implementation_workflow`.
   - Add tests for success, failed test, blocked, and needs-replan paths.
   - Commit this slice.

5. Upgrade the smoke workflow from no-op to real patch.

   - Extend fake mode to write a tiny patch and run a test.
   - Smoke workflow must verify changed diff and test result.
   - Commit this slice.

6. Continue `src/eval/swe_bench.py`.

   - Implement official image pull/start, patch extraction/application, oracle execution, baseline execution, and results JSON.
   - Add a five-task filtered subset mode.
   - Commit this slice.

Rules for the next chat:

- Use TDD for behavior changes: write failing tests first, watch them fail, then implement.
- Commit after each completed part with a clear message.
- Run `ruff format`, `ruff check`, and relevant tests before every commit.
- Before claiming completion, run fresh verification and report exact commands/results.
- Do not revert user changes.
