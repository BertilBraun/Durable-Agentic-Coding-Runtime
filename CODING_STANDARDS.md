# Coding Standards

## Type Annotations

All functions must have full type annotations on parameters and return types. No exceptions.

```python
# correct
async def build_contract(request: TaskRequest) -> TaskContract: ...

# wrong
async def build_contract(request, contract_type="bugfix"): ...
```

### Generic structured output

When a function produces a type determined by a caller-supplied type, use a TypeVar:

```python
from typing import TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

async def generate_structured(
    self,
    messages: list[Message],
    output_type: type[T],
) -> T:
    ...
```

The return type is always the concrete type the caller passed in — never `BaseModel` or `Any`.

---

## No String Keys

Never use raw dicts or string key access to pass structured data between functions. Always use a dataclass, Pydantic model, or NamedTuple.

```python
# correct
@dataclass
class WorkerResult:
    status: WorkerStatus
    patch_id: str
    diff_summary: str
    tests_run: list[str]
    test_results: list[TestResult]
    discovered_issues: list[str]
    confidence: Confidence
    replan_suggestion: str | None

# wrong
result = {
    "status": "success",
    "patch_id": "abc",
    ...
}
```

This applies to: activity inputs/outputs, workflow inputs/outputs, LLM structured outputs, tool inputs/outputs, config, event payloads.

---

## Enums over Literal Strings

Use `enum.Enum` (or `enum.StrEnum` for serialization compatibility) for any value that belongs to a fixed set.

```python
class WorkerStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    NEEDS_REPLAN = "needs_replan"

class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
```

---

## Dataclasses and Pydantic

- Use `@dataclass` for internal, non-serialized data structures.
- Use `pydantic.BaseModel` for anything that crosses a serialization boundary: LLM structured outputs, Temporal-Light activity inputs/outputs, HTTP payloads.
- All Pydantic models use `model_config = ConfigDict(frozen=True)` unless mutation is explicitly needed.

---

## No Abbreviations

Full descriptive names everywhere. No `ctx`, `cfg`, `req`, `res`, `msg`, `impl`, `fn`, `cb`, `tmp`, `val`.

```python
# correct
workflow_context, model_configuration, task_request, llm_response

# wrong
ctx, cfg, req, res
```

---

## No silent defaults

Never use default parameter values that could hide bugs. All parameters must be explicitly passed by the caller - do not rely on defaults to fill in missing values.

```python
# correct
def create_task(request: TaskRequest) -> TaskContract: ...
# wrong
def create_task(request: TaskRequest, contract_type="bugfix") -> TaskContract: ...

#correct
important_value = config.get("important_value")
if important_value is None:
    raise ValueError("important_value is required in config")
# wrong
important_value = config.get("important_value", "default_value")
```

---

## Dependencies

Once dependencies are added, they may be assumed to be present and do not require defensive checks or fallbacks.

```python
# correct
from httpx import AsyncClient

# wrong
try:
    import httpx
except ImportError:
    class AsyncClient:
        def __init__(*args, **kwargs):
            raise ImportError("httpx is required for AsyncClient")
```

---

## match/case over isinstance chains

```python
# correct
match node:
    case FunctionDefinition(name=name):
        ...
    case ClassDefinition(name=name):
        ...

# wrong
if isinstance(node, FunctionDefinition):
    ...
elif isinstance(node, ClassDefinition):
    ...
```

---

## Error Handling

- `assert` for internal invariants that signal bugs.
- `raise ValueError` for invalid inputs at system boundaries.
- No defensive checks for things that cannot happen given the type system.

---

## Comments

No comments unless the WHY is non-obvious. Never describe what the code does. One line maximum.

---

## Formatting

Run `ruff format` and `ruff check --fix` before any commit. All warnings must be resolved.

---

## Testing

- Parametrize similar cases with `@pytest.mark.parametrize`.
- Integration tests that require external services (Docker, Temporal-Light, Postgres) are marked `@pytest.mark.integration` and skipped unless the relevant env vars are set.

---

## Python Version

Target Python 3.10+. Use `match/case`, `X | Y` union syntax.
