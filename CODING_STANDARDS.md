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

## Trust Typed Provider APIs

Use precise third-party SDK types in protocols, helper signatures, and return values. Let those types remove impossible branches.

```python
# correct
from openai.types.chat import ChatCompletion, ParsedChatCompletion

def _extract_parsed(response: ParsedChatCompletion[T]) -> T: ...

# wrong
def _extract_parsed(response: object, output_type: type[T]) -> T:
    if isinstance(response, dict):
        ...
```

Do not add broad `object`, `Any`, `dict`, `getattr`, or `isinstance` fallbacks for shapes that the SDK type already guarantees. If a provider violates its declared type, let the boundary fail clearly instead of turning the client into a permissive parser.

---

## Test Doubles Stay Out of Production Code

Production modules must not contain fake-mode branches, canned LLM responses, fake clients, or test fixtures. Use dependency injection and test-local fakes instead.

```python
# correct
llm_client = LLMClient(async_openai_client=fake_openai_client)

# wrong
if os.getenv("LLM_FAKE_MODE") == "1":
    return '{"status":"success"}'
```

Temporary smoke harnesses may have explicit fake implementations, but they must live under `tests/`, `examples/`, or an evaluation harness module, not in the production client.

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

### LLM Structured Outputs

- Model LLM responses with Pydantic models, not raw dictionaries or nullable catch-all payloads.
- Add `Field(description=...)` to LLM-facing fields when the model needs parameter guidance.
- Prefer tagged unions for tool-call variants so each tool exposes only its own parameters.
- Do not expose irrelevant nullable fields to the model.

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

## No Dynamic Attribute Access - EVER

Never use `getattr`, `setattr`, `hasattr`, or any form of dynamic attribute access. Always use explicit attributes defined on a class or dataclass. If you find yourself needing dynamic attribute access, it's a sign that your data structure should be redesigned.

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

## Imports

All imports go at the top of the module. Never import inside a function, a `try/except` block, or a `match` case.

Use the imported type in annotations — never fall back to `object` when the real type is available.

```python
# correct
import docker

def _docker_client() -> docker.DockerClient:
    return docker.from_env()

# wrong
def _docker_client() -> object:
    import docker
    return docker.from_env()
```

---

## match/case over isinstance chains

Use `match/case` for branching on types instead of `isinstance` chains. This is more concise, readable, and extensible. The match statement also provides exhaustiveness checking, so if a new type is added to the union, you'll get a warning to handle it. But - keep in mind, that this should also only be used sparingly, when the design genuinely calls for branching on multiple types. If you find yourself needing to use `match/case` frequently, it may be a sign that your design could be improved by using polymorphism or a more explicit data structure instead of a union of many types.

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
