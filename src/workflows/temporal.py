from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

WorkflowFunction = TypeVar("WorkflowFunction", bound=Callable[..., Awaitable[object]])


try:
    from temporal_light import spawn_child, wait_for_child, wait_for_signal, workflow
except ImportError:

    def workflow(workflow_function: WorkflowFunction) -> WorkflowFunction:
        return workflow_function

    async def spawn_child(workflow_name: str, **keyword_arguments: object) -> str:
        raise RuntimeError(f"Temporal-Light is not installed; cannot spawn {workflow_name}.")

    async def wait_for_child(child_id: str) -> object:
        raise RuntimeError(f"Temporal-Light is not installed; cannot wait for {child_id}.")

    async def wait_for_signal(signal_type: str) -> object:
        raise RuntimeError(f"Temporal-Light is not installed; cannot wait for {signal_type}.")
