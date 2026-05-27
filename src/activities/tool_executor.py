from __future__ import annotations

from src.activities.temporal import durable_activity
from src.activities.workspace_manager import ToolExecutionRequest, ToolResult, run_tool


@durable_activity(retries=0, timeout=300)
async def execute_tool(request: ToolExecutionRequest) -> ToolResult:
    return await run_tool(request)
