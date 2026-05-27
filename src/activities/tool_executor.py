from __future__ import annotations

from src.activities.temporal import durable_activity
from src.activities.workspace_manager import ToolResult, WorkspaceInfo, run_tool
from src.tools.definitions import Tool


@durable_activity(retries=0, timeout=300)
async def execute_tool(workspace_info: WorkspaceInfo, tool: Tool) -> ToolResult:
    return await run_tool(workspace_info, tool)
