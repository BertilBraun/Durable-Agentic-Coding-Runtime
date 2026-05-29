from __future__ import annotations

import os

from temporal_light import Worker

from src.activities.human_approval import present_plan_to_human
from src.activities.implementation import get_full_diff
from src.activities.repo_indexer import build_repo_index
from src.activities.report_builder import (
    collect_llm_usage_summary,
    reset_llm_usage_summary,
)
from src.activities.workspace_manager import create_workspace, destroy_workspace, run_tool
from src.llm.client import generate_completion, generate_structured_completion
from src.workflows.implementation_workflow import implementation_workflow
from src.workflows.main_workflow import main_workflow


def main() -> None:
    database_url = os.environ['TEMPORAL_DATABASE_URL']
    worker = Worker(
        database_url=database_url,
        workflow_functions=[main_workflow, implementation_workflow],
        activity_functions=[
            build_repo_index,
            collect_llm_usage_summary,
            create_workspace,
            destroy_workspace,
            generate_completion,
            generate_structured_completion,
            get_full_diff,
            present_plan_to_human,
            reset_llm_usage_summary,
            run_tool,
        ],
    )
    worker.run()


if __name__ == '__main__':
    main()
