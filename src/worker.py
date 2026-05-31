from __future__ import annotations

import os

from temporal_light import Worker

from src.activities.human_approval import present_plan_to_human
from src.activities.implementation import get_full_diff
from src.activities.repo_indexer import build_repo_index
from src.activities.workspace_manager import (
    begin_candidate,
    finalize_winner,
    run_tool,
    setup_environment,
    teardown_environment,
)
from src.llm.client import generate_completion, generate_structured_completion
from src.workflows.implementation_workflow import implementation_workflow
from src.workflows.main_workflow import main_workflow


def main() -> None:
    database_url = os.environ['TEMPORAL_DATABASE_URL']
    worker = Worker(
        database_url=database_url,
        workflow_functions=[main_workflow, implementation_workflow],
        activity_functions=[
            begin_candidate,
            build_repo_index,
            finalize_winner,
            generate_completion,
            generate_structured_completion,
            get_full_diff,
            present_plan_to_human,
            run_tool,
            setup_environment,
            teardown_environment,
        ],
    )
    worker.run()


if __name__ == '__main__':
    main()
