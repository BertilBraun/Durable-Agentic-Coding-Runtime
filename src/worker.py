from __future__ import annotations

import os

from temporal_light import Worker

from src.activities.complexity_assessor import assess_complexity
from src.activities.context_gatherer import gather_context
from src.activities.contract_builder import build_contract
from src.activities.human_approval import present_plan_to_human
from src.activities.implementation import generate_implementation_agent_turn, get_full_diff
from src.activities.planner import build_plan
from src.activities.repo_indexer import build_repo_index
from src.activities.report_builder import (
    build_final_report,
    collect_llm_usage_summary,
    reset_llm_usage_summary,
)
from src.activities.reviewer import review_patch
from src.activities.workspace_manager import create_workspace, destroy_workspace, run_tool
from src.workflows.implementation_workflow import implementation_workflow
from src.workflows.main_workflow import main_workflow


def main() -> None:
    database_url = os.environ['TEMPORAL_DATABASE_URL']
    worker = Worker(
        database_url=database_url,
        workflow_functions=[main_workflow, implementation_workflow],
        activity_functions=[
            assess_complexity,
            build_contract,
            build_final_report,
            build_plan,
            build_repo_index,
            collect_llm_usage_summary,
            create_workspace,
            destroy_workspace,
            gather_context,
            generate_implementation_agent_turn,
            get_full_diff,
            present_plan_to_human,
            review_patch,
            reset_llm_usage_summary,
            run_tool,
        ],
    )
    worker.run()


if __name__ == '__main__':
    main()
