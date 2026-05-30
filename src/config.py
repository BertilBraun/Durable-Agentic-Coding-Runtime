from __future__ import annotations

import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from src.runtime_enums import StrEnum

MODELS_CSV_PATH = Path(__file__).parent / 'llm' / 'models.csv'

load_dotenv()


class ModelRole(StrEnum):
    CONTRACT_BUILDER = 'contract_builder'
    PLANNER = 'planner'
    COMPLEXITY_ASSESSOR = 'complexity_assessor'
    CONTEXT_GATHERER = 'context_gatherer'
    IMPLEMENTATION = 'implementation'
    REVIEWER = 'reviewer'
    SUMMARIZER = 'summarizer'


class ModelEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    context_limit_tokens: int
    input_price_usd_per_mtok: float
    output_price_usd_per_mtok: float
    cache_read_price_usd_per_mtok: float


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    llm_api_key: str | None
    llm_base_url: str | None

    workspace_root: str
    workspace_image: str
    artifacts_root: str

    implementation_max_tool_rounds: int
    context_gatherer_max_tool_calls: int
    context_utilization_stop_threshold: float
    context_utilization_hard_stop_threshold: float

    tool_output_max_characters: int
    tool_output_compact_head_characters: int
    tool_output_compact_tail_characters: int

    models_by_id: dict[str, ModelEntry]
    model_by_role: dict[ModelRole, str]

    def model_for_role(self, role: ModelRole) -> str:
        return self.model_by_role[role]

    def context_limit_for_role(self, role: ModelRole) -> int:
        return self.model_entry(self.model_for_role(role)).context_limit_tokens

    def model_entry(self, model_id: str) -> ModelEntry:
        if model_id not in self.models_by_id:
            raise ValueError(f'Unknown model id: {model_id!r}. Add it to {MODELS_CSV_PATH}.')
        return self.models_by_id[model_id]


def load_settings() -> Settings:
    models_by_id = _load_models_csv(MODELS_CSV_PATH)
    model_by_role = _load_model_role_bindings(models_by_id)

    return Settings(
        llm_api_key=os.getenv('LLM_API_KEY'),
        llm_base_url=os.getenv('LLM_BASE_URL'),
        workspace_root=os.getenv('WORKSPACE_ROOT', '.agentic-workspaces'),
        workspace_image=os.getenv('WORKSPACE_IMAGE', 'durable-agentic-workspace:latest'),
        artifacts_root=os.getenv('ARTIFACTS_ROOT', '.agentic-artifacts'),
        implementation_max_tool_rounds=int(os.getenv('IMPLEMENTATION_MAX_TOOL_ROUNDS', '12')),
        context_gatherer_max_tool_calls=int(os.getenv('CONTEXT_GATHERER_MAX_TOOL_CALLS', '10')),
        context_utilization_stop_threshold=float(
            os.getenv('CONTEXT_UTILIZATION_STOP_THRESHOLD', '0.80')
        ),
        context_utilization_hard_stop_threshold=float(
            os.getenv('CONTEXT_UTILIZATION_HARD_STOP_THRESHOLD', '0.95')
        ),
        tool_output_max_characters=int(os.getenv('TOOL_OUTPUT_MAX_CHARACTERS', '20000')),
        tool_output_compact_head_characters=int(
            os.getenv('TOOL_OUTPUT_COMPACT_HEAD_CHARACTERS', '8000')
        ),
        tool_output_compact_tail_characters=int(
            os.getenv('TOOL_OUTPUT_COMPACT_TAIL_CHARACTERS', '4000')
        ),
        models_by_id=models_by_id,
        model_by_role=model_by_role,
    )


def _load_models_csv(path: Path) -> dict[str, ModelEntry]:
    models_by_id: dict[str, ModelEntry] = {}
    with path.open(encoding='utf-8', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            entry = ModelEntry(
                id=row['id'],
                context_limit_tokens=int(row['context_limit_tokens']),
                input_price_usd_per_mtok=float(row['input_price_usd_per_mtok']),
                output_price_usd_per_mtok=float(row['output_price_usd_per_mtok']),
                cache_read_price_usd_per_mtok=float(row['cache_read_price_usd_per_mtok']),
            )
            models_by_id[entry.id] = entry
    return models_by_id


def _load_model_role_bindings(models_by_id: dict[str, ModelEntry]) -> dict[ModelRole, str]:
    bindings: dict[ModelRole, tuple[str, str]] = {
        ModelRole.CONTRACT_BUILDER: ('MODEL_CONTRACT_BUILDER', 'claude-opus-4-7'),
        ModelRole.PLANNER: ('MODEL_PLANNER', 'claude-opus-4-7'),
        ModelRole.COMPLEXITY_ASSESSOR: ('MODEL_COMPLEXITY_ASSESSOR', 'claude-opus-4-7'),
        ModelRole.CONTEXT_GATHERER: ('MODEL_CONTEXT_GATHERER', 'claude-haiku-4-5-20251001'),
        ModelRole.IMPLEMENTATION: ('MODEL_IMPLEMENTATION', 'claude-sonnet-4-6'),
        ModelRole.REVIEWER: ('MODEL_REVIEWER', 'claude-sonnet-4-6'),
        ModelRole.SUMMARIZER: ('MODEL_SUMMARIZER', 'claude-haiku-4-5-20251001'),
    }
    resolved: dict[ModelRole, str] = {}
    for role, (env_name, default_model_id) in bindings.items():
        model_id = os.getenv(env_name, default_model_id)
        if model_id not in models_by_id:
            raise ValueError(
                f'Model {model_id!r} for role {role.value!r} is not registered in '
                f'{MODELS_CSV_PATH}. Add it to the CSV or set {env_name} to a known id.'
            )
        resolved[role] = model_id
    return resolved


CONFIG = load_settings()
