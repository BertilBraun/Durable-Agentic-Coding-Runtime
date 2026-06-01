from __future__ import annotations

import csv
import os
from pathlib import Path

from dotenv import load_dotenv

from src.models.frozen_base_model import FrozenBaseModel
from src.runtime_enums import StrEnum

MODELS_CSV_PATH = Path(__file__).parent / 'llm' / 'models.csv'

load_dotenv()


class ModelRole(StrEnum):
    CONTRACT_BUILDER = 'contract_builder'
    PLANNER = 'planner'
    PLAN_REVIEWER = 'plan_reviewer'
    COMPLEXITY_ASSESSOR = 'complexity_assessor'
    CONTEXT_GATHERER = 'context_gatherer'
    REPRODUCER = 'reproducer'
    IMPLEMENTATION = 'implementation'
    REVIEWER = 'reviewer'
    SUMMARIZER = 'summarizer'


class ReasoningEffort(StrEnum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    XHIGH = 'xhigh'


class ModelEntry(FrozenBaseModel):
    id: str
    context_limit_tokens: int
    input_price_usd_per_mtok: float
    output_price_usd_per_mtok: float
    cache_read_price_usd_per_mtok: float


class Settings(FrozenBaseModel):
    llm_api_key: str | None
    llm_base_url: str | None

    artifacts_root: str
    human_approval_enabled: bool
    cleanup_candidate_branches: bool

    implementation_max_tool_rounds: int
    reproducer_max_tool_rounds: int
    max_replan_attempts: int
    max_plan_review_rounds: int
    context_gatherer_max_tool_calls: int
    context_utilization_stop_threshold: float
    context_utilization_hard_stop_threshold: float

    tool_output_max_characters: int
    tool_output_compact_head_characters: int
    tool_output_compact_tail_characters: int

    context_pack_max_characters: int

    models_by_id: dict[str, ModelEntry]
    model_by_role: dict[ModelRole, str]
    reasoning_effort_by_role: dict[ModelRole, ReasoningEffort | None]

    def model_for_role(self, role: ModelRole) -> str:
        return self.model_by_role[role]

    def reasoning_effort_for_role(self, role: ModelRole) -> ReasoningEffort | None:
        return self.reasoning_effort_by_role[role]

    def context_limit_for_role(self, role: ModelRole) -> int:
        return self.model_entry(self.model_for_role(role)).context_limit_tokens

    def model_entry(self, model_id: str) -> ModelEntry:
        if model_id not in self.models_by_id:
            raise ValueError(f'Unknown model id: {model_id!r}. Add it to {MODELS_CSV_PATH}.')
        return self.models_by_id[model_id]


def load_settings() -> Settings:
    models_by_id = _load_models_csv(MODELS_CSV_PATH)
    model_by_role = _load_model_role_bindings(models_by_id)
    reasoning_effort_by_role = _load_reasoning_efforts()

    return Settings(
        llm_api_key=os.getenv('LLM_API_KEY'),
        llm_base_url=os.getenv('LLM_BASE_URL'),
        artifacts_root=os.getenv('ARTIFACTS_ROOT', '.agentic-artifacts'),
        human_approval_enabled=_parse_bool(os.getenv('HUMAN_APPROVAL_ENABLED'), default=True),
        cleanup_candidate_branches=_parse_bool(
            os.getenv('CLEANUP_CANDIDATE_BRANCHES'), default=False
        ),
        implementation_max_tool_rounds=int(os.getenv('IMPLEMENTATION_MAX_TOOL_ROUNDS', '12')),
        reproducer_max_tool_rounds=int(os.getenv('REPRODUCER_MAX_TOOL_ROUNDS', '12')),
        max_replan_attempts=int(os.getenv('MAX_REPLAN_ATTEMPTS', '3')),
        max_plan_review_rounds=int(os.getenv('MAX_PLAN_REVIEW_ROUNDS', '2')),
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
        context_pack_max_characters=int(os.getenv('CONTEXT_PACK_MAX_CHARACTERS', '16000')),
        models_by_id=models_by_id,
        model_by_role=model_by_role,
        reasoning_effort_by_role=reasoning_effort_by_role,
    )


def _parse_bool(raw_value: str | None, default: bool) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {'1', 'true', 'yes', 'on'}


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
        ModelRole.PLAN_REVIEWER: ('MODEL_PLAN_REVIEWER', 'claude-opus-4-7'),
        ModelRole.COMPLEXITY_ASSESSOR: ('MODEL_COMPLEXITY_ASSESSOR', 'claude-opus-4-7'),
        ModelRole.CONTEXT_GATHERER: ('MODEL_CONTEXT_GATHERER', 'claude-haiku-4-5-20251001'),
        ModelRole.REPRODUCER: ('MODEL_REPRODUCER', 'claude-sonnet-4-6'),
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


def _load_reasoning_efforts() -> dict[ModelRole, ReasoningEffort | None]:
    default_effort_by_role: dict[ModelRole, ReasoningEffort] = {
        ModelRole.CONTRACT_BUILDER: ReasoningEffort.HIGH,
        ModelRole.PLANNER: ReasoningEffort.HIGH,
        ModelRole.PLAN_REVIEWER: ReasoningEffort.HIGH,
    }
    efforts: dict[ModelRole, ReasoningEffort | None] = {}
    for role in ModelRole:
        raw_value = os.getenv(f'REASONING_EFFORT_{role.name}')
        if raw_value is None:
            efforts[role] = default_effort_by_role.get(role)
        else:
            efforts[role] = _parse_reasoning_effort(raw_value)
    return efforts


def _parse_reasoning_effort(raw_value: str) -> ReasoningEffort | None:
    normalized = raw_value.strip().lower()
    if normalized in {'', 'off', 'none'}:
        return None
    return ReasoningEffort(normalized)


CONFIG = load_settings()
