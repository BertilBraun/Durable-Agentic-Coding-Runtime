from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict

from src.runtime_enums import StrEnum

DEFAULT_CONTEXT_LIMIT_TOKENS = 200_000
CONTEXT_UTILIZATION_STOP_THRESHOLD = 0.80

# TODO collect all settings for this project here - don't distribute the settings througout the files - that way we have a single source of truth for all settings, and it's easier to manage and update them - right now we have some settings in env vars, some hardcoded in code, some in json files, it's a mess - we should unify this and have a clear structure for how settings are defined and loaded in this project


class ModelRole(StrEnum):
    CONTRACT_BUILDER = 'contract_builder'
    PLANNER = 'planner'
    COMPLEXITY_ASSESSOR = 'complexity_assessor'
    CONTEXT_GATHERER = 'context_gatherer'
    IMPLEMENTATION = 'implementation'
    REVIEWER = 'reviewer'
    SUMMARIZER = 'summarizer'


# TODO Have a model definition - with context limits, costs, model name/id, etc - and load that from a json/db rather than hardcoding it in code and env vars, that way we can easily add new models and update context limits and costs without changing code
class ModelContextLimit(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    context_limit_tokens: int


class ModelConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract_builder_model: str
    planner_model: str
    complexity_assessor_model: str
    context_gatherer_model: str
    implementation_model: str
    reviewer_model: str
    summarizer_model: str
    model_context_limits: list[ModelContextLimit]

    def model_for_role(self, role: ModelRole) -> str:
        match role:
            case ModelRole.CONTRACT_BUILDER:
                return self.contract_builder_model
            case ModelRole.PLANNER:
                return self.planner_model
            case ModelRole.COMPLEXITY_ASSESSOR:
                return self.complexity_assessor_model
            case ModelRole.CONTEXT_GATHERER:
                return self.context_gatherer_model
            case ModelRole.IMPLEMENTATION:
                return self.implementation_model
            case ModelRole.REVIEWER:
                return self.reviewer_model
            case ModelRole.SUMMARIZER:
                return self.summarizer_model
            case _:
                raise AssertionError(f'No model configured for role: {role}')

    def context_limit_for_role(self, role: ModelRole) -> int:
        model = self.model_for_role(role)
        for model_context_limit in self.model_context_limits:
            if model_context_limit.model == model:
                return model_context_limit.context_limit_tokens
        raise ValueError(f'Context limit is required for model: {model}')


def load_model_configuration() -> ModelConfiguration:
    # TODO these should be required env args, no defaults, and we should validate that the models are valid/available before starting the service - i.e. verify, that they are in the json/db mentioned below.
    contract_builder_model = os.getenv('MODEL_CONTRACT_BUILDER', 'claude-opus-4-7')
    planner_model = os.getenv('MODEL_PLANNER', 'claude-opus-4-7')
    complexity_assessor_model = os.getenv('MODEL_COMPLEXITY_ASSESSOR', 'claude-opus-4-7')
    context_gatherer_model = os.getenv(
        'MODEL_CONTEXT_GATHERER',
        'claude-haiku-4-5-20251001',
    )
    implementation_model = os.getenv('MODEL_IMPLEMENTATION', 'claude-sonnet-4-6')
    reviewer_model = os.getenv('MODEL_REVIEWER', 'claude-sonnet-4-6')
    summarizer_model = os.getenv('MODEL_SUMMARIZER', 'claude-haiku-4-5-20251001')
    return ModelConfiguration(
        contract_builder_model=contract_builder_model,
        planner_model=planner_model,
        complexity_assessor_model=complexity_assessor_model,
        context_gatherer_model=context_gatherer_model,
        implementation_model=implementation_model,
        reviewer_model=reviewer_model,
        summarizer_model=summarizer_model,
        model_context_limits=_context_limits_for_models(
            [
                contract_builder_model,
                planner_model,
                complexity_assessor_model,
                context_gatherer_model,
                implementation_model,
                reviewer_model,
                summarizer_model,
            ]
        ),
    )


def _context_limits_for_models(models: list[str]) -> list[ModelContextLimit]:
    return [
        # TODO we need a json/db where we define the models, their context limits, costs, and other relevant metadata, rather than hardcoding this in code.
        ModelContextLimit(model=model, context_limit_tokens=DEFAULT_CONTEXT_LIMIT_TOKENS)
        for model in sorted(set(models))
    ]
