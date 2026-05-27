from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict

from src.runtime_enums import StrEnum


class ModelRole(StrEnum):
    CONTRACT_BUILDER = "contract_builder"
    PLANNER = "planner"
    COMPLEXITY_ASSESSOR = "complexity_assessor"
    CONTEXT_GATHERER = "context_gatherer"
    IMPLEMENTATION = "implementation"
    REVIEWER = "reviewer"
    SUMMARIZER = "summarizer"


class ModelConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract_builder_model: str
    planner_model: str
    complexity_assessor_model: str
    context_gatherer_model: str
    implementation_model: str
    reviewer_model: str
    summarizer_model: str

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


def load_model_configuration() -> ModelConfiguration:
    return ModelConfiguration(
        contract_builder_model=os.getenv("MODEL_CONTRACT_BUILDER", "claude-opus-4-7"),
        planner_model=os.getenv("MODEL_PLANNER", "claude-opus-4-7"),
        complexity_assessor_model=os.getenv("MODEL_COMPLEXITY_ASSESSOR", "claude-opus-4-7"),
        context_gatherer_model=os.getenv(
            "MODEL_CONTEXT_GATHERER",
            "claude-haiku-4-5-20251001",
        ),
        implementation_model=os.getenv("MODEL_IMPLEMENTATION", "claude-sonnet-4-6"),
        reviewer_model=os.getenv("MODEL_REVIEWER", "claude-sonnet-4-6"),
        summarizer_model=os.getenv("MODEL_SUMMARIZER", "claude-haiku-4-5-20251001"),
    )
