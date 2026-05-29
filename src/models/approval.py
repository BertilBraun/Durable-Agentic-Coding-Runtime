from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from src.runtime_enums import StrEnum


class ApprovalDecision(StrEnum):
    APPROVE = 'approve'
    REVISE = 'revise'


class HumanApprovalSignal(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: ApprovalDecision
    feedback: str | None = None

    @model_validator(mode='after')
    def validate_revision_feedback(self) -> HumanApprovalSignal:
        if self.decision == ApprovalDecision.REVISE and not self.feedback:
            raise ValueError('feedback is required when decision is revise')
        return self


class ComplexityVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    reasoning: str
    requires_human_approval: bool
