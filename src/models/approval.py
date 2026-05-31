from __future__ import annotations

from pydantic import model_validator

from src.models.frozen_base_model import FrozenBaseModel
from src.runtime_enums import StrEnum


class ApprovalDecision(StrEnum):
    APPROVE = 'approve'
    REVISE = 'revise'


class HumanApprovalSignal(FrozenBaseModel):
    decision: ApprovalDecision
    feedback: str | None = None

    @model_validator(mode='after')
    def validate_revision_feedback(self) -> HumanApprovalSignal:
        if self.decision == ApprovalDecision.REVISE and not self.feedback:
            raise ValueError('feedback is required when decision is revise')
        return self
