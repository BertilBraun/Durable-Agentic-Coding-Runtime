from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.runtime_enums import StrEnum


class ReviewDecision(StrEnum):
    ACCEPT = 'accept'
    REVISE = 'revise'
    REJECT = 'reject'
    NEEDS_HUMAN = 'needs_human'


# TODO keep the models close to where they are used! i.e. the review verdict should be defined in the reviewer activity file, not here in the models directory which should be reserved for more general purpose models that are used across multiple activities
# Same for a lot of the other models that are only used in a single activity, they should be defined in the activity file, not here, f.e. approval, etc

# TODO !!Important!! Always have reasoning / evidence fields BEFORE any final decision fields in the model - only this way can the LLM reason about the decision it is making before outputting the final decision, if the decision field comes first the LLM will be biased to fill that out first and then struggle to come up with reasoning / evidence after the fact


class ReviewVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    verdict: ReviewDecision
    blocking_issues: list[str] = Field(default_factory=list)
    non_blocking_issues: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    missing_tests: list[str] = Field(default_factory=list)
    regression_risks: list[str] = Field(default_factory=list)
    minimality_assessment: str
    recommended_next_action: str
