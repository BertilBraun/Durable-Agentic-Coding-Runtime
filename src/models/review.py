from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.runtime_enums import StrEnum


class ReviewDecision(StrEnum):
    ACCEPT = "accept"
    REVISE = "revise"
    REJECT = "reject"
    NEEDS_HUMAN = "needs_human"


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
