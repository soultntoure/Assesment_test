from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.solution.constants import (
    NARRATIVE_STRENGTH_VALUES,
    SENTIMENT_VALUES,
    TEAM_VALUES,
    TOPIC_VALUES,
    URGENCY_VALUES,
)


def _validate_controlled_value(value: str, allowed: tuple[str, ...], field_name: str) -> str:
    if value not in allowed:
        allowed_values = ", ".join(allowed)
        raise ValueError(f"{field_name} must be one of: {allowed_values}")
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RawPost(StrictModel):
    id: str
    platform: str
    text: str
    timestamp: str
    engagement: dict[str, int | float] = Field(default_factory=dict)


class PreprocessedPost(StrictModel):
    post_id: str
    platform: str | None = None
    timestamp: str | None = None
    engagement: dict[str, int | float] = Field(default_factory=dict)
    original_text: str
    text_for_classification: str
    original_language: str
    translated: bool


class ClassifiedPost(StrictModel):
    post_id: str
    sentiment: str
    topic: str
    urgency: str
    contains_legal_threat: bool = False
    contains_competitor_mention: bool = False
    original_language: str
    translated: bool

    @field_validator("sentiment")
    @classmethod
    def sentiment_must_be_controlled(cls, value: str) -> str:
        return _validate_controlled_value(value, SENTIMENT_VALUES, "sentiment")

    @field_validator("topic")
    @classmethod
    def topic_must_be_controlled(cls, value: str) -> str:
        return _validate_controlled_value(value, TOPIC_VALUES, "topic")

    @field_validator("urgency")
    @classmethod
    def urgency_must_be_controlled(cls, value: str) -> str:
        return _validate_controlled_value(value, URGENCY_VALUES, "urgency")


class Narrative(StrictModel):
    narrative_id: str
    title: str
    supporting_post_ids: list[str]
    narrative_strength: str
    estimated_hours_until_trending: int = Field(ge=0)
    recommended_action: str

    @field_validator("narrative_strength")
    @classmethod
    def strength_must_be_controlled(cls, value: str) -> str:
        return _validate_controlled_value(
            value,
            NARRATIVE_STRENGTH_VALUES,
            "narrative_strength",
        )


class RiskScore(StrictModel):
    post_id: str
    urgency: str
    raw_engagement: float
    engagement_multiplier: float
    narrative_count: int = Field(ge=0)
    contains_legal_threat: bool
    risk_score: float

    @field_validator("urgency")
    @classmethod
    def urgency_must_be_controlled(cls, value: str) -> str:
        return _validate_controlled_value(value, URGENCY_VALUES, "urgency")


class RoutedEscalation(StrictModel):
    post_id: str
    teams: list[str]
    briefing_note: str

    @field_validator("teams")
    @classmethod
    def teams_must_be_controlled(cls, values: list[str]) -> list[str]:
        for value in values:
            _validate_controlled_value(value, TEAM_VALUES, "team")
        return values


class ResponseDraft(StrictModel):
    post_id: str
    platform: str
    draft_response: str
    send_gate_note: str


class PipelineArtifacts(StrictModel):
    raw_posts: list[RawPost] = Field(default_factory=list)
    preprocessed_posts: list[PreprocessedPost] = Field(default_factory=list)
    classified_posts: list[ClassifiedPost] = Field(default_factory=list)
    narratives: list[Narrative] = Field(default_factory=list)
    risk_scores: list[RiskScore] = Field(default_factory=list)
    routing: list[RoutedEscalation] = Field(default_factory=list)
    drafts: list[ResponseDraft] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
