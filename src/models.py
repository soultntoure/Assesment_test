from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator

from src.vocab import (
    NARRATIVE_STRENGTH,
    SENTIMENT,
    TEAMS,
    TOPICS,
    URGENCY,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _validate_vocab(value: str, allowed: tuple[str, ...], field_name: str) -> str:
    if value not in allowed:
        allowed_values = ", ".join(allowed)
        raise ValueError(f"{field_name} must be one of: {allowed_values}")
    return value


class Engagement(StrictModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    likes: int = Field(default=0, ge=0)
    replies: int = Field(default=0, ge=0)
    reposts: int = Field(default=0, ge=0)
    upvotes: int = Field(default=0, ge=0)
    comments: int = Field(default=0, ge=0)
    helpful_votes: int = Field(default=0, ge=0)
    reactions: int = Field(default=0, ge=0)


class RawPost(StrictModel):
    id: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    text: str = Field(min_length=1)
    timestamp: datetime
    engagement: Engagement = Field(default_factory=Engagement)

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must include timezone information")
        return value


class RawPosts(RootModel[list[RawPost]]):
    @model_validator(mode="after")
    def post_ids_must_be_unique(self) -> RawPosts:
        ids = [post.id for post in self.root]
        if len(ids) != len(set(ids)):
            raise ValueError("post ids must be unique")
        return self


class PreprocessedPost(StrictModel):
    post_id: str = Field(min_length=1)
    original_text: str = Field(min_length=1)
    text_for_classification: str = Field(min_length=1)
    original_language: str = Field(min_length=2)
    translated: bool

    @model_validator(mode="after")
    def translated_posts_need_changed_text(self) -> PreprocessedPost:
        if self.translated and self.original_text == self.text_for_classification:
            raise ValueError("translated posts must include translated classification text")
        return self


class PreprocessedPosts(RootModel[list[PreprocessedPost]]):
    @model_validator(mode="after")
    def post_ids_must_be_unique(self) -> PreprocessedPosts:
        ids = [post.post_id for post in self.root]
        if len(ids) != len(set(ids)):
            raise ValueError("preprocessed post ids must be unique")
        return self


class ClassifiedPost(StrictModel):
    post_id: str = Field(min_length=1)
    sentiment: str
    topic: str
    urgency: str
    contains_legal_threat: bool
    contains_competitor_mention: bool
    original_language: str = Field(min_length=2)
    translated: bool

    @field_validator("sentiment")
    @classmethod
    def sentiment_must_be_controlled(cls, value: str) -> str:
        return _validate_vocab(value, SENTIMENT, "sentiment")

    @field_validator("topic")
    @classmethod
    def topic_must_be_controlled(cls, value: str) -> str:
        return _validate_vocab(value, TOPICS, "topic")

    @field_validator("urgency")
    @classmethod
    def urgency_must_be_controlled(cls, value: str) -> str:
        return _validate_vocab(value, URGENCY, "urgency")


class ClassificationResult(StrictModel):
    posts: list[ClassifiedPost]

    @model_validator(mode="after")
    def post_ids_must_be_unique(self) -> ClassificationResult:
        ids = [post.post_id for post in self.posts]
        if len(ids) != len(set(ids)):
            raise ValueError("classified post ids must be unique")
        return self


class NarrativeInputPost(StrictModel):
    post_id: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    timestamp: datetime
    engagement: Engagement = Field(default_factory=Engagement)
    text_for_classification: str = Field(min_length=1)
    sentiment: str
    topic: str
    urgency: str
    contains_legal_threat: bool
    contains_competitor_mention: bool
    original_language: str = Field(min_length=2)
    translated: bool

    @field_validator("sentiment")
    @classmethod
    def sentiment_must_be_controlled(cls, value: str) -> str:
        return _validate_vocab(value, SENTIMENT, "sentiment")

    @field_validator("topic")
    @classmethod
    def topic_must_be_controlled(cls, value: str) -> str:
        return _validate_vocab(value, TOPICS, "topic")

    @field_validator("urgency")
    @classmethod
    def urgency_must_be_controlled(cls, value: str) -> str:
        return _validate_vocab(value, URGENCY, "urgency")


class NarrativeDetectionInput(StrictModel):
    posts: list[NarrativeInputPost]

    @model_validator(mode="after")
    def post_ids_must_be_unique(self) -> NarrativeDetectionInput:
        ids = [post.post_id for post in self.posts]
        if len(ids) != len(set(ids)):
            raise ValueError("narrative input post ids must be unique")
        return self


class Narrative(StrictModel):
    narrative_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    supporting_post_ids: list[str] = Field(min_length=1)
    narrative_strength: str
    estimated_hours_until_trending: int = Field(ge=0)
    recommended_action: str = Field(min_length=1)

    @field_validator("narrative_strength")
    @classmethod
    def narrative_strength_must_be_controlled(cls, value: str) -> str:
        return _validate_vocab(value, NARRATIVE_STRENGTH, "narrative_strength")

    @field_validator("supporting_post_ids")
    @classmethod
    def supporting_post_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("supporting_post_ids must be unique within a narrative")
        return value


class NarrativeDetectionResult(StrictModel):
    narratives: list[Narrative]

    @model_validator(mode="after")
    def narrative_ids_must_be_unique(self) -> NarrativeDetectionResult:
        ids = [narrative.narrative_id for narrative in self.narratives]
        if len(ids) != len(set(ids)):
            raise ValueError("narrative ids must be unique")
        return self


class RiskScore(StrictModel):
    post_id: str = Field(min_length=1)
    urgency: str
    base_risk: int = Field(ge=0)
    raw_engagement: float = Field(ge=0)
    engagement_multiplier: float = Field(ge=1.0, le=3.0)
    legal_threat_bonus: int = Field(ge=0)
    narrative_membership_count: int = Field(ge=0)
    narrative_membership_bonus: int = Field(ge=0)
    risk_score: float = Field(ge=0)
    escalated: bool = False
    rank: int | None = Field(default=None, ge=1)

    @field_validator("urgency")
    @classmethod
    def urgency_must_be_controlled(cls, value: str) -> str:
        return _validate_vocab(value, URGENCY, "urgency")


class RiskScoringResult(StrictModel):
    scores: list[RiskScore]

    @model_validator(mode="after")
    def top_escalations_must_be_ranked(self) -> RiskScoringResult:
        escalated = [score for score in self.scores if score.escalated]
        if len(escalated) > 5:
            raise ValueError("no more than five posts may be escalated")
        if escalated and any(score.rank is None for score in escalated):
            raise ValueError("escalated posts must include a rank")
        return self


class EscalationRoute(StrictModel):
    post_id: str = Field(min_length=1)
    teams: list[str] = Field(min_length=1)
    briefing_note: str = Field(min_length=1)

    @field_validator("teams")
    @classmethod
    def teams_must_be_controlled_and_unique(cls, value: list[str]) -> list[str]:
        for team in value:
            _validate_vocab(team, TEAMS, "teams")
        if len(value) != len(set(value)):
            raise ValueError("teams must not contain duplicates")
        return value


class EscalationRoutingResult(StrictModel):
    routes: list[EscalationRoute]


class ResponseDraft(StrictModel):
    post_id: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    draft_response: str = Field(min_length=1)
    send_gate_note: str = Field(min_length=1)


class ResponseDraftResult(StrictModel):
    drafts: list[ResponseDraft]


class SentimentTimeBucket(StrictModel):
    start: datetime
    end: datetime
    counts: dict[str, int]

    @field_validator("counts")
    @classmethod
    def counts_must_use_controlled_sentiments(cls, value: dict[str, int]) -> dict[str, int]:
        for sentiment, count in value.items():
            _validate_vocab(sentiment, SENTIMENT, "sentiment bucket")
            if count < 0:
                raise ValueError("sentiment counts must be non-negative")
        return value


class SentimentTrend(StrictModel):
    sentiment_distribution: dict[str, int]
    time_buckets: list[SentimentTimeBucket]
    inflection_point: str | None = None
    shift_driver: str | None = None

    @field_validator("sentiment_distribution")
    @classmethod
    def distribution_must_use_controlled_sentiments(cls, value: dict[str, int]) -> dict[str, int]:
        for sentiment, count in value.items():
            _validate_vocab(sentiment, SENTIMENT, "sentiment distribution")
            if count < 0:
                raise ValueError("sentiment distribution counts must be non-negative")
        return value


class CompetitorSignal(StrictModel):
    post_id: str = Field(min_length=1)
    implied_competitor_type: str = Field(min_length=1)
    switching_trigger: str = Field(min_length=1)
    retention_argument: str = Field(min_length=1)


class CompetitorSignalResult(StrictModel):
    signals: list[CompetitorSignal]


class LLMCallLog(StrictModel):
    stage: str = Field(min_length=1)
    timestamp: datetime
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_hash: str = Field(min_length=1)
    input_artifacts: list[str] = Field(min_length=1)
    output_artifact: str = Field(min_length=1)
