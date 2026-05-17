from datetime import datetime, UTC

import pytest
from pydantic import ValidationError

from src.schemas import (
    ClassifiedPost,
    EscalationRoute,
    NarrativeDetectionInput,
    PreprocessedPost,
    RawPost,
)
from src.state import PipelineState


def test_raw_post_accepts_source_schema() -> None:
    post = RawPost(
        id="P01",
        platform="Twitter/X",
        text="Withdrawal delayed for nine days.",
        timestamp="2025-04-10T08:15:00Z",
        engagement={"likes": 12, "replies": 4, "reposts": 8},
    )

    assert post.timestamp == datetime(2025, 4, 10, 8, 15, tzinfo=UTC)
    assert post.engagement.likes == 12
    assert post.engagement.replies == 4
    assert post.engagement.reposts == 8
    assert post.engagement.comments == 0


def test_raw_post_ignores_unknown_engagement_metrics() -> None:
    post = RawPost(
        id="P02",
        platform="Twitter/X",
        text="Post with an unsupported engagement metric.",
        timestamp="2025-04-10T08:15:00Z",
        engagement={"likes": 10, "shares": 4},
    )

    assert post.engagement.likes == 10
    assert not hasattr(post.engagement, "shares")


def test_raw_post_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ValidationError):
        RawPost(
            id="P03",
            platform="Twitter/X",
            text="Timestamp without timezone.",
            timestamp="2025-04-10T08:15:00",
            engagement={"likes": 1},
        )


def test_translated_preprocessed_post_requires_classification_text_change() -> None:
    with pytest.raises(ValidationError):
        PreprocessedPost(
            post_id="P10",
            original_text="Adakah sesiapa tahu kenapa Deriv minta dokumen tambahan?",
            text_for_classification="Adakah sesiapa tahu kenapa Deriv minta dokumen tambahan?",
            original_language="ms",
            translated=True,
        )


def test_classified_post_enforces_controlled_vocabularies() -> None:
    with pytest.raises(ValidationError):
        ClassifiedPost(
            post_id="P01",
            sentiment="angry",
            topic="withdrawal",
            urgency="high",
            contains_legal_threat=False,
            contains_competitor_mention=False,
            original_language="en",
            translated=False,
        )


def test_narrative_detection_input_keeps_stage_two_context() -> None:
    narrative_input = NarrativeDetectionInput(
        posts=[
            {
                "post_id": "P01",
                "platform": "Twitter/X",
                "timestamp": "2025-04-10T08:15:00Z",
                "engagement": {"likes": 12, "replies": 4, "reposts": 8},
                "text_for_classification": "Withdrawal delayed for nine days.",
                "sentiment": "negative",
                "topic": "withdrawal",
                "urgency": "high",
                "contains_legal_threat": False,
                "contains_competitor_mention": False,
                "original_language": "en",
                "translated": False,
            }
        ]
    )

    post = narrative_input.posts[0]
    assert post.platform == "Twitter/X"
    assert post.engagement.likes == 12
    assert post.topic == "withdrawal"


def test_escalation_route_enforces_controlled_teams() -> None:
    with pytest.raises(ValidationError):
        EscalationRoute(
            post_id="P07",
            teams=["Legal", "Social Media"],
            briefing_note="Legal threat around locked account balance.",
        )


def test_pipeline_state_enforces_core_stage_names() -> None:
    with pytest.raises(ValidationError):
        PipelineState(current_stage="SENTIMENT_TREND_ANALYSIS")
