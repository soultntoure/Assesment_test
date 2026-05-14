import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solution.artifacts import read_json, write_json
from src.solution.constants import SENTIMENT_VALUES, STAGES, TEAM_VALUES, TOPIC_VALUES
from src.solution.schemas import ClassifiedPost, Narrative, RoutedEscalation


def test_invalid_sentiment_topic_and_team_values_fail_validation():
    valid_classification = {
        "post_id": "P01",
        "sentiment": "negative",
        "topic": "withdrawal",
        "urgency": "high",
        "contains_legal_threat": False,
        "contains_competitor_mention": False,
        "original_language": "en",
        "translated": False,
    }

    with pytest.raises(ValidationError):
        ClassifiedPost(**{**valid_classification, "sentiment": "furious"})

    with pytest.raises(ValidationError):
        ClassifiedPost(**{**valid_classification, "topic": "made_up_topic"})

    with pytest.raises(ValidationError):
        RoutedEscalation(
            post_id="P01",
            teams=["Customer Happiness", "Unlisted Team"],
            briefing_note="Escalate the withdrawal delay with context.",
        )


def test_controlled_vocabularies_match_challenge_spec():
    assert SENTIMENT_VALUES == ("positive", "negative", "neutral", "mixed")
    assert TOPIC_VALUES == (
        "withdrawal",
        "account_suspension",
        "spread_pricing",
        "product_feedback",
        "regulatory",
        "technical",
        "deposit",
        "kyc",
        "general",
    )
    assert TEAM_VALUES == (
        "Customer Support",
        "Legal",
        "Compliance",
        "PR/Comms",
        "Product",
        "Engineering",
        "Finance",
    )


def test_required_output_schemas_accept_challenge_fields():
    classification = ClassifiedPost(
        post_id="P01",
        sentiment="negative",
        topic="withdrawal",
        urgency="high",
        contains_legal_threat=False,
        contains_competitor_mention=False,
        original_language="en",
        translated=False,
    )
    assert classification.topic == "withdrawal"

    narrative = Narrative(
        narrative_id="N01",
        title="Withdrawal delays",
        supporting_post_ids=["P01"],
        narrative_strength="strong",
        estimated_hours_until_trending=6,
        recommended_action="Prepare a comms update.",
    )
    assert narrative.estimated_hours_until_trending == 6

    route = RoutedEscalation(
        post_id="P01",
        teams=["Legal", "PR/Comms"],
        briefing_note="Legal threat and public allegation need approved response.",
    )
    assert route.teams == ["Legal", "PR/Comms"]


def test_json_artifact_helpers_round_trip_list_of_objects(tmp_path):
    path = tmp_path / "artifact.json"
    records = [{"post_id": "P01", "score": 42}, {"post_id": "P02", "score": 7}]

    write_json(path, records)

    assert read_json(path) == records


def test_stages_are_in_required_order():
    assert STAGES == [
        "INIT",
        "POSTS_LOADED",
        "MULTILINGUAL_PREPROCESSING_COMPLETE",
        "POSTS_CLASSIFIED",
        "NARRATIVES_DETECTED",
        "RISK_SCORES_COMPUTED",
        "ESCALATIONS_SELECTED",
        "ROUTING_COMPLETE",
        "RESPONSE_DRAFTS_GENERATED",
        "VALIDATION_COMPLETE",
        "RESULTS_FINALISED",
    ]
