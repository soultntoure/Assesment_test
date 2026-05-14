import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solution.risk import compute_risk_scores, raw_engagement, top_escalation_post_ids


def classified(post_id, urgency, legal=False):
    return {
        "post_id": post_id,
        "sentiment": "negative",
        "topic": "withdrawal",
        "urgency": urgency,
        "contains_legal_threat": legal,
        "contains_competitor_mention": False,
        "original_language": "en",
        "translated": False,
    }


def preprocessed(post_id, engagement):
    return {
        "post_id": post_id,
        "platform": "Twitter/X",
        "timestamp": "2025-04-10T08:00:00Z",
        "engagement": engagement,
        "original_text": "Original",
        "text_for_classification": "Original",
        "original_language": "en",
        "translated": False,
    }


def test_raw_engagement_uses_required_weights():
    assert raw_engagement(
        {
            "likes": 10,
            "reposts": 3,
            "comments": 4,
            "replies": 2,
            "upvotes": 5,
            "helpful_votes": 7,
            "reactions": 11,
            "ignored": 999,
        }
    ) == 48.0


def test_compute_risk_scores_exact_formula_and_sort_order():
    classified_posts = [
        classified("P01", "critical", legal=True),
        classified("P02", "high"),
        classified("P03", "low"),
    ]
    preprocessed_posts = [
        preprocessed("P01", {"likes": 10}),
        preprocessed("P02", {"likes": 20}),
        preprocessed("P03", {"likes": 30}),
    ]
    narratives = [
        {
            "narrative_id": "N01",
            "title": "Withdrawal delays",
            "supporting_post_ids": ["P01", "P02"],
            "narrative_strength": "strong",
            "estimated_hours_until_trending": 6,
            "recommended_action": "Act now.",
        },
        {
            "narrative_id": "N02",
            "title": "Account access",
            "supporting_post_ids": ["P01"],
            "narrative_strength": "moderate",
            "estimated_hours_until_trending": 12,
            "recommended_action": "Monitor.",
        },
    ]

    scores = compute_risk_scores(classified_posts, preprocessed_posts, narratives)

    by_id = {score.post_id: score for score in scores}
    assert by_id["P01"].raw_engagement == 10.0
    assert by_id["P01"].engagement_multiplier == 1.0
    assert by_id["P01"].narrative_count == 2
    assert by_id["P01"].risk_score == 90.0

    assert by_id["P02"].engagement_multiplier == 2.0
    assert by_id["P02"].risk_score == 65.0

    assert by_id["P03"].engagement_multiplier == 3.0
    assert by_id["P03"].risk_score == 9.0
    assert [score.post_id for score in scores] == ["P01", "P02", "P03"]


def test_same_engagement_uses_multiplier_one_for_all_posts():
    scores = compute_risk_scores(
        [classified("P01", "medium"), classified("P02", "medium")],
        [preprocessed("P01", {"likes": 5}), preprocessed("P02", {"likes": 5})],
        [],
    )

    assert [score.engagement_multiplier for score in scores] == [1.0, 1.0]
    assert [score.risk_score for score in scores] == [10.0, 10.0]


def test_top_escalation_post_ids_returns_first_five_by_score():
    scores = compute_risk_scores(
        [
            classified("P01", "low"),
            classified("P02", "medium"),
            classified("P03", "high"),
            classified("P04", "critical"),
            classified("P05", "low"),
            classified("P06", "medium"),
        ],
        [
            preprocessed("P01", {"likes": 1}),
            preprocessed("P02", {"likes": 2}),
            preprocessed("P03", {"likes": 3}),
            preprocessed("P04", {"likes": 4}),
            preprocessed("P05", {"likes": 5}),
            preprocessed("P06", {"likes": 6}),
        ],
        [],
    )

    assert top_escalation_post_ids(scores) == ["P04", "P03", "P06", "P02", "P05"]
