import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solution.optional_analysis import analyze_sentiment_trend, extract_competitor_signals


def classified(post_id, sentiment):
    return {
        "post_id": post_id,
        "sentiment": sentiment,
        "topic": "general",
        "urgency": "low",
        "contains_legal_threat": False,
        "contains_competitor_mention": False,
        "original_language": "en",
        "translated": False,
    }


def preprocessed(post_id, timestamp, text):
    return {
        "post_id": post_id,
        "platform": "Reddit",
        "timestamp": timestamp,
        "engagement": {},
        "original_text": text,
        "text_for_classification": text,
        "original_language": "en",
        "translated": False,
    }


def test_analyze_sentiment_trend_buckets_by_hour_and_finds_inflection():
    trend = analyze_sentiment_trend(
        [
            classified("P01", "positive"),
            classified("P02", "negative"),
            classified("P03", "mixed"),
        ],
        [
            preprocessed("P01", "2025-04-10T08:15:00Z", "Good."),
            preprocessed("P02", "2025-04-10T09:00:00Z", "Bad."),
            preprocessed("P03", "2025-04-10T09:30:00Z", "Mixed."),
        ],
    )

    assert trend["sentiment_distribution_over_time"]["2025-04-10T08:00:00Z"] == {"positive": 1}
    assert trend["sentiment_distribution_over_time"]["2025-04-10T09:00:00Z"] == {
        "negative": 1,
        "mixed": 1,
    }
    assert trend["inflection_point"]["post_id"] == "P03"
    assert trend["shift_driver_post_ids"] == ["P02", "P03"]


def test_extract_competitor_signals_detects_named_and_implied_alternatives():
    signals = extract_competitor_signals(
        [
            preprocessed(
                "P01",
                "2025-04-10T09:00:00Z",
                "Spreads widened, starting to look at alternatives.",
            ),
            preprocessed(
                "P02",
                "2025-04-10T10:00:00Z",
                "Any serious trader moved to proper platforms years ago.",
            ),
            preprocessed("P03", "2025-04-10T11:00:00Z", "Fast execution."),
        ]
    )

    assert [signal["post_id"] for signal in signals] == ["P01", "P02"]
    assert signals[0]["switching_trigger"] == "spread widening or pricing concern"
    assert signals[1]["implied_competitor_type"] == "higher-trust trading platform"
