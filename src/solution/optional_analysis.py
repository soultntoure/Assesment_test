from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from src.solution.artifacts import write_json
from src.solution.constants import COMPETITOR_SIGNALS_PATH, SENTIMENT_TREND_PATH
from src.solution.schemas import ClassifiedPost, PreprocessedPost


COMPETITOR_PATTERNS = (
    ("alternatives", "alternative broker/platform consideration"),
    ("alternative", "alternative broker/platform consideration"),
    ("moved to proper platforms", "higher-trust trading platform"),
    ("proper platforms", "higher-trust trading platform"),
    ("starting to look", "alternative broker/platform consideration"),
)


def write_sentiment_trend(
    classified_posts: Iterable[ClassifiedPost | dict],
    preprocessed_posts: Iterable[PreprocessedPost | dict],
    *,
    output_artifact: str | Path = SENTIMENT_TREND_PATH,
) -> dict:
    trend = analyze_sentiment_trend(classified_posts, preprocessed_posts)
    write_json(output_artifact, trend)
    return trend


def analyze_sentiment_trend(
    classified_posts: Iterable[ClassifiedPost | dict],
    preprocessed_posts: Iterable[PreprocessedPost | dict],
) -> dict:
    classified_by_id = {
        post.post_id: post for post in (ClassifiedPost.model_validate(post) for post in classified_posts)
    }
    rows = []
    for post_like in preprocessed_posts:
        post = PreprocessedPost.model_validate(post_like)
        if post.post_id not in classified_by_id or post.timestamp is None:
            continue
        rows.append((post.timestamp, post.post_id, classified_by_id[post.post_id].sentiment))

    rows.sort(key=lambda row: row[0])
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    cumulative = Counter()
    first_negative_majority = None
    negative_cluster_posts: list[str] = []

    for timestamp, post_id, sentiment in rows:
        hour = _hour_bucket(timestamp)
        buckets[hour][sentiment] += 1
        cumulative[sentiment] += 1
        if sentiment in {"negative", "mixed"}:
            negative_cluster_posts.append(post_id)
        if first_negative_majority is None:
            negative = cumulative["negative"] + cumulative["mixed"]
            positive = cumulative["positive"]
            if negative > positive and negative >= 2:
                first_negative_majority = {
                    "timestamp": timestamp,
                    "post_id": post_id,
                    "description": "Negative or mixed posts became the dominant cumulative sentiment.",
                }

    return {
        "sentiment_distribution_over_time": {
            hour: dict(counter) for hour, counter in sorted(buckets.items())
        },
        "overall_distribution": dict(cumulative),
        "inflection_point": first_negative_majority,
        "shift_driver_post_ids": negative_cluster_posts[:10],
    }


def write_competitor_signals(
    preprocessed_posts: Iterable[PreprocessedPost | dict],
    *,
    output_artifact: str | Path = COMPETITOR_SIGNALS_PATH,
) -> list[dict[str, str]]:
    signals = extract_competitor_signals(preprocessed_posts)
    write_json(output_artifact, signals)
    return signals


def extract_competitor_signals(preprocessed_posts: Iterable[PreprocessedPost | dict]) -> list[dict[str, str]]:
    signals: list[dict[str, str]] = []
    for post_like in preprocessed_posts:
        post = PreprocessedPost.model_validate(post_like)
        text = post.text_for_classification.casefold()
        matched_type = next((signal_type for pattern, signal_type in COMPETITOR_PATTERNS if pattern in text), None)
        if matched_type is None:
            continue

        signals.append(
            {
                "post_id": post.post_id,
                "implied_competitor_type": matched_type,
                "switching_trigger": _switching_trigger(text),
                "retention_argument": _retention_argument(text),
            }
        )
    return signals


def _hour_bucket(timestamp: str) -> str:
    value = timestamp.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(value)
    return parsed.strftime("%Y-%m-%dT%H:00:00Z")


def _switching_trigger(text: str) -> str:
    if "spread" in text:
        return "spread widening or pricing concern"
    if "support" in text:
        return "support or service confidence concern"
    if "proper platforms" in text:
        return "brand trust and platform credibility concern"
    return "user signaled interest in alternatives"


def _retention_argument(text: str) -> str:
    if "spread" in text:
        return "Provide transparent pricing context and a path for account-specific review."
    if "proper platforms" in text:
        return "Reinforce platform reliability, regulatory controls, and product strengths without arguing."
    return "Acknowledge the concern and offer a concrete support path before the user switches."
