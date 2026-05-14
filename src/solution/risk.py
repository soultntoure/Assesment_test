from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from src.solution.artifacts import write_json
from src.solution.constants import (
    LEGAL_THREAT_BONUS,
    NARRATIVE_MEMBERSHIP_BONUS,
    NARRATIVES_PATH,
    PREPROCESSED_POSTS_PATH,
    RISK_SCORES_PATH,
    URGENCY_BASE_RISK,
)
from src.solution.schemas import ClassifiedPost, Narrative, PreprocessedPost, RiskScore


ENGAGEMENT_WEIGHTS = {
    "likes": 1.0,
    "reposts": 2.0,
    "comments": 1.5,
    "replies": 1.5,
    "upvotes": 1.0,
    "helpful_votes": 1.0,
    "reactions": 1.0,
}


def raw_engagement(engagement: dict[str, int | float]) -> float:
    return sum(float(engagement.get(field, 0)) * weight for field, weight in ENGAGEMENT_WEIGHTS.items())


def engagement_multipliers(raw_values: dict[str, float]) -> dict[str, float]:
    if not raw_values:
        return {}

    minimum = min(raw_values.values())
    maximum = max(raw_values.values())
    if minimum == maximum:
        return {post_id: 1.0 for post_id in raw_values}

    spread = maximum - minimum
    return {post_id: 1.0 + ((value - minimum) / spread) * 2.0 for post_id, value in raw_values.items()}


def narrative_membership_counts(narratives: Iterable[Narrative | dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for narrative_like in narratives:
        narrative = Narrative.model_validate(narrative_like)
        for post_id in narrative.supporting_post_ids:
            counts[post_id] += 1
    return counts


def compute_risk_scores(
    classified_posts: Iterable[ClassifiedPost | dict],
    preprocessed_posts: Iterable[PreprocessedPost | dict],
    narratives: Iterable[Narrative | dict],
) -> list[RiskScore]:
    classified = [ClassifiedPost.model_validate(post) for post in classified_posts]
    preprocessed_by_id = {
        post.post_id: post for post in (PreprocessedPost.model_validate(post) for post in preprocessed_posts)
    }
    missing_engagement = sorted({post.post_id for post in classified} - set(preprocessed_by_id))
    if missing_engagement:
        raise ValueError(f"Missing preprocessed engagement for posts: {', '.join(missing_engagement)}")

    raw_values = {
        post.post_id: raw_engagement(preprocessed_by_id[post.post_id].engagement) for post in classified
    }
    multipliers = engagement_multipliers(raw_values)
    narrative_counts = narrative_membership_counts(narratives)

    risk_scores: list[RiskScore] = []
    for post in classified:
        base = URGENCY_BASE_RISK[post.urgency]
        legal_bonus = LEGAL_THREAT_BONUS if post.contains_legal_threat else 0
        narrative_bonus = narrative_counts[post.post_id] * NARRATIVE_MEMBERSHIP_BONUS
        score = (base * multipliers[post.post_id]) + legal_bonus + narrative_bonus
        risk_scores.append(
            RiskScore(
                post_id=post.post_id,
                urgency=post.urgency,
                raw_engagement=raw_values[post.post_id],
                engagement_multiplier=multipliers[post.post_id],
                narrative_count=narrative_counts[post.post_id],
                contains_legal_threat=post.contains_legal_threat,
                risk_score=score,
            )
        )

    return sorted(risk_scores, key=lambda item: (-item.risk_score, item.post_id))


def top_escalation_post_ids(risk_scores: Iterable[RiskScore | dict], limit: int = 5) -> list[str]:
    scores = [RiskScore.model_validate(score) for score in risk_scores]
    sorted_scores = sorted(scores, key=lambda item: (-item.risk_score, item.post_id))
    return [score.post_id for score in sorted_scores[:limit]]


def write_risk_scores(risk_scores: Iterable[RiskScore], path: str | Path = RISK_SCORES_PATH) -> None:
    write_json(path, [score.model_dump() for score in risk_scores])


def run_risk_scoring(
    classified_posts: Iterable[ClassifiedPost | dict],
    preprocessed_posts: Iterable[PreprocessedPost | dict],
    narratives: Iterable[Narrative | dict],
    *,
    output_artifact: str | Path = RISK_SCORES_PATH,
) -> list[RiskScore]:
    scores = compute_risk_scores(classified_posts, preprocessed_posts, narratives)
    write_risk_scores(scores, output_artifact)
    return scores


def default_input_artifacts() -> tuple[Path, Path]:
    return PREPROCESSED_POSTS_PATH, NARRATIVES_PATH
