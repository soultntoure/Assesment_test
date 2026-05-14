from __future__ import annotations

from typing import Any, TypedDict

from src.solution.constants import STAGES


class PipelineState(TypedDict, total=False):
    stage: str
    posts: list[Any]
    preprocessed_posts: list[Any]
    classified_posts: list[Any]
    narratives: list[Any]
    risk_scores: list[Any]
    top_escalation_post_ids: list[str]
    routing: list[Any]
    drafts: list[Any]
    generated_artifacts: list[str]


def advance_stage(current: str, expected_current: str, next_stage: str) -> str:
    if current != expected_current:
        raise ValueError(
            f"Stage transition out-of-order: expected {expected_current}, got {current}; "
            f"cannot advance to {next_stage}"
        )

    try:
        current_index = STAGES.index(current)
        next_index = STAGES.index(next_stage)
    except ValueError as exc:
        raise ValueError(f"Unknown pipeline stage in transition: {current} -> {next_stage}") from exc

    if next_index != current_index + 1:
        raise ValueError(f"Stage transition out-of-order: {current} cannot advance to {next_stage}")

    return next_stage
