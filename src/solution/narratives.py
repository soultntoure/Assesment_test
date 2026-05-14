from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from src.solution.artifacts import write_json
from src.solution.constants import (
    CLASSIFIED_POSTS_PATH,
    NARRATIVE_STRENGTH_VALUES,
    NARRATIVES_DETECTED,
    NARRATIVES_PATH,
)
from src.solution.llm import StructuredLLM
from src.solution.schemas import ClassifiedPost, Narrative, PreprocessedPost


class NarrativeDetectionResult(BaseModel):
    narratives: list[Narrative] = Field(default_factory=list)


def detect_narratives(
    classified_posts: Iterable[ClassifiedPost | dict],
    post_context: Iterable[PreprocessedPost | dict],
    *,
    llm=None,
    input_artifact: str | Path = CLASSIFIED_POSTS_PATH,
    output_artifact: str | Path = NARRATIVES_PATH,
) -> list[Narrative]:
    classified = [ClassifiedPost.model_validate(post) for post in classified_posts]
    context = [PreprocessedPost.model_validate(post) for post in post_context]
    _ensure_context_for_classified_posts(classified, context)

    prompt = _build_narrative_prompt(classified, context)
    detector = llm or StructuredLLM(schema=NarrativeDetectionResult)
    result = detector.invoke(
        stage=NARRATIVES_DETECTED,
        prompt=prompt,
        input_artifacts=[input_artifact],
        output_artifact=output_artifact,
    )

    narratives = [Narrative.model_validate(narrative) for narrative in result.narratives]
    _ensure_supporting_ids_exist(narratives, {post.post_id for post in classified})
    write_json(output_artifact, [narrative.model_dump() for narrative in narratives])
    return narratives


def _build_narrative_prompt(
    classified_posts: list[ClassifiedPost],
    post_context: list[PreprocessedPost],
) -> str:
    context_by_id = {post.post_id: post for post in post_context}
    narrative_input = []
    for classified in classified_posts:
        context = context_by_id[classified.post_id]
        narrative_input.append(
            {
                "post_id": classified.post_id,
                "platform": context.platform,
                "timestamp": context.timestamp,
                "engagement": context.engagement,
                "text_for_classification": context.text_for_classification,
                "classification": classified.model_dump(),
            }
        )

    allowed_strengths = ", ".join(NARRATIVE_STRENGTH_VALUES)
    payload = json.dumps(narrative_input, ensure_ascii=False, indent=2)
    return f"""You are detecting emerging narratives in Deriv social and forum mentions.

Use Stage 1 classified data plus post context. Do not cluster from raw post text alone.
Identify systemic issue clusters that may trend even when individual posts look minor.

Allowed narrative_strength values: {allowed_strengths}. Do not invent categories.

Return JSON matching this schema:
{{
  "narratives": [
    {{
      "narrative_id": "string",
      "title": "string",
      "supporting_post_ids": ["P01"],
      "narrative_strength": "strong",
      "estimated_hours_until_trending": 6,
      "recommended_action": "string"
    }}
  ]
}}

Input posts:
{payload}
"""


def _ensure_context_for_classified_posts(
    classified_posts: list[ClassifiedPost],
    post_context: list[PreprocessedPost],
) -> None:
    classified_ids = {post.post_id for post in classified_posts}
    context_ids = {post.post_id for post in post_context}
    if classified_ids != context_ids:
        missing = sorted(classified_ids - context_ids)
        extra = sorted(context_ids - classified_ids)
        raise ValueError(
            "post_context must include exactly one context record for each classified post; "
            f"missing={missing}, extra={extra}"
        )


def _ensure_supporting_ids_exist(narratives: list[Narrative], classified_ids: set[str]) -> None:
    unknown_ids = sorted(
        {
            post_id
            for narrative in narratives
            for post_id in narrative.supporting_post_ids
            if post_id not in classified_ids
        }
    )
    if unknown_ids:
        raise ValueError(f"Narratives reference unknown supporting post IDs: {unknown_ids}")
