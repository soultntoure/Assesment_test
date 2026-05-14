from __future__ import annotations

import json
from typing import Iterable

from src.solution.constants import SENTIMENT_VALUES, TOPIC_VALUES, URGENCY_VALUES
from src.solution.schemas import PreprocessedPost


def _csv(values: tuple[str, ...]) -> str:
    return ", ".join(values)


def build_classification_prompt(posts: Iterable[PreprocessedPost | dict]) -> str:
    payload = []
    for post_like in posts:
        post = PreprocessedPost.model_validate(post_like)
        payload.append(
            {
                "post_id": post.post_id,
                "text_for_classification": post.text_for_classification,
                "original_language": post.original_language,
                "translated": post.translated,
            }
        )

    return f"""You are Stage 1 of a replayable social-listening pipeline for Deriv.

Classify every preprocessed post below. Return structured JSON matching this schema:
{{
  "posts": [
    {{
      "post_id": "string",
      "sentiment": "positive",
      "topic": "withdrawal",
      "urgency": "high",
      "contains_legal_threat": false,
      "contains_competitor_mention": false,
      "original_language": "en",
      "translated": false
    }}
  ]
}}

Controlled vocabularies:
- sentiment: {_csv(SENTIMENT_VALUES)}
- topic: {_csv(TOPIC_VALUES)}
- urgency: {_csv(URGENCY_VALUES)}

Do not invent sentiment, topic, or urgency categories. Use exactly one controlled value for each field.
Classify based on text_for_classification. Preserve each post_id, original_language, and translated value exactly.
Set contains_legal_threat true for lawyers, lawsuits, regulator complaints, chargebacks, or formal legal/regulatory threats.
Set contains_competitor_mention true when the post names competitors or implies switching to alternatives.

Preprocessed posts:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""
