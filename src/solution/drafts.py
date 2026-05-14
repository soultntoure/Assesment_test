from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from src.solution.artifacts import write_text
from src.solution.constants import (
    RESPONSE_DRAFTS_GENERATED,
    RESPONSE_DRAFTS_PATH,
    RISK_SCORES_PATH,
)
from src.solution.llm import StructuredLLM
from src.solution.schemas import ResponseDraft


class PublicDraftResult(BaseModel):
    drafts: list[ResponseDraft] = Field(default_factory=list)


def generate_public_drafts(
    posts_requiring_response: Iterable[dict[str, Any]],
    *,
    llm=None,
    input_artifacts: Iterable[str | Path] = (RISK_SCORES_PATH,),
    output_artifact: str | Path = RESPONSE_DRAFTS_PATH,
) -> list[ResponseDraft]:
    selected_posts = [
        dict(post)
        for post in posts_requiring_response
        if post.get("urgency") == "critical" or bool(post.get("contains_legal_threat"))
    ]
    prompt = _build_draft_prompt(selected_posts)
    drafter = llm or StructuredLLM(schema=PublicDraftResult)
    result = drafter.invoke(
        stage=RESPONSE_DRAFTS_GENERATED,
        prompt=prompt,
        input_artifacts=input_artifacts,
        output_artifact=output_artifact,
    )

    drafts = [ResponseDraft.model_validate(draft) for draft in result.drafts]
    _ensure_draft_for_each_selected_post(drafts, {str(post.get("post_id")) for post in selected_posts})
    write_text(output_artifact, _format_drafts_markdown(drafts))
    return drafts


def _build_draft_prompt(selected_posts: list[dict[str, Any]]) -> str:
    return f"""Draft public-facing Deriv responses for posts that are critical or include legal-threat language.

Each draft must:
- acknowledge the issue
- avoid admitting liability
- provide next steps
- match platform tone
- avoid disclosing account details or other private information
- include send_gate_note describing what internal information is required before posting

Return JSON matching this schema:
{{
  "drafts": [
    {{
      "post_id": "P01",
      "platform": "Twitter/X",
      "draft_response": "string",
      "send_gate_note": "string"
    }}
  ]
}}

Selected posts:
{json.dumps(selected_posts, ensure_ascii=False, indent=2)}
"""


def _ensure_draft_for_each_selected_post(drafts: list[ResponseDraft], selected_post_ids: set[str]) -> None:
    draft_ids = {draft.post_id for draft in drafts}
    if draft_ids != selected_post_ids:
        missing = sorted(selected_post_ids - draft_ids)
        extra = sorted(draft_ids - selected_post_ids)
        raise ValueError(
            "Expected a public draft for every selected post; "
            f"missing={missing}, extra={extra}"
        )


def _format_drafts_markdown(drafts: list[ResponseDraft]) -> str:
    lines = ["# Public Response Drafts", ""]
    for draft in drafts:
        lines.extend(
            [
                f"## {draft.post_id}",
                "",
                f"Platform: {draft.platform}",
                "",
                f"Draft response: {draft.draft_response}",
                "",
                f"Send-gate note: {draft.send_gate_note}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
