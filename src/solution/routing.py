from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from src.solution.artifacts import write_json
from src.solution.constants import (
    ESCALATION_ROUTING_PATH,
    NARRATIVES_PATH,
    RISK_SCORES_PATH,
    ROUTING_COMPLETE,
    TEAM_VALUES,
)
from src.solution.llm import StructuredLLM
from src.solution.schemas import Narrative, RoutedEscalation


class EscalationRoutingResult(BaseModel):
    routes: list[RoutedEscalation] = Field(default_factory=list)


def route_escalations(
    top_posts: Iterable[dict[str, Any]],
    narratives: Iterable[Narrative | dict],
    *,
    llm=None,
    input_artifacts: Iterable[str | Path] = (RISK_SCORES_PATH, NARRATIVES_PATH),
    output_artifact: str | Path = ESCALATION_ROUTING_PATH,
) -> list[RoutedEscalation]:
    posts = [dict(post) for post in top_posts]
    if len(posts) > 5:
        raise ValueError("route_escalations must receive only the top 5 flagged posts")

    detected_narratives = [Narrative.model_validate(narrative) for narrative in narratives]
    prompt = _build_routing_prompt(posts, detected_narratives)
    router = llm or StructuredLLM(schema=EscalationRoutingResult)
    result = router.invoke(
        stage=ROUTING_COMPLETE,
        prompt=prompt,
        input_artifacts=input_artifacts,
        output_artifact=output_artifact,
    )

    routes = [RoutedEscalation.model_validate(route) for route in result.routes]
    _ensure_routes_reference_top_posts(routes, {str(post.get("post_id")) for post in posts})
    write_json(output_artifact, [route.model_dump() for route in routes])
    return routes


def _build_routing_prompt(top_posts: list[dict[str, Any]], narratives: list[Narrative]) -> str:
    teams = ", ".join(TEAM_VALUES)
    payload = {
        "top_5_flagged_posts": top_posts,
        "narratives": [narrative.model_dump() for narrative in narratives],
    }
    return f"""Route Deriv escalation items for internal follow-up.

Input only the top 5 flagged posts and all detected narratives. Assign one or more teams from this controlled vocabulary only:
{teams}

Write each briefing_note as a concise internal Slack-style update for the relevant teams.
Do not invent teams.

Return JSON matching this schema:
{{
  "routes": [
    {{
      "post_id": "P01",
      "teams": ["Legal", "PR/Comms"],
      "briefing_note": "Concise internal update."
    }}
  ]
}}

Routing input:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def _ensure_routes_reference_top_posts(routes: list[RoutedEscalation], top_post_ids: set[str]) -> None:
    unknown_ids = sorted({route.post_id for route in routes if route.post_id not in top_post_ids})
    if unknown_ids:
        raise ValueError(f"Routes reference posts outside the top escalation posts: {unknown_ids}")
