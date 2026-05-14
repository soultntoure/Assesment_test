import json
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solution.routing import EscalationRoutingResult, route_escalations
from src.solution.schemas import Narrative, RoutedEscalation


class FakeRoutingLLM:
    def __init__(self, result: BaseModel) -> None:
        self.result = result
        self.calls = []

    def invoke(self, *, stage, prompt, input_artifacts, output_artifact):
        self.calls.append(
            {
                "stage": stage,
                "prompt": prompt,
                "input_artifacts": input_artifacts,
                "output_artifact": output_artifact,
            }
        )
        return self.result


def sample_top_posts():
    return [
        {
            "post_id": "P07",
            "platform": "Trustpilot review",
            "urgency": "critical",
            "topic": "account_suspension",
            "risk_score": 93.5,
            "contains_legal_threat": True,
            "text_for_classification": "Account suspended and lawyer notified.",
        },
        {
            "post_id": "P01",
            "platform": "Twitter/X",
            "urgency": "high",
            "topic": "withdrawal",
            "risk_score": 61.0,
            "contains_legal_threat": False,
            "text_for_classification": "Waiting 9 days for withdrawal.",
        },
    ]


def sample_narratives():
    return [
        Narrative(
            narrative_id="N1",
            title="Withdrawal and account lock concerns",
            supporting_post_ids=["P07", "P01"],
            narrative_strength="strong",
            estimated_hours_until_trending=6,
            recommended_action="Coordinate support and comms response.",
        )
    ]


def test_route_escalations_writes_artifact_and_uses_top_posts_only(tmp_path):
    output_path = tmp_path / "escalation_routing.json"
    llm = FakeRoutingLLM(
        EscalationRoutingResult(
            routes=[
                RoutedEscalation(
                    post_id="P07",
                    teams=["Legal", "PR/Comms", "Customer Support"],
                    briefing_note="Legal/Comms: customer is threatening formal action; verify case status before public response.",
                )
            ]
        )
    )

    routes = route_escalations(
        sample_top_posts(),
        sample_narratives(),
        llm=llm,
        output_artifact=output_path,
    )

    assert [route.post_id for route in routes] == ["P07"]
    assert len(llm.calls) == 1
    assert llm.calls[0]["stage"] == "ROUTING_COMPLETE"
    assert "Customer Support" in llm.calls[0]["prompt"]
    assert "top 5" in llm.calls[0]["prompt"]
    assert "P07" in llm.calls[0]["prompt"]

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written[0]["teams"] == ["Legal", "PR/Comms", "Customer Support"]


def test_routed_escalation_rejects_unknown_team():
    with pytest.raises(ValidationError):
        RoutedEscalation(
            post_id="P01",
            teams=["Social Team"],
            briefing_note="Invalid destination.",
        )


def test_route_escalations_rejects_route_for_post_outside_top_five(tmp_path):
    llm = FakeRoutingLLM(
        EscalationRoutingResult(
            routes=[
                RoutedEscalation(
                    post_id="P99",
                    teams=["PR/Comms"],
                    briefing_note="This post was not supplied.",
                )
            ]
        )
    )

    with pytest.raises(ValueError, match="top escalation posts"):
        route_escalations(
            sample_top_posts(),
            sample_narratives(),
            llm=llm,
            output_artifact=tmp_path / "escalation_routing.json",
        )
