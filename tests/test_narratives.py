import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solution.narratives import NarrativeDetectionResult, detect_narratives
from src.solution.schemas import ClassifiedPost, Narrative, PreprocessedPost


class FakeNarrativeLLM:
    def __init__(self, result: NarrativeDetectionResult) -> None:
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


def sample_classified_posts():
    return [
        ClassifiedPost(
            post_id="P01",
            sentiment="negative",
            topic="withdrawal",
            urgency="high",
            contains_legal_threat=False,
            contains_competitor_mention=False,
            original_language="en",
            translated=False,
        ),
        ClassifiedPost(
            post_id="P07",
            sentiment="negative",
            topic="account_suspension",
            urgency="critical",
            contains_legal_threat=True,
            contains_competitor_mention=False,
            original_language="en",
            translated=False,
        ),
    ]


def sample_post_context():
    return [
        PreprocessedPost(
            post_id="P01",
            platform="Twitter/X",
            timestamp="2025-04-10T08:15:00Z",
            engagement={"likes": 12, "reposts": 8},
            original_text="Waiting 9 days for withdrawal.",
            text_for_classification="Waiting 9 days for withdrawal.",
            original_language="en",
            translated=False,
        ),
        PreprocessedPost(
            post_id="P07",
            platform="Trustpilot review",
            timestamp="2025-04-10T12:15:00Z",
            engagement={"helpful_votes": 18},
            original_text="Account suspended and lawyer notified.",
            text_for_classification="Account suspended and lawyer notified.",
            original_language="en",
            translated=False,
        ),
    ]


def test_detect_narratives_writes_artifact_and_includes_classified_context(tmp_path):
    output_path = tmp_path / "narratives.json"
    llm = FakeNarrativeLLM(
        NarrativeDetectionResult(
            narratives=[
                Narrative(
                    narrative_id="N1",
                    title="Withdrawal and account lock concerns",
                    supporting_post_ids=["P01", "P07"],
                    narrative_strength="strong",
                    estimated_hours_until_trending=6,
                    recommended_action="Coordinate support and comms response.",
                )
            ]
        )
    )

    narratives = detect_narratives(
        sample_classified_posts(),
        sample_post_context(),
        llm=llm,
        input_artifact="classified_posts.json",
        output_artifact=output_path,
    )

    assert [narrative.narrative_id for narrative in narratives] == ["N1"]
    assert len(llm.calls) == 1
    assert llm.calls[0]["stage"] == "NARRATIVES_DETECTED"
    assert llm.calls[0]["input_artifacts"] == ["classified_posts.json"]
    assert "text_for_classification" in llm.calls[0]["prompt"]
    assert "sentiment" in llm.calls[0]["prompt"]
    assert "engagement" in llm.calls[0]["prompt"]

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written[0]["supporting_post_ids"] == ["P01", "P07"]


def test_detect_narratives_rejects_unknown_supporting_post_id(tmp_path):
    llm = FakeNarrativeLLM(
        NarrativeDetectionResult(
            narratives=[
                Narrative(
                    narrative_id="N1",
                    title="Unknown post cluster",
                    supporting_post_ids=["P01", "P99"],
                    narrative_strength="moderate",
                    estimated_hours_until_trending=12,
                    recommended_action="Investigate.",
                )
            ]
        )
    )

    with pytest.raises(ValueError, match="unknown supporting post IDs"):
        detect_narratives(
            sample_classified_posts(),
            sample_post_context(),
            llm=llm,
            output_artifact=tmp_path / "narratives.json",
        )


def test_detect_narratives_requires_context_for_each_classified_post(tmp_path):
    with pytest.raises(ValueError, match="post_context"):
        detect_narratives(
            sample_classified_posts(),
            sample_post_context()[:1],
            llm=FakeNarrativeLLM(NarrativeDetectionResult(narratives=[])),
            output_artifact=tmp_path / "narratives.json",
        )
