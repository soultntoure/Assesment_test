import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solution.classification import classify_posts
from src.solution.constants import SENTIMENT_VALUES, TOPIC_VALUES, URGENCY_VALUES
from src.solution.prompts import build_classification_prompt
from src.solution.schemas import ClassificationResult, ClassifiedPost, PreprocessedPost


class FakeClassificationLLM:
    def __init__(self) -> None:
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
        return ClassificationResult(
            posts=[
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
                    post_id="P10",
                    sentiment="negative",
                    topic="kyc",
                    urgency="medium",
                    contains_legal_threat=False,
                    contains_competitor_mention=False,
                    original_language="ms",
                    translated=True,
                ),
            ]
        )


def sample_preprocessed_posts():
    return [
        PreprocessedPost(
            post_id="P01",
            platform="Twitter/X",
            timestamp="2025-04-10T08:15:00Z",
            engagement={"likes": 12},
            original_text="Waiting 9 days for withdrawal.",
            text_for_classification="Waiting 9 days for withdrawal.",
            original_language="en",
            translated=False,
        ),
        PreprocessedPost(
            post_id="P10",
            platform="Facebook group (Forex Malaysia)",
            timestamp="2025-04-10T14:00:00Z",
            engagement={"reactions": 29},
            original_text="Adakah sesiapa tahu kenapa Deriv minta dokumen tambahan tiba-tiba?",
            text_for_classification="Does anyone know why Deriv suddenly asked for additional documents?",
            original_language="ms",
            translated=True,
        ),
    ]


def test_classification_prompt_includes_schema_and_controlled_vocabularies():
    prompt = build_classification_prompt(sample_preprocessed_posts())

    for value in SENTIMENT_VALUES:
        assert value in prompt
    for value in TOPIC_VALUES:
        assert value in prompt
    for value in URGENCY_VALUES:
        assert value in prompt

    assert "contains_legal_threat" in prompt
    assert "contains_competitor_mention" in prompt
    assert "Do not invent" in prompt
    assert "text_for_classification" in prompt


def test_classify_posts_makes_one_llm_call_and_writes_artifact(tmp_path):
    llm = FakeClassificationLLM()
    output_path = tmp_path / "classified_posts.json"

    classified = classify_posts(
        sample_preprocessed_posts(),
        llm=llm,
        input_artifact="preprocessed_posts.json",
        output_artifact=output_path,
    )

    assert len(classified) == 2
    assert [post.post_id for post in classified] == ["P01", "P10"]
    assert len(llm.calls) == 1
    assert llm.calls[0]["stage"] == "POSTS_CLASSIFIED"
    assert llm.calls[0]["input_artifacts"] == ["preprocessed_posts.json"]
    assert llm.calls[0]["output_artifact"] == output_path

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written[0]["topic"] == "withdrawal"
    assert written[1]["original_language"] == "ms"
    assert written[1]["translated"] is True


def test_classify_posts_rejects_missing_or_extra_post_ids(tmp_path):
    class MissingPostLLM:
        def invoke(self, *, stage, prompt, input_artifacts, output_artifact):
            return ClassificationResult(
                posts=[
                    ClassifiedPost(
                        post_id="P01",
                        sentiment="negative",
                        topic="withdrawal",
                        urgency="high",
                        contains_legal_threat=False,
                        contains_competitor_mention=False,
                        original_language="en",
                        translated=False,
                    )
                ]
            )

    with pytest.raises(ValueError, match="exactly one classification"):
        classify_posts(
            sample_preprocessed_posts(),
            llm=MissingPostLLM(),
            output_artifact=tmp_path / "classified_posts.json",
        )


def test_classified_post_rejects_unknown_controlled_values():
    with pytest.raises(ValidationError):
        ClassifiedPost(
            post_id="P99",
            sentiment="angry",
            topic="withdrawal",
            urgency="high",
            contains_legal_threat=False,
            contains_competitor_mention=False,
            original_language="en",
            translated=False,
        )
