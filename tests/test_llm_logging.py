import json
import sys
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solution.llm import (
    StructuredLLM,
    hash_prompt,
    log_llm_call,
    normalise_input_artifacts,
)


class ExampleResponse(BaseModel):
    value: str


class FakeStructuredModel:
    def invoke(self, messages):
        assert messages == "classify this"
        return ExampleResponse(value="ok")


def test_hash_prompt_is_deterministic_sha256():
    first = hash_prompt("same prompt")
    second = hash_prompt("same prompt")
    different = hash_prompt("different prompt")

    assert first == second
    assert first != different
    assert len(first) == 64
    assert all(char in "0123456789abcdef" for char in first)


def test_normalise_input_artifacts_serialises_paths_as_strings(tmp_path):
    path = tmp_path / "preprocessed_posts.json"

    assert normalise_input_artifacts([path, "classified_posts.json"]) == [
        str(path),
        "classified_posts.json",
    ]


def test_log_llm_call_appends_required_jsonl_record(tmp_path):
    log_path = tmp_path / "llm_calls.jsonl"

    log_llm_call(
        stage="POSTS_CLASSIFIED",
        provider="openrouter",
        model="anthropic/claude-haiku-4-5",
        prompt="Classify these posts",
        input_artifacts=[tmp_path / "preprocessed_posts.json"],
        output_artifact=tmp_path / "classified_posts.json",
        log_path=log_path,
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["stage"] == "POSTS_CLASSIFIED"
    assert record["provider"] == "openrouter"
    assert record["model"] == "anthropic/claude-haiku-4-5"
    assert record["prompt_hash"] == hash_prompt("Classify these posts")
    assert record["input_artifacts"] == [str(tmp_path / "preprocessed_posts.json")]
    assert record["output_artifact"] == str(tmp_path / "classified_posts.json")
    assert "T" in record["timestamp"]


def test_structured_llm_invokes_model_and_logs_call(tmp_path):
    log_path = tmp_path / "llm_calls.jsonl"
    llm = StructuredLLM(
        schema=ExampleResponse,
        model_name="test/model",
        provider="fake-provider",
        log_path=log_path,
        structured_model=FakeStructuredModel(),
    )

    result = llm.invoke(
        stage="POSTS_CLASSIFIED",
        prompt="classify this",
        input_artifacts=["preprocessed_posts.json"],
        output_artifact="classified_posts.json",
    )

    assert result == ExampleResponse(value="ok")
    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["stage"] == "POSTS_CLASSIFIED"
    assert record["provider"] == "fake-provider"
    assert record["model"] == "test/model"
