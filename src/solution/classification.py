from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.solution.artifacts import write_json
from src.solution.constants import CLASSIFIED_POSTS_PATH, PREPROCESSED_POSTS_PATH, POSTS_CLASSIFIED
from src.solution.llm import StructuredLLM
from src.solution.prompts import build_classification_prompt
from src.solution.schemas import ClassificationResult, ClassifiedPost, PreprocessedPost


def classify_posts(
    preprocessed_posts: Iterable[PreprocessedPost | dict],
    *,
    llm=None,
    input_artifact: str | Path = PREPROCESSED_POSTS_PATH,
    output_artifact: str | Path = CLASSIFIED_POSTS_PATH,
) -> list[ClassifiedPost]:
    posts = [PreprocessedPost.model_validate(post) for post in preprocessed_posts]
    prompt = build_classification_prompt(posts)
    classifier = llm or StructuredLLM(schema=ClassificationResult)

    result = classifier.invoke(
        stage=POSTS_CLASSIFIED,
        prompt=prompt,
        input_artifacts=[input_artifact],
        output_artifact=output_artifact,
    )

    classified = [ClassifiedPost.model_validate(post) for post in result.posts]
    _ensure_one_classification_per_post(posts, classified)
    write_json(output_artifact, [post.model_dump() for post in classified])
    return classified


def _ensure_one_classification_per_post(
    preprocessed_posts: list[PreprocessedPost],
    classified_posts: list[ClassifiedPost],
) -> None:
    expected_ids = [post.post_id for post in preprocessed_posts]
    actual_ids = [post.post_id for post in classified_posts]
    if sorted(expected_ids) != sorted(actual_ids) or len(actual_ids) != len(set(actual_ids)):
        raise ValueError("Expected exactly one classification for every preprocessed post")
