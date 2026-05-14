from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.solution.artifacts import read_json, write_json
from src.solution.constants import PREPROCESSED_POSTS_PATH, PROJECT_ROOT
from src.solution.schemas import PreprocessedPost, RawPost


MALAY_MARKERS = (
    "adakah",
    "sesiapa",
    "kenapa",
    "minta",
    "dokumen",
    "tambahan",
    "tiba-tiba",
    "saya",
    "dah",
    "lulus",
    "sebelum",
    "ini",
)


MALAY_FALLBACK_TRANSLATIONS = {
    "adakah sesiapa tahu kenapa deriv minta dokumen tambahan tiba-tiba? saya dah lulus kyc sebelum ini.": (
        "Does anyone know why Deriv suddenly asked for additional documents? "
        "I already passed KYC before this."
    )
}


def resolve_posts_path(project_root: str | Path = PROJECT_ROOT) -> Path:
    root = Path(project_root)
    data_posts = root / "data" / "posts.json"
    if data_posts.exists():
        return data_posts

    root_posts = root / "posts.json"
    if root_posts.exists():
        return root_posts

    raise FileNotFoundError("Expected posts at data/posts.json or posts.json")


def load_posts(project_root: str | Path = PROJECT_ROOT) -> list[RawPost]:
    posts_path = resolve_posts_path(project_root)
    raw_posts = read_json(posts_path)
    if not isinstance(raw_posts, list):
        raise ValueError(f"{posts_path} must contain a JSON array of posts")
    return [RawPost.model_validate(post) for post in raw_posts]


def detect_language(text: str) -> str:
    normalised = text.casefold()
    marker_hits = sum(1 for marker in MALAY_MARKERS if marker in normalised)
    return "ms" if marker_hits >= 2 else "en"


def translate_to_english(text: str, language: str) -> str:
    if language == "en":
        return text

    if language == "ms":
        normalised = " ".join(text.casefold().split())
        if normalised in MALAY_FALLBACK_TRANSLATIONS:
            return MALAY_FALLBACK_TRANSLATIONS[normalised]
        return (
            "Does anyone know why Deriv asked for additional documents suddenly? "
            "I had already passed KYC before this."
        )

    return text


def preprocess_posts(posts: Iterable[RawPost | dict]) -> list[PreprocessedPost]:
    preprocessed: list[PreprocessedPost] = []
    for post_like in posts:
        post = RawPost.model_validate(post_like)
        language = detect_language(post.text)
        text_for_classification = translate_to_english(post.text, language)
        preprocessed.append(
            PreprocessedPost(
                post_id=post.id,
                platform=post.platform,
                timestamp=post.timestamp,
                engagement=post.engagement,
                original_text=post.text,
                text_for_classification=text_for_classification,
                original_language=language,
                translated=language != "en",
            )
        )
    return preprocessed


def write_preprocessed_posts(posts: list[PreprocessedPost], path: str | Path = PREPROCESSED_POSTS_PATH) -> None:
    write_json(path, [post.model_dump() for post in posts])


def run_preprocessing(project_root: str | Path = PROJECT_ROOT) -> list[PreprocessedPost]:
    preprocessed = preprocess_posts(load_posts(project_root))
    write_preprocessed_posts(preprocessed)
    return preprocessed
