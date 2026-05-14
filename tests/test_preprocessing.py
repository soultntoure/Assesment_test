import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solution.preprocessing import (
    detect_language,
    load_posts,
    preprocess_posts,
    resolve_posts_path,
    translate_to_english,
)


MALAY_TEXT = (
    "Adakah sesiapa tahu kenapa Deriv minta dokumen tambahan tiba-tiba? "
    "Saya dah lulus KYC sebelum ini."
)


def test_resolve_posts_path_prefers_data_folder(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_posts = data_dir / "posts.json"
    root_posts = tmp_path / "posts.json"
    data_posts.write_text("[]", encoding="utf-8")
    root_posts.write_text("[]", encoding="utf-8")

    assert resolve_posts_path(tmp_path) == data_posts


def test_resolve_posts_path_falls_back_to_root_posts(tmp_path):
    root_posts = tmp_path / "posts.json"
    root_posts.write_text("[]", encoding="utf-8")

    assert resolve_posts_path(tmp_path) == root_posts


def test_load_posts_validates_required_shape(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    posts_path = data_dir / "posts.json"
    posts_path.write_text(
        """
        [
          {
            "id": "P01",
            "platform": "Twitter/X",
            "text": "Delayed withdrawal",
            "timestamp": "2025-04-10T08:15:00Z",
            "engagement": {"likes": 3}
          }
        ]
        """,
        encoding="utf-8",
    )

    posts = load_posts(tmp_path)

    assert len(posts) == 1
    assert posts[0].id == "P01"


def test_detect_language_marks_malay_fixture_text():
    assert detect_language(MALAY_TEXT) == "ms"
    assert detect_language("Deriv withdrawals have been delayed.") == "en"


def test_translate_to_english_has_deterministic_malay_fallback():
    translated = translate_to_english(MALAY_TEXT, "ms")

    assert "additional documents" in translated
    assert "KYC" in translated


def test_preprocess_posts_preserves_original_text_and_translation():
    raw_posts = [
        {
            "id": "P10",
            "platform": "Facebook group (Forex Malaysia)",
            "text": MALAY_TEXT,
            "timestamp": "2025-04-10T14:00:00Z",
            "engagement": {"reactions": 29, "comments": 15},
        }
    ]

    preprocessed = preprocess_posts(raw_posts)

    assert len(preprocessed) == 1
    post = preprocessed[0]
    assert post.post_id == "P10"
    assert post.original_text == MALAY_TEXT
    assert post.original_language == "ms"
    assert post.translated is True
    assert post.text_for_classification != MALAY_TEXT
    assert "additional documents" in post.text_for_classification
    assert post.platform == "Facebook group (Forex Malaysia)"
    assert post.engagement == {"reactions": 29, "comments": 15}
