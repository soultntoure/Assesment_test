import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solution.risk import compute_risk_scores
from src.solution.validation import validate_artifacts


def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def base_artifacts(tmp_path: Path):
    (tmp_path / "data").mkdir()
    raw_posts = [
        {
            "id": "P01",
            "platform": "Twitter/X",
            "text": "Withdrawal delayed.",
            "timestamp": "2025-04-10T08:00:00Z",
            "engagement": {"likes": 10},
        },
        {
            "id": "P02",
            "platform": "Facebook",
            "text": "Adakah sesiapa tahu kenapa Deriv minta dokumen tambahan tiba-tiba?",
            "timestamp": "2025-04-10T09:00:00Z",
            "engagement": {"comments": 2},
        },
        {
            "id": "P03",
            "platform": "Reddit",
            "text": "Good platform.",
            "timestamp": "2025-04-10T10:00:00Z",
            "engagement": {"upvotes": 1},
        },
        {
            "id": "P04",
            "platform": "Trustpilot",
            "text": "Lawyer notified.",
            "timestamp": "2025-04-10T11:00:00Z",
            "engagement": {"helpful_votes": 3},
        },
        {
            "id": "P05",
            "platform": "Telegram",
            "text": "Deposits failing.",
            "timestamp": "2025-04-10T12:00:00Z",
            "engagement": {"reactions": 4},
        },
    ]
    preprocessed = [
        {
            "post_id": "P01",
            "platform": "Twitter/X",
            "timestamp": "2025-04-10T08:00:00Z",
            "engagement": {"likes": 10},
            "original_text": "Withdrawal delayed.",
            "text_for_classification": "Withdrawal delayed.",
            "original_language": "en",
            "translated": False,
        },
        {
            "post_id": "P02",
            "platform": "Facebook",
            "timestamp": "2025-04-10T09:00:00Z",
            "engagement": {"comments": 2},
            "original_text": "Adakah sesiapa tahu kenapa Deriv minta dokumen tambahan tiba-tiba?",
            "text_for_classification": "Does anyone know why Deriv asked for additional documents suddenly?",
            "original_language": "ms",
            "translated": True,
        },
        {
            "post_id": "P03",
            "platform": "Reddit",
            "timestamp": "2025-04-10T10:00:00Z",
            "engagement": {"upvotes": 1},
            "original_text": "Good platform.",
            "text_for_classification": "Good platform.",
            "original_language": "en",
            "translated": False,
        },
        {
            "post_id": "P04",
            "platform": "Trustpilot",
            "timestamp": "2025-04-10T11:00:00Z",
            "engagement": {"helpful_votes": 3},
            "original_text": "Lawyer notified.",
            "text_for_classification": "Lawyer notified.",
            "original_language": "en",
            "translated": False,
        },
        {
            "post_id": "P05",
            "platform": "Telegram",
            "timestamp": "2025-04-10T12:00:00Z",
            "engagement": {"reactions": 4},
            "original_text": "Deposits failing.",
            "text_for_classification": "Deposits failing.",
            "original_language": "en",
            "translated": False,
        },
    ]
    classified = [
        {
            "post_id": "P01",
            "sentiment": "negative",
            "topic": "withdrawal",
            "urgency": "critical",
            "contains_legal_threat": False,
            "contains_competitor_mention": False,
            "original_language": "en",
            "translated": False,
        },
        {
            "post_id": "P02",
            "sentiment": "negative",
            "topic": "kyc",
            "urgency": "high",
            "contains_legal_threat": False,
            "contains_competitor_mention": False,
            "original_language": "ms",
            "translated": True,
        },
        {
            "post_id": "P03",
            "sentiment": "positive",
            "topic": "product_feedback",
            "urgency": "low",
            "contains_legal_threat": False,
            "contains_competitor_mention": False,
            "original_language": "en",
            "translated": False,
        },
        {
            "post_id": "P04",
            "sentiment": "negative",
            "topic": "account_suspension",
            "urgency": "critical",
            "contains_legal_threat": True,
            "contains_competitor_mention": False,
            "original_language": "en",
            "translated": False,
        },
        {
            "post_id": "P05",
            "sentiment": "negative",
            "topic": "deposit",
            "urgency": "medium",
            "contains_legal_threat": False,
            "contains_competitor_mention": False,
            "original_language": "en",
            "translated": False,
        },
    ]
    narratives = [
        {
            "narrative_id": "N01",
            "title": "Compliance friction",
            "supporting_post_ids": ["P01", "P02", "P04"],
            "narrative_strength": "strong",
            "estimated_hours_until_trending": 6,
            "recommended_action": "Coordinate response.",
        }
    ]
    risk_scores = [score.model_dump() for score in compute_risk_scores(classified, preprocessed, narratives)]
    routing = [
        {"post_id": score["post_id"], "teams": ["PR/Comms"], "briefing_note": "Internal update."}
        for score in risk_scores[:5]
    ]

    write_json(tmp_path / "data" / "posts.json", raw_posts)
    write_json(tmp_path / "preprocessed_posts.json", preprocessed)
    write_json(tmp_path / "classified_posts.json", classified)
    write_json(tmp_path / "narratives.json", narratives)
    write_json(tmp_path / "risk_scores.json", risk_scores)
    write_json(tmp_path / "escalation_routing.json", routing)
    (tmp_path / "response_drafts.md").write_text(
        "## P01\nDraft response.\nSend-gate note: confirm case status.\n\n"
        "## P04\nDraft response.\nSend-gate note: confirm legal/compliance approval.\n",
        encoding="utf-8",
    )
    (tmp_path / "llm_calls.jsonl").write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "stage": "POSTS_CLASSIFIED",
                    "timestamp": "2026-05-14T00:00:00+00:00",
                    "provider": "test",
                    "model": "test",
                    "prompt_hash": "a",
                    "input_artifacts": ["preprocessed_posts.json"],
                    "output_artifact": "classified_posts.json",
                },
                {
                    "stage": "NARRATIVES_DETECTED",
                    "timestamp": "2026-05-14T00:00:00+00:00",
                    "provider": "test",
                    "model": "test",
                    "prompt_hash": "b",
                    "input_artifacts": ["classified_posts.json"],
                    "output_artifact": "narratives.json",
                },
                {
                    "stage": "ROUTING_COMPLETE",
                    "timestamp": "2026-05-14T00:00:00+00:00",
                    "provider": "test",
                    "model": "test",
                    "prompt_hash": "c",
                    "input_artifacts": ["risk_scores.json"],
                    "output_artifact": "escalation_routing.json",
                },
                {
                    "stage": "RESPONSE_DRAFTS_GENERATED",
                    "timestamp": "2026-05-14T00:00:00+00:00",
                    "provider": "test",
                    "model": "test",
                    "prompt_hash": "d",
                    "input_artifacts": ["classified_posts.json"],
                    "output_artifact": "response_drafts.md",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return risk_scores


def test_validate_artifacts_passes_for_complete_contract(tmp_path):
    base_artifacts(tmp_path)

    report = validate_artifacts(tmp_path)

    assert report.ok, report.errors


def test_validate_artifacts_rejects_non_deterministic_risk_score(tmp_path):
    risk_scores = base_artifacts(tmp_path)
    risk_scores[0]["risk_score"] += 1
    write_json(tmp_path / "risk_scores.json", risk_scores)

    report = validate_artifacts(tmp_path)

    assert not report.ok
    assert any("risk_score" in error for error in report.errors)


def test_validate_artifacts_requires_narrative_llm_call_to_use_classified_posts(tmp_path):
    base_artifacts(tmp_path)
    lines = (tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    for record in records:
        if record["stage"] == "NARRATIVES_DETECTED":
            record["input_artifacts"] = ["preprocessed_posts.json"]
    (tmp_path / "llm_calls.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    report = validate_artifacts(tmp_path)

    assert not report.ok
    assert any("Narrative detection" in error for error in report.errors)
