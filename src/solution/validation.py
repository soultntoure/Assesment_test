from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.solution.artifacts import read_json
from src.solution.constants import (
    CLASSIFIED_POSTS_PATH,
    ESCALATION_ROUTING_PATH,
    LLM_CALLS_PATH,
    NARRATIVES_DETECTED,
    NARRATIVES_PATH,
    POSTS_CLASSIFIED,
    PREPROCESSED_POSTS_PATH,
    RESPONSE_DRAFTS_GENERATED,
    RESPONSE_DRAFTS_PATH,
    RISK_SCORES_PATH,
    ROUTING_COMPLETE,
    TEAM_VALUES,
)
from src.solution.preprocessing import resolve_posts_path
from src.solution.risk import compute_risk_scores, top_escalation_post_ids
from src.solution.schemas import (
    ClassifiedPost,
    Narrative,
    PreprocessedPost,
    RawPost,
    RiskScore,
    RoutedEscalation,
)


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)

    def raise_for_errors(self) -> None:
        if not self.ok:
            raise ValueError("\n".join(self.errors))


def validate_artifacts(project_root: str | Path = Path.cwd()) -> ValidationReport:
    root = Path(project_root)
    errors: list[str] = []

    paths = _artifact_paths(root)
    for name, path in paths.items():
        if not path.exists():
            errors.append(f"Missing required artifact: {path}")

    if errors:
        return ValidationReport(ok=False, errors=errors)

    try:
        raw_posts = [RawPost.model_validate(post) for post in read_json(resolve_posts_path(root))]
        preprocessed = [PreprocessedPost.model_validate(post) for post in read_json(paths["preprocessed"])]
        classified = [ClassifiedPost.model_validate(post) for post in read_json(paths["classified"])]
        narratives = [Narrative.model_validate(item) for item in read_json(paths["narratives"])]
        risk_scores = [RiskScore.model_validate(item) for item in _risk_score_items(read_json(paths["risk"]))]
        routing = [RoutedEscalation.model_validate(item) for item in read_json(paths["routing"])]
        llm_calls = _read_jsonl(paths["llm_calls"])
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        return ValidationReport(ok=False, errors=[f"Artifact parsing or schema validation failed: {exc}"])

    _check_post_coverage(raw_posts, preprocessed, classified, errors)
    _check_translations(raw_posts, preprocessed, classified, errors)
    _check_narratives_reference_classified_posts(narratives, classified, errors)
    _check_risk_scores(classified, preprocessed, narratives, risk_scores, routing, errors)
    _check_routing_teams(routing, errors)
    _check_response_drafts(classified, paths["drafts"], errors)
    _check_llm_calls(llm_calls, errors)

    return ValidationReport(ok=not errors, errors=errors)


def _artifact_paths(root: Path) -> dict[str, Path]:
    return {
        "preprocessed": root / PREPROCESSED_POSTS_PATH.name,
        "classified": root / CLASSIFIED_POSTS_PATH.name,
        "narratives": root / NARRATIVES_PATH.name,
        "risk": root / RISK_SCORES_PATH.name,
        "routing": root / ESCALATION_ROUTING_PATH.name,
        "drafts": root / RESPONSE_DRAFTS_PATH.name,
        "llm_calls": root / LLM_CALLS_PATH.name,
    }


def _risk_score_items(data: Any) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("risk_scores"), list):
        return data["risk_scores"]
    raise ValueError("risk_scores.json must be a list or contain a risk_scores list")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSONL") from exc
    return records


def _check_post_coverage(
    raw_posts: list[RawPost],
    preprocessed: list[PreprocessedPost],
    classified: list[ClassifiedPost],
    errors: list[str],
) -> None:
    raw_ids = {post.id for post in raw_posts}
    preprocessed_ids = {post.post_id for post in preprocessed}
    classified_ids = {post.post_id for post in classified}
    if raw_ids != preprocessed_ids:
        errors.append("Preprocessed posts do not match input post IDs")
    if raw_ids != classified_ids:
        errors.append("Classified posts do not match input post IDs")


def _check_translations(
    raw_posts: list[RawPost],
    preprocessed: list[PreprocessedPost],
    classified: list[ClassifiedPost],
    errors: list[str],
) -> None:
    raw_by_id = {post.id: post for post in raw_posts}
    classified_by_id = {post.post_id: post for post in classified}
    for post in preprocessed:
        if post.post_id in raw_by_id and post.original_text != raw_by_id[post.post_id].text:
            errors.append(f"{post.post_id} did not preserve original_text")
        if post.original_language != "en":
            if not post.translated:
                errors.append(f"{post.post_id} is non-English but translated is false")
            if post.text_for_classification == post.original_text:
                errors.append(f"{post.post_id} does not include translated classification text")
        classified_post = classified_by_id.get(post.post_id)
        if classified_post and (
            classified_post.original_language != post.original_language
            or classified_post.translated != post.translated
        ):
            errors.append(f"{post.post_id} classification language metadata does not match preprocessing")


def _check_narratives_reference_classified_posts(
    narratives: list[Narrative],
    classified: list[ClassifiedPost],
    errors: list[str],
) -> None:
    valid_ids = {post.post_id for post in classified}
    for narrative in narratives:
        invalid = sorted(set(narrative.supporting_post_ids) - valid_ids)
        if invalid:
            errors.append(f"{narrative.narrative_id} references unknown posts: {', '.join(invalid)}")


def _check_risk_scores(
    classified: list[ClassifiedPost],
    preprocessed: list[PreprocessedPost],
    narratives: list[Narrative],
    risk_scores: list[RiskScore],
    routing: list[RoutedEscalation],
    errors: list[str],
) -> None:
    expected = compute_risk_scores(classified, preprocessed, narratives)
    expected_by_id = {score.post_id: score for score in expected}
    actual_by_id = {score.post_id: score for score in risk_scores}
    if set(expected_by_id) != set(actual_by_id):
        errors.append("Risk scores do not cover the classified post IDs")
        return

    actual_order = [score.post_id for score in risk_scores]
    expected_order = [score.post_id for score in expected]
    if actual_order != expected_order:
        errors.append("Risk scores are not sorted by deterministic risk score descending")

    for post_id, expected_score in expected_by_id.items():
        actual = actual_by_id[post_id]
        if abs(actual.risk_score - expected_score.risk_score) > 1e-9:
            errors.append(f"{post_id} risk_score does not match deterministic computation")
        if abs(actual.raw_engagement - expected_score.raw_engagement) > 1e-9:
            errors.append(f"{post_id} raw_engagement does not match deterministic computation")

    top_ids = set(top_escalation_post_ids(expected))
    routed_ids = {route.post_id for route in routing}
    if routed_ids != top_ids:
        errors.append("Escalation routing post IDs do not match the computed top 5 risk posts")


def _check_routing_teams(routing: list[RoutedEscalation], errors: list[str]) -> None:
    allowed = set(TEAM_VALUES)
    for route in routing:
        invalid = sorted(set(route.teams) - allowed)
        if invalid:
            errors.append(f"{route.post_id} has uncontrolled routing teams: {', '.join(invalid)}")


def _check_response_drafts(classified: list[ClassifiedPost], drafts_path: Path, errors: list[str]) -> None:
    text = drafts_path.read_text(encoding="utf-8")
    required_ids = {
        post.post_id for post in classified if post.urgency == "critical" or post.contains_legal_threat
    }
    for post_id in required_ids:
        if post_id not in text:
            errors.append(f"Missing public response draft for {post_id}")
    if required_ids and not re.search(r"send[-_ ]?gate|send_gate_note", text, re.IGNORECASE):
        errors.append("Response drafts do not include send-gate notes")


def _check_llm_calls(llm_calls: list[dict[str, Any]], errors: list[str]) -> None:
    required = {
        POSTS_CLASSIFIED,
        NARRATIVES_DETECTED,
        ROUTING_COMPLETE,
        RESPONSE_DRAFTS_GENERATED,
    }
    stages = {record.get("stage") for record in llm_calls}
    missing = sorted(required - stages)
    if missing:
        errors.append(f"Missing LLM call log stages: {', '.join(missing)}")

    narrative_records = [record for record in llm_calls if record.get("stage") == NARRATIVES_DETECTED]
    if not any(
        any("classified_posts.json" in str(path) for path in record.get("input_artifacts", []))
        for record in narrative_records
    ):
        errors.append("Narrative detection LLM call did not declare classified_posts.json as an input artifact")
