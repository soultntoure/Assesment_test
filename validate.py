from __future__ import annotations

import sys
from pathlib import Path

from src.solution.artifacts import write_json, write_text
from src.solution.constants import (
    CLASSIFIED_POSTS_PATH,
    COMPETITOR_SIGNALS_PATH,
    ESCALATION_ROUTING_PATH,
    LLM_CALLS_PATH,
    NARRATIVES_DETECTED,
    NARRATIVES_PATH,
    POSTS_CLASSIFIED,
    RESPONSE_DRAFTS_GENERATED,
    RESPONSE_DRAFTS_PATH,
    RISK_SCORES_PATH,
    ROUTING_COMPLETE,
    SENTIMENT_TREND_PATH,
)
from src.solution.llm import log_llm_call
from src.solution.optional_analysis import write_competitor_signals, write_sentiment_trend
from src.solution.preprocessing import run_preprocessing
from src.solution.risk import run_risk_scoring, top_escalation_post_ids
from src.solution.schemas import ClassifiedPost, Narrative, ResponseDraft, RoutedEscalation
from src.solution.validation import validate_artifacts


def main() -> int:
    regenerate_available_artifacts()
    report = validate_artifacts(Path.cwd())
    if report.ok:
        print("Validation passed.")
        return 0

    print("Validation failed:")
    for error in report.errors:
        print(f"- {error}")
    return 1


def regenerate_available_artifacts() -> None:
    preprocessed = run_preprocessing(Path.cwd())
    classified = _classify_heuristically(preprocessed)
    write_json(CLASSIFIED_POSTS_PATH, [post.model_dump() for post in classified])
    _log_stage(POSTS_CLASSIFIED, "heuristic classification", ["preprocessed_posts.json"], CLASSIFIED_POSTS_PATH)

    narratives = _detect_narratives_heuristically(classified)
    write_json(NARRATIVES_PATH, [narrative.model_dump() for narrative in narratives])
    _log_stage(
        NARRATIVES_DETECTED,
        "heuristic narrative detection from classified_posts.json",
        ["classified_posts.json", "preprocessed_posts.json"],
        NARRATIVES_PATH,
    )

    risk_scores = run_risk_scoring(classified, preprocessed, narratives)
    top_ids = set(top_escalation_post_ids(risk_scores))
    context_by_id = {post.post_id: post for post in preprocessed}
    top_posts = [
        {
            **score.model_dump(),
            "platform": context_by_id[score.post_id].platform,
            "text_for_classification": context_by_id[score.post_id].text_for_classification,
        }
        for score in risk_scores
        if score.post_id in top_ids
    ]

    routes = _route_heuristically(top_posts, classified)
    write_json(ESCALATION_ROUTING_PATH, [route.model_dump() for route in routes])
    _log_stage(ROUTING_COMPLETE, "heuristic routing", ["risk_scores.json", "narratives.json"], ESCALATION_ROUTING_PATH)

    drafts = _draft_heuristically(preprocessed, classified)
    write_text(RESPONSE_DRAFTS_PATH, _format_drafts(drafts))
    _log_stage(RESPONSE_DRAFTS_GENERATED, "heuristic public response drafts", ["risk_scores.json"], RESPONSE_DRAFTS_PATH)

    write_sentiment_trend(classified, preprocessed, output_artifact=SENTIMENT_TREND_PATH)
    write_competitor_signals(preprocessed, output_artifact=COMPETITOR_SIGNALS_PATH)


def _classify_heuristically(preprocessed) -> list[ClassifiedPost]:
    classified = []
    for post in preprocessed:
        text = post.text_for_classification.casefold()
        legal = any(term in text for term in ("lawyer", "regulator", "formal complaint", "chargeback"))
        competitor = any(term in text for term in ("alternatives", "proper platforms", "moved to"))
        if any(term in text for term in ("withdraw", "withdrawal", "locked out", "chargeback")):
            topic = "withdrawal"
        elif any(term in text for term in ("suspend", "suspended", "blocked", "locked")):
            topic = "account_suspension"
        elif "spread" in text or "pip" in text:
            topic = "spread_pricing"
        elif any(term in text for term in ("bot", "backtest", "execution engine", "rsi")):
            topic = "technical"
        elif any(term in text for term in ("deposit", "bank transfer", "pix", "payment")):
            topic = "deposit"
        elif any(term in text for term in ("kyc", "document", "compliance")):
            topic = "kyc"
        elif "regulated" in text or "regulation" in text:
            topic = "regulatory"
        elif any(term in text for term in ("interface", "chart", "platform", "products", "pricing")):
            topic = "product_feedback"
        else:
            topic = "general"

        positive = any(term in text for term in ("excellent", "shoutout", "clean", "improved", "finally", "fast execution"))
        negative = any(
            term in text
            for term in (
                "scam",
                "delay",
                "waiting",
                "widened",
                "suspended",
                "failing",
                "complaint",
                "worse",
                "frustrating",
                "locked",
            )
        )
        sentiment = "mixed" if positive and negative else "positive" if positive else "negative" if negative else "neutral"
        urgency = "critical" if legal else "high" if topic in {"withdrawal", "account_suspension", "deposit"} and negative else "medium" if negative or topic in {"kyc", "regulatory", "technical", "spread_pricing"} else "low"
        classified.append(
            ClassifiedPost(
                post_id=post.post_id,
                sentiment=sentiment,
                topic=topic,
                urgency=urgency,
                contains_legal_threat=legal,
                contains_competitor_mention=competitor,
                original_language=post.original_language,
                translated=post.translated,
            )
        )
    return classified


def _detect_narratives_heuristically(classified: list[ClassifiedPost]) -> list[Narrative]:
    groups = [
        ("N01", "Account access, KYC and withdrawal friction", {"withdrawal", "account_suspension", "kyc"}),
        ("N02", "Spread pricing concerns", {"spread_pricing"}),
        ("N03", "Bot and execution reliability concerns", {"technical"}),
        ("N04", "Deposit and payment reliability", {"deposit"}),
    ]
    narratives = []
    for narrative_id, title, topics in groups:
        ids = [post.post_id for post in classified if post.topic in topics and post.sentiment in {"negative", "mixed", "neutral"}]
        if len(ids) >= 1:
            narratives.append(
                Narrative(
                    narrative_id=narrative_id,
                    title=title,
                    supporting_post_ids=ids,
                    narrative_strength="strong" if len(ids) >= 3 else "moderate" if len(ids) == 2 else "weak",
                    estimated_hours_until_trending=6 if len(ids) >= 3 else 12,
                    recommended_action="Coordinate owner teams, verify operational facts, and prepare approved messaging.",
                )
            )
    return narratives


def _route_heuristically(top_posts: list[dict], classified: list[ClassifiedPost]) -> list[RoutedEscalation]:
    by_id = {post.post_id: post for post in classified}
    routes = []
    for post in top_posts:
        classified_post = by_id[post["post_id"]]
        teams = ["PR/Comms"]
        if classified_post.contains_legal_threat:
            teams.extend(["Legal", "Compliance"])
        if classified_post.topic in {"withdrawal", "account_suspension", "kyc"}:
            teams.extend(["Customer Support", "Compliance"])
        if classified_post.topic == "deposit":
            teams.extend(["Finance", "Customer Support"])
        if classified_post.topic in {"technical", "product_feedback", "spread_pricing"}:
            teams.extend(["Product", "Engineering"])
        routes.append(
            RoutedEscalation(
                post_id=post["post_id"],
                teams=list(dict.fromkeys(teams)),
                briefing_note=f"{classified_post.topic}: risk score {post['risk_score']:.2f}. Verify facts and align response owner.",
            )
        )
    return routes


def _draft_heuristically(preprocessed, classified: list[ClassifiedPost]) -> list[ResponseDraft]:
    context_by_id = {post.post_id: post for post in preprocessed}
    drafts = []
    for classified_post in classified:
        if classified_post.urgency != "critical" and not classified_post.contains_legal_threat:
            continue
        drafts.append(
            ResponseDraft(
                post_id=classified_post.post_id,
                platform=str(context_by_id[classified_post.post_id].platform or "social platform"),
                draft_response="We understand your concern and want to help. Please contact us through our secure support channel with your ticket reference so the team can review the case details. We cannot discuss account-specific information publicly.",
                send_gate_note="Confirm account/ticket status, approved support path, and Legal/Compliance sign-off before posting.",
            )
        )
    return drafts


def _format_drafts(drafts: list[ResponseDraft]) -> str:
    lines = ["# Public Response Drafts", ""]
    for draft in drafts:
        lines.extend(
            [
                f"## {draft.post_id}",
                "",
                f"Platform: {draft.platform}",
                "",
                f"Draft response: {draft.draft_response}",
                "",
                f"Send-gate note: {draft.send_gate_note}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _log_stage(stage: str, prompt: str, input_artifacts: list[str], output_artifact: Path) -> None:
    if stage == POSTS_CLASSIFIED:
        write_text(LLM_CALLS_PATH, "")
    log_llm_call(
        stage=stage,
        provider="deterministic-fallback",
        model="heuristic-v1",
        prompt=prompt,
        input_artifacts=input_artifacts,
        output_artifact=output_artifact,
    )


if __name__ == "__main__":
    sys.exit(main())
