from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.solution.classification import classify_posts
from src.solution.constants import (
    CLASSIFIED_POSTS_PATH,
    COMPETITOR_SIGNALS_PATH,
    ESCALATION_ROUTING_PATH,
    INIT,
    LLM_CALLS_PATH,
    MULTILINGUAL_PREPROCESSING_COMPLETE,
    NARRATIVES_DETECTED,
    NARRATIVES_PATH,
    POSTS_CLASSIFIED,
    POSTS_LOADED,
    PREPROCESSED_POSTS_PATH,
    PROJECT_ROOT,
    RESPONSE_DRAFTS_GENERATED,
    RESPONSE_DRAFTS_PATH,
    RISK_SCORES_COMPUTED,
    RISK_SCORES_PATH,
    ROUTING_COMPLETE,
    SENTIMENT_TREND_PATH,
    ESCALATIONS_SELECTED,
    VALIDATION_COMPLETE,
    RESULTS_FINALISED,
)
from src.solution.drafts import PublicDraftResult, generate_public_drafts
from src.solution.llm import log_llm_call
from src.solution.narratives import NarrativeDetectionResult, detect_narratives
from src.solution.optional_analysis import write_competitor_signals, write_sentiment_trend
from src.solution.preprocessing import load_posts, preprocess_posts, write_preprocessed_posts
from src.solution.risk import run_risk_scoring, top_escalation_post_ids
from src.solution.routing import EscalationRoutingResult, route_escalations
from src.solution.schemas import (
    ClassificationResult,
    ClassifiedPost,
    Narrative,
    ResponseDraft,
    RoutedEscalation,
)
from src.solution.state import PipelineState, advance_stage
from src.solution.validation import validate_artifacts


class DeterministicLLM:
    def __init__(self, result: Any) -> None:
        self.result = result

    def invoke(
        self,
        *,
        stage: str,
        prompt: str,
        input_artifacts,
        output_artifact,
        **_: Any,
    ) -> Any:
        log_llm_call(
            stage=stage,
            provider="deterministic-fallback",
            model="heuristic-v1",
            prompt=prompt,
            input_artifacts=input_artifacts,
            output_artifact=output_artifact,
        )
        return self.result


def load_posts_node(state: PipelineState) -> PipelineState:
    stage = advance_stage(state.get("stage", INIT), INIT, POSTS_LOADED)
    posts = load_posts(PROJECT_ROOT)
    return {
        **state,
        "stage": stage,
        "posts": posts,
        "generated_artifacts": list(state.get("generated_artifacts", [])),
    }


def multilingual_preprocessing_node(state: PipelineState) -> PipelineState:
    stage = advance_stage(state["stage"], POSTS_LOADED, MULTILINGUAL_PREPROCESSING_COMPLETE)
    preprocessed = preprocess_posts(state["posts"])
    write_preprocessed_posts(preprocessed, PREPROCESSED_POSTS_PATH)
    return _with_artifact(state, stage, "preprocessed_posts", preprocessed, PREPROCESSED_POSTS_PATH)


def classify_posts_node(state: PipelineState) -> PipelineState:
    stage = advance_stage(state["stage"], MULTILINGUAL_PREPROCESSING_COMPLETE, POSTS_CLASSIFIED)
    classified = _classify_heuristically(state["preprocessed_posts"])
    result = ClassificationResult(posts=classified)
    classified = classify_posts(
        state["preprocessed_posts"],
        llm=DeterministicLLM(result),
        input_artifact=PREPROCESSED_POSTS_PATH,
        output_artifact=CLASSIFIED_POSTS_PATH,
    )
    return _with_artifact(state, stage, "classified_posts", classified, CLASSIFIED_POSTS_PATH)


def detect_narratives_node(state: PipelineState) -> PipelineState:
    stage = advance_stage(state["stage"], POSTS_CLASSIFIED, NARRATIVES_DETECTED)
    narratives = _detect_narratives_heuristically(state["classified_posts"])
    result = NarrativeDetectionResult(narratives=narratives)
    narratives = detect_narratives(
        state["classified_posts"],
        state["preprocessed_posts"],
        llm=DeterministicLLM(result),
        input_artifact=CLASSIFIED_POSTS_PATH,
        output_artifact=NARRATIVES_PATH,
    )
    return _with_artifact(state, stage, "narratives", narratives, NARRATIVES_PATH)


def compute_risk_scores_node(state: PipelineState) -> PipelineState:
    stage = advance_stage(state["stage"], NARRATIVES_DETECTED, RISK_SCORES_COMPUTED)
    risk_scores = run_risk_scoring(
        state["classified_posts"],
        state["preprocessed_posts"],
        state["narratives"],
        output_artifact=RISK_SCORES_PATH,
    )
    return _with_artifact(state, stage, "risk_scores", risk_scores, RISK_SCORES_PATH)


def select_escalations_node(state: PipelineState) -> PipelineState:
    stage = advance_stage(state["stage"], RISK_SCORES_COMPUTED, ESCALATIONS_SELECTED)
    top_ids = top_escalation_post_ids(state["risk_scores"])
    return {**state, "stage": stage, "top_escalation_post_ids": top_ids}


def route_escalations_node(state: PipelineState) -> PipelineState:
    stage = advance_stage(state["stage"], ESCALATIONS_SELECTED, ROUTING_COMPLETE)
    top_posts = _top_posts_for_routing(state)
    routes = _route_heuristically(top_posts, state["classified_posts"])
    result = EscalationRoutingResult(routes=routes)
    routes = route_escalations(
        top_posts,
        state["narratives"],
        llm=DeterministicLLM(result),
        input_artifacts=(RISK_SCORES_PATH, NARRATIVES_PATH),
        output_artifact=ESCALATION_ROUTING_PATH,
    )
    return _with_artifact(state, stage, "routing", routes, ESCALATION_ROUTING_PATH)


def generate_response_drafts_node(state: PipelineState) -> PipelineState:
    stage = advance_stage(state["stage"], ROUTING_COMPLETE, RESPONSE_DRAFTS_GENERATED)
    selected_posts = _posts_for_public_drafts(state)
    drafts = _draft_heuristically(state["preprocessed_posts"], state["classified_posts"])
    result = PublicDraftResult(drafts=drafts)
    drafts = generate_public_drafts(
        selected_posts,
        llm=DeterministicLLM(result),
        input_artifacts=(RISK_SCORES_PATH,),
        output_artifact=RESPONSE_DRAFTS_PATH,
    )
    return _with_artifact(state, stage, "drafts", drafts, RESPONSE_DRAFTS_PATH)


def validate_results_node(state: PipelineState) -> PipelineState:
    stage = advance_stage(state["stage"], RESPONSE_DRAFTS_GENERATED, VALIDATION_COMPLETE)
    write_sentiment_trend(state["classified_posts"], state["preprocessed_posts"], output_artifact=SENTIMENT_TREND_PATH)
    write_competitor_signals(state["preprocessed_posts"], output_artifact=COMPETITOR_SIGNALS_PATH)
    generated_artifacts = [
        *state.get("generated_artifacts", []),
        str(SENTIMENT_TREND_PATH),
        str(COMPETITOR_SIGNALS_PATH),
    ]
    report = validate_artifacts(PROJECT_ROOT)
    report.raise_for_errors()
    return {**state, "stage": stage, "generated_artifacts": generated_artifacts}


def finalise_results_node(state: PipelineState) -> PipelineState:
    stage = advance_stage(state["stage"], VALIDATION_COMPLETE, RESULTS_FINALISED)
    return {**state, "stage": stage}


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("load_posts", load_posts_node)
    graph.add_node("multilingual_preprocessing", multilingual_preprocessing_node)
    graph.add_node("classify_posts", classify_posts_node)
    graph.add_node("detect_narratives", detect_narratives_node)
    graph.add_node("compute_risk_scores", compute_risk_scores_node)
    graph.add_node("select_escalations", select_escalations_node)
    graph.add_node("route_escalations", route_escalations_node)
    graph.add_node("generate_response_drafts", generate_response_drafts_node)
    graph.add_node("validate_results", validate_results_node)
    graph.add_node("finalise_results", finalise_results_node)

    graph.add_edge(START, "load_posts")
    graph.add_edge("load_posts", "multilingual_preprocessing")
    graph.add_edge("multilingual_preprocessing", "classify_posts")
    graph.add_edge("classify_posts", "detect_narratives")
    graph.add_edge("detect_narratives", "compute_risk_scores")
    graph.add_edge("compute_risk_scores", "select_escalations")
    graph.add_edge("select_escalations", "route_escalations")
    graph.add_edge("route_escalations", "generate_response_drafts")
    graph.add_edge("generate_response_drafts", "validate_results")
    graph.add_edge("validate_results", "finalise_results")
    graph.add_edge("finalise_results", END)
    return graph.compile()


def run_pipeline(initial_state: PipelineState | None = None) -> PipelineState:
    Path(LLM_CALLS_PATH).write_text("", encoding="utf-8")
    state = initial_state or {"stage": INIT, "generated_artifacts": []}
    return build_graph().invoke(state)


def _with_artifact(
    state: PipelineState,
    stage: str,
    key: str,
    value: Any,
    artifact_path: Path,
) -> PipelineState:
    return {
        **state,
        "stage": stage,
        key: value,
        "generated_artifacts": [*state.get("generated_artifacts", []), str(artifact_path)],
    }


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
        urgency = (
            "critical"
            if legal
            else "high"
            if topic in {"withdrawal", "account_suspension", "deposit"} and negative
            else "medium"
            if negative or topic in {"kyc", "regulatory", "technical", "spread_pricing"}
            else "low"
        )
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
    grouped_topics = [
        ("N01", "Account access, KYC and withdrawal friction", {"withdrawal", "account_suspension", "kyc"}),
        ("N02", "Spread pricing concerns", {"spread_pricing"}),
        ("N03", "Bot and execution reliability concerns", {"technical"}),
        ("N04", "Deposit and payment reliability", {"deposit"}),
    ]
    narratives = []
    for narrative_id, title, topics in grouped_topics:
        ids = [
            post.post_id
            for post in classified
            if post.topic in topics and post.sentiment in {"negative", "mixed", "neutral"}
        ]
        if ids:
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


def _top_posts_for_routing(state: PipelineState) -> list[dict[str, Any]]:
    top_ids = set(state["top_escalation_post_ids"])
    context_by_id = {post.post_id: post for post in state["preprocessed_posts"]}
    classified_by_id = {post.post_id: post for post in state["classified_posts"]}
    return [
        {
            **score.model_dump(),
            **classified_by_id[score.post_id].model_dump(),
            "platform": context_by_id[score.post_id].platform,
            "timestamp": context_by_id[score.post_id].timestamp,
            "text_for_classification": context_by_id[score.post_id].text_for_classification,
        }
        for score in state["risk_scores"]
        if score.post_id in top_ids
    ]


def _route_heuristically(top_posts: list[dict[str, Any]], classified: list[ClassifiedPost]) -> list[RoutedEscalation]:
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


def _posts_for_public_drafts(state: PipelineState) -> list[dict[str, Any]]:
    context_by_id = {post.post_id: post for post in state["preprocessed_posts"]}
    rows = []
    for post in state["classified_posts"]:
        context = context_by_id[post.post_id]
        rows.append(
            {
                **post.model_dump(),
                "platform": context.platform,
                "text_for_classification": context.text_for_classification,
            }
        )
    return rows


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
                draft_response=(
                    "We understand your concern and want to help. Please contact us through our secure support "
                    "channel with your ticket reference so the team can review the case details. We cannot discuss "
                    "account-specific information publicly."
                ),
                send_gate_note="Confirm account/ticket status, approved support path, and Legal/Compliance sign-off before posting.",
            )
        )
    return drafts


def narrative_membership_by_post(narratives: list[Narrative]) -> dict[str, list[str]]:
    memberships: dict[str, list[str]] = defaultdict(list)
    for narrative in narratives:
        for post_id in narrative.supporting_post_ids:
            memberships[post_id].append(narrative.narrative_id)
    return dict(memberships)
