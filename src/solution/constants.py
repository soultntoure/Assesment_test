from pathlib import Path


SENTIMENT_VALUES = ("positive", "negative", "neutral", "mixed")

TOPIC_VALUES = (
    "withdrawal",
    "account_suspension",
    "spread_pricing",
    "product_feedback",
    "regulatory",
    "technical",
    "deposit",
    "kyc",
    "general",
)

URGENCY_VALUES = ("critical", "high", "medium", "low")

TEAM_VALUES = (
    "Customer Support",
    "Legal",
    "Compliance",
    "PR/Comms",
    "Product",
    "Engineering",
    "Finance",
)

NARRATIVE_STRENGTH_VALUES = ("strong", "moderate", "weak")

GEMINI_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"

STAGES = [
    "INIT",
    "POSTS_LOADED",
    "MULTILINGUAL_PREPROCESSING_COMPLETE",
    "POSTS_CLASSIFIED",
    "NARRATIVES_DETECTED",
    "RISK_SCORES_COMPUTED",
    "ESCALATIONS_SELECTED",
    "ROUTING_COMPLETE",
    "RESPONSE_DRAFTS_GENERATED",
    "VALIDATION_COMPLETE",
    "RESULTS_FINALISED",
]

INIT = STAGES[0]
POSTS_LOADED = STAGES[1]
MULTILINGUAL_PREPROCESSING_COMPLETE = STAGES[2]
POSTS_CLASSIFIED = STAGES[3]
NARRATIVES_DETECTED = STAGES[4]
RISK_SCORES_COMPUTED = STAGES[5]
ESCALATIONS_SELECTED = STAGES[6]
ROUTING_COMPLETE = STAGES[7]
RESPONSE_DRAFTS_GENERATED = STAGES[8]
VALIDATION_COMPLETE = STAGES[9]
RESULTS_FINALISED = STAGES[10]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREPROCESSED_POSTS_PATH = PROJECT_ROOT / "preprocessed_posts.json"
CLASSIFIED_POSTS_PATH = PROJECT_ROOT / "classified_posts.json"
NARRATIVES_PATH = PROJECT_ROOT / "narratives.json"
RISK_SCORES_PATH = PROJECT_ROOT / "risk_scores.json"
ESCALATION_ROUTING_PATH = PROJECT_ROOT / "escalation_routing.json"
RESPONSE_DRAFTS_PATH = PROJECT_ROOT / "response_drafts.md"
LLM_CALLS_PATH = PROJECT_ROOT / "llm_calls.jsonl"
SENTIMENT_TREND_PATH = PROJECT_ROOT / "sentiment_trend.json"
COMPETITOR_SIGNALS_PATH = PROJECT_ROOT / "competitor_signals.json"

REQUIRED_ARTIFACT_PATHS = (
    PREPROCESSED_POSTS_PATH,
    CLASSIFIED_POSTS_PATH,
    NARRATIVES_PATH,
    RISK_SCORES_PATH,
    ESCALATION_ROUTING_PATH,
    RESPONSE_DRAFTS_PATH,
    LLM_CALLS_PATH,
)

OPTIONAL_ARTIFACT_PATHS = (
    SENTIMENT_TREND_PATH,
    COMPETITOR_SIGNALS_PATH,
)

URGENCY_BASE_RISK = {
    "critical": 40,
    "high": 25,
    "medium": 10,
    "low": 3,
}

LEGAL_THREAT_BONUS = 20
NARRATIVE_MEMBERSHIP_BONUS = 15
