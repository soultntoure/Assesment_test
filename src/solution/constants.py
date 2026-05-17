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
