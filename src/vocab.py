SENTIMENT = ("positive", "negative", "neutral", "mixed")

TOPICS = (
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

URGENCY = ("critical", "high", "medium", "low")

TEAMS = (
    "Customer Support",
    "Legal",
    "Compliance",
    "PR/Comms",
    "Product",
    "Engineering",
    "Finance",
)

NARRATIVE_STRENGTH = ("strong", "moderate", "weak")

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
