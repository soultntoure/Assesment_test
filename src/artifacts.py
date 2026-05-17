import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

POSTS_PATH = DATA_DIR / "posts.json"

PREPROCESSED_POSTS_PATH = ARTIFACT_DIR / "preprocessed_posts.json"
CLASSIFIED_POSTS_PATH = ARTIFACT_DIR / "classified_posts.json"
NARRATIVES_PATH = ARTIFACT_DIR / "narratives.json"
RISK_SCORES_PATH = ARTIFACT_DIR / "risk_scores.json"
ESCALATION_ROUTING_PATH = ARTIFACT_DIR / "escalation_routing.json"
RESPONSE_DRAFTS_PATH = ARTIFACT_DIR / "response_drafts.md"
LLM_CALLS_PATH = ARTIFACT_DIR / "llm_calls.jsonl"

SENTIMENT_TREND_PATH = ARTIFACT_DIR / "sentiment_trend.json"
COMPETITOR_SIGNALS_PATH = ARTIFACT_DIR / "competitor_signals.json"
CRISIS_RATING_PATH = ARTIFACT_DIR / "crisis_rating.json"
MONITORING_PLAN_PATH = ARTIFACT_DIR / "monitoring_plan.md"

ARTIFACT_PATHS = (
    PREPROCESSED_POSTS_PATH,
    CLASSIFIED_POSTS_PATH,
    NARRATIVES_PATH,
    RISK_SCORES_PATH,
    ESCALATION_ROUTING_PATH,
    RESPONSE_DRAFTS_PATH,
    LLM_CALLS_PATH,
    SENTIMENT_TREND_PATH,
    COMPETITOR_SIGNALS_PATH,
    CRISIS_RATING_PATH,
    MONITORING_PLAN_PATH,
)


def ensure_artifact_dir() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> Any:
    artifact_path = Path(path)
    with artifact_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, data: Any) -> None:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False))
        file.write("\n")


def write_text(path: str | Path, text: str) -> None:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(text, encoding="utf-8")
