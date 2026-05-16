# Deriv Social Media Intelligence Pipeline

A replayable, staged pipeline that ingests social media and forum mentions of Deriv, classifies each post by sentiment and topic, detects emerging narratives before they trend, computes deterministic engagement-weighted risk scores, routes escalations to internal teams, and drafts public responses with human send gates.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Stages](#pipeline-stages)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the Pipeline](#running-the-pipeline)
- [Validation](#validation)
- [Output Artifacts](#output-artifacts)
- [Controlled Vocabularies](#controlled-vocabularies)
- [Risk Scoring Formula](#risk-scoring-formula)
- [LLM Configuration](#llm-configuration)
- [Architecture Notes](#architecture-notes)

---

## Overview

This pipeline processes raw social media posts (`posts.json`) through a series of deterministic, staged steps:

1. **Multilingual Preprocessing** — detects and translates non-English posts while preserving originals
2. **Post Classification** — classifies each post by sentiment, topic, urgency, and flags (legal threats, competitor mentions)
3. **Narrative Detection** — clusters classified posts into emerging narratives that may trend
4. **Risk Scoring** — computes engagement-weighted risk scores deterministically in code (no LLM)
5. **Escalation Routing** — selects the top 5 highest-risk posts and routes them to appropriate internal teams
6. **Response Drafting** — generates human-gated public response drafts for critical/legal-threat posts
7. **Optional Analyses** — sentiment trend arc, competitor signal extraction, 24-hour monitoring plan

All intermediate artifacts are preserved on disk. LLM calls are logged to `llm_calls.jsonl`.

---

## Pipeline Stages

```
INIT
 → POSTS_LOADED
 → MULTILINGUAL_PREPROCESSING_COMPLETE
 → POSTS_CLASSIFIED
 → NARRATIVES_DETECTED
 → RISK_SCORES_COMPUTED
 → ESCALATIONS_SELECTED
 → ROUTING_COMPLETE
 → RESPONSE_DRAFTS_GENERATED
 → VALIDATION_COMPLETE
 → RESULTS_FINALISED
```

Each stage depends on the output of the previous stage. Narrative detection receives **Stage 1 classified data**, not raw post text. Risk scoring occurs **after** narrative detection because narrative membership contributes to the score.

---

## Project Structure

```
.
├── posts.json                   # Input: raw social media posts
├── preprocessed_posts.json      # Stage output: translated posts
├── classified_posts.json        # Stage output: sentiment + topic + risk scores
├── narratives.json              # Stage output: emerging narrative clusters
├── risk_scores.json             # Stage output: deterministic risk scores
├── escalation_routing.json      # Stage output: top-5 posts routed to teams
├── response_drafts.md           # Stage output: human-gated public responses
├── sentiment_trend.json         # Optional: sentiment arc over the day
├── competitor_signals.json      # Optional: posts signalling switching intent
├── llm_calls.jsonl              # LLM call log (one record per call)
│
├── main.py                      # Entrypoint (full LLM pipeline)
├── validate.py                  # Validation + deterministic fallback pipeline
├── pyproject.toml               # Project metadata and dependencies
├── .env                         # API keys (not committed)
│
└── src/
    └── solution/
        ├── constants.py         # Controlled vocabularies, stage names, paths
        ├── schemas.py           # Pydantic models for all artifacts
        ├── preprocessing.py     # Multilingual detection and translation
        ├── classification.py    # Stage 1 LLM call: post classification
        ├── narratives.py        # Stage 2 LLM call: narrative detection
        ├── risk.py              # Deterministic engagement-weighted risk scoring
        ├── routing.py           # Stage 3 LLM call: escalation routing
        ├── drafts.py            # Stage 4 LLM call: public response drafting
        ├── optional_analysis.py # Sentiment trend + competitor signal extraction
        ├── llm.py               # LLM client (Gemini) + call logger
        ├── prompts.py           # Prompt templates for each LLM stage
        ├── artifacts.py         # JSON/text artifact read/write helpers
        ├── validation.py        # Artifact validation checks
        ├── base_agent.py        # Shared LLM agent base
        └── graph.py             # Pipeline graph definition
```

---

## Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** (recommended package manager)
- A **Gemini API key from Google AI Studio** with access to your chosen model

---

## Setup

### 1. Clone and navigate to the project

```bash
git clone <repository-url>
cd Assesment_test
```

### 2. Install dependencies

Using `uv`:

```bash
uv sync
```

Or using pip:

```bash
pip install -e .
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_ai_studio_api_key_here
```

> **Model selection:** The default provider and model are selected by `BaseAgent` in `src/solution/base_agent.py`. Change `BaseAgent.default_model_name`, or pass an explicit model when constructing the LLM helper, to use another Gemini model.

### 4. Provide input data

Ensure `posts.json` exists in the project root with the expected schema:

```json
[
  {
    "id": "P01",
    "platform": "Twitter/X",
    "text": "...",
    "timestamp": "2025-04-10T08:15:00Z",
    "engagement": { "likes": 12, "replies": 4, "reposts": 8 }
  }
]
```

---

## Running the Pipeline

### Full LLM Pipeline

```bash
uv run python main.py
```

This executes the complete staged pipeline, making real LLM calls for classification, narrative detection, escalation routing, and response drafting.

### Deterministic Fallback (for validation / CI)

```bash
uv run python validate.py
```

Runs a heuristic-based version of the full pipeline without LLM calls, regenerates all artifacts, then validates them. Useful for testing the pipeline structure without consuming API credits.

---

## Validation

Run the validation command to verify all required artifacts are present, structurally valid, and meet the pipeline constraints:

```bash
uv run python validate.py
```

The validator checks:

| Check | Description |
|---|---|
| Required artifacts exist | All output files are present |
| Valid JSON | All `.json` files parse correctly |
| All posts processed | No posts skipped during classification |
| Multilingual preservation | Non-English posts retain `original_text` + `translated: true` |
| Controlled vocabularies | Sentiment, topic, urgency values are within allowed sets |
| Narrative input dependency | Narratives reference classified post IDs |
| Deterministic risk scores | Scores computed after narrative detection |
| Top-5 escalation selection | Escalated posts are the 5 highest-risk |
| Team vocabulary | Routing teams use only allowed team names |
| Critical/legal drafts | Public drafts exist for all critical/legal-threat posts |
| Send-gate notes | Every draft includes a send-gate condition |
| LLM log completeness | Separate log records exist for each required stage |

Exit code `0` = all checks passed. Exit code `1` = validation failed (errors printed to stdout).

---

## Output Artifacts

| File | Description |
|---|---|
| `preprocessed_posts.json` | Posts with language detection and English translations |
| `classified_posts.json` | Per-post sentiment, topic, urgency, flags, and risk scores |
| `narratives.json` | Emerging narrative clusters with strength and ETA to trending |
| `risk_scores.json` | Engagement-weighted risk scores for all posts |
| `escalation_routing.json` | Top-5 posts routed to internal teams with briefing notes |
| `response_drafts.md` | Human-gated public response drafts for critical/legal posts |
| `sentiment_trend.json` | Sentiment arc across the day with inflection point analysis |
| `competitor_signals.json` | Posts signalling switching intent + retention arguments |
| `llm_calls.jsonl` | Audit log of all LLM calls (stage, model, prompt hash, I/O) |

---

## Controlled Vocabularies

All LLM outputs are validated against these fixed value sets (defined in `src/solution/constants.py`):

**Sentiment:** `positive` · `negative` · `neutral` · `mixed`

**Topic:** `withdrawal` · `account_suspension` · `spread_pricing` · `product_feedback` · `regulatory` · `technical` · `deposit` · `kyc` · `general`

**Urgency:** `critical` · `high` · `medium` · `low`

**Internal Teams:** `Customer Support` · `Legal` · `Compliance` · `PR/Comms` · `Product` · `Engineering` · `Finance`

**Narrative Strength:** `strong` · `moderate` · `weak`

---

## Risk Scoring Formula

Risk scores are computed **deterministically in code** — the LLM is not used for scoring.

```
base_risk:
  critical = 40 | high = 25 | medium = 10 | low = 3

raw_engagement = likes + reposts×2 + comments×1.5 + replies×1.5
               + upvotes + helpful_votes + reactions

engagement_multiplier = normalised(raw_engagement) → [1.0, 3.0]

legal_threat_bonus  = +20  (if contains_legal_threat is true)
narrative_bonus     = +15  (per narrative the post belongs to)

risk_score = base_risk × engagement_multiplier
           + legal_threat_bonus
           + narrative_bonus
```

The **top 5 posts by risk score** are flagged for escalation.

---

## LLM Configuration

The pipeline uses [Google Gemini](https://ai.google.dev/gemini-api/docs) to access LLMs. Configure via `.env`:

```env
GEMINI_API_KEY=your_ai_studio_api_key_here
```

Each LLM call is logged to `llm_calls.jsonl` with the following schema:

```json
{
  "stage": "POSTS_CLASSIFIED",
  "timestamp": "2025-04-10T08:00:00Z",
  "provider": "gemini",
  "model": "gemini-2.5-flash-lite",
  "prompt_hash": "sha256:...",
  "input_artifacts": ["preprocessed_posts.json"],
  "output_artifact": "classified_posts.json"
}
```

Separate log records are required for: classification, narrative detection, escalation routing, response drafting, and any optional analysis stages.

---

## Architecture Notes

- **Staged execution** — Each stage writes its output to disk before the next stage reads it. Intermediate artifacts are always preserved.
- **Controlled vocabularies enforced at runtime** — Pydantic models validate all LLM outputs against allowed value sets. Invalid outputs cause the pipeline to fail early.
- **Deterministic risk scoring** — Risk scores are purely computed from the classified data and narrative membership. Replacing `posts.json` with equivalent fixtures produces consistent, reproducible scores.
- **Human send gates** — Every public response draft includes a `send_gate_note` specifying what internal approvals are required before the draft may be posted.
- **Evaluator compatibility** — The pipeline reads from `posts.json` and regenerates all artifacts on every run. Static precomputed outputs are not used.
