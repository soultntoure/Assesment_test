# Deriv Social Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a replayable LangGraph pipeline that ingests Deriv social posts, preprocesses multilingual text, classifies posts, detects narratives, computes deterministic risk scores, routes escalations, drafts gated public responses, and validates all required artifacts.

**Architecture:** Use LangGraph as the stage orchestrator and LangChain/Gemini for the four required LLM stages. Keep deterministic work in plain Python modules, preserve every intermediate artifact on disk, and validate stage order through metadata plus artifact checks.

**Tech Stack:** Python 3.13, LangGraph, LangChain, langchain-google-genai, Pydantic, python-dotenv, pytest.

---

## File Map

- Modify `main.py`: command-line entry point that runs the graph.
- Create `validate.py`: command-line validation entry point.
- Create `src/solution/__init__.py`: package marker.
- Create `src/solution/constants.py`: controlled vocabularies, stage names, artifact paths, risk constants.
- Modify `src/solution/schemas.py`: Pydantic schemas for posts, classifications, narratives, routes, drafts, state.
- Modify `src/solution/artifacts.py`: JSON/Markdown/JSONL read-write helpers.
- Modify `src/solution/llm.py`: Gemini model factory, prompt hashing, LLM call logging, structured invocation helper.
- Modify `src/solution/prompts.py`: prompts for classification, narrative detection, routing, and response drafts.
- Create `src/solution/preprocessing.py`: load posts, detect non-English text, translate or deterministic fallback for Malay.
- Create `src/solution/classification.py`: Stage 1 LLM classification.
- Create `src/solution/narratives.py`: Stage 2 LLM narrative detection.
- Create `src/solution/risk.py`: deterministic engagement and risk scoring.
- Create `src/solution/routing.py`: Stage 3 LLM escalation routing.
- Create `src/solution/drafts.py`: Stage 4 LLM public response drafts.
- Create `src/solution/optional_analysis.py`: sentiment trend and competitor signals if time allows.
- Modify `src/solution/state.py`: LangGraph state type and stage transition guard.
- Modify `src/solution/graph.py`: LangGraph nodes and edges.
- Create `src/solution/validation.py`: validator used by `validate.py` and final graph stage.
- Create tests in `tests/`: focused tests for preprocessing, risk scoring, validation, and stage order.

## Artifact Policy

Write generated artifacts to the repository root because the challenge names exact filenames:

- `preprocessed_posts.json`
- `classified_posts.json`
- `narratives.json`
- `risk_scores.json`
- `escalation_routing.json`
- `response_drafts.md`
- `llm_calls.jsonl`
- `sentiment_trend.json` if attempted
- `competitor_signals.json` if attempted

Read input from `data/posts.json` first, with fallback to root `posts.json`. This supports your local layout and evaluator replacement.

## LangGraph Flow

Use this exact graph:

```text
load_posts
 -> preprocess_posts
 -> classify_posts
 -> detect_narratives
 -> compute_risk_scores
 -> select_escalations
 -> route_escalations
 -> generate_response_drafts
 -> validate_outputs
 -> finalize
```

Map these nodes to required stages:

```text
INIT
POSTS_LOADED
MULTILINGUAL_PREPROCESSING_COMPLETE
POSTS_CLASSIFIED
NARRATIVES_DETECTED
RISK_SCORES_COMPUTED
ESCALATIONS_SELECTED
ROUTING_COMPLETE
RESPONSE_DRAFTS_GENERATED
VALIDATION_COMPLETE
RESULTS_FINALISED
```

Each node should assert it received the expected previous stage before doing work.

---

## Task 1: Constants, Schemas, and Artifacts

**Files:**
- Create: `src/solution/__init__.py`
- Create: `src/solution/constants.py`
- Modify: `src/solution/schemas.py`
- Modify: `src/solution/artifacts.py`
- Test: `tests/test_artifacts_and_schemas.py`

- [ ] Add controlled vocabularies exactly as specified: sentiment, topics, urgency, teams, narrative strengths.
- [ ] Add stage constants in the exact required order.
- [ ] Add artifact path constants for every required output.
- [ ] Define Pydantic models for raw posts, preprocessed posts, classified posts, narratives, risk scores, routed escalations, and response drafts.
- [ ] Add validators that reject uncontrolled vocabulary values.
- [ ] Add helpers `read_json`, `write_json`, `append_jsonl`, and `write_text`.
- [ ] Test that invalid sentiment/topic/team values fail validation.
- [ ] Test that JSON artifact helpers round-trip a list of objects.
- [ ] Run `pytest tests/test_artifacts_and_schemas.py -v`.

## Task 2: Preprocessing

**Files:**
- Create: `src/solution/preprocessing.py`
- Test: `tests/test_preprocessing.py`

- [ ] Implement `resolve_posts_path()` that prefers `data/posts.json` and falls back to `posts.json`.
- [ ] Implement `load_posts()` that validates every input post has `id`, `platform`, `text`, `timestamp`, and `engagement`.
- [ ] Implement `detect_language(text)` with a deterministic Malay keyword fallback for fixture-like data.
- [ ] Implement `translate_to_english(text, language)` with a deterministic Malay fallback for P10-style text.
- [ ] Implement `preprocess_posts(posts)` that emits `post_id`, `original_text`, `text_for_classification`, `original_language`, and `translated`.
- [ ] Write `preprocessed_posts.json` after preprocessing.
- [ ] Test that the Malay fixture text is marked `original_language="ms"`, `translated=true`, and preserves original text.
- [ ] Run `pytest tests/test_preprocessing.py -v`.

## Task 3: LLM Wrapper and Logging

**Files:**
- Modify: `src/solution/llm.py`
- Reuse: `src/solution/base_agent.py`
- Test: `tests/test_llm_logging.py`

- [ ] Create a LangChain/Gemini structured invocation helper using `ChatGoogleGenerativeAI`.
- [ ] Read only `GEMINI_API_KEY` via `python-dotenv`; keep Gemini model selection in code via `DEFAULT_GEMINI_MODEL`.
- [ ] Keep `temperature=0` for reproducibility.
- [ ] Add `hash_prompt(prompt: str)` using SHA-256.
- [ ] Add `log_llm_call(stage, provider, model, prompt, input_artifacts, output_artifact)`.
- [ ] Ensure each log record includes `stage`, ISO timestamp, `provider`, `model`, `prompt_hash`, `input_artifacts`, and `output_artifact`.
- [ ] Test that a fake call appends one valid JSON object to `llm_calls.jsonl`.
- [ ] Run `pytest tests/test_llm_logging.py -v`.

## Task 4: Stage 1 Classification

**Files:**
- Modify: `src/solution/prompts.py`
- Create: `src/solution/classification.py`
- Test: `tests/test_classification_validation.py`

- [ ] Create a classification prompt that includes all allowed sentiment, topic, and urgency values.
- [ ] The prompt must tell the model not to invent categories.
- [ ] The prompt must classify all preprocessed posts in one call.
- [ ] Implement `classify_posts(preprocessed_posts)` returning one classification per post.
- [ ] Validate the response with Pydantic controlled vocabularies.
- [ ] Write `classified_posts.json`.
- [ ] Log the LLM call with stage `POSTS_CLASSIFIED`.
- [ ] Test that classifications with unknown topic or urgency are rejected.
- [ ] Run `pytest tests/test_classification_validation.py -v`.

## Task 5: Stage 2 Narrative Detection

**Files:**
- Modify: `src/solution/prompts.py`
- Create: `src/solution/narratives.py`
- Test: `tests/test_narratives.py`

- [ ] Build narrative detection input from classified posts plus platform, timestamp, engagement summary, and classification text.
- [ ] Assert raw posts alone are not accepted by the narrative function.
- [ ] Create a narrative prompt that asks for emerging systemic clusters.
- [ ] Validate `narrative_strength` against `strong`, `moderate`, and `weak`.
- [ ] Validate that every supporting post ID exists in the classified dataset.
- [ ] Write `narratives.json`.
- [ ] Log the LLM call with stage `NARRATIVES_DETECTED` and `classified_posts.json` as an input artifact.
- [ ] Test that an unknown supporting post ID fails validation.
- [ ] Run `pytest tests/test_narratives.py -v`.

## Task 6: Deterministic Risk Scoring

**Files:**
- Create: `src/solution/risk.py`
- Test: `tests/test_risk.py`

- [ ] Implement raw engagement formula exactly:

```text
likes + reposts*2 + comments*1.5 + replies*1.5 + upvotes + helpful_votes + reactions
```

- [ ] Implement min-max engagement multiplier normalized from `1.0` to `3.0`.
- [ ] If all posts have equal engagement, use multiplier `1.0` for all posts.
- [ ] Implement base risk mapping: `critical=40`, `high=25`, `medium=10`, `low=3`.
- [ ] Add `20` legal threat bonus when `contains_legal_threat=true`.
- [ ] Add `15` per narrative membership.
- [ ] Compute `risk_score` in code, never through LLM.
- [ ] Sort descending by `risk_score`.
- [ ] Select the top 5 for escalation.
- [ ] Write `risk_scores.json`.
- [ ] Test exact numeric scoring with a small three-post fixture.
- [ ] Test that top 5 IDs equal the top 5 sorted risk scores.
- [ ] Run `pytest tests/test_risk.py -v`.

## Task 7: Stage 3 Escalation Routing

**Files:**
- Modify: `src/solution/prompts.py`
- Create: `src/solution/routing.py`
- Test: `tests/test_routing_validation.py`

- [ ] Create a routing prompt that includes only the allowed internal team vocabulary.
- [ ] Input only the top 5 flagged posts plus all narratives.
- [ ] Implement `route_escalations(top_posts, narratives)`.
- [ ] Validate every team against the controlled team list.
- [ ] Require concise internal Slack-style briefing notes.
- [ ] Write `escalation_routing.json`.
- [ ] Log the LLM call with stage `ROUTING_COMPLETE`.
- [ ] Test that unknown team names fail validation.
- [ ] Run `pytest tests/test_routing_validation.py -v`.

## Task 8: Stage 4 Public Response Drafts

**Files:**
- Modify: `src/solution/prompts.py`
- Create: `src/solution/drafts.py`
- Test: `tests/test_drafts.py`

- [ ] Select posts where `urgency == "critical"` or `contains_legal_threat == true`.
- [ ] Create a response drafting prompt requiring acknowledgement, no liability admission, next steps, platform tone, no account details, and send-gate note.
- [ ] Implement `generate_public_drafts(selected_posts)`.
- [ ] Write `response_drafts.md`.
- [ ] Include one section per post with `post_id`, `platform`, `draft_response`, and `send_gate_note`.
- [ ] Log the LLM call with stage `RESPONSE_DRAFTS_GENERATED`.
- [ ] Test that every critical or legal-threat post has a draft section.
- [ ] Test that every draft includes a send-gate note.
- [ ] Run `pytest tests/test_drafts.py -v`.

## Task 9: LangGraph Orchestration

**Files:**
- Modify: `src/solution/state.py`
- Modify: `src/solution/graph.py`
- Modify: `main.py`
- Test: `tests/test_graph_stages.py`

- [ ] Define `PipelineState` as a typed dict containing `stage`, `posts`, `preprocessed_posts`, `classified_posts`, `narratives`, `risk_scores`, `top_escalation_post_ids`, `routing`, and `drafts`.
- [ ] Implement `advance_stage(current, expected_current, next_stage)` that raises if stages are out of order.
- [ ] Add one LangGraph node per required stage transition.
- [ ] Wire graph edges in the exact required order.
- [ ] `main.py` should build and invoke the graph from `INIT`.
- [ ] Print a concise success message listing generated artifacts.
- [ ] Test that trying to compute risk before narratives raises a stage-order error.
- [ ] Run `pytest tests/test_graph_stages.py -v`.

## Task 10: Validation Command

**Files:**
- Create: `src/solution/validation.py`
- Create: `validate.py`
- Test: `tests/test_validation.py`

- [ ] Implement artifact existence checks.
- [ ] Validate all JSON files parse.
- [ ] Confirm all input posts were preprocessed and classified.
- [ ] Confirm non-English posts preserve original text and use translated classification text.
- [ ] Confirm all classification vocabularies are controlled.
- [ ] Confirm narratives reference valid classified post IDs.
- [ ] Confirm `risk_scores.json` exists and top 5 escalation IDs match computed top 5.
- [ ] Confirm routing teams are controlled.
- [ ] Confirm drafts exist for critical or legal-threat posts.
- [ ] Confirm every public draft includes a send-gate note.
- [ ] Confirm `llm_calls.jsonl` has separate records for `POSTS_CLASSIFIED`, `NARRATIVES_DETECTED`, `ROUTING_COMPLETE`, and `RESPONSE_DRAFTS_GENERATED`.
- [ ] Make `python validate.py` exit with code `0` on success and nonzero on failure.
- [ ] Run `pytest tests/test_validation.py -v`.

## Task 11: Optional Should-Attempt Items

**Files:**
- Create: `src/solution/optional_analysis.py`
- Modify: `src/solution/graph.py`
- Test: `tests/test_optional_analysis.py`

- [ ] Implement deterministic sentiment trend analysis from timestamps and classified sentiment.
- [ ] Write `sentiment_trend.json`.
- [ ] Implement competitor signal extraction from classification text using phrases such as `alternatives`, `moved to`, `proper platforms`, and `look at alternatives`.
- [ ] Write `competitor_signals.json`.
- [ ] Keep these after the core pipeline works.
- [ ] Run `pytest tests/test_optional_analysis.py -v`.

## Task 12: Final Verification

**Files:**
- Modify: `README.md`

- [ ] Document setup:

```bash
uv sync
```

- [ ] Document required environment variables:

```bash
GEMINI_API_KEY=...
```

- [ ] Document run commands:

```bash
python main.py
python validate.py
```

- [ ] Delete generated artifacts and rerun the full pipeline.
- [ ] Run all tests:

```bash
pytest -v
```

- [ ] Run validation:

```bash
python validate.py
```

- [ ] Confirm generated files include all required artifacts.

## Implementation Order for Agentic Coding

Use this order:

1. Task 1: constants, schemas, artifact helpers.
2. Task 2: preprocessing.
3. Task 6: deterministic risk scoring.
4. Task 10: validation skeleton.
5. Task 3: LLM wrapper and logging.
6. Task 4: classification.
7. Task 5: narratives.
8. Task 7: routing.
9. Task 8: drafts.
10. Task 9: LangGraph orchestration.
11. Task 11: optional should-attempt items.
12. Task 12: README and final verification.

This order gets deterministic foundations in place before relying on the LLM.

## Self-Review

- Spec coverage: The plan covers preprocessing, one-call classification, separate narrative detection, deterministic risk scoring after narratives, top 5 escalation selection, separate routing, public response drafts, LLM logging, validation, and optional should-attempt analyses.
- Placeholder scan: No task depends on unspecified implementation details; each task names files, behavior, commands, and expected checks.
- Type consistency: The artifact names and stage names match the challenge requirements and are reused consistently across graph, logging, and validation.

