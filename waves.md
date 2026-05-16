# Parallel Implementation Waves

This file splits the work after Task 4 of `docs/2026-05-14-deriv-social-pipeline.md` into two execution waves. Each lane has explicit file ownership so you can open two terminals, start two agents, and avoid overwrites.

## Starting Assumption

Before starting these waves, Tasks 1-4 are complete:

- `src/solution/constants.py`
- `src/solution/schemas.py`
- `src/solution/artifacts.py`
- `src/solution/preprocessing.py`
- `src/solution/classification.py`
- `src/solution/llm.py`
- `src/solution/prompts.py`
- `main.py` may still be minimal
- `data/posts.json` exists

If Task 4 is not complete yet, do not start these lanes. Finish classification first because later work depends on stable schemas, artifact helpers, and the LLM logging contract.

## Parallel Safety Rules

- Do not edit files outside your lane ownership.
- If you need a shared schema or constant changed, stop and coordinate before editing.
- Do not run broad formatting over the full repository from a lane.
- Keep tests lane-specific until Wave 2.
- Do not edit `.env`, `uv.lock`, or `pyproject.toml` unless explicitly assigned.
- Prefer small commits per lane so integration is easy.

## Wave 1: Independent Stage Implementations

Run Lane 1 and Lane 2 in parallel after Task 4.

### Wave 1, Lane 1: LLM Decision Stages

**Goal:** Implement the remaining LLM stages after classification: narrative detection, escalation routing, and public response drafting.

**Owns these files only:**

- `src/solution/narratives.py`
- `src/solution/routing.py`
- `src/solution/drafts.py`
- `tests/test_narratives.py`
- `tests/test_routing_validation.py`
- `tests/test_drafts.py`

**Allowed to read but not edit:**

- `src/solution/constants.py`
- `src/solution/schemas.py`
- `src/solution/artifacts.py`
- `src/solution/llm.py`
- `src/solution/classification.py`
- `src/solution/prompts.py`
- `data/posts.json`
- `docs/2026-05-14-deriv-social-pipeline.md`

**Important constraint:**

Do not edit `src/solution/prompts.py` in this lane. Put stage-specific prompt strings inside `narratives.py`, `routing.py`, and `drafts.py` for now. A later cleanup can consolidate prompts if time allows.

**Agent prompt to paste into Terminal 1:**

```text
You are implementing Wave 1 Lane 1 of waves.md.

Context:
- Tasks 1-4 are assumed complete.
- Do not edit files outside this lane.
- You own only:
  - src/solution/narratives.py
  - src/solution/routing.py
  - src/solution/drafts.py
  - tests/test_narratives.py
  - tests/test_routing_validation.py
  - tests/test_drafts.py

Goal:
Implement the remaining LLM decision stages for the Deriv social listening pipeline.

Requirements:
1. narratives.py:
   - Implement detect_narratives(classified_posts, post_context).
   - The input must include classified posts, post IDs, platform, timestamp, engagement summary, and translated text used for classification.
   - Validate narrative_strength against the controlled vocabulary.
   - Validate supporting_post_ids exist in the classified post set.
   - Write narratives.json.
   - Log an LLM call with stage NARRATIVES_DETECTED.

2. routing.py:
   - Implement route_escalations(top_posts, narratives).
   - Input only top 5 flagged posts plus all narratives.
   - Validate teams against the controlled internal team vocabulary.
   - Write escalation_routing.json.
   - Log an LLM call with stage ROUTING_COMPLETE.

3. drafts.py:
   - Implement generate_public_drafts(posts_requiring_response).
   - Select callers can pass posts where urgency is critical or contains_legal_threat is true.
   - Drafts must acknowledge the issue, avoid admitting liability, give next steps, match platform tone, avoid private account details, and include send_gate_note.
   - Write response_drafts.md.
   - Log an LLM call with stage RESPONSE_DRAFTS_GENERATED.

4. Tests:
   - Add tests for invalid narrative support IDs.
   - Add tests for invalid routing teams.
   - Add tests that every critical/legal-threat post gets a draft section and send-gate note.

Do not edit prompts.py. Keep prompts local to the module. Return a concise summary of changed files and test commands run.
```

**Lane-local verification:**

```bash
pytest tests/test_narratives.py tests/test_routing_validation.py tests/test_drafts.py -v
```

### Wave 1, Lane 2: Deterministic Computation and Validation

**Goal:** Implement non-LLM deterministic risk scoring, validation, and optional should-attempt analysis.

**Owns these files only:**

- `src/solution/risk.py`
- `src/solution/validation.py`
- `src/solution/optional_analysis.py`
- `validate.py`
- `tests/test_risk.py`
- `tests/test_validation.py`
- `tests/test_optional_analysis.py`

**Allowed to read but not edit:**

- `src/solution/constants.py`
- `src/solution/schemas.py`
- `src/solution/artifacts.py`
- `src/solution/preprocessing.py`
- `src/solution/classification.py`
- `data/posts.json`
- `docs/2026-05-14-deriv-social-pipeline.md`

**Important constraint:**

Do not edit `src/solution/graph.py`, `main.py`, or any LLM stage modules in this lane. Validation should check artifact contracts, not import graph orchestration.

**Agent prompt to paste into Terminal 2:**

```text
You are implementing Wave 1 Lane 2 of waves.md.

Context:
- Tasks 1-4 are assumed complete.
- Do not edit files outside this lane.
- You own only:
  - src/solution/risk.py
  - src/solution/validation.py
  - src/solution/optional_analysis.py
  - validate.py
  - tests/test_risk.py
  - tests/test_validation.py
  - tests/test_optional_analysis.py

Goal:
Implement deterministic computation and validation for the Deriv social listening pipeline.

Requirements:
1. risk.py:
   - Implement raw engagement exactly:
     likes + reposts*2 + comments*1.5 + replies*1.5 + upvotes + helpful_votes + reactions
   - Normalize engagement multiplier from 1.0 to 3.0 across all posts.
   - If all posts have the same engagement, use 1.0 for all multipliers.
   - Use base risk:
     critical=40, high=25, medium=10, low=3.
   - Add legal threat bonus of 20.
   - Add 15 per narrative containing the post.
   - Compute risk_score in code, never via LLM.
   - Sort descending and select top 5.
   - Write risk_scores.json.

2. validation.py and validate.py:
   - Implement python validate.py.
   - Check required artifacts exist.
   - Check JSON files parse.
   - Confirm all input posts were preprocessed and classified.
   - Confirm non-English posts preserve original text and use translated classification text.
   - Confirm controlled vocabularies.
   - Confirm narratives reference valid classified post IDs.
   - Confirm risk scores are sorted and top 5 escalation IDs match computed scores.
   - Confirm routing teams are controlled.
   - Confirm drafts exist for critical or legal-threat posts and include send-gate notes.
   - Confirm llm_calls.jsonl has separate records for POSTS_CLASSIFIED, NARRATIVES_DETECTED, ROUTING_COMPLETE, and RESPONSE_DRAFTS_GENERATED.

3. optional_analysis.py:
   - Implement deterministic sentiment trend analysis and write sentiment_trend.json.
   - Implement deterministic competitor signal extraction and write competitor_signals.json.

4. Tests:
   - Add exact numeric tests for risk scoring.
   - Add validation tests using temporary artifact files.
   - Add optional analysis tests.

Do not edit graph.py or main.py. Return a concise summary of changed files and test commands run.
```

**Lane-local verification:**

```bash
pytest tests/test_risk.py tests/test_validation.py tests/test_optional_analysis.py -v
```

## Wave 1 Integration Check

After both lanes finish, run:

```bash
pytest tests/test_narratives.py tests/test_routing_validation.py tests/test_drafts.py tests/test_risk.py tests/test_validation.py tests/test_optional_analysis.py -v
```

If both agents respected file ownership, there should be no merge conflicts.

## Wave 2: Graph Integration and End-to-End Hardening

Start Wave 2 only after Wave 1 is merged or both lane changes are present in the same working tree.

Run Lane 3 and Lane 4 in parallel.

### Wave 2, Lane 3: LangGraph Orchestration

**Goal:** Wire completed modules into the required staged LangGraph pipeline.

**Owns these files only:**

- `src/solution/state.py`
- `src/solution/graph.py`
- `main.py`
- `tests/test_graph_stages.py`

**Allowed to read but not edit:**

- All `src/solution/*.py` modules from Wave 1
- `validate.py`
- `data/posts.json`

**Important constraint:**

Do not modify `risk.py`, `validation.py`, `narratives.py`, `routing.py`, or `drafts.py`. If a module interface is awkward, adapt in `graph.py` rather than changing lane-owned modules.

**Agent prompt to paste into Terminal 1 for Wave 2:**

```text
You are implementing Wave 2 Lane 3 of waves.md.

Context:
- Wave 1 is complete.
- Do not edit files outside this lane.
- You own only:
  - src/solution/state.py
  - src/solution/graph.py
  - main.py
  - tests/test_graph_stages.py

Goal:
Wire the Deriv social listening pipeline using LangGraph with explicit required stage transitions.

Required stages:
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

Requirements:
1. state.py:
   - Define PipelineState with fields for stage, posts, preprocessed_posts, classified_posts, narratives, risk_scores, top_escalation_post_ids, routing, drafts, and generated_artifacts.
   - Implement advance_stage(current, expected_current, next_stage) that raises ValueError on out-of-order transitions.

2. graph.py:
   - Define one LangGraph node per stage transition.
   - Wire edges in the exact required order.
   - Ensure narrative detection receives classified data, not raw text alone.
   - Ensure risk scoring happens after narratives.
   - Ensure escalation routing receives top 5 risk posts and narratives.

3. main.py:
   - Invoke the graph from INIT.
   - Print concise generated artifact summary.

4. tests:
   - Test valid stage order.
   - Test risk-before-narratives raises a stage-order error.

Do not change implementation modules from Wave 1. Return a concise summary of changed files and test commands run.
```

**Lane-local verification:**

```bash
pytest tests/test_graph_stages.py -v
```

### Wave 2, Lane 4: README and End-to-End Smoke Tests

**Goal:** Add documentation and smoke tests without touching graph internals.

**Owns these files only:**

- `README.md`
- `tests/test_e2e_contract.py`

**Allowed to read but not edit:**

- All source files
- `data/posts.json`
- `validate.py`

**Important constraint:**

Do not edit `main.py`, `graph.py`, or any implementation modules. This lane documents and tests contracts only.

**Agent prompt to paste into Terminal 2 for Wave 2:**

```text
You are implementing Wave 2 Lane 4 of waves.md.

Context:
- Wave 1 is complete.
- Wave 2 Lane 3 may be editing graph.py/main.py in parallel.
- Do not edit files outside this lane.
- You own only:
  - README.md
  - tests/test_e2e_contract.py

Goal:
Document the project and add end-to-end contract smoke tests without changing implementation code.

Requirements:
1. README.md:
   - Explain the purpose of the pipeline.
   - Document setup:
     uv sync
   - Document env vars:
     GEMINI_API_KEY
   - Document run commands:
     python main.py
     python validate.py
   - List generated artifacts.
   - Explain that risk scoring is deterministic code and LLM calls are logged.

2. tests/test_e2e_contract.py:
   - Add tests that inspect source/artifact contracts without requiring live Gemini calls or a model env var.
   - Verify data/posts.json exists and contains posts.
   - Verify README mentions python main.py and python validate.py.
   - Verify required artifact names are documented.
   - If graph.py exists, verify required stage names appear in source text.

Do not edit graph.py or main.py. Return a concise summary of changed files and test commands run.
```

**Lane-local verification:**

```bash
pytest tests/test_e2e_contract.py -v
```

## Wave 2 Integration Check

After both Wave 2 lanes finish, run:

```bash
pytest -v
```

Then run the actual pipeline:

```bash
python main.py
```

Then validate:

```bash
python validate.py
```

## Final Manual Checklist

- `preprocessed_posts.json` exists.
- `classified_posts.json` exists.
- `narratives.json` exists.
- `risk_scores.json` exists.
- `escalation_routing.json` exists.
- `response_drafts.md` exists.
- `llm_calls.jsonl` exists.
- `llm_calls.jsonl` has separate records for classification, narratives, routing, and drafts.
- Risk scoring is implemented in `src/solution/risk.py`, not in an LLM prompt.
- Narrative detection uses classified data.
- Top 5 escalation posts come from sorted deterministic risk scores.
- Public drafts include send-gate notes.

