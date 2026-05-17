## BUILD

Build a replayable pipeline that ingests social media and forum mentions of Deriv, classifies each post by sentiment and topic, detects emerging narratives before they trend, computes deterministic engagement-weighted risk scores, routes escalations to internal teams, and drafts public responses with human send gates.

This is not a one-shot social media summary task. The evaluator will run your pipeline from a clean checkout, may replace the post dataset with equivalent fixtures, and will verify that classification, narrative detection, risk scoring, routing, and response drafting are separated into staged steps.

The pipeline must preserve intermediate artifacts, enforce controlled vocabularies, log LLM calls, and compute risk scores deterministically in code.

---

## INPUT FILES

Your pipeline must read posts from disk:

- `posts.json`

The sample posts below are provided for local testing. The evaluator may replace `posts.json` with equivalent social or forum posts using the same schema. Your implementation must not depend on exact post IDs, ordering, wording, platforms, or expected final rankings from the public fixture.

---

## SAMPLE `posts.json`

```json
[
  {
    "id": "P01",
    "platform": "Twitter/X",
    "text": "Been waiting 9 days for my Deriv withdrawal. Support keeps saying 'under review'. I'm posting here because they're ignoring my tickets. #Deriv #scam",
    "timestamp": "2025-04-10T08:15:00Z",
    "engagement": { "likes": 12, "replies": 4, "reposts": 8 }
  },
  {
    "id": "P02",
    "platform": "Reddit r/Forex",
    "text": "Deriv's spreads on EUR/USD have noticeably widened over the last 2 weeks. Anyone else seeing this? Starting to look at alternatives.",
    "timestamp": "2025-04-10T09:02:00Z",
    "engagement": { "upvotes": 47, "comments": 23 }
  },
  {
    "id": "P03",
    "platform": "Twitter/X",
    "text": "Shoutout to @Deriv_Official — their support team sorted my issue in under an hour. Rare to see that level of service from a broker.",
    "timestamp": "2025-04-10T10:45:00Z",
    "engagement": { "likes": 31, "reposts": 5 }
  },
  {
    "id": "P04",
    "platform": "Telegram (public group)",
    "text": "Is Deriv regulated in South Africa? My friend said they got blocked from withdrawing because of some new regulation.",
    "timestamp": "2025-04-10T11:00:00Z",
    "engagement": { "reactions": 6 }
  },
  {
    "id": "P05",
    "platform": "Reddit r/binaryoptions",
    "text": "Anyone noticed Deriv Bot is giving inconsistent results compared to backtests? My live strategy is performing about 40% worse than the backtest showed.",
    "timestamp": "2025-04-10T11:30:00Z",
    "engagement": { "upvotes": 89, "comments": 41 }
  },
  {
    "id": "P06",
    "platform": "Twitter/X",
    "text": "Deriv just added PIX payment in Brazil finally!! Been waiting for this for so long. @Deriv_Official",
    "timestamp": "2025-04-10T12:00:00Z",
    "engagement": { "likes": 204, "reposts": 67 }
  },
  {
    "id": "P07",
    "platform": "Trustpilot review",
    "text": "My account was suspended without explanation 3 weeks ago. I have over $12,000 in there. My lawyer has been notified and I will be filing a formal complaint with the regulator if this is not resolved within 48 hours.",
    "timestamp": "2025-04-10T12:15:00Z",
    "engagement": { "helpful_votes": 18 }
  },
  {
    "id": "P08",
    "platform": "Twitter/X",
    "text": "lol at people still using Deriv. Any serious trader moved to proper platforms years ago. #amateurhour",
    "timestamp": "2025-04-10T13:00:00Z",
    "engagement": { "likes": 3, "replies": 7 }
  },
  {
    "id": "P09",
    "platform": "Reddit r/Forex",
    "text": "Second post I've seen this week about Deriv spreads widening on major pairs. Is this a policy change or a technical issue?",
    "timestamp": "2025-04-10T13:30:00Z",
    "engagement": { "upvotes": 34, "comments": 12 }
  },
  {
    "id": "P10",
    "platform": "Facebook group (Forex Malaysia)",
    "text": "Adakah sesiapa tahu kenapa Deriv minta dokumen tambahan tiba-tiba? Saya dah lulus KYC sebelum ini.",
    "timestamp": "2025-04-10T14:00:00Z",
    "engagement": { "reactions": 29, "comments": 15 }
  },
  {
    "id": "P11",
    "platform": "Twitter/X",
    "text": "Deriv's new trading interface is actually really clean. The chart tools have improved massively in the last update.",
    "timestamp": "2025-04-10T14:30:00Z",
    "engagement": { "likes": 88, "reposts": 12 }
  },
  {
    "id": "P12",
    "platform": "Reddit r/binaryoptions",
    "text": "Third person I know this month who had their Deriv account suspended with no reason given and large balance locked. Something is going on. Are they doing a mass KYC sweep?",
    "timestamp": "2025-04-10T15:00:00Z",
    "engagement": { "upvotes": 156, "comments": 67 }
  },
  {
    "id": "P13",
    "platform": "Trustpilot review",
    "text": "Excellent platform for synthetic indices trading. Fast execution, fair pricing. Been using for 3 years.",
    "timestamp": "2025-04-10T15:15:00Z",
    "engagement": { "helpful_votes": 7 }
  },
  {
    "id": "P14",
    "platform": "Twitter/X",
    "text": "Serious question: has anyone filed a successful chargeback against Deriv? Asking for a friend who's been locked out of $5k for 6 weeks.",
    "timestamp": "2025-04-10T15:45:00Z",
    "engagement": { "likes": 22, "replies": 31, "reposts": 14 }
  },
  {
    "id": "P15",
    "platform": "Reddit r/Forex",
    "text": "EUR/USD spread on Deriv is now 1.8 pips during London session. A month ago it was 0.9. Anyone have an explanation?",
    "timestamp": "2025-04-10T16:00:00Z",
    "engagement": { "upvotes": 61, "comments": 28 }
  },
  {
    "id": "P16",
    "platform": "Facebook group (Nigeria Forex)",
    "text": "Deriv Nigeria deposits via bank transfer have been failing for 3 days. Anyone else?",
    "timestamp": "2025-04-10T16:30:00Z",
    "engagement": { "reactions": 44, "comments": 37 }
  },
  {
    "id": "P17",
    "platform": "Twitter/X",
    "text": "My Deriv account was suspended unfairly. I've been a customer for 4 years. #DerivScam #ConsumerRights",
    "timestamp": "2025-04-10T17:00:00Z",
    "engagement": { "likes": 19, "reposts": 11 }
  },
  {
    "id": "P18",
    "platform": "Telegram (public group)",
    "text": "Anyone else using Deriv Bot with the RSI strategy getting weird results since last week's update? Feels like something changed in the execution engine.",
    "timestamp": "2025-04-10T17:30:00Z",
    "engagement": { "reactions": 33 }
  },
  {
    "id": "P19",
    "platform": "Reddit r/investing",
    "text": "Balanced take: Deriv has good products but their compliance processes create a lot of friction. Worth it if you trade synthetic indices but frustrating for deposits/withdrawals.",
    "timestamp": "2025-04-10T18:00:00Z",
    "engagement": { "upvotes": 72, "comments": 19 }
  },
  {
    "id": "P20",
    "platform": "Twitter/X",
    "text": "Heads up: seeing multiple posts about Deriv account suspensions and withdrawal delays in the last 48 hours. Could be a mass KYC sweep or compliance event. Deriv comms team needs to get ahead of this.",
    "timestamp": "2025-04-10T18:30:00Z",
    "engagement": { "likes": 91, "reposts": 44 }
  }
]
```

---

## CONTROLLED VOCABULARIES

Define these vocabularies in code and validate model outputs against them.

Allowed sentiment values:

```text
positive
negative
neutral
mixed
```

Allowed topics:

```text
withdrawal
account_suspension
spread_pricing
product_feedback
regulatory
technical
deposit
kyc
general
```

Allowed urgency values:

```text
critical
high
medium
low
```

Allowed internal teams:

```text
Customer Support
Legal
Compliance
PR/Comms
Product
Engineering
Finance
```

Allowed narrative strengths:

```text
strong
moderate
weak
```

---

## PIPELINE STAGES

Your implementation must enforce these stages in code:

```text
INIT
 -> POSTS_LOADED
 -> MULTILINGUAL_PREPROCESSING_COMPLETE
 -> POSTS_CLASSIFIED
 -> NARRATIVES_DETECTED
 -> RISK_SCORES_COMPUTED
 -> ESCALATIONS_SELECTED
 -> ROUTING_COMPLETE
 -> RESPONSE_DRAFTS_GENERATED
 -> VALIDATION_COMPLETE
 -> RESULTS_FINALISED
```

Narrative detection must receive Stage 1 classified data, not raw post text alone.

Risk scoring must occur after narrative detection because narrative membership contributes to the score.

---

## MUST COMPLETE

### 1. Multilingual Preprocessing

Before classification, detect any non-English posts and translate them to English.

For the public fixture, P10 must be translated from Malay before classification.

Each preprocessed post must preserve:

```json
{
  "post_id": "string",
  "original_text": "string",
  "text_for_classification": "string",
  "original_language": "string",
  "translated": true
}
```

Save output to `preprocessed_posts.json`.

---

### 2. Post-Level Classification

Make one Stage 1 LLM call using all preprocessed posts.

The prompt must define all schema fields and controlled vocabularies.

For each post, output:

```json
{
  "post_id": "string",
  "sentiment": "positive",
  "topic": "withdrawal",
  "urgency": "high",
  "contains_legal_threat": false,
  "contains_competitor_mention": false,
  "original_language": "en",
  "translated": false
}
```

Save output to `classified_posts.json`.

The model must not invent new sentiment, topic, or urgency categories.

---

### 3. Narrative Detection

Make a separate Stage 2 LLM call.

This call must receive:

- Stage 1 classified posts
- post IDs
- platform
- timestamp
- engagement summary
- translated text used for classification

The purpose of this stage is to identify emerging narratives: clusters that suggest a systemic issue even if individual posts look minor.

For each narrative, output:

```json
{
  "narrative_id": "string",
  "title": "string",
  "supporting_post_ids": ["P01", "P07"],
  "narrative_strength": "strong",
  "estimated_hours_until_trending": 6,
  "recommended_action": "string"
}
```

At least 3 distinct narratives should be identified for the public fixture.

Save output to `narratives.json`.

---

### 4. Engagement-Weighted Risk Scoring

Compute post risk scores deterministically in code.

Do not use the LLM to assign risk scores.

Use this formula:

```text
base risk:
critical = 40
high = 25
medium = 10
low = 3

raw engagement = likes + reposts*2 + comments*1.5 + replies*1.5 + upvotes + helpful_votes + reactions

engagement multiplier = normalised value from 1.0 to 3.0 across all posts

legal threat bonus = 20 if contains_legal_threat is true

narrative membership bonus = 15 per narrative containing the post

risk_score = base * engagement_multiplier + legal_threat_bonus + narrative_membership_bonus
```

Sort posts by risk score.

Flag the top 5 for escalation.

Save a separate `risk_scores.json`.

---

### 5. Escalation Routing

Make a separate Stage 3 LLM call.

The call must include:

- top 5 flagged posts
- all detected narratives
- controlled internal team vocabulary

For each routed item, output:

```json
{
  "post_id": "string",
  "teams": ["Legal", "PR/Comms"],
  "briefing_note": "string"
}
```

Briefing notes must be written as concise internal Slack-style updates for the relevant team.

Multiple teams may receive the same post if relevant.

Save output to `escalation_routing.json`.

---

### 6. Public Response Drafts

For any post with `urgency: critical` or `contains_legal_threat: true`, generate a public-facing draft response.

Drafts must:

- acknowledge the issue
- avoid admitting liability
- provide next steps
- match platform tone
- avoid disclosing account details
- include a send-gate note

Each draft must include:

```json
{
  "post_id": "string",
  "platform": "string",
  "draft_response": "string",
  "send_gate_note": "string"
}
```

The send-gate note must specify what internal information is required before posting.

Save output to `response_drafts.md`.

---

## SHOULD ATTEMPT

### 7. Sentiment Trend Analysis

Characterise the sentiment arc across the day using timestamps and classified sentiment.

Identify:

- sentiment distribution over time
- inflection point, if any
- post or narrative cluster that caused the shift

Save output to `sentiment_trend.json`.

---

### 8. Competitor Signal Extraction

Identify posts that suggest users are considering alternatives, even without naming a specific competitor.

For each signal, output:

```json
{
  "post_id": "string",
  "implied_competitor_type": "string",
  "switching_trigger": "string",
  "retention_argument": "string"
}
```

Save output to `competitor_signals.json`.

---


### 10. 24-Hour Monitoring Plan

Based on the identified narratives, produce a monitoring plan for the next 24 hours.

Include:

- keywords to track
- platforms to prioritise
- escalation signals
- de-escalation signals
- owner team

Save output to `monitoring_plan.md`.

---

## REQUIRED ARTIFACTS

Your repository must produce:

- `posts.json`
- `preprocessed_posts.json`
- `classified_posts.json`
- `narratives.json`
- `risk_scores.json`
- `escalation_routing.json`
- `response_drafts.md`
- `sentiment_trend.json`, if attempted
- `competitor_signals.json`, if attempted
- `crisis_rating.json`, if attempted
- `monitoring_plan.md`, if attempted
- `llm_calls.jsonl`

---

## `llm_calls.jsonl` REQUIREMENTS

Log one JSON object per LLM call.

Each record must include:

```json
{
  "stage": "string",
  "timestamp": "ISO-8601 timestamp",
  "provider": "string",
  "model": "string",
  "prompt_hash": "string",
  "input_artifacts": ["path"],
  "output_artifact": "path"
}
```

There must be separate records for:

- post classification
- narrative detection
- escalation routing
- public response drafting
- optional analysis stages, if attempted

---

## VALIDATION REQUIREMENTS

The repository must include a validation command, for example:

```bash
make validate
```

or:

```bash
python validate.py
```

The validation command must check that:

- required artifacts exist
- JSON files are valid
- all posts were processed
- non-English posts preserve original text and include translated classification text
- classified outputs use only controlled vocabularies
- narrative detection uses classified posts as input
- risk scores are computed deterministically after narrative detection
- top 5 escalation posts are selected from computed scores
- routing teams use only the controlled team vocabulary
- public drafts exist for critical or legal-threat posts
- each public draft includes a send-gate note
- LLM call logs contain separate records for required stages

---

## EXECUTION REQUIREMENTS

The evaluator will run the pipeline from a clean checkout.

Generated artifacts may be deleted before evaluation.

The evaluator may replace `posts.json` with equivalent social-listening data using the same schema.

Static precomputed outputs are not sufficient.

The solution must actually run the staged pipeline and regenerate the required artifacts.

---

## TOOLS

Any programming language may be used.

Any LLM provider or AI tooling may be used.

---

## TECHNICAL CONSTRAINTS

- Read `posts.json` from disk.
- Translate non-English posts before classification while preserving original text.
- Risk scoring must be deterministic code.
- Stage 2 narrative detection must receive Stage 1 classified data.
- Classification, narrative detection, routing, and response drafting must be separate stages.
- Public responses must avoid admitting liability.
- Public responses must not disclose private account information.
- Static precomputed outputs are not sufficient.