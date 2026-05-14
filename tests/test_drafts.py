import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solution.drafts import PublicDraftResult, generate_public_drafts
from src.solution.schemas import ResponseDraft


class FakeDraftLLM:
    def __init__(self, result: PublicDraftResult) -> None:
        self.result = result
        self.calls = []

    def invoke(self, *, stage, prompt, input_artifacts, output_artifact):
        self.calls.append(
            {
                "stage": stage,
                "prompt": prompt,
                "input_artifacts": input_artifacts,
                "output_artifact": output_artifact,
            }
        )
        return self.result


def posts_requiring_response():
    return [
        {
            "post_id": "P07",
            "platform": "Trustpilot review",
            "urgency": "critical",
            "contains_legal_threat": True,
            "text_for_classification": "My account was suspended and my lawyer has been notified.",
        },
        {
            "post_id": "P14",
            "platform": "Twitter/X",
            "urgency": "high",
            "contains_legal_threat": True,
            "text_for_classification": "Has anyone filed a successful chargeback against Deriv?",
        },
    ]


def test_generate_public_drafts_writes_section_and_send_gate_for_each_post(tmp_path):
    output_path = tmp_path / "response_drafts.md"
    llm = FakeDraftLLM(
        PublicDraftResult(
            drafts=[
                ResponseDraft(
                    post_id="P07",
                    platform="Trustpilot review",
                    draft_response="We are sorry to hear about this experience. Please contact us via the secure support channel so the team can review the case details.",
                    send_gate_note="Confirm account review status and approved case reference before posting.",
                ),
                ResponseDraft(
                    post_id="P14",
                    platform="Twitter/X",
                    draft_response="@user We understand your concern. Please DM us your ticket number so our team can check the case through secure channels.",
                    send_gate_note="Confirm the ticket path and approved public wording before posting.",
                ),
            ]
        )
    )

    drafts = generate_public_drafts(
        posts_requiring_response(),
        llm=llm,
        output_artifact=output_path,
    )

    assert [draft.post_id for draft in drafts] == ["P07", "P14"]
    assert len(llm.calls) == 1
    assert llm.calls[0]["stage"] == "RESPONSE_DRAFTS_GENERATED"
    assert "avoid admitting liability" in llm.calls[0]["prompt"]
    assert "send_gate_note" in llm.calls[0]["prompt"]

    markdown = output_path.read_text(encoding="utf-8")
    for post_id in ("P07", "P14"):
        assert f"## {post_id}" in markdown
    assert markdown.count("Send-gate note:") == 2


def test_generate_public_drafts_rejects_missing_required_post_draft(tmp_path):
    llm = FakeDraftLLM(
        PublicDraftResult(
            drafts=[
                ResponseDraft(
                    post_id="P07",
                    platform="Trustpilot review",
                    draft_response="Please contact support through secure channels.",
                    send_gate_note="Confirm case owner before posting.",
                )
            ]
        )
    )

    try:
        generate_public_drafts(
            posts_requiring_response(),
            llm=llm,
            output_artifact=tmp_path / "response_drafts.md",
        )
    except ValueError as exc:
        assert "draft for every selected post" in str(exc)
    else:
        raise AssertionError("Expected missing draft validation to fail")


def test_generate_public_drafts_keeps_only_critical_or_legal_posts(tmp_path):
    mixed_posts = posts_requiring_response() + [
        {
            "post_id": "P03",
            "platform": "Twitter/X",
            "urgency": "low",
            "contains_legal_threat": False,
            "text_for_classification": "Support solved my issue.",
        }
    ]
    llm = FakeDraftLLM(
        PublicDraftResult(
            drafts=[
                ResponseDraft(
                    post_id="P07",
                    platform="Trustpilot review",
                    draft_response="Please contact support through secure channels.",
                    send_gate_note="Confirm case owner before posting.",
                ),
                ResponseDraft(
                    post_id="P14",
                    platform="Twitter/X",
                    draft_response="Please DM us your ticket number.",
                    send_gate_note="Confirm social handle and ticket workflow before posting.",
                ),
            ]
        )
    )

    generate_public_drafts(mixed_posts, llm=llm, output_artifact=tmp_path / "response_drafts.md")

    assert "P03" not in llm.calls[0]["prompt"]
