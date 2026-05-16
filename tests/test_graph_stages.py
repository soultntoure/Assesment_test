import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solution.constants import (
    INIT,
    NARRATIVES_DETECTED,
    POSTS_CLASSIFIED,
    RISK_SCORES_COMPUTED,
    STAGES,
)
from src.solution.graph import compute_risk_scores_node
from src.solution.state import advance_stage


def test_advance_stage_accepts_required_order():
    stage = INIT
    for next_stage in STAGES[1:]:
        stage = advance_stage(stage, STAGES[STAGES.index(next_stage) - 1], next_stage)
        assert stage == next_stage


def test_advance_stage_rejects_out_of_order_transition():
    with pytest.raises(ValueError, match="out-of-order"):
        advance_stage(POSTS_CLASSIFIED, NARRATIVES_DETECTED, RISK_SCORES_COMPUTED)


def test_risk_before_narratives_raises_stage_order_error():
    with pytest.raises(ValueError, match="out-of-order"):
        compute_risk_scores_node(
            {
                "stage": POSTS_CLASSIFIED,
                "classified_posts": [],
                "preprocessed_posts": [],
                "narratives": [],
            }
        )
