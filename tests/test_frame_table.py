"""The frame_breakdown table (figure 7): per-frame means of P7's harm_std, clustered by prompt."""

from __future__ import annotations

import numpy as np
import pytest

from margins.lattice import DeclaredLattice, load_lattice_config
from margins.metrics.metric import triangulate
from margins.report.frames import frame_breakdown_rows, frame_of
from tests.p11_synthetic import OPEN_GEN_ROSTER, grid, model_scores

BENCHMARK = "open_gen_templates"


def _scores(seed: int = 0):  # type: ignore[no-untyped-def]
    cells = grid(DeclaredLattice(load_lattice_config("confirmatory_v1")), ("age", "ability"))
    return model_scores(
        cells,
        benchmark=BENCHMARK,
        model_id="m/x",
        roster=OPEN_GEN_ROSTER,
        n_items=3,
        seeds=(1, 2),
        gamma_scale=0.5,
        rng=np.random.default_rng(seed),
    )


def test_frame_of_reads_the_template_field() -> None:
    assert frame_of("describe::age=older|ability=blind::person::abc123") == "describe"
    with pytest.raises(ValueError):
        frame_of("no-separator")


def test_one_row_per_model_harm_frame_with_the_mean_of_harm_std() -> None:
    scores = _scores()
    rows = frame_breakdown_rows(
        scores, benchmark=BENCHMARK, n_resamples=50, ci_level=0.95, rng=np.random.default_rng(1)
    )
    assert {(r["harm_type"], r["frame"]) for r in rows} == {
        (h, f"frame{i}") for h in OPEN_GEN_ROSTER for i in range(3)
    }
    # The mean is over P7's triangulated harm_std, not over raw scorer rows.
    _, observations = triangulate([s for s in scores if s.harm_type == "regard"])
    frame0 = [o.combined for o in observations if o.prompt_id.startswith("frame0::")]
    row = next(r for r in rows if r["harm_type"] == "regard" and r["frame"] == "frame0")
    assert row["mean_standardized"] == pytest.approx(float(np.mean(frame0)))
    assert row["n_items"] == len(
        {o.prompt_id for o in observations if o.prompt_id.startswith("frame0::")}
    )
    assert all(r["ci_low"] <= r["mean_standardized"] <= r["ci_high"] for r in rows)
    assert {r["data_status"] for r in rows} == {"real"}


def test_another_benchmark_is_refused() -> None:
    scores = [s.model_copy(update={"benchmark": "redteam"}) for s in _scores()]
    with pytest.raises(ValueError, match="redteam"):
        frame_breakdown_rows(
            scores, benchmark=BENCHMARK, n_resamples=10, ci_level=0.95, rng=np.random.default_rng(0)
        )
