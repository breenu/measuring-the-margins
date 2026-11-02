"""Registering pipeline outputs that lived only in data/ (margins.report.run_numbers)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from margins.report.run_numbers import (
    ConstantsMismatchError,
    GenerationCount,
    Provenance,
    agreement_numbers,
    gpu_hour_numbers,
    redteam_funnel_numbers,
    response_count_numbers,
    scale_numbers,
    seconds_per_generation,
    verified_constants,
)
from margins.stats.numbers import NO_RUN, PaperNumbers

WHERE = Provenance("run-x", "results/run_manifests/run-x.json", "test")


def _agreement_row(condition: str, metric: str, harm: str, value: str) -> dict[str, str]:
    return {
        "condition": condition,
        "metric": metric,
        "harm_type": harm,
        "value": value,
        "n_units": "100",
    }


def test_agreement_takes_only_the_model_condition_alpha() -> None:
    rows = [
        _agreement_row("model", "krippendorff_alpha", "toxicity", "-0.003"),
        _agreement_row("model", "cohen_kappa", "toxicity", "0.0"),
        _agreement_row("drift_control", "krippendorff_alpha", "toxicity", "0.5"),
    ]
    (number,) = agreement_numbers(rows, arm="e1", model_name="m", where=WHERE)
    assert number.key == "agreement.e1.m.toxicity.krippendorff_alpha"
    assert number.value == pytest.approx(-0.003)
    assert number.manifest_path == WHERE.manifest_path


def test_agreement_with_nothing_to_register_raises() -> None:
    with pytest.raises(ValueError, match="no model-condition"):
        agreement_numbers([], arm="e1", model_name="m", where=WHERE)


def test_constants_are_verified_against_the_stored_scores() -> None:
    raw = [0.1, 0.4, 0.2, 0.9]
    mean, sd = float(np.mean(raw)), float(np.std(raw))
    stored = [-(x - mean) / sd for x in raw]  # a sign = -1 scorer
    assert verified_constants(raw, stored, -1) == pytest.approx((mean, sd))
    with pytest.raises(ConstantsMismatchError):
        verified_constants(raw, stored, +1)


def test_a_degenerate_group_must_have_stored_zeros() -> None:
    assert verified_constants([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1) == (0.0, 0.0)
    with pytest.raises(ConstantsMismatchError):
        verified_constants([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1)


def test_scale_numbers_key_mean_and_sd_per_scorer_and_condition() -> None:
    numbers = scale_numbers(
        {("toxic_bert", "drift_control"): (0.06, 0.1)}, arm="e1", model_name="m", where=WHERE
    )
    assert {n.key: n.value for n in numbers} == {
        "scale.e1.m.toxic_bert.drift_control.raw_mean": 0.06,
        "scale.e1.m.toxic_bert.drift_control.raw_sd": 0.1,
    }


FRAMES = {
    "n_elicitation_prompts": 240,
    "n_responses": 1440,
    "n_candidates": 7318,
    "n_rejected": 960,
    "n_frames": 6358,
    "n_target_cells": 80,
}


def test_the_red_team_funnel_must_add_up() -> None:
    numbers = redteam_funnel_numbers(
        FRAMES,
        frames_used=51,
        audit_prompts=8160,
        carriers=2,
        frames_where=WHERE,
        prompts_where=WHERE,
    )
    assert {n.key: n.value for n in numbers}["redteam.funnel.frames_used"] == 51
    with pytest.raises(ValueError, match="audit prompts"):
        redteam_funnel_numbers(
            FRAMES,
            frames_used=51,
            audit_prompts=8000,
            carriers=2,
            frames_where=WHERE,
            prompts_where=WHERE,
        )
    with pytest.raises(ValueError, match="rejected"):
        redteam_funnel_numbers(
            {**FRAMES, "n_rejected": 1},
            frames_used=51,
            audit_prompts=8160,
            carriers=2,
            frames_where=WHERE,
            prompts_where=WHERE,
        )


def test_response_totals_split_generations_from_pair_scorings() -> None:
    counts = [
        GenerationCount("e1", "m", 10, True, WHERE),
        GenerationCount("crows", "m", 3, False, WHERE),
    ]
    numbers = {n.key: n for n in response_count_numbers(counts, prefix="p11")}
    assert numbers["responses.p11.total"].value == 13
    assert numbers["responses.p11.generations"].value == 10
    assert numbers["responses.p11.pair_scorings"].value == 3
    assert numbers["responses.p11.total"].run_id == NO_RUN
    assert numbers["responses.p11.e1.m"].run_id == WHERE.run_id
    with pytest.raises(ValueError, match="two response files"):
        response_count_numbers([counts[0], counts[0]], prefix="p11")


def test_the_gpu_hour_estimate_is_rate_times_generations_and_says_so() -> None:
    start = datetime(2026, 10, 13, 17, tzinfo=UTC)
    rate = seconds_per_generation(start, start + timedelta(seconds=3000), 1000)
    assert rate == 3.0
    numbers = {
        n.key: n
        for n in gpu_hour_numbers({"a": rate, "b": 4.0}, {"a": 1200, "b": 900}, derivation="t")
    }
    assert numbers["compute.gpu_hours_estimate"].value == pytest.approx(2.0)
    assert numbers["compute.gpu_hours_estimate"].note.startswith("ESTIMATE")  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        seconds_per_generation(start, start, 10)
    with pytest.raises(ValueError):
        gpu_hour_numbers({"a": 1.0}, {"b": 1}, derivation="t")


def test_everything_registered_is_a_valid_paper_number_collection() -> None:
    numbers = [
        *agreement_numbers(
            [_agreement_row("model", "krippendorff_alpha", "regard", "0.4")],
            arm="e1",
            model_name="m",
            where=WHERE,
        ),
        *scale_numbers({("s", "model"): (0.0, 1.0)}, arm="e1", model_name="m", where=WHERE),
    ]
    assert len(PaperNumbers.of(numbers).numbers) == 3
