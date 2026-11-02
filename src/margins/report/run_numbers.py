"""Paper numbers the P11 pipeline computed but never registered (P12).

Four quantities the manuscript needs were produced by the pipeline and then left only in the
gitignored ``data/`` tree, which the anonymous review mirror does not carry:

* **Inter-scorer agreement** (Krippendorff's alpha per model x harm type), written by
  ``run_scoring.py`` to ``agreement.csv``. Constraint 6 requires it to be *reported*, and on this
  run toxicity's two scorers barely agree at all -- a result, not a footnote.
* **The standardization constants** (raw mean and SD per scorer x condition). They are what
  ``standardize_within_benchmark`` divides by, and they carry the scale check behind the drift
  ratio: on control text the neural scorers' raw spread is *wider* than on model output, so the
  within-condition z-scoring cannot have inflated the ratio. The constants are re-derived from the
  stored raw scores and **verified against the stored standardized values**, so a constant that
  is not the one the pipeline used raises rather than being registered.
* **The red-team funnel**, from the frames manifest plus the frame cap the audit prompts were
  built under (``--max-frames``, chosen by ``run_p11.py budget``).
* **Response counts** per arm and model, each tied to the generation manifest that wrote them.

Also the R2 lexicon hits (P12, Gate 1 decision G4): how often the LDNOOBW leg fired in each R2
rerun, and where each hit lands on the original run's scale. ``docs/r2_diagnosis.md`` computed
these by hand; registering them lets the paper cite them by key.

Plus one ESTIMATE, labelled as one: GPU-hours from logged throughput. No manifest records a
duration, but every manifest records when its stage started. Within one ``run_p11.py e4`` command
the ``t10`` arm is followed immediately by the ``greedy`` arm (the orchestrator runs them in
sequence), so the gap between those two manifests is one uninterrupted ``t10`` arm of known size.
That interval also contains the next model load (about a minute), so the rate is slightly
pessimistic. Every other gap between manifests can contain idle time -- two lab sessions were
interrupted and resumed -- and is not used.

Everything here is pure: callers pass parsed rows and values, nothing reads a file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from margins.stats.numbers import NO_RUN, PaperNumber

__all__ = [
    "ConstantsMismatchError",
    "GenerationCount",
    "Provenance",
    "agreement_numbers",
    "gpu_hour_numbers",
    "largest_coefficient_numbers",
    "lexicon_hit_numbers",
    "redteam_funnel_numbers",
    "response_count_numbers",
    "scale_numbers",
    "seconds_per_generation",
    "verified_constants",
]

#: How closely a re-derived constant must reproduce every stored standardized value.
CONSTANTS_TOLERANCE = 1e-9


class ConstantsMismatchError(ValueError):
    """Re-derived standardization constants do not reproduce the stored standardized scores."""


@dataclass(frozen=True)
class Provenance:
    """The run a number came from."""

    run_id: str
    manifest_path: str
    source: str


def _number(key: str, value: float | int, unit: str, where: Provenance, note: str) -> PaperNumber:
    return PaperNumber(
        key=key,
        value=value,
        unit=unit,
        run_id=where.run_id,
        manifest_path=where.manifest_path,
        source=where.source,
        note=note,
    )


# --------------------------------------------------------------------------------------
# agreement
# --------------------------------------------------------------------------------------


def agreement_numbers(
    rows: Sequence[Mapping[str, str]], *, arm: str, model_name: str, where: Provenance
) -> list[PaperNumber]:
    """Krippendorff's alpha per harm type, on the MODEL condition, from ``agreement.csv`` rows."""
    out: list[PaperNumber] = []
    for row in rows:
        if row["condition"] != "model" or row["metric"] != "krippendorff_alpha":
            continue
        out.append(
            _number(
                f"agreement.{arm}.{model_name}.{row['harm_type']}.krippendorff_alpha",
                float(row["value"]),
                "none",
                where,
                (
                    "Krippendorff's alpha (interval) between the harm type's scorers, over "
                    f"{row['n_units']} responses. Agreement is not validity: two scorers can "
                    "agree and be wrong together."
                ),
            )
        )
    if not out:
        raise ValueError(f"no model-condition Krippendorff alpha for {arm}/{model_name}")
    return out


# --------------------------------------------------------------------------------------
# standardization constants
# --------------------------------------------------------------------------------------


def verified_constants(
    raw: Sequence[float], standardized: Sequence[float], sign: int
) -> tuple[float, float]:
    """``(raw_mean, raw_sd)`` of one standardization group, proven to be the ones used.

    Re-derived exactly as ``standardize_within_benchmark`` derives them (population SD), then
    checked against every stored standardized value. A degenerate group stored all zeros.
    """
    values = np.asarray(raw, dtype=float)
    stored = np.asarray(standardized, dtype=float)
    if values.shape != stored.shape or values.size == 0:
        raise ValueError("raw and standardized must be non-empty and the same length")
    mean = float(values.mean())
    sd = float(values.std(ddof=0))
    expected = np.zeros_like(values) if sd == 0.0 else sign * (values - mean) / sd
    worst = float(np.max(np.abs(expected - stored)))
    if worst > CONSTANTS_TOLERANCE:
        raise ConstantsMismatchError(
            f"re-derived constants (mean {mean:.6g}, sd {sd:.6g}, sign {sign:+d}) miss the stored "
            f"standardized values by up to {worst:.3g}; these are not the constants the "
            "pipeline used"
        )
    return mean, sd


def scale_numbers(
    constants: Mapping[tuple[str, str], tuple[float, float]],
    *,
    arm: str,
    model_name: str,
    where: Provenance,
) -> list[PaperNumber]:
    """Register ``(scorer, condition) -> (mean, sd)`` as ``scale.<arm>.<model>.<scorer>.<cond>``."""
    out: list[PaperNumber] = []
    note = (
        "Standardization constant (population SD / mean of the RAW score) for this scorer and "
        "condition, verified against the stored standardized scores."
    )
    for (scorer, condition), (mean, sd) in sorted(constants.items()):
        base = f"scale.{arm}.{model_name}.{scorer}.{condition}"
        out.append(_number(f"{base}.raw_mean", mean, "none", where, note))
        out.append(_number(f"{base}.raw_sd", sd, "none", where, note))
    return out


# --------------------------------------------------------------------------------------
# the largest emitted interaction coefficient, dependent ones included
# --------------------------------------------------------------------------------------


def largest_coefficient_numbers(
    rows: Sequence[Mapping[str, str]],
    *,
    scope: str,
    model_name: str,
    harm_type: str,
    where: Provenance,
) -> list[PaperNumber]:
    """The interaction coefficient of largest magnitude among ``rows``, with its bootstrap CI.

    ``rows`` are ``emergence_effects`` interaction rows for one model x harm type, ALL emitted
    coefficients and not only the free contrasts: which contrasts are "free" depends on which level
    of each axis is dropped, so a family-only maximum can be smaller than a coefficient a figure
    draws. The pre-registered power target (a concentrated 0.5 SD interaction) is a statement
    about exactly this quantity, the largest |gamma| in a pair's table.
    """
    interactions = [row for row in rows if row["kind"] == "interaction"]
    if not interactions:
        raise ValueError(f"no interaction rows for {model_name}/{harm_type}")
    top = max(interactions, key=lambda row: abs(float(row["estimate"])))
    kind = "free contrast" if top["family_member"] == "true" else "dependent coefficient"
    base = f"gamma.largest.{scope}.{model_name}.{harm_type}"
    note = (
        f"Largest |gamma| over all {len(interactions)} emitted interaction coefficients "
        f"({scope}), dependent ones included: {top['term']} ({top['axis_a']} x {top['axis_b']}), "
        f"a {kind}. Signed estimate, 95% cluster-bootstrap percentile CI. Descriptive: no p, no q."
    )
    within = sum(abs(float(row["estimate"])) <= 0.15 for row in interactions)
    return [
        PaperNumber(
            key=f"{base}.estimate",
            value=float(top["estimate"]),
            unit="sd",
            run_id=where.run_id,
            manifest_path=where.manifest_path,
            source=where.source,
            ci_low=float(top["ci_low"]),
            ci_high=float(top["ci_high"]),
            note=note,
        ),
        _number(f"{base}.n_coefficients", len(interactions), "count", where, note),
        _number(
            f"{base}.n_within_0_15",
            within,
            "count",
            where,
            "Emitted interaction coefficients with |gamma| <= 0.15 SD (" + scope + ").",
        ),
    ]


# --------------------------------------------------------------------------------------
# R2: the lexicon hits behind the free-tier failure
# --------------------------------------------------------------------------------------


def lexicon_hit_numbers(
    rerun_raw: Sequence[float],
    original_raw: Sequence[float],
    *,
    mean: float,
    sd: float,
    sign: int,
    machine: str,
    model_name: str,
    rerun_where: Provenance,
    original_where: Provenance,
) -> list[PaperNumber]:
    """How often the lexicon fired in an R2 rerun, and how far each hit sits on the ORIGINAL scale.

    ``mean``/``sd`` are the original run's standardization constants (the ones the R2 comparison
    maps the rerun through); ``original_raw`` is the original run restricted to the rerun's cells,
    so the two hit counts compare like with like. A hit is any non-zero raw score.
    """
    if sd <= 0.0:
        raise ValueError("the original constants are degenerate; a hit has no finite z-score")
    rerun = np.asarray(rerun_raw, dtype=float)
    original = np.asarray(original_raw, dtype=float)
    hits = rerun[rerun != 0.0]
    base = f"r2.lexicon.{machine}.{model_name}"
    note = (
        "LDNOOBW lexicon leg of toxicity in the R2 sampled rerun; a hit is a non-zero raw score. "
        "z-scores are through the ORIGINAL run's constants, as the R2 comparison standardizes."
    )
    out = [
        _number(f"{base}.n_responses", int(rerun.size), "count", rerun_where, note),
        _number(f"{base}.n_hits", int(hits.size), "count", rerun_where, note),
        _number(
            f"r2.lexicon.original.{model_name}.n_hits",
            int(np.count_nonzero(original)),
            "count",
            original_where,
            "LDNOOBW hits in the original E1 responses of the cells R2 reruns.",
        ),
    ]
    if hits.size:
        z = sign * (hits - mean) / sd
        out.append(_number(f"{base}.z_min", float(z.min()), "sd", rerun_where, note))
        out.append(_number(f"{base}.z_max", float(z.max()), "sd", rerun_where, note))
    return out


# --------------------------------------------------------------------------------------
# the red-team funnel
# --------------------------------------------------------------------------------------


def redteam_funnel_numbers(
    frames_block: Mapping[str, object],
    *,
    frames_used: int,
    audit_prompts: int,
    carriers: int,
    frames_where: Provenance,
    prompts_where: Provenance,
) -> list[PaperNumber]:
    """Elicitation -> candidates -> frames -> the capped subset -> audit prompts.

    Raises if the stages do not add up: candidates minus rejections must equal the frame pool,
    and the audit prompts must be every used frame in every target cell under every carrier.
    """
    elicitation_prompts = int(frames_block["n_elicitation_prompts"])  # type: ignore[call-overload]
    responses = int(frames_block["n_responses"])  # type: ignore[call-overload]
    candidates = int(frames_block["n_candidates"])  # type: ignore[call-overload]
    rejected = int(frames_block["n_rejected"])  # type: ignore[call-overload]
    pool = int(frames_block["n_frames"])  # type: ignore[call-overload]
    cells = int(frames_block["n_target_cells"])  # type: ignore[call-overload]
    if candidates - rejected != pool:
        raise ValueError(f"{candidates} candidates - {rejected} rejected != {pool} frames")
    if frames_used > pool:
        raise ValueError(f"{frames_used} frames used but the pool holds only {pool}")
    if frames_used * cells * carriers != audit_prompts:
        raise ValueError(
            f"{frames_used} frames x {cells} cells x {carriers} carriers != {audit_prompts} "
            "audit prompts"
        )
    note = "Red-team funnel, descriptive arm (powered=False, outside the FDR family)."
    return [
        _number(
            "redteam.funnel.elicitation_prompts", elicitation_prompts, "count", frames_where, note
        ),
        _number("redteam.funnel.elicitation_responses", responses, "count", frames_where, note),
        _number("redteam.funnel.candidates", candidates, "count", frames_where, note),
        _number("redteam.funnel.rejected", rejected, "count", frames_where, note),
        _number("redteam.funnel.frame_pool", pool, "count", frames_where, note),
        _number("redteam.funnel.target_cells", cells, "count", frames_where, note),
        _number(
            "redteam.funnel.frames_used",
            frames_used,
            "count",
            prompts_where,
            "The frame cap (--max-frames) the audit prompts were built under, sized from "
            "measured speed against the ~20 GPU-h red-team budget; chosen by content-ordered "
            "round-robin over (angle x pair).",
        ),
        _number("redteam.funnel.carriers", carriers, "count", prompts_where, note),
        _number(
            "redteam.funnel.audit_prompts_per_model", audit_prompts, "count", prompts_where, note
        ),
    ]


# --------------------------------------------------------------------------------------
# response counts
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationCount:
    """One response file: its arm, model, size, and whether its rows are generated text."""

    arm: str
    model_name: str
    n: int
    generated: bool
    where: Provenance


def response_count_numbers(counts: Sequence[GenerationCount], *, prefix: str) -> list[PaperNumber]:
    """One key per (arm, model), plus totals split into generations and pair scorings."""
    out: list[PaperNumber] = []
    seen: set[tuple[str, str]] = set()
    for count in counts:
        if (count.arm, count.model_name) in seen:
            raise ValueError(f"two response files for {count.arm}/{count.model_name}")
        seen.add((count.arm, count.model_name))
        kind = "generated responses" if count.generated else "sentence-pair log-likelihood scorings"
        out.append(
            _number(
                f"responses.{prefix}.{count.arm}.{count.model_name}",
                count.n,
                "count",
                count.where,
                f"Rows in the run's response file: {kind}.",
            )
        )
    totals = Provenance(
        NO_RUN, "", "scripts/emit_run_numbers.py: sum of the per-run response counts"
    )
    generated = sum(c.n for c in counts if c.generated)
    scorings = sum(c.n for c in counts if not c.generated)
    for key, value, note in (
        ("total", generated + scorings, "All model responses, generations plus pair scorings."),
        ("generations", generated, "Generated responses only."),
        ("pair_scorings", scorings, "Sentence-pair log-likelihood scorings (no text generated)."),
    ):
        out.append(
            PaperNumber(
                key=f"responses.{prefix}.{key}",
                value=value,
                unit="count",
                run_id=totals.run_id,
                manifest_path=None,
                source=totals.source,
                note=note,
            )
        )
    return out


# --------------------------------------------------------------------------------------
# GPU-hours: an ESTIMATE from logged throughput
# --------------------------------------------------------------------------------------


def seconds_per_generation(started: datetime, next_started: datetime, n: int) -> float:
    """Seconds per generation of a stage, from its start and the start of the stage after it.

    Valid only when nothing ran between the two (the stages are consecutive in one command); the
    interval then also holds the next stage's model load, so the rate errs slow.
    """
    if n <= 0:
        raise ValueError("a stage must have generated something")
    seconds = (next_started - started).total_seconds()
    if seconds <= 0:
        raise ValueError("the next stage must start after this one")
    return seconds / n


def gpu_hour_numbers(
    rates: Mapping[str, float],
    generations: Mapping[str, int],
    *,
    derivation: str,
) -> list[PaperNumber]:
    """Per-model seconds/generation and the summed GPU-hour estimate. Labelled an ESTIMATE."""
    if set(rates) != set(generations):
        raise ValueError(f"rates for {sorted(rates)} but generations for {sorted(generations)}")
    source = f"scripts/emit_run_numbers.py: {derivation}"

    def estimate(key: str, value: float, unit: str, note: str) -> PaperNumber:
        return PaperNumber(
            key=key, value=value, unit=unit, run_id=NO_RUN, manifest_path=None, source=source,
            note=f"ESTIMATE, not a measurement of the whole run. {note}",
        )  # fmt: skip

    out = [
        estimate(
            f"compute.seconds_per_generation.{model}",
            rate,
            "none",
            "Throughput of one uninterrupted E4 t10 arm (1,296 generations, batch 8).",
        )
        for model, rate in sorted(rates.items())
    ]
    hours = sum(rates[m] * generations[m] for m in rates) / 3600.0
    out.append(
        estimate(
            "compute.gpu_hours_estimate",
            hours,
            "gpu_hours",
            "Per-model throughput x that model's generations in the P11 lab run. Excludes the "
            "smoke run, GPU verification, the minutes of CrowS scoring, and R2.",
        )
    )
    return out
