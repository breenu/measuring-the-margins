"""Register the P11 quantities that lived only in ``data/`` into ``results/paper_numbers.json``.

    uv run python scripts/emit_run_numbers.py

The manuscript cites numbers by key, and the anonymous review mirror carries ``results/`` but not
the gitignored ``data/`` tree. This script moves four pipeline outputs across that seam, each tied
to the manifest of the run that produced it (see ``margins.report.run_numbers`` for why each one
matters and how it is checked):

* inter-scorer agreement per model x harm type (``agreement.csv`` from ``run_scoring.py``);
* the standardization constants, re-derived and verified against the stored standardized scores;
* the red-team funnel, including the frame cap the audit prompts were built under;
* response counts per arm and model, from the response files the generation manifests name;
* the R2 lexicon hits per rerun, standardized through the original run's constants.

It also registers a GPU-hour ESTIMATE from logged throughput, labelled as an estimate.

Idempotent: re-running with unchanged inputs writes identical bytes, and a changed value for an
existing key is refused by ``PaperNumbers.merged`` rather than silently replaced.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from margins.inference.harness import read_responses
from margins.inference.models import load_model_config
from margins.lattice.config import DEFAULT_LATTICE, load_lattice_config
from margins.report.run_numbers import (
    GenerationCount,
    Provenance,
    agreement_numbers,
    gpu_hour_numbers,
    largest_coefficient_numbers,
    lexicon_hit_numbers,
    redteam_funnel_numbers,
    response_count_numbers,
    scale_numbers,
    seconds_per_generation,
    verified_constants,
)
from margins.scoring.config import DEFAULT_SCORING, load_scoring_config
from margins.scoring.export import (
    AGREEMENT_CSV_FILENAME,
    DRIFT_FILENAME,
    SCORES_FILENAME,
    read_drift_rows,
    read_scores,
)
from margins.scoring.standardize import signs_from_specs
from margins.stats.numbers import PAPER_NUMBERS_PATH, read_paper_numbers, write_paper_numbers

MODELS = ("llama31_8b_instruct", "mistral7b_instruct_v03")
LEXICON = "ldnoobw_lexicon"
SCRIPT = "scripts/emit_run_numbers.py"

#: (directory under data/responses, key segment, generated text?) for the P11 lab run and R2.
P11_ARMS = (
    ("e1", "e1", True),
    ("crows", "crows", False),
    ("e4_t10", "e4_t10", True),
    ("e4_greedy", "e4_greedy", True),
    ("redteam_elicitation", "redteam_elicitation", True),
    ("redteam", "redteam_audit", True),
)
R2_ARMS = (
    ("r2_kaggle_sampled", "kaggle_sampled", True),
    ("r2_kaggle_greedy", "kaggle_greedy", True),
    ("r2_lab_sampled", "lab_sampled", True),
    ("r2_lab_greedy", "lab_greedy", True),
)


class Manifests:
    """The committed run manifests, looked up by run id or by a unique name prefix."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.paths = {p.stem: p for p in sorted(directory.glob("*.json"))}

    def load(self, run_id: str) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads(self.paths[run_id].read_text(encoding="utf-8"))
        return loaded

    def unique(self, prefix: str) -> str:
        matches = [run_id for run_id in self.paths if run_id.startswith(prefix)]
        if len(matches) != 1:
            raise SystemExit(f"expected exactly one manifest starting {prefix!r}, found {matches}")
        return matches[0]

    def provenance(self, run_id: str, source: str) -> Provenance:
        if run_id not in self.paths:
            raise SystemExit(f"no committed manifest for run {run_id!r} in {self.directory}")
        return Provenance(run_id, self.paths[run_id].as_posix(), source)

    def started(self, run_id: str) -> datetime:
        return datetime.fromisoformat(self.load(run_id)["created_at"].replace("Z", "+00:00"))


def _one(directory: Path, pattern: str) -> Path:
    found = sorted(directory.glob(pattern))
    if len(found) != 1:
        raise SystemExit(f"expected exactly one {pattern} in {directory}, found {found}")
    return found[0]


def _response_counts(
    data: Path, manifests: Manifests, arms: tuple[tuple[str, str, bool], ...]
) -> list[GenerationCount]:
    counts: list[GenerationCount] = []
    for directory, key, generated in arms:
        for model_dir in sorted((data / "responses" / directory).iterdir()):
            path = _one(model_dir, "*.jsonl")
            run_id = path.stem
            responses = read_responses(path)
            if len({r.response_id for r in responses}) != len(responses):
                raise SystemExit(f"{path} repeats a response_id")
            model_name = manifests.load(run_id)["config"]["model_name"]
            counts.append(
                GenerationCount(
                    arm=key,
                    model_name=model_name,
                    n=len(responses),
                    generated=generated,
                    where=manifests.provenance(
                        run_id, f"{SCRIPT}: rows of {directory}/{model_name}"
                    ),
                )
            )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--manifests", type=Path, default=Path("results") / "run_manifests")
    parser.add_argument("--tables", type=Path, default=Path("results") / "tables")
    parser.add_argument("--scoring", default=DEFAULT_SCORING)
    parser.add_argument("--out", type=Path, default=PAPER_NUMBERS_PATH)
    args = parser.parse_args(argv)

    manifests = Manifests(args.manifests)
    signs = signs_from_specs(load_scoring_config(args.scoring).scorers)
    entries = []
    e1_constants: dict[str, dict[tuple[str, str], tuple[float, float]]] = {}

    # --- agreement + standardization constants -------------------------------------------
    for model in MODELS:
        for arm in ("e1", "crows"):
            run_id = manifests.unique(f"scoring-{arm}-{model}-")
            scoring_dir = args.data / "scoring" / arm / model
            with (scoring_dir / AGREEMENT_CSV_FILENAME).open(encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            where = manifests.provenance(run_id, f"run_scoring.py -> {arm}/{model}/agreement.csv")
            entries += agreement_numbers(rows, arm=arm, model_name=model, where=where)

        run_id = manifests.unique(f"scoring-e1-{model}-")
        scoring_dir = args.data / "scoring" / "e1" / model
        groups: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
        for score in read_scores(scoring_dir / SCORES_FILENAME):
            raw, std = groups.setdefault((score.scorer_id, score.condition), ([], []))
            raw.append(score.raw)
            std.append(score.standardized)
        for drift in read_drift_rows(scoring_dir / DRIFT_FILENAME):
            raw, std = groups.setdefault((drift.scorer_id, drift.condition), ([], []))
            raw.append(drift.raw)
            std.append(drift.standardized)
        constants = {
            key: verified_constants(raw, std, signs[key[0]]) for key, (raw, std) in groups.items()
        }
        where = manifests.provenance(
            run_id, f"{SCRIPT}: standardization constants of e1/{model}, verified"
        )
        entries += scale_numbers(constants, arm="e1", model_name=model, where=where)
        e1_constants[model] = constants

    # --- the red-team funnel ---------------------------------------------------------------
    frames_run = manifests.unique("build-redteam-frames-")
    prompts_run = manifests.unique("build-redteam-prompts-")
    audit_path = args.data / "prompts" / "cell_targeted_v1.jsonl"
    with audit_path.open(encoding="utf-8") as fh:
        audit_ids = [json.loads(line)["prompt_id"] for line in fh if line.strip()]
    frames_used = len({prompt_id.split("::")[0] for prompt_id in audit_ids})
    carriers = len({prompt_id.split("::")[2] for prompt_id in audit_ids})
    with (args.data / "prompts" / "cell_targeted_v1_frame_selection.csv").open(
        encoding="utf-8", newline=""
    ) as fh:
        selected = len(list(csv.DictReader(fh)))
    if selected != frames_used:
        raise SystemExit(
            f"frame selection lists {selected} frames, audit prompts use {frames_used}"
        )
    entries += redteam_funnel_numbers(
        manifests.load(frames_run)["redteam"],
        frames_used=frames_used,
        audit_prompts=len(audit_ids),
        carriers=carriers,
        frames_where=manifests.provenance(frames_run, "build_redteam_frames.py manifest"),
        prompts_where=manifests.provenance(prompts_run, f"{SCRIPT}: {audit_path.as_posix()}"),
    )

    # --- response counts -------------------------------------------------------------------
    p11 = _response_counts(args.data, manifests, P11_ARMS)
    entries += response_count_numbers(p11, prefix="p11")
    entries += response_count_numbers(_response_counts(args.data, manifests, R2_ARMS), prefix="r2")

    # --- the largest emitted interaction coefficient, dependent ones included ----------------
    # Read from the COMMITTED result table (written from the stats run), so the anonymous mirror
    # can check it. Two scopes: the confirmatory pairs the family covers, and every declared pair.
    exploratory = {
        p.axes for p in load_lattice_config(DEFAULT_LATTICE).pairs if p.status == "exploratory"
    }
    table = args.tables / "emergence_effects.csv"
    with table.open(encoding="utf-8", newline="") as fh:
        effects = [row for row in csv.DictReader(fh) if row["benchmark"] == "open_gen_templates"]
    stats_where = manifests.provenance(
        manifests.unique("stats-e1-"), f"{SCRIPT}: {table.as_posix()} interaction rows"
    )
    for model in MODELS:
        model_id = load_model_config(model).model_id
        for harm in sorted({row["harm_type"] for row in effects}):
            rows = [r for r in effects if r["model_id"] == model_id and r["harm_type"] == harm]
            confirmatory = [r for r in rows if (r["axis_a"], r["axis_b"]) not in exploratory]
            for scope, subset in (("confirmatory_pairs", confirmatory), ("all_pairs", rows)):
                entries += largest_coefficient_numbers(
                    subset, scope=scope, model_name=model, harm_type=harm, where=stats_where
                )

    # --- R2: lexicon hits, on the original scale ----------------------------------------------
    for model in MODELS:
        e1_run = manifests.unique(f"scoring-e1-{model}-")
        original = [
            s
            for s in read_scores(args.data / "scoring" / "e1" / model / SCORES_FILENAME)
            if s.scorer_id == LEXICON and s.condition == "model"
        ]
        mean, sd = e1_constants[model][(LEXICON, "model")]
        for machine in ("kaggle", "lab"):
            arm = f"r2_{machine}_sampled"
            rerun = [
                s
                for s in read_scores(args.data / "scoring" / arm / model / SCORES_FILENAME)
                if s.scorer_id == LEXICON
            ]
            cells = {s.cell_id for s in rerun}
            entries += lexicon_hit_numbers(
                [s.raw for s in rerun],
                [s.raw for s in original if s.cell_id in cells],
                mean=mean,
                sd=sd,
                sign=signs[LEXICON],
                machine=machine,
                model_name=model,
                rerun_where=manifests.provenance(
                    manifests.unique(f"scoring-{arm}-{model}-"), f"{SCRIPT}: {arm}/{model} scores"
                ),
                original_where=manifests.provenance(
                    e1_run, f"{SCRIPT}: e1/{model} scores, R2 cells"
                ),
            )

    # --- GPU-hours, an ESTIMATE ------------------------------------------------------------
    rates = {
        model: seconds_per_generation(
            manifests.started(manifests.unique(f"e4-t10-{model}-")),
            manifests.started(manifests.unique(f"e4-greedy-{model}-")),
            next(c.n for c in p11 if c.arm == "e4_t10" and c.model_name == model),
        )
        for model in MODELS
    }
    generations = {
        model: sum(c.n for c in p11 if c.generated and c.model_name == model) for model in MODELS
    }
    entries += gpu_hour_numbers(
        rates,
        generations,
        derivation=(
            "per model, (start of the e4-greedy manifest - start of the e4-t10 manifest) / "
            "e4-t10 generations, the two arms run back to back by one run_p11.py e4 command; "
            "times that model's P11 generations"
        ),
    )

    numbers = read_paper_numbers(args.out).merged(entries)
    write_paper_numbers(numbers, args.out)
    print(f"registered {len(entries)} numbers -> {args.out} ({len(numbers.numbers)} keys)")
    for entry in entries:
        if entry.key.startswith(
            ("agreement.", "compute.", "redteam.funnel.", "responses.p11.", "gamma.")
        ):
            print(f"  {entry.key:<62} {entry.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
