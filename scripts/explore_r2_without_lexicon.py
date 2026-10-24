"""EXPLORATORY, POST-HOC: R2's comparison with the LDNOOBW lexicon leg removed from toxicity.

    uv run python scripts/explore_r2_without_lexicon.py --machine kaggle

**This is not R2.** R2's verdict is the pre-registered one (``scripts/run_r2_analysis.py``) and it
stands. This script was written AFTER the Kaggle rerun failed, to measure how much of that failure
the lexicon leg accounts for (``docs/r2_diagnosis.md``). It is exploratory by construction, logged
in the pre-registration's §10 as post-hoc, and its numbers are registered under ``exploratory.*``,
never as a claim.

**It also breaks a framing constraint on purpose, so say so wherever it is cited:** toxicity scored
by ``toxic_bert`` alone is single-classifier (constraint 6). That is acceptable for a diagnostic of
the instrument and is never acceptable for a finding about a model.

**Apples to apples.** BOTH sides are refitted without the lexicon: the original class x race scores
(as standardized in the reported run) and the rerun (through the original run's constants). The
original intervals therefore change too; comparing a lexicon-free rerun to lexicon-bearing
intervals would be meaningless. Regard does not use the lexicon, so its rows are the pre-registered
ones, refitted identically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from margins.lattice.build import DeclaredLattice
from margins.lattice.config import load_lattice_config
from margins.metrics.config import DEFAULT_METRICS, load_metrics_config
from margins.orchestrate.config import load_p11_config
from margins.orchestrate.plan import OPEN_GEN_BENCHMARK, Layout, r2_arm_dir
from margins.orchestrate.smoke import pair_cells
from margins.schemas import ItemScore
from margins.scoring.config import DEFAULT_SCORING, load_scoring_config
from margins.scoring.export import SCORES_FILENAME, read_scores
from margins.scoring.standardize import (
    signs_from_specs,
    standardize_with_reference,
    standardize_within_benchmark,
)
from margins.sensitivity import compare_to_original, fit_pair, free_terms, restrict_to_cells
from margins.stats.numbers import (
    NO_RUN,
    PAPER_NUMBERS_PATH,
    PaperNumber,
    read_paper_numbers,
    write_paper_numbers,
)

LEXICON = "ldnoobw_lexicon"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--p11", default="p11_v1")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--scoring", default=DEFAULT_SCORING)
    parser.add_argument("--paper-numbers", type=Path, default=PAPER_NUMBERS_PATH)
    parser.add_argument("--no-paper-numbers", action="store_true")
    args = parser.parse_args(argv)

    p11 = load_p11_config(args.p11)
    layout = Layout(data=args.data_root)
    lattice = DeclaredLattice(load_lattice_config(p11.lattice))
    cells = pair_cells(lattice, p11.r2.pair)
    axis_pair = (cells[0].axes[0][0], cells[0].axes[1][0])
    terms = free_terms(cells)
    metrics = load_metrics_config(args.metrics)
    signs = signs_from_specs(load_scoring_config(args.scoring).scorers)
    min_cell_n = lattice.config.power.min_cell_n

    original: list[ItemScore] = []
    rerun: list[ItemScore] = []
    for model in p11.e1.models:
        full = read_scores(layout.scoring("e1", model) / SCORES_FILENAME)
        _, reference = standardize_within_benchmark(full, benchmark=OPEN_GEN_BENCHMARK, signs=signs)
        sampled = layout.scoring(r2_arm_dir(args.machine, "sampled"), model) / SCORES_FILENAME
        for target, rows in (
            (original, restrict_to_cells(full, cells)),
            (rerun, restrict_to_cells(read_scores(sampled), cells)),
        ):
            mapped = standardize_with_reference(
                rows, benchmark=OPEN_GEN_BENCHMARK, reference=reference
            )
            target.extend(s for s in mapped if s.scorer_id != LEXICON)

    before = fit_pair(original, cells, config=metrics, min_cell_n=min_cell_n)
    after = fit_pair(rerun, cells, config=metrics, min_cell_n=min_cell_n)
    comparisons = compare_to_original(
        before.effects, after.effects, axis_pair=axis_pair, terms=terms
    )

    print("EXPLORATORY, POST-HOC -- not R2. Toxicity = toxic_bert ALONE (single-classifier).\n")
    summary: dict[str, float] = {}
    for harm in ("toxicity", "regard"):
        rows = [c for c in comparisons if c.harm_type == harm]
        inside = sum(1 for c in rows if c.inside_original_ci)
        worst = max(rows, key=lambda c: c.difference_in_se)
        print(
            f"  {harm:<9} inside CI {inside}/{len(rows)}   "
            f"max |diff| {worst.difference_in_se:.2f} SE"
            f"  ({worst.model_id.split('/')[-1]}, {worst.term})"
        )
        summary[f"{harm}.n_inside_ci"] = inside
        summary[f"{harm}.n_terms"] = len(rows)
        summary[f"{harm}.max_abs_diff_se"] = worst.difference_in_se
    total = sum(1 for c in comparisons if c.inside_original_ci)
    print(f"\n  all terms inside CI {total}/{len(comparisons)}")
    for c in sorted(comparisons, key=lambda c: -c.difference_in_se)[:5]:
        print(
            f"    {c.model_id.split('/')[-1]:<24} {c.harm_type:<9} {c.term:<55} "
            f"{c.difference_in_se:5.2f} SE  in CI: {c.inside_original_ci}"
        )

    if not args.no_paper_numbers:
        prefix = f"exploratory.r2_{args.machine}_no_lexicon"
        common = {
            "run_id": NO_RUN,
            "source": "scripts/explore_r2_without_lexicon.py",
            "claimable": False,
            "note": "POST-HOC EXPLORATORY diagnostic of the R2 failure; toxicity = toxic_bert "
            "alone (single-classifier). Not R2's verdict.",
        }
        registered = [
            PaperNumber(key=f"{prefix}.n_inside_ci", value=total, unit="count", **common),
            PaperNumber(key=f"{prefix}.n_terms", value=len(comparisons), unit="count", **common),
            *(
                PaperNumber(
                    key=f"{prefix}.{key}",
                    value=value,
                    unit="ratio" if key.endswith("_se") else "count",
                    **common,
                )
                for key, value in summary.items()
            ),
        ]
        write_paper_numbers(
            read_paper_numbers(args.paper_numbers).merged(registered), args.paper_numbers
        )
        print(f"\npaper      {len(registered)} exploratory numbers -> {args.paper_numbers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
