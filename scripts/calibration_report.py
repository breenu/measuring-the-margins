"""Score the scorers against the human labels (P6).

    uv run python scripts/calibration_report.py --labels data/scoring/labels_toxicity.jsonl

Reads hand-filled labels, computes each scorer's precision / recall / F1 at its declared threshold,
and writes ``data/scoring/calibration_report.md``.

**What this answers, and what it does not.** Calibration asks whether a scorer finds what a person
calls harm. It says nothing about whether the scorer responds to identity terms independently of
harm -- that is the drift control -- nor about whether two scorers behave alike, which is the
agreement table. A harm type needs all three before it is load-bearing, and the generated report
says so on its face rather than leaving a reader to remember it.

The sample was deliberately drawn to over-weight scorer disagreement and order-2 cells, so precision
measured on it is **not** an estimate of precision on the corpus. The report states that too.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from margins.scoring.calibration import (
    CALIBRATION_REPORT_FILENAME,
    calibration_metrics,
    read_labels,
    render_calibration_report,
)
from margins.scoring.config import DEFAULT_SCORING, default_scoring_config_dir, load_scoring_config
from margins.scoring.export import DEFAULT_SCORING_DIR, SCORES_FILENAME, read_scores
from margins.stats.numbers import (
    NO_RUN,
    PAPER_NUMBERS_PATH,
    PaperNumber,
    read_paper_numbers,
    write_paper_numbers,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        nargs="+",
        required=True,
        help="Hand-filled label JSONL(s), e.g. one per harm type. They form ONE report.",
    )
    parser.add_argument(
        "--scores", type=Path, nargs="+", default=[DEFAULT_SCORING_DIR / SCORES_FILENAME]
    )
    parser.add_argument("--scoring", default=DEFAULT_SCORING)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SCORING_DIR)
    parser.add_argument("--paper-numbers", type=Path, default=PAPER_NUMBERS_PATH)
    parser.add_argument("--no-paper-numbers", action="store_true")
    parser.add_argument(
        "--labeler-note",
        default="one human labeller",
        help="Recorded on every registered number (who labelled, how many labellers).",
    )
    args = parser.parse_args(argv)

    config = load_scoring_config(args.scoring, config_dir=default_scoring_config_dir())
    # One report over every harm type labelled: run per file, the second report used to overwrite
    # the first (found on the first real labels, 24 Oct 2026).
    labels = [label for path in args.labels for label in read_labels(path)]
    scores = [row for path in args.scores for row in read_scores(path)]

    metrics = calibration_metrics(scores, labels, config)

    print(f"labels     {len(labels)}")
    print(f"scores     {len(scores)}\n")
    for metric in metrics:
        precision = "—" if metric.precision is None else f"{metric.precision:.3f}"
        recall = "—" if metric.recall is None else f"{metric.recall:.3f}"
        print(
            f"  {metric.harm_type:<10} {metric.scorer_id:<20} "
            f"n={metric.n:<5} precision={precision:<7} recall={recall}"
        )

    text = render_calibration_report(
        metrics, scoring_version=config.scoring_version, n_labels=len(labels)
    )
    path = args.out_dir / CALIBRATION_REPORT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)

    print(f"\nwrote {path}")

    if not args.no_paper_numbers:
        common = {
            "run_id": NO_RUN,
            "source": "scripts/calibration_report.py",
            "claimable": False,
            "note": f"Human calibration, {args.labeler_note}; a stratified sample that "
            "over-weights scorer disagreement, so NOT an estimate of corpus prevalence.",
        }
        registered: list[PaperNumber] = []
        for harm in sorted({m.harm_type for m in metrics if m.n}):
            evaluated = [m for m in metrics if m.harm_type == harm and m.n]
            for key, value in (
                ("n_labeled", evaluated[0].n),
                ("n_positive", evaluated[0].n_positive),
            ):
                registered.append(
                    PaperNumber(
                        key=f"calibration.{harm}.{key}", value=value, unit="count", **common
                    )
                )
            for m in evaluated:
                base = f"calibration.{harm}.{m.scorer_id}"
                registered += [
                    PaperNumber(
                        key=f"{base}.false_positives",
                        value=m.false_positives,
                        unit="count",
                        **common,
                    ),
                    PaperNumber(
                        key=f"{base}.precision", value=m.precision, unit="probability", **common
                    ),
                    PaperNumber(key=f"{base}.recall", value=m.recall, unit="probability", **common),
                ]
        write_paper_numbers(
            read_paper_numbers(args.paper_numbers).merged(registered), args.paper_numbers
        )
        print(f"paper      {len(registered)} numbers -> {args.paper_numbers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
