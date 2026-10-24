"""Compute the on-target % from the human labels and write the report (P8).

Gate P8 requires "human on-target % on a sample reported". ``scripts/build_redteam_frames.py``
writes the unlabeled sample; **the labeling is the human's work**; this turns the returned labels
into the reported number.

The labeler answers one question per frame: *would this frame plausibly expose disparate treatment
of its named target?* Not *is it harmful* — a frame naming an identity term was rejected before it
reached the sample, so every frame here is neutral text with one injected slot.

No model is loaded and nothing is auto-labeled. Asking a model whether a model-generated prompt is
on-target would be a scorer judging a scorer, which is what CLAUDE.md's guardrails forbid and what
would make the number worthless.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from margins.redteam import (
    DEFAULT_REDTEAM_SET,
    ON_TARGET_REPORT_FILENAME,
    ON_TARGET_SAMPLE_FILENAME,
    default_redteam_config_dir,
    load_redteam_config,
    on_target_metrics,
    read_on_target_labels,
    render_on_target_report,
)
from margins.redteam.on_target import OnTargetUnit
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
        required=True,
        help="Hand-filled JSONL: one {frame_id, on_target, labeler} record per line.",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("data/redteam") / ON_TARGET_SAMPLE_FILENAME,
        help="The sample the labels belong to. Labels outside it are refused.",
    )
    parser.add_argument("--redteam", default=DEFAULT_REDTEAM_SET)
    parser.add_argument("--redteam-config-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/redteam"))
    parser.add_argument("--paper-numbers", type=Path, default=PAPER_NUMBERS_PATH)
    parser.add_argument("--no-paper-numbers", action="store_true")
    parser.add_argument("--labeler-note", default="one human labeller")
    args = parser.parse_args(argv)

    config = load_redteam_config(
        args.redteam, config_dir=args.redteam_config_dir or default_redteam_config_dir()
    )

    if not args.sample.is_file():
        raise SystemExit(
            f"sample not found: {args.sample}. Run scripts/build_redteam_frames.py first."
        )
    units = [
        OnTargetUnit.model_validate_json(line)
        for line in args.sample.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = read_on_target_labels(args.labels, units=units)
    metrics = on_target_metrics(units, labels)

    report = render_on_target_report(
        metrics, redteam_version=config.redteam_version, n_labels=len(labels)
    )
    out_path = args.out_dir / ON_TARGET_REPORT_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(report)

    print(f"redteam_version:  {config.redteam_version}")
    print(f"sampled:          {len(units)}")
    print(f"labeled:          {len(labels)}")
    print("\non-target rate")
    for metric in metrics:
        rate = "—" if metric.on_target_rate is None else f"{metric.on_target_rate:.1%}"
        print(f"  {metric.stratum:<36} {metric.n_on_target}/{metric.n_labeled}  {rate}")
    print(f"\nwrote {out_path}")

    unlabeled = len(units) - len(labels)
    if unlabeled:
        print(
            f"\nNOTE: {unlabeled} sampled frame(s) carry no label. They count toward no "
            "denominator -- an unlabeled frame is not a miss."
        )

    if not args.no_paper_numbers:
        common = {
            "run_id": NO_RUN,
            "source": "scripts/redteam_on_target_report.py",
            "claimable": False,
            "note": f"Human on-target audit of red-team frames, {args.labeler_note}; descriptive.",
        }
        total = next(m for m in metrics if m.stratum == "ALL")
        registered = [
            PaperNumber(
                key="redteam.on_target.n_labeled", value=total.n_labeled, unit="count", **common
            ),
            PaperNumber(
                key="redteam.on_target.n_on_target", value=total.n_on_target, unit="count", **common
            ),
            PaperNumber(
                key="redteam.on_target.rate",
                value=total.on_target_rate,
                unit="probability",
                **common,
            ),
        ]
        write_paper_numbers(
            read_paper_numbers(args.paper_numbers).merged(registered), args.paper_numbers
        )
        print(f"paper      {len(registered)} numbers -> {args.paper_numbers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
