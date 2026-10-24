"""Build the `frame_breakdown` result table (figure 7) from the E1 item scores.

    uv run python scripts/build_frame_table.py --scores data/scoring/e1/<model>/scores.parquet ...

Mean standardized harm per template frame, per model and harm type, with a cluster-bootstrap CI
over prompts (:mod:`margins.report.frames`). Descriptive: a level per TEMPLATE, never an identity
interaction. ``run_p11.py process`` runs this in its tables stage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from margins.metrics.config import DEFAULT_METRICS, load_metrics_config
from margins.report.frames import frame_breakdown_rows
from margins.report.tables import (
    DEFAULT_TABLE_DIR,
    read_table,
    spec_for,
    write_table_csv,
    write_table_latex,
)
from margins.repro import set_global_seed
from margins.scoring.export import read_scores

BENCHMARK = "open_gen_templates"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--metrics", default=DEFAULT_METRICS, help="Supplies the bootstrap.")
    parser.add_argument("--n-resamples", type=int, default=None, help="Smoke runs only.")
    parser.add_argument("--no-latex", action="store_true")
    args = parser.parse_args(argv)

    bootstrap = load_metrics_config(args.metrics).bootstrap
    n_resamples = args.n_resamples if args.n_resamples is not None else bootstrap.n_resamples
    _, rng = set_global_seed(bootstrap.seed)
    scores = [row for path in args.scores for row in read_scores(path)]
    rows = frame_breakdown_rows(
        scores,
        benchmark=BENCHMARK,
        n_resamples=n_resamples,
        ci_level=bootstrap.ci_level,
        rng=rng,
    )
    path = write_table_csv(spec_for("frame_breakdown"), rows, args.out_dir / "frame_breakdown.csv")
    print(f"  frame_breakdown {len(rows):>5} rows  -> {path}")
    if not args.no_latex:
        write_table_latex(
            read_table("frame_breakdown", directory=args.out_dir),
            args.out_dir / "frame_breakdown.tex",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
