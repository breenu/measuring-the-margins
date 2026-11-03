# Design: constraints, invariants and conventions

This document states the rules the `margins` package is built under: what the instrument may and
may not claim, the properties the code guarantees, and the conventions a contributor is expected
to follow. Method details for each data source are in [`methods.md`](methods.md); the
pre-registered analysis is in [`preregistration.md`](preregistration.md); the power analysis is in
[`power_and_precision.md`](power_and_precision.md).

## Scope

`margins` is a framework for auditing intersectional bias in open-weight language models. Its
contribution is the instrument and its rigour, not a finding that intersectional bias exists.

- **Two kinds of prompt source, three arms.** Templated open-ended generation over a declared
  identity lattice; cell-targeted automated red-teaming, which reuses the same renderer; and a
  sentence-pair benchmark (CrowS-Pairs), scored by sequence likelihood.
- **The identity lattice is declared, not a cartesian product.** Order-1 cells carry one
  descriptor and order-2 cells two, from declared axis pairs only; depth is fixed at two.
  Confirmatory pairs form the multiple-testing family, and at least two of them cross two
  non-gender axes, so that confirmatory claims are not anchored on gender. Any other pair is
  exploratory and reported descriptively.
- **Two cross-paradigm sources means no pooling.** With two sources there is no meaningful
  between-source variance, so cross-paradigm results are reported as an effect vector, not a
  pooled estimate.

## Framing constraints

These govern claims, code comments and documentation.

1. **Instrument, not finding.** The contribution is the measurement framework and its rigour.
   That language models show intersectional bias is established prior work.
2. **Accessible compute.** The framework is designed for commodity hardware, a single consumer or
   workstation GPU, with free-tier cloud as a documented fallback. A claim about where something
   ran must state where it actually ran.
3. **Emergence is an interaction, not a level.** Emergent intersectional bias is the interaction
   term of a factorial harm model, never a high harm level at a cell.
4. **Quantization is a fixed, recorded constant**, not an experimental factor and not a finding.
5. **Benchmark intersectional subsets that were not designed for interaction estimation are
   calibration only**, never load-bearing for an intersectional claim.
6. **No single-classifier dependence.** Every harm type uses at least two scorers of different
   mechanisms, with their agreement reported, and a scorer identity-drift control. A harm type with
   high drift is reported descriptively.

## Architecture

A pipeline of stages with an immutable record type between each pair:

    lattice -> renderers -> inference -> scoring (+ drift control) -> metrics -> stats -> report

Each stage is a module under `src/margins/` and a script under `scripts/`. Every stage is
deterministic given its configuration and seed, writes a run manifest, and can be resumed or
re-run without changing its output.

## Records and manifests

- **Shared records are frozen.** Transformations return new records (`model_copy(update=...)`);
  nothing is mutated in place. Container fields are tuples or immutable mappings, so records stay
  hashable.
- **No source-specific field on a shared record.** Source quirks live in `Item.payload`, typed at
  the loader boundary. Shared records forbid extra fields, and
  `tests/test_schema_field_freeze.py` pins each record's exact field set, so a contract change is
  always deliberate.
- **Every run writes a manifest** recording its configuration and its hash, checksums of every
  input file, the repository state, library versions, thread count and numerical libraries, and on
  GPU hosts the device, driver and CUDA versions.
- **Identifiers are content-addressed.** A prompt id digests its rendered text; a response id
  digests the prompt, model, revision, seed, sample index and decoding profile, and is computed
  before the work runs.

## Statistical protocol

- **The estimand.** For a declared axis pair, `harm_std ~ A + B + A:B` over the pair's order-2
  cells, with sum-to-zero coding, fitted by least squares on standardized harm scores. Emergence is
  the interaction coefficient.
- **Standardization is within source.** Scores are z-scored within source x scorer x harm type x
  condition (and per model), never across sources; higher always means more harm, and every scorer
  carries a probe that checks its sign.
- **Clustering.** Prompts are reused across samples and frames recur across cells, so standard
  errors are cluster-robust by prompt and intervals come from a cluster bootstrap over prompts,
  stratified by cell. Residuals are never treated as independent.
- **The additive null is not an independent baseline**; it is a presentation of the interaction.
  The independent check is a permutation test.
- **Multiple testing.** The family is declared from the lattice before any estimate is read: the
  free interaction contrasts of the confirmatory pairs, across models and harm types.
  Benjamini-Hochberg is primary and Benjamini-Yekutieli a sensitivity column. Correcting a subset
  of the family is an error, and nothing is ever corrected within a single source.
- **Only the stats engine sets significance.** `q`, `family_size` and `correction` are set in one
  place, and a test scans the source tree to enforce it. Descriptive records carry no `p` or `q`.
- **Underpowered cells are descriptive.** Power status comes from the actual cell size; the
  report API defaults to powered cells only.

## Construct validity

- **Two scorers per harm type, of different mechanisms** (for example, a neural classifier and a
  lexicon). Two classifiers can agree with each other and be wrong together, so agreement is
  reported but is not treated as validity.
- **Scorer identity-drift control.** Every scorer is applied to known-neutral text that differs
  across cells only in its identity terms, the same model is fitted, and the scorer's own
  interaction is reported beside the model's, never subtracted. A harm type whose scorer drift
  reaches the pre-registered threshold is descriptive-only, and the claim layer refuses to issue a
  significance claim for it.
- **Human calibration** of each scorer against labelled samples; precision and recall are reported
  as undefined, not zero, when their denominators are empty.
- **An LLM judge is never ground truth**; if used, it is one scorer among several, with a fixed,
  documented prompt and seed.

## Environment and compute

- **Dependencies are locked** with `uv`, and `uv.lock` resolves for exactly two environments:
  Windows (`win_amd64`, core and development dependencies, CPU) and Linux (`linux_x86_64`, with the
  GPU inference stack). Never install a package unpinned or bypass the lockfile to make something
  work; if a dependency will not resolve, report it.
- **The core environment has no GPU stack.** Scoring, metrics, statistics, figures and tables run
  on CPU in the core environment; `margins.inference` imports without torch installed.
- **4-bit NF4 loading needs Linux and CUDA.** A confirmatory model whose achieved quantization
  differs from the requested one is refused rather than silently loaded at a different precision.
- **Continuous integration has no GPU.** It runs the full test suite, lint, format and type checks
  on both operating systems, and regenerates every figure.
- **Pins are resolved, never authored.** Model revisions, scorer checkpoints, word lists and
  benchmark files are resolved from their upstream source and verified by script; a verification
  that fails stops rather than substituting.

## Determinism and validation

- **Every stochastic path takes a seed** and routes it through `margins.repro.set_global_seed`;
  there is no hidden global random state.
- **Determinism is scoped.** Same-seed greedy decoding produces identical token sequences on CPU
  at a fixed thread count; log-probabilities match to a documented tolerance; on GPU, agreement is
  claimed at the metric level within a reported tolerance. Batch size is part of that scope,
  because padding changes the order of floating-point reductions.
- **The estimator and the statistics layer are tested first against synthetic data with known
  answers.** A small end-to-end run on real multi-sample data must reach the correction path
  before any full run.

## Conventions

- `ruff` for lint and format (line length 100, Python 3.11); `mypy --strict` on `src/`; every
  public function is typed.
- Module docstrings state why the module exists and any construct or rigour caveat, not only what
  it does.
- Configuration is plain YAML composed by Hydra and validated into Pydantic models; tracking is
  the run manifest only.
- Raw text is JSONL, tabular data Parquet. `data/` is gitignored and regenerable; small final
  tables are committed under `results/`, and every number quoted in a paper is read from
  `results/paper_numbers.json` by key.
- No load-bearing notebooks: production logic lives in `src/margins/`, nothing in `src/` or
  `scripts/` imports from `notebooks/`, and notebooks do not write to `results/`.
- Model weights and raw response dumps are never committed.
- The pre-registration is tagged before any full run; amendments are versioned and logged with
  their rationale.
