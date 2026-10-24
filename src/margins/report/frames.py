"""The per-frame breakdown table (figure 7): mean standardized harm by template frame.

**What it is.** For each model and harm type, the triangulated harm score (P7's own
:func:`~margins.metrics.metric.triangulate`, so the same ``harm_std`` every estimate is built on),
averaged over every response whose prompt came from a frame, with a cluster-bootstrap CI that
resamples whole PROMPTS within the frame. A prompt is the item; its seeds are not independent.

**What it is not.** A level, not an interaction, and about templates, not identity cells. It makes
no emergence claim. It shows how much the frames differ, which is the between-cluster variance
that bounds the design's precision (``docs/power_and_precision.md``).

Pure: records in, rows out. ``scripts/build_frame_table.py`` reads and writes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from margins.metrics.estimator import percentile_interval
from margins.metrics.metric import triangulate
from margins.schemas import ItemScore

__all__ = ["frame_breakdown_rows", "frame_of"]


def frame_of(prompt_id: str) -> str:
    """The template frame of a prompt: the first field of ``<template>::<cell>::<carrier>::…``."""
    frame = prompt_id.split("::", 1)[0]
    if not frame or frame == prompt_id:
        raise ValueError(f"{prompt_id!r} is not a <template>::<cell>::... prompt id")
    return frame


def frame_breakdown_rows(
    scores: Sequence[ItemScore],
    *,
    benchmark: str,
    n_resamples: int,
    ci_level: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """One row per (model, harm type, frame), in that sorted order.

    Raises:
        ValueError: on rows from another benchmark, or no model-condition rows at all.
    """
    model_rows = [s for s in scores if s.condition == "model"]
    foreign = sorted({s.benchmark for s in model_rows if s.benchmark != benchmark})
    if foreign:
        raise ValueError(f"frame breakdown for {benchmark!r} was given rows from {foreign}")
    if not model_rows:
        raise ValueError("no model-condition scores to summarize")

    rows: list[dict[str, Any]] = []
    groups = sorted({(s.model_id, s.harm_type) for s in model_rows})
    for model_id, harm in groups:
        _, observations = triangulate(
            [s for s in model_rows if (s.model_id, s.harm_type) == (model_id, harm)]
        )
        by_frame: dict[str, dict[str, list[float]]] = {}
        for obs in observations:
            prompts = by_frame.setdefault(frame_of(obs.prompt_id), {})
            prompts.setdefault(obs.prompt_id, []).append(obs.combined)
        for frame in sorted(by_frame):
            items = [np.asarray(values) for _, values in sorted(by_frame[frame].items())]
            mean = float(np.concatenate(items).mean())
            replicates = np.empty(n_resamples)
            for r in range(n_resamples):
                draws = rng.integers(0, len(items), size=len(items))
                replicates[r] = float(np.concatenate([items[d] for d in draws]).mean())
            low, high = percentile_interval(replicates, ci_level)
            rows.append(
                {
                    "benchmark": benchmark,
                    "model_id": model_id,
                    "harm_type": harm,
                    "frame": frame,
                    "n_items": len(items),
                    "mean_standardized": mean,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "data_status": "real",
                }
            )
    return rows
