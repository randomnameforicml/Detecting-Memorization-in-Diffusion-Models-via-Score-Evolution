"""Aggregate per-prompt DSM scores into ROC-AUC and TPR@1%FPR."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import roc_auc_score

from .io_utils import write_csv


def tpr_at_max_fpr(labels: Iterable[int], scores: Iterable[float], max_fpr: float = 0.01) -> Dict[str, float]:
    """Return the best TPR under FPR<=max_fpr by scanning score thresholds.

    This is equivalent to using a high negative-score quantile threshold, but
    scanning unique thresholds handles ties transparently and guarantees the
    reported FPR is not above the requested bound.
    """
    y = np.asarray(list(labels), dtype=np.int64)
    s = np.asarray(list(scores), dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    thresholds = np.unique(s)[::-1]
    best_tpr, best_fpr, best_threshold = 0.0, 0.0, float("inf")
    for threshold in thresholds:
        fpr = float(np.mean(neg >= threshold)) if len(neg) else 0.0
        if fpr <= max_fpr + 1e-12:
            tpr = float(np.mean(pos >= threshold)) if len(pos) else 0.0
            if tpr > best_tpr or (tpr == best_tpr and fpr < best_fpr):
                best_tpr, best_fpr, best_threshold = tpr, fpr, float(threshold)
    return {"tpr_at_1pct_fpr": best_tpr, "fpr": best_fpr, "threshold": best_threshold}


def aggregate_rows(rows: List[Dict[str, str]], metric_names: List[str]) -> List[Dict[str, object]]:
    out = []
    labels = [int(r["label"]) for r in rows]
    model = rows[0].get("model", "") if rows else ""
    num_steps = rows[0].get("num_steps", "") if rows else ""
    seed = rows[0].get("seed", "") if rows else ""
    for metric_name in metric_names:
        scores = [float(r[metric_name]) for r in rows]
        roc_auc = float("nan") if len(set(labels)) < 2 else float(roc_auc_score(labels, scores))
        tail = tpr_at_max_fpr(labels, scores)
        out.append(
            {
                "model": model,
                "num_steps": num_steps,
                "seed": seed,
                "metric_name": metric_name,
                "roc_auc": roc_auc,
                "tpr_at_1pct_fpr": tail["tpr_at_1pct_fpr"],
                "actual_fpr": tail["fpr"],
                "threshold": tail["threshold"],
                "num_positive": int(sum(labels)),
                "num_negative": int(len(labels) - sum(labels)),
            }
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate DSM detection metrics.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--metrics", nargs="+", default=["dsm_0_1", "dsm_1_2"])
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    with Path(args.input).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = aggregate_rows(rows, args.metrics)
    write_csv(
        args.output,
        out,
        [
            "model",
            "num_steps",
            "seed",
            "metric_name",
            "roc_auc",
            "tpr_at_1pct_fpr",
            "actual_fpr",
            "threshold",
            "num_positive",
            "num_negative",
        ],
    )


if __name__ == "__main__":
    main()
