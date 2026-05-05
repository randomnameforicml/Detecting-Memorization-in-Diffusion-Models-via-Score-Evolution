"""DSM score definitions."""

from __future__ import annotations

from typing import Iterable, Mapping

import torch


def squared_l2(x: torch.Tensor) -> torch.Tensor:
    return x.float().flatten().pow(2).sum()


def compute_dsm(score_gaps: Iterable[torch.Tensor]) -> Mapping[str, float]:
    gaps = list(score_gaps)
    if len(gaps) < 3:
        raise ValueError("DSM(1->2) requires at least three score gaps.")
    dsm_0_1 = squared_l2(gaps[1] - gaps[0]).item()
    dsm_1_2 = squared_l2(gaps[2] - gaps[1]).item()
    return {"dsm_0_1": dsm_0_1, "dsm_1_2": dsm_1_2}


def compute_dt_normalized_dsm(
    score_gaps: Iterable[torch.Tensor],
    timesteps: Iterable[int],
) -> Mapping[str, float]:
    gaps = list(score_gaps)
    ts = list(timesteps)
    if len(gaps) < 3 or len(ts) < 3:
        raise ValueError("dt-normalized DSM requires three gaps and timesteps.")
    dt_0_1 = max(abs(ts[1] - ts[0]), 1)
    dt_1_2 = max(abs(ts[2] - ts[1]), 1)
    return {
        "dsm_0_1_dt_normalized": squared_l2((gaps[1] - gaps[0]) / dt_0_1).item(),
        "dsm_1_2_dt_normalized": squared_l2((gaps[2] - gaps[1]) / dt_1_2).item(),
    }
