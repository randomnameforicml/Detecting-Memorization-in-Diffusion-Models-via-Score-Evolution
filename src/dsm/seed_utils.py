"""Seed helpers for deterministic DSM evaluation."""

from __future__ import annotations

import torch


def make_generator(seed: int, device: torch.device | str) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator
