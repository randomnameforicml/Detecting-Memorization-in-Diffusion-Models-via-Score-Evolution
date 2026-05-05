"""Prediction-to-score conversion for Stable Diffusion schedulers."""

from __future__ import annotations

from typing import Any, Tuple

import torch


def timestep_to_index(timestep: torch.Tensor | int) -> int:
    if isinstance(timestep, torch.Tensor):
        return int(timestep.detach().cpu().item())
    return int(timestep)


def alpha_sigma(
    scheduler: Any,
    timestep: torch.Tensor | int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return sqrt(alpha_bar_t) and sqrt(1-alpha_bar_t)."""
    idx = timestep_to_index(timestep)
    alpha_bar = scheduler.alphas_cumprod.to(device=device, dtype=dtype)[idx]
    alpha = alpha_bar.sqrt()
    sigma = (1.0 - alpha_bar).sqrt()
    return alpha, sigma


def scheduler_prediction_type(scheduler: Any) -> str:
    config = getattr(scheduler, "config", None)
    return getattr(config, "prediction_type", "epsilon")


def model_output_to_epsilon(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    """Convert a UNet output into epsilon.

    Diffusers/DDIM ``v_prediction`` uses v = alpha * eps - sigma * x0.
    Combined with x_t = alpha * x0 + sigma * eps, this gives:

        eps = alpha * v + sigma * x_t.

    The conversion is kept isolated so scheduler-specific output handling stays
    in one place.
    """
    if prediction_type in {"epsilon", "sample_epsilon"}:
        return model_output
    if prediction_type == "v_prediction":
        return alpha * model_output + sigma * sample
    raise ValueError(f"Unsupported prediction_type: {prediction_type}")


def epsilon_to_score(epsilon: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return -epsilon / sigma


def model_output_to_score(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    epsilon = model_output_to_epsilon(model_output, sample, alpha, sigma, prediction_type)
    return epsilon_to_score(epsilon, sigma)
