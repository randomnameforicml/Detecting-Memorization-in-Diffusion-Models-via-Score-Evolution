"""Small diffusers wrapper used by DSM scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

from .dsm_metric import compute_dsm, compute_dt_normalized_dsm
from .score_utils import alpha_sigma, model_output_to_epsilon, model_output_to_score, scheduler_prediction_type
from .seed_utils import make_generator


@dataclass
class PromptResult:
    row: Dict[str, object]


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def load_pipeline(model_name: str, dtype: torch.dtype, device: str, *, allow_download: bool = False):
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    print(f"Loading pipeline: {model_name}...")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=not allow_download,
        )
    except OSError as exc:
        if allow_download:
            raise
        raise OSError(
            "Could not load the Stable Diffusion checkpoint from local files only. "
            "Pass a local checkpoint directory to --model, or explicitly add "
            "--allow-download if you intentionally want diffusers to fetch files "
            "from the Hugging Face Hub."
        ) from exc
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    print(f"Moving pipeline to {device}...")
    pipe.to(device)
    print(f"Pipeline successfully moved to {device}.")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def encode_prompt(pipe: Any, prompt: str, device: str) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    tokenizer = pipe.tokenizer
    max_length = tokenizer.model_max_length
    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    untruncated = tokenizer([prompt], padding="longest", return_tensors="pt")
    was_truncated = untruncated.input_ids.shape[-1] > text_inputs.input_ids.shape[-1]
    input_ids = text_inputs.input_ids.to(device)
    if getattr(pipe.text_encoder.config, "use_attention_mask", False):
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    prompt_embeds = pipe.text_encoder(input_ids, attention_mask=attention_mask)[0]

    uncond_inputs = tokenizer(
        [""],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_ids = uncond_inputs.input_ids.to(device)
    if getattr(pipe.text_encoder.config, "use_attention_mask", False):
        uncond_mask = uncond_inputs.attention_mask.to(device)
    else:
        uncond_mask = None
    uncond_embeds = pipe.text_encoder(uncond_ids, attention_mask=uncond_mask)[0]
    return uncond_embeds, prompt_embeds, bool(was_truncated)


def latent_shape(pipe: Any, height: int, width: int) -> Tuple[int, int, int, int]:
    vae_scale_factor = getattr(pipe, "vae_scale_factor", 8)
    channels = pipe.unet.config.in_channels
    return (1, channels, height // vae_scale_factor, width // vae_scale_factor)


def initial_latents(
    pipe: Any,
    *,
    seed: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    generator = make_generator(seed, device)
    latents = torch.randn(latent_shape(pipe, height, width), generator=generator, device=device, dtype=dtype)
    return latents * pipe.scheduler.init_noise_sigma


def unet_uncond_cond(
    pipe: Any,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    uncond_embeds: torch.Tensor,
    cond_embeds: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    latent_model_input = torch.cat([latents, latents], dim=0)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
    embeds = torch.cat([uncond_embeds, cond_embeds], dim=0)
    model_output = pipe.unet(latent_model_input, timestep, encoder_hidden_states=embeds, return_dict=False)[0]
    return model_output.chunk(2)


@torch.no_grad()
def compute_prompt_dsm(
    pipe: Any,
    *,
    prompt_row: Dict[str, str],
    model_name: str,
    seed: int,
    num_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    dtype_name: str,
    device: str,
    run_full_generation: bool = False,
) -> Dict[str, object]:
    dtype = dtype_from_name(dtype_name)
    pipe.scheduler.set_timesteps(num_steps, device=device)
    timesteps = list(pipe.scheduler.timesteps)
    if len(timesteps) < 3:
        raise ValueError("DSM(1->2) requires num_steps >= 3.")
    prediction_type = scheduler_prediction_type(pipe.scheduler)

    latents = initial_latents(pipe, seed=seed, height=height, width=width, dtype=dtype, device=device)
    uncond_embeds, cond_embeds, was_truncated = encode_prompt(pipe, prompt_row["prompt"], device)
    score_gaps: List[torch.Tensor] = []
    used_timesteps: List[int] = []

    for step_idx, timestep in enumerate(timesteps):
        out_uncond, out_cond = unet_uncond_cond(pipe, latents, timestep, uncond_embeds, cond_embeds)
        alpha, sigma = alpha_sigma(pipe.scheduler, timestep, device=latents.device, dtype=latents.dtype)
        score_uncond = model_output_to_score(out_uncond, latents, alpha, sigma, prediction_type)
        score_cond = model_output_to_score(out_cond, latents, alpha, sigma, prediction_type)
        if step_idx < 3:
            score_gaps.append((score_cond - score_uncond).detach().cpu())
            used_timesteps.append(int(timestep.detach().cpu().item()))

        guided_output = out_uncond + guidance_scale * (out_cond - out_uncond)
        latents = pipe.scheduler.step(guided_output, timestep, latents, return_dict=False)[0]
        if step_idx >= 2 and not run_full_generation:
            break

    raw = compute_dsm(score_gaps)
    normalized = compute_dt_normalized_dsm(score_gaps, used_timesteps)
    return {
        "prompt_id": prompt_row["prompt_id"],
        "prompt": prompt_row["prompt"],
        "label": int(prompt_row["label"]),
        "source": prompt_row.get("source", ""),
        "split": prompt_row.get("split", ""),
        "model_group": prompt_row.get("model_group", ""),
        "model": model_name,
        "seed": seed,
        "num_steps": num_steps,
        "timesteps": " ".join(str(t) for t in used_timesteps),
        "prediction_type": prediction_type,
        "guidance_scale": guidance_scale,
        "dtype": dtype_name,
        "tokenizer_truncated": int(was_truncated),
        "dsm_0_1": raw["dsm_0_1"],
        "dsm_1_2": raw["dsm_1_2"],
        "dsm_0_1_dt_normalized": normalized["dsm_0_1_dt_normalized"],
        "dsm_1_2_dt_normalized": normalized["dsm_1_2_dt_normalized"],
    }


def mock_prompt_result(prompt_row: Dict[str, str], prompt_index: int, seed: int) -> Dict[str, object]:
    label = int(prompt_row["label"])
    base = (seed % 17) * 0.01 + prompt_index * 0.001
    dsm_0_1 = 2.0 + base if label else 0.5 + base
    dsm_1_2 = 2.5 + base if label else 0.6 + base
    return {
        "prompt_id": prompt_row["prompt_id"],
        "prompt": prompt_row["prompt"],
        "label": label,
        "source": prompt_row.get("source", "mock"),
        "split": prompt_row.get("split", "smoke"),
        "model_group": prompt_row.get("model_group", "mock"),
        "model": "mock",
        "seed": seed,
        "num_steps": 3,
        "timesteps": "2 1 0",
        "prediction_type": "epsilon",
        "guidance_scale": 1.0,
        "dtype": "float32",
        "tokenizer_truncated": 0,
        "dsm_0_1": dsm_0_1,
        "dsm_1_2": dsm_1_2,
        "dsm_0_1_dt_normalized": dsm_0_1,
        "dsm_1_2_dt_normalized": dsm_1_2,
    }


def conditional_x0_estimate(
    pipe: Any,
    *,
    prompt: str,
    seed: int,
    num_steps: int,
    height: int,
    width: int,
    dtype_name: str,
    device: str,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    dtype = dtype_from_name(dtype_name)
    pipe.scheduler.set_timesteps(num_steps, device=device)
    timestep = pipe.scheduler.timesteps[0]
    latents = initial_latents(pipe, seed=seed, height=height, width=width, dtype=dtype, device=device)
    _, cond_embeds, was_truncated = encode_prompt(pipe, prompt, device)
    latent_in = pipe.scheduler.scale_model_input(latents, timestep)
    model_output = pipe.unet(latent_in, timestep, encoder_hidden_states=cond_embeds, return_dict=False)[0]
    alpha, sigma = alpha_sigma(pipe.scheduler, timestep, device=latents.device, dtype=latents.dtype)
    prediction_type = scheduler_prediction_type(pipe.scheduler)
    eps = model_output_to_epsilon(model_output, latents, alpha, sigma, prediction_type)
    x0_hat = (latents - sigma * eps) / alpha
    meta = {
        "timestep": int(timestep.detach().cpu().item()),
        "prediction_type": prediction_type,
        "tokenizer_truncated": int(was_truncated),
    }
    return x0_hat, meta


def decode_latents(pipe: Any, latents: torch.Tensor):
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    from PIL import Image

    return [Image.fromarray((img * 255).round().astype("uint8")) for img in image]
