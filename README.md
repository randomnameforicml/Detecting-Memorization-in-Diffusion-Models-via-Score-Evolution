# DSM Memorization Detection

This repository contains a clean public implementation of the DSM components
for diffusion memorization detection. It intentionally implements only the DSM
experiments and does not include baseline-method implementations.

The repo supports three reproduction targets:

1. DSM(0->1) and DSM(1->2) memorization scores.
2. First-step DDIM `x0_hat` prediction visualization.
3. Appendix A.1 DDIM sampling-step scan.

## Setup

```bash
pip install -r requirements.txt
```

Model loading is local-files-only by default. The experiment commands below do
not silently download Stable Diffusion weights. Prepare checkpoint directories
before running full experiments, for example:

```bash
mkdir -p checkpoints
hf auth login
hf download CompVis/stable-diffusion-v1-4 --local-dir checkpoints/sd-v1-4
hf download runwayml/stable-diffusion-v1-5 --local-dir checkpoints/sd-v1-5
hf download stabilityai/stable-diffusion-2-base --local-dir checkpoints/sd-2-base
```

You can also point `--model` to any existing local diffusers-format checkpoint
directory. To intentionally let diffusers fetch a model from the Hub, pass both
a Hub model id and `--allow-download`; this opt-in is disabled by default.

For local development:

```bash
pip install -e ".[test]"
pytest
```

The test suite uses a deterministic mock pipeline and does not require
downloading Stable Diffusion checkpoints.

## Prompt Files

The `prompts/` directory includes the prompt CSV files used by the experiments:

- `prompts/sd14_sd15_eval_prompts.csv`: 500 memorized prompts and 500 generalized prompts.
- `prompts/sd20base_eval_prompts.csv`: 219 SD v2 memorized prompts and 500 generalized prompts.
- `prompts/x0_visualization_examples.csv`: a small visualization subset.

Each prompt CSV has stable columns:

```text
prompt_id,prompt,label,source,split,model_group
```

`label=1` denotes memorized prompts and `label=0` denotes generalized prompts.

## DSM Definition

At the first DDIM reverse timesteps, the code computes conditional and
unconditional scores:

```text
s_c,t_k(x_t_k), s_empty,t_k(x_t_k)
```

and the score gap:

```text
Delta s_k = s_c,t_k(x_t_k) - s_empty,t_k(x_t_k).
```

The default reported metrics are the raw squared finite differences:

```text
DSM(0->1) = ||Delta s_1 - Delta s_0||_2^2
DSM(1->2) = ||Delta s_2 - Delta s_1||_2^2
```

The CSV also stores optional `dt`-normalized diagnostics, but the default
outputs and aggregate scripts use raw DSM.

## Score Conversion

The code does not assume all Stable Diffusion checkpoints are epsilon
prediction models. It reads `scheduler.config.prediction_type` and converts the
UNet output to epsilon before computing the VP/DDPM score:

```text
s_t(x_t) ~= -epsilon_theta(x_t,t) / sqrt(1 - alpha_bar_t).
```

For `prediction_type == "epsilon"`, the UNet output is epsilon.

For `prediction_type == "v_prediction"`, the code uses the standard diffusers
conversion:

```text
epsilon = sqrt(alpha_bar_t) * v + sqrt(1 - alpha_bar_t) * x_t.
```

This is the important SD2-family guardrail.

## Run DSM Evaluation

SD1.4:

```bash
python -m dsm.eval_dsm \
  --model checkpoints/sd-v1-4 \
  --prompt-file prompts/sd14_sd15_eval_prompts.csv \
  --num-steps 50 \
  --guidance-scale 7.5 \
  --seed 42 \
  --output results/sd14_seed42_dsm.csv
```

SD1.5:

```bash
python -m dsm.eval_dsm \
  --model checkpoints/sd-v1-5 \
  --prompt-file prompts/sd14_sd15_eval_prompts.csv \
  --num-steps 50 \
  --guidance-scale 7.5 \
  --seed 42 \
  --output results/sd15_seed42_dsm.csv
```

SD2.0-base:

```bash
python -m dsm.eval_dsm \
  --model checkpoints/sd-2-base \
  --prompt-file prompts/sd20base_eval_prompts.csv \
  --num-steps 50 \
  --guidance-scale 7.5 \
  --seed 42 \
  --output results/sd20b_seed42_dsm.csv
```

For a quick smoke test without model downloads:

```bash
python -m dsm.eval_dsm \
  --model mock \
  --prompt-file prompts/x0_visualization_examples.csv \
  --max-prompts 4 \
  --output results/mock_dsm.csv
```

## Aggregate Metrics

```bash
python -m dsm.aggregate_metrics \
  --input results/sd14_seed42_dsm.csv \
  --metrics dsm_0_1 dsm_1_2 \
  --output results/sd14_seed42_metrics.csv
```

The aggregation reports ROC-AUC and TPR@1%FPR. TPR@1%FPR is computed by
scanning score thresholds and selecting the highest TPR among thresholds with
negative-prompt FPR <= 1%.

## DDIM Step Scan

```bash
python -m dsm.step_scan \
  --model checkpoints/sd-v1-4 \
  --prompt-file prompts/sd14_sd15_eval_prompts.csv \
  --steps 5 10 20 30 50 \
  --guidance-scale 7.5 \
  --seed 42 \
  --output-dir results/step_scan_sd14
```

The same initial latent is used for every prompt for a given `--seed`; for
example, `--seed 42` means all prompts start from the exact same `x_T`.

## First-Step DDIM `x0_hat` Visualization

```bash
python -m dsm.visualize_x0 \
  --model checkpoints/sd-v1-4 \
  --prompt-file prompts/x0_visualization_examples.csv \
  --num-steps 50 \
  --guidance-scale 7.5 \
  --seed 42 \
  --output-dir figures/x0_one_step
```

The script computes the conditional first-step DDIM clean estimate:

```text
x0_hat = (x_t - sqrt(1-alpha_bar_t) * epsilon_theta(x_t,c)) / sqrt(alpha_bar_t)
```

and decodes it with the VAE. It saves individual images and `x0_hat_grid.png`.

## Output Columns

`dsm.eval_dsm` writes one row per prompt with:

```text
prompt_id,prompt,label,source,split,model_group,model,seed,
num_steps,timesteps,prediction_type,guidance_scale,dtype,
tokenizer_truncated,dsm_0_1,dsm_1_2,dsm_0_1_dt_normalized,
dsm_1_2_dt_normalized
```

## Runtime Notes

Default inference uses `float16`, batch size 1, DDIMScheduler, and guidance
scale 7.5. DSM only runs the first three reverse steps by default; it does not
decode final images, does not train, and does not use gradients, Hessians, JVPs,
or backpropagation.

Full experiment scripts are reproducibility-safe: they call diffusers with
`local_files_only=True` unless the user explicitly passes `--allow-download`.

Typical GPU memory depends on checkpoint and image size. A 16 GB GPU is usually
comfortable for 512x512 batch-size-1 evaluation; use `--dtype float32` only when
debugging numerical behavior and expect higher memory use.

## Artifact Checklist

- Prompt CSVs are included.
- SD1.4, SD1.5, and SD2.0-base are supported.
- DSM(0->1) and DSM(1->2) are both output.
- First-step DDIM `x0_hat` images can be generated.
- DDIM step scan can be run with one command.
- Aggregate metrics report ROC-AUC and TPR@1%FPR.
- The repo does not implement baseline methods.
- The repo does not require training.
- The repo does not require Hessians, JVPs, or backpropagation.
