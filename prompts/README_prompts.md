# Prompt Sets

This repository includes the prompt text needed to reproduce the DSM memorization
detection experiments. It does not include training images.

## Files

- `sd14_sd15_eval_prompts.csv`
  - 500 Webster memorized prompts for Stable Diffusion v1.x evaluation.
  - 500 generalized prompts used as non-memorized controls.
  - Columns: `prompt_id,prompt,label,source,split,model_group`.

- `sd20base_eval_prompts.csv`
  - 219 Webster SD v2 memorized prompts for Stable Diffusion 2.0-base evaluation.
  - 500 generalized prompts used as non-memorized controls.
  - Columns: `prompt_id,prompt,label,source,split,model_group`.

- `x0_visualization_examples.csv`
  - A small subset for first-step DDIM `x0_hat` visualization.
  - It uses the same columns and adds `display_group`.

## Source Notes

The memorized prompts follow the Webster memorized prompt lists used in recent
diffusion memorization detection work. The generalized prompts are the prompt
controls used with the same evaluation setup. The prompt CSVs are provided so
the DSM scores can be reproduced without access to any training image.

Please cite the relevant memorization prompt source in papers using these files:

- Webster, Ryan, et al. "A Reproducible Extraction of Training Images from
  Diffusion Models." 2023.

If you replace these prompts with another public prompt source, keep the same
CSV schema and record source/license information here.
