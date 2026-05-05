#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${DSM_SD14_MODEL:-checkpoints/sd-v1-4}"

python -m dsm.step_scan \
  --model "$MODEL_DIR" \
  --prompt-file prompts/sd14_sd15_eval_prompts.csv \
  --steps 5 10 20 30 50 \
  --guidance-scale 7.5 \
  --seed 42 \
  --output-dir results/step_scan_sd14
