#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${DSM_SD15_MODEL:-checkpoints/sd-v1-5}"

python -m dsm.eval_dsm \
  --model "$MODEL_DIR" \
  --prompt-file prompts/sd14_sd15_eval_prompts.csv \
  --num-steps 50 \
  --guidance-scale 7.5 \
  --seed 42 \
  --output results/sd15_seed42_dsm.csv
