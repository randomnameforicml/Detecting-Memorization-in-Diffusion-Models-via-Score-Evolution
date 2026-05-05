#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${DSM_SD14_MODEL:-checkpoints/sd-v1-4}"

python -m dsm.eval_dsm \
  --model "$MODEL_DIR" \
  --prompt-file prompts/sd14_sd15_eval_prompts.csv \
  --num-steps 50 \
  --guidance-scale 7.5 \
  --seed 42 \
  --output results/sd14_seed42_dsm.csv
