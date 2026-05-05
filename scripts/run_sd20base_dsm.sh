#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${DSM_SD20BASE_MODEL:-checkpoints/sd-2-base}"

python -m dsm.eval_dsm \
  --model "$MODEL_DIR" \
  --prompt-file prompts/sd20base_eval_prompts.csv \
  --num-steps 50 \
  --guidance-scale 7.5 \
  --seed 42 \
  --output results/sd20b_seed42_dsm.csv
