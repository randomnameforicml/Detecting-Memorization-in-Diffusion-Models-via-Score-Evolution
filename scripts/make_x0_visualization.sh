#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${DSM_SD14_MODEL:-checkpoints/sd-v1-4}"

python -m dsm.visualize_x0 \
  --model "$MODEL_DIR" \
  --prompt-file prompts/x0_visualization_examples.csv \
  --num-steps 50 \
  --guidance-scale 7.5 \
  --seed 42 \
  --output-dir figures/x0_one_step
