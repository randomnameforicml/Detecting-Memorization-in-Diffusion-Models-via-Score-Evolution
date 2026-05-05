"""CLI for per-prompt DSM evaluation."""

from __future__ import annotations

import argparse

from .diffusion import compute_prompt_dsm, dtype_from_name, load_pipeline, mock_prompt_result
from .io_utils import read_prompt_csv, write_csv


OUTPUT_COLUMNS = [
    "prompt_id",
    "prompt",
    "label",
    "source",
    "split",
    "model_group",
    "model",
    "seed",
    "num_steps",
    "timesteps",
    "prediction_type",
    "guidance_scale",
    "dtype",
    "tokenizer_truncated",
    "dsm_0_1",
    "dsm_1_2",
    "dsm_0_1_dt_normalized",
    "dsm_1_2_dt_normalized",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute DSM(0->1) and DSM(1->2).")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--run-full-generation", action="store_true")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Opt in to downloading model files from the Hugging Face Hub. "
        "By default, model loading is local-files-only for reproducibility.",
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prompt_rows = read_prompt_csv(args.prompt_file, args.max_prompts)
    rows = []
    if args.model == "mock":
        for idx, prompt_row in enumerate(prompt_rows):
            rows.append(mock_prompt_result(prompt_row, idx, args.seed))
    else:
        pipe = load_pipeline(args.model, dtype_from_name(args.dtype), args.device, allow_download=args.allow_download)
        num_prompts = len(prompt_rows)
        for idx, prompt_row in enumerate(prompt_rows):
            if idx % 10 == 0:
                print(f"[{args.model}] Processing prompt {idx}/{num_prompts}...")
            rows.append(
                compute_prompt_dsm(
                    pipe,
                    prompt_row=prompt_row,
                    model_name=args.model,
                    seed=args.seed,
                    num_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    dtype_name=args.dtype,
                    device=args.device,
                    run_full_generation=args.run_full_generation,
                )
            )
    write_csv(args.output, rows, OUTPUT_COLUMNS)


if __name__ == "__main__":
    main()
