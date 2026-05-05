"""Appendix A.1 DDIM sampling-step scan."""

from __future__ import annotations

import argparse
from pathlib import Path

from .aggregate_metrics import aggregate_rows
from .diffusion import compute_prompt_dsm, dtype_from_name, load_pipeline, mock_prompt_result
from .eval_dsm import OUTPUT_COLUMNS
from .io_utils import read_prompt_csv, write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DSM while scanning DDIM step counts.")
    parser.add_argument("--model", default="checkpoints/sd-v1-4")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[5, 10, 20, 30, 50])
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Opt in to downloading model files from the Hugging Face Hub. "
        "By default, model loading is local-files-only for reproducibility.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prompt_rows = read_prompt_csv(args.prompt_file, args.max_prompts)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_all = []
    pipe = None
    if args.model != "mock":
        pipe = load_pipeline(args.model, dtype_from_name(args.dtype), args.device, allow_download=args.allow_download)

    for num_steps in args.steps:
        rows = []
        if args.model == "mock":
            for idx, prompt_row in enumerate(prompt_rows):
                row = mock_prompt_result(prompt_row, idx, args.seed)
                row["num_steps"] = num_steps
                rows.append(row)
        else:
            assert pipe is not None
            for idx, prompt_row in enumerate(prompt_rows):
                rows.append(
                    compute_prompt_dsm(
                        pipe,
                        prompt_row=prompt_row,
                        model_name=args.model,
                        seed=args.seed,
                        num_steps=num_steps,
                        guidance_scale=args.guidance_scale,
                        height=args.height,
                        width=args.width,
                        dtype_name=args.dtype,
                        device=args.device,
                    )
                )
        step_scores_path = out_dir / f"dsm_steps_{num_steps}.csv"
        write_csv(step_scores_path, rows, OUTPUT_COLUMNS)
        aggregate_all.extend(aggregate_rows(rows, ["dsm_0_1", "dsm_1_2"]))

    write_csv(
        out_dir / "aggregate_metrics.csv",
        aggregate_all,
        [
            "model",
            "num_steps",
            "seed",
            "metric_name",
            "roc_auc",
            "tpr_at_1pct_fpr",
            "actual_fpr",
            "threshold",
            "num_positive",
            "num_negative",
        ],
    )


if __name__ == "__main__":
    main()
