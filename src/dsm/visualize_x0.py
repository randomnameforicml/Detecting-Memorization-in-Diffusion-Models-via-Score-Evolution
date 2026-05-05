"""First-step DDIM x0-hat visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from .diffusion import conditional_x0_estimate, decode_latents, dtype_from_name, load_pipeline
from .io_utils import read_prompt_csv


def make_grid(images, labels, output_path: Path, cols: int = 4) -> None:
    if not images:
        raise ValueError("No images to grid.")
    w, h = images[0].size
    label_h = 24
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w, rows * (h + label_h)), "white")
    draw = ImageDraw.Draw(grid)
    for i, img in enumerate(images):
        x = (i % cols) * w
        y = (i // cols) * (h + label_h)
        grid.paste(img.convert("RGB"), (x, y + label_h))
        draw.text((x + 4, y + 4), labels[i][:40], fill=(0, 0, 0))
    grid.save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode first-step DDIM x0-hat images.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max-examples", type=int, default=8)
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
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(args.model, dtype_from_name(args.dtype), args.device, allow_download=args.allow_download)
    rows = read_prompt_csv(args.prompt_file, args.max_examples)
    images = []
    labels = []
    for idx, row in enumerate(rows):
        x0_hat, meta = conditional_x0_estimate(
            pipe,
            prompt=row["prompt"],
            seed=args.seed,
            num_steps=args.num_steps,
            height=args.height,
            width=args.width,
            dtype_name=args.dtype,
            device=args.device,
        )
        image = decode_latents(pipe, x0_hat)[0]
        group = row.get("display_group") or ("memorized" if int(row["label"]) else "generalized")
        image_path = out_dir / f"{idx:03d}_{group}_seed{args.seed}.png"
        image.save(image_path)
        images.append(image)
        labels.append(f"{group}: {row['prompt_id']}")
        meta_path = out_dir / f"{idx:03d}_{group}_seed{args.seed}.txt"
        meta_path.write_text(
            f"prompt_id={row['prompt_id']}\nseed={args.seed}\ntimestep={meta['timestep']}\n"
            f"prediction_type={meta['prediction_type']}\ntruncated={meta['tokenizer_truncated']}\n",
            encoding="utf-8",
        )
    make_grid(images, labels, out_dir / "x0_hat_grid.png")


if __name__ == "__main__":
    main()
