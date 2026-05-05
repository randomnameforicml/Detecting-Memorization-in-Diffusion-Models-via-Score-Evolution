"""CSV helpers with stable public-facing columns."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List


PROMPT_COLUMNS = ["prompt_id", "prompt", "label", "source", "split", "model_group"]


def read_prompt_csv(path: str | Path, max_prompts: int | None = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in PROMPT_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Prompt CSV is missing columns: {missing}")
        for row in reader:
            rows.append(row)
            if max_prompts is not None and len(rows) >= max_prompts:
                break
    return rows


def write_csv(path: str | Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
