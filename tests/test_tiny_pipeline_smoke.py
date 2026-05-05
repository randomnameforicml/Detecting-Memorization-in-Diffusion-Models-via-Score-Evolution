import csv
import os
import subprocess
import sys
from pathlib import Path


def test_eval_script_mock_smoke(tmp_path: Path):
    prompt_file = tmp_path / "prompts.csv"
    with prompt_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompt_id", "prompt", "label", "source", "split", "model_group"],
        )
        writer.writeheader()
        writer.writerow({"prompt_id": "p0", "prompt": "memorized toy", "label": 1, "source": "test", "split": "memorized", "model_group": "mock"})
        writer.writerow({"prompt_id": "p1", "prompt": "generalized toy", "label": 0, "source": "test", "split": "generalized", "model_group": "mock"})
        writer.writerow({"prompt_id": "p2", "prompt": "another toy", "label": 0, "source": "test", "split": "generalized", "model_group": "mock"})

    output = tmp_path / "out.csv"
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dsm.eval_dsm",
            "--model",
            "mock",
            "--prompt-file",
            str(prompt_file),
            "--output",
            str(output),
            "--max-prompts",
            "3",
        ],
        check=True,
        env=env,
    )

    with output.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert "dsm_0_1" in rows[0]
    assert "dsm_1_2" in rows[0]
    assert "seed" + "_policy" not in rows[0]
    assert "base" + "_seed" not in rows[0]
    assert {row["seed"] for row in rows} == {"42"}
