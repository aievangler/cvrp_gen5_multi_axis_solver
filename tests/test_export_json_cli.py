from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from scripts.run_gen3_v3 import main as runner_main
from scripts.export_json import main as export_main


def test_export_json_cli(tmp_path: Path, capsys):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    shutil.copy("tests/tmp_tiny.vrp", dataset_dir / "tmp_tiny.vrp")
    db_path = tmp_path / "runs.db"
    # Run solver to populate DB
    runner_argv = [
        "--dataset",
        str(dataset_dir),
        "--distance-type",
        "1",
        "--time-limit",
        "0.5",
        "--db",
        str(db_path),
    ]
    assert runner_main(runner_argv) == 0
    # Export to JSON
    out_json = tmp_path / "out.json"
    assert export_main(["--db", str(db_path), "--run-id", "1", "--out", str(out_json)]) == 0
    data = json.loads(out_json.read_text())
    assert data["run_id"] == 1
    assert data["dataset"] == str(dataset_dir)
    assert len(data["instances"]) == 1
    assert data["instances"][0]["instance"] == "tmp_tiny.vrp"
