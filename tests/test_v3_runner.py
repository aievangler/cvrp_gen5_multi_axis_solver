from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from scripts.run_gen3_v3 import main as runner_main
from scripts.export_json import export_run


def test_runner_inserts_rows(tmp_path: Path, capsys):
    # Prepare a tiny dataset directory with a single small instance
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    shutil.copy("tests/tmp_tiny.vrp", dataset_dir / "tmp_tiny.vrp")

    db_path = tmp_path / "runs.db"
    argv = [
        "--dataset",
        str(dataset_dir),
        "--distance-type",
        "1",
        "--time-limit",
        "0.5",
        "--db",
        str(db_path),
    ]
    rc = runner_main(argv)
    assert rc == 0

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM runs")
    runs_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM solver_instances")
    sol_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM instances")
    inst_count = cur.fetchone()[0]
    conn.close()
    assert runs_count == 1
    assert sol_count == 1
    assert inst_count == 1
    captured = capsys.readouterr()
    # stderr log should mention run id
    assert "run" in captured.err

    # Export JSON for the run and verify structure
    conn = sqlite3.connect(db_path)
    data = export_run(conn, 1)
    conn.close()
    assert data["run_id"] == 1
    assert data["dataset"] == str(dataset_dir)
    assert len(data["instances"]) == 1
    assert data["instances"][0]["instance"] == "tmp_tiny.vrp"
