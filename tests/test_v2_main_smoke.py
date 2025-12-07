import subprocess
import sys
from pathlib import Path


def test_v2_main_smoke():
    # Use a tiny A instance for a quick smoke; limit time to 1s.
    vrp = Path("data/instances/A/A-n32-k5.vrp")
    assert vrp.exists()
    cmd = [sys.executable, "-m", "gen3.v2.main", str(vrp), "1", "1"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0
    # stdout should contain at least one Route line and a Cost line
    assert "Route #1" in proc.stdout
    assert "Cost" in proc.stdout
    # stderr should remain empty for normal runs
    assert proc.stderr.strip() == ""


def test_v2_main_bad_args():
    cmd = [sys.executable, "-m", "gen3.v2.main"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert proc.returncode != 0
    assert "Usage" in proc.stderr
