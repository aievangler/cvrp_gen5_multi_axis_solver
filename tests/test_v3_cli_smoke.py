from __future__ import annotations

import io

from gen3.v3.main import main


def test_cli_smoke_emits_solution(monkeypatch, capsys):
    argv = ["tests/tmp_tiny.vrp", "1", "1.0"]
    monkeypatch.setenv("PYTHONPATH", ".")
    # capture stdout via capsys; main prints to stdout directly
    rc = main(argv)
    assert rc == 0
    captured = capsys.readouterr()
    out = captured.out.strip()
    assert "Route #1:" in out
    assert "Cost" in out
