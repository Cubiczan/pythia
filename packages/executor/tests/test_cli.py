"""CLI smoke tests for the ``pythia`` entry point.

These tests confirm the argparse tree compiles, parses, and dispatches to
the right subcommand handler. They do NOT exercise the full pipeline (the
real ATT / LLM calls are mocked at the import boundary in test_pipeline.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pythia_executor import cli

def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """``pythia --help`` exits 0 and mentions the executor subcommand."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "executor" in captured.out

def test_executor_help_shows_delphi(capsys: pytest.CaptureFixture[str]) -> None:
    """``pythia executor --help`` mentions the delphi subcommand."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["executor", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "delphi" in captured.out

def test_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """``pythia`` with no args prints help and returns 0."""
    rc = cli.main([])
    assert rc == 0
    captured = capsys.readouterr()
    assert "executor" in captured.out

def test_paper_trade_help(capsys: pytest.CaptureFixture[str]) -> None:
    """``pythia executor delphi paper-trade --help`` shows the flags."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["executor", "delphi", "paper-trade", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--market" in captured.out
    assert "--analysts" in captured.out
    assert "--consensus-threshold" in captured.out
    assert "--max-stake-usd" in captured.out
    assert "--audit-log" in captured.out

def test_run_help(capsys: pytest.CaptureFixture[str]) -> None:
    """``pythia executor delphi run --help`` shows the flags."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["executor", "delphi", "run", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--config" in captured.out
    assert "--risk-max-drawdown-pct" in captured.out
    assert "--log-level" in captured.out
    assert "--once" in captured.out

def test_replay_help(capsys: pytest.CaptureFixture[str]) -> None:
    """``pythia executor delphi replay --help`` shows the flags."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["executor", "delphi", "replay", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "audit_log_path" in captured.out
    assert "--line" in captured.out

def test_replay_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    """``replay`` on a non-existent path returns 1 with a clear message."""
    rc = cli.main(["executor", "delphi", "replay", "/tmp/does-not-exist.jsonl"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err

def test_replay_prints_decision_chain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``replay`` on a real audit line prints the full decision chain."""
    # Write a synthetic audit line.
    payload = {
        "market_id": "dphi_test_001",
        "timestamp": "2026-02-14T12:00:00Z",
        "skipped_reason": None,
        "signature": "abc123def456",
        "signature_algo": "hmac-sha256",
        "signed_by": "test-key-fp",
        "estimates": [
            {
                "market_id": "dphi_test_001",
                "probability": 0.72,
                "confidence": 0.8,
                "rationale": "polls favor YES",
                "evidence": [],
                "analyst_id": "politics",
                "timestamp": "2026-02-14T11:59:00Z",
            }
        ],
        "decision": {
            "market_id": "dphi_test_001",
            "consensus_prob": 0.72,
            "agreement_score": 0.85,
            "gate": "trade",
            "contributor_ids": ["politics"],
            "method": "logit-mean",
            "weights_used": {"politics": 1.0},
            "timestamp": "2026-02-14T11:59:30Z",
        },
        "plan": {
            "market_id": "dphi_test_001",
            "side": "YES",
            "size_usd": 25.0,
            "limit_price": 0.55,
            "rationale": "quarter-Kelly, edge=+0.17",
            "risk_flags": [],
            "decision": "APPROVE",
            "timestamp": "2026-02-14T11:59:45Z",
        },
        "receipt": {
            "market_id": "dphi_test_001",
            "side": "YES",
            "size_usd": 25.0,
            "fill_price": 0.55,
            "att_order_id": "paper-xyz",
            "status": "PAPER",
            "signed_by": "paper-mode",
            "timestamp": "2026-02-14T12:00:00Z",
        },
    }
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    rc = cli.main(["executor", "delphi", "replay", str(audit_path), "--line", "1"])
    assert rc == 0
    captured = capsys.readouterr()
    out = captured.out
    # All four sections of the decision chain should be printed.
    assert "PipelineResult" in out
    assert "dphi_test_001" in out
    assert "Estimates" in out
    assert "politics" in out
    assert "ConsensusDecision" in out
    assert "gate:" in out
    assert "trade" in out
    assert "TradePlan" in out
    assert "APPROVE" in out
    assert "TradeReceipt" in out
    assert "PAPER" in out

def test_replay_line_out_of_range(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``replay --line 99`` on a 1-line file returns 1."""
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text('{"market_id": "x"}\n', encoding="utf-8")
    rc = cli.main(["executor", "delphi", "replay", str(audit_path), "--line", "99"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "out of range" in captured.err

def test_replay_empty_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``replay`` on an empty file returns 1."""
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text("", encoding="utf-8")
    rc = cli.main(["executor", "delphi", "replay", str(audit_path)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "empty" in captured.err
