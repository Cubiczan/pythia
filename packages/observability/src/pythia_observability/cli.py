"""`pythia-replay` CLI — replay server + audit-log inspection.

Subcommands
-----------
- `serve`        — start the FastAPI replay UI on http://127.0.0.1:8088
- `stats`        — print aggregate stats as JSON
- `achievements` — print achievement unlock status as JSON
- `export`       — dump the full audit log as a single JSON array

Examples
--------
    pythia-replay serve --log ./logs/audit.jsonl --port 8088
    pythia-replay stats --log ./logs/audit.jsonl
    pythia-replay achievements --log ./logs/audit.jsonl \\
        --config configs/achievements.toml
    pythia-replay export --log ./logs/audit.jsonl --out ./logs/export.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .achievements import AchievementsEvaluator
from .audit_reader import AuditLogReader
from .server import ReplayServer

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pythia-replay",
        description=(
            "Pythia observability CLI — replay the audit log via a FastAPI UI, "
            "inspect stats, evaluate achievements, or export the log."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- serve ------------------------------------------------------ #
    p_serve = sub.add_parser(
        "serve",
        help="Start the FastAPI replay UI (dashboard + JSON API).",
    )
    p_serve.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to the audit JSONL file (one entry per line).",
    )
    p_serve.add_argument(
        "--achievements-config",
        type=Path,
        default=None,
        help="Path to achievements.toml (optional — omit to hide the achievements grid).",
    )
    p_serve.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1).",
    )
    p_serve.add_argument(
        "--port",
        type=int,
        default=8088,
        help="Bind port (default: 8088).",
    )

    # --- stats ------------------------------------------------------ #
    p_stats = sub.add_parser(
        "stats",
        help="Print aggregate stats (P&L, win rate, bankroll, drawdown, Brier).",
    )
    p_stats.add_argument(
        "--log", type=Path, required=True, help="Path to the audit JSONL file."
    )

    # --- achievements ----------------------------------------------- #
    p_ach = sub.add_parser(
        "achievements",
        help="Evaluate achievements against the audit log; print unlock status.",
    )
    p_ach.add_argument(
        "--log", type=Path, required=True, help="Path to the audit JSONL file."
    )
    p_ach.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to achievements.toml (same format as the upstream icohangar-ops/achievements repo).",
    )

    # --- export ----------------------------------------------------- #
    p_exp = sub.add_parser(
        "export",
        help="Export the full audit log as a single JSON array (one pretty-printed file).",
    )
    p_exp.add_argument(
        "--log", type=Path, required=True, help="Path to the audit JSONL file."
    )
    p_exp.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path (will be overwritten if it exists).",
    )

    return parser

def _cmd_serve(args: argparse.Namespace) -> int:
    server = ReplayServer(
        log_path=args.log,
        achievements_config_path=args.achievements_config,
    )
    print(
        f"Pythia Replay → http://{args.host}:{args.port}/  "
        f"(log: {args.log}, achievements: {args.achievements_config or 'none'})",
        file=sys.stderr,
    )
    server.run(host=args.host, port=args.port)
    return 0

def _cmd_stats(args: argparse.Namespace) -> int:
    reader = AuditLogReader(args.log)
    stats = reader.compute_stats()
    json.dump(stats, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0

def _cmd_achievements(args: argparse.Namespace) -> int:
    reader = AuditLogReader(args.log)
    evaluator = AchievementsEvaluator(args.config)
    unlocked = evaluator.evaluate(reader)
    payload = [a.model_dump(mode="json") for a in unlocked]
    # Summary line on stderr, full JSON on stdout.
    n_unlocked = sum(1 for a in unlocked if a.unlocked_at is not None)
    print(
        f"achievements: {n_unlocked}/{len(unlocked)} unlocked",
        file=sys.stderr,
    )
    json.dump(payload, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0

def _cmd_export(args: argparse.Namespace) -> int:
    reader = AuditLogReader(args.log)
    entries = reader.read_all()
    payload = [e.model_dump(mode="json") for e in entries]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        f"exported {len(payload)} entries → {args.out}",
        file=sys.stderr,
    )
    return 0

def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the `pythia-replay` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        return _cmd_serve(args)
    if args.command == "stats":
        return _cmd_stats(args)
    if args.command == "achievements":
        return _cmd_achievements(args)
    if args.command == "export":
        return _cmd_export(args)

    parser.error(f"unknown command: {args.command!r}")
    return 2  # unreachable

if __name__ == "__main__":
    sys.exit(main())
