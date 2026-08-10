"""CLI entrypoint for pythia-consensus.

Usage
-----
    pythia-consensus fuse <estimates.json> [--method logit-mean] [--threshold 0.65]
    pythia-consensus explain <decision.json>

`estimates.json` is a JSON list of `Estimate` dicts. `decision.json` is a
serialised `ConsensusDecision` (as produced by `fuse`'s `--out` flag, or by
`decision.model_dump_json()`).

The CLI exists for ad-hoc testing and for the demo replay UI — it lets a judge
fuse a saved set of estimates and see the explanation without writing Python.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .engine import ConsensusEngine
from .fusion import fuse
from .types import ConsensusConfig, ConsensusDecision, Estimate


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
        raise SystemExit(2)
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON in {path}: {exc}", file=sys.stderr)
        raise SystemExit(2)


def _cmd_fuse(args: argparse.Namespace) -> int:
    raw = _load_json(Path(args.estimates))
    if not isinstance(raw, list) or not raw:
        print("error: estimates file must be a non-empty JSON list.", file=sys.stderr)
        return 2

    try:
        estimates = [Estimate.model_validate(item) for item in raw]
    except Exception as exc:  # pydantic validation error
        print(f"error: could not parse Estimate records: {exc}", file=sys.stderr)
        return 2

    config = ConsensusConfig(
        method=args.method,
        agreement_threshold=args.threshold,
        min_analysts=args.min_analysts,
    )

    if args.weights:
        weights_raw = _load_json(Path(args.weights))
        if not isinstance(weights_raw, dict):
            print("error: --weights file must be a JSON object {analyst_id: weight}.", file=sys.stderr)
            return 2
        config = config.model_copy(update={"weights": weights_raw})

    decision = fuse(estimates, config)

    if args.engine:
        engine = ConsensusEngine(config)
        decision = engine.decide(estimates)
        print(engine.explain(decision), file=sys.stderr)

    out = decision.model_dump_json(indent=2)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"wrote decision to {args.out}", file=sys.stderr)
    print(out)
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    raw = _load_json(Path(args.decision))
    try:
        decision = ConsensusDecision.model_validate(raw)
    except Exception as exc:
        print(f"error: could not parse ConsensusDecision: {exc}", file=sys.stderr)
        return 2
    engine = ConsensusEngine(ConsensusConfig(method=decision.method))
    print(engine.explain(decision))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pythia-consensus",
        description="Fuse analyst estimates into a signed consensus decision.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fuse = sub.add_parser("fuse", help="Fuse a list of Estimate records into a ConsensusDecision.")
    p_fuse.add_argument("estimates", help="Path to a JSON file containing a list of Estimate dicts.")
    p_fuse.add_argument(
        "--method",
        choices=["logit-mean", "median", "trimmed-mean"],
        default="logit-mean",
        help="Fusion method (default: logit-mean).",
    )
    p_fuse.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Agreement threshold below which gate flips to 'skip' (default: 0.65).",
    )
    p_fuse.add_argument(
        "--min-analysts",
        type=int,
        default=2,
        help="Minimum number of analysts for gate='trade' (default: 2).",
    )
    p_fuse.add_argument(
        "--weights",
        default=None,
        help="Optional JSON file {analyst_id: weight} overriding equal weights.",
    )
    p_fuse.add_argument(
        "--engine",
        action="store_true",
        help="Also route through ConsensusEngine.explain() and print to stderr.",
    )
    p_fuse.add_argument(
        "--out",
        default=None,
        help="Optional path to write the decision JSON to.",
    )
    p_fuse.set_defaults(func=_cmd_fuse)

    p_explain = sub.add_parser(
        "explain",
        help="Print a human-readable explanation of a saved ConsensusDecision.",
    )
    p_explain.add_argument("decision", help="Path to a JSON file with a serialised ConsensusDecision.")
    p_explain.set_defaults(func=_cmd_explain)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
