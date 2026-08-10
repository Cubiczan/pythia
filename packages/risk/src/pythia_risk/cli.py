"""Command-line interface for pythia-risk.

Two subcommands:

    pythia-risk size --consensus 0.7 --price 0.5 --bankroll 1000
        Print the fractional-Kelly dollar stake (quarter-Kelly by default).

    pythia-risk evaluate <decision.json> <market.json> --bankroll 1000 --exposure 200
        Build a RiskEngine with a default config, load the decision + market
        from JSON, run the full 6-gate pipeline, and print the TradePlan as
        JSON.

JSON shapes
-----------

``decision.json`` (mirrors ConsensusDecision):
    {
        "market_id": "delphi-1234",
        "consensus_prob": 0.72,
        "agreement_score": 0.81,
        "gate": "trade",
        "contributor_ids": ["politics", "crypto", "niche"],
        "method": "logit-mean",
        "timestamp": "2026-08-10T12:00:00Z"
    }

``market.json`` (mirrors Market):
    {
        "market_id": "delphi-1234",
        "yes_price": 0.55,
        "category": "politics",
        "question": "Will the Fed cut rates in September?"
    }

The evaluate subcommand builds a default ``RiskConfig`` with quarter-Kelly,
$50 per-market cap, $500 total exposure, 5% drawdown breaker, 30-min cool-down.
Override via flags if needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any

from pythia_risk.engine import RiskEngine
from pythia_risk.sizing import size_trade_kelly
from pythia_risk.types import (
    BankrollState,
    ConsensusDecision,
    Market,
    MarketTypeRules,
    RiskConfig,
)

def _default_config() -> RiskConfig:
    """Default config for the CLI (mirrors configs/live-mvp.toml [risk])."""
    return RiskConfig(
        sizing="kelly-fractional",
        kelly_fraction=0.25,
        max_stake_per_market_usd=50.0,
        max_total_exposure_usd=500.0,
        max_drawdown_pct=5.0,
        cool_down_min_after_loss=30.0,
        market_type_rules={
            "politics": MarketTypeRules(max_stake_usd=50.0, allowed=True),
            "crypto": MarketTypeRules(max_stake_usd=40.0, allowed=True),
            "sports": MarketTypeRules(max_stake_usd=20.0, allowed=True),
            "niche": MarketTypeRules(max_stake_usd=30.0, allowed=True),
            "subjective": MarketTypeRules(max_stake_usd=25.0, allowed=True),
        },
    )

def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def _cmd_size(args: argparse.Namespace) -> int:
    stake = size_trade_kelly(
        p_consensus=args.consensus,
        market_price=args.price,
        bankroll_usd=args.bankroll,
        kelly_fraction=args.fraction,
        max_stake_usd=args.max_stake,
    )
    print(f"{stake:.2f}")
    return 0

def _cmd_evaluate(args: argparse.Namespace) -> int:
    decision_data = _load_json(args.decision)
    market_data = _load_json(args.market)

    decision = ConsensusDecision(**decision_data)
    market = Market(**market_data)

    config = _default_config()
    # CLI override for the per-market cap.
    if args.max_stake is not None:
        config = config.model_copy(
            update={"max_stake_per_market_usd": float(args.max_stake)}
        )

    # `--bankroll` is the total current value (cash + open positions).
    # `--exposure` is the open-positions portion (current exposure to risk).
    # If exposure is None, we assume 0 open positions (cash == bankroll).
    open_positions = float(args.exposure) if args.exposure is not None else 0.0
    bankroll = BankrollState(
        cash_usd=max(0.0, float(args.bankroll) - open_positions),
        open_positions_usd=open_positions,
        peak_bankroll_usd=float(args.bankroll),
        current_bankroll_usd=float(args.bankroll),
        drawdown_pct=0.0,
        last_loss_at=None,
    )

    engine = RiskEngine(config)
    plan = engine.evaluate(decision=decision, market=market, current_bankroll=bankroll)

    output = plan.model_dump()
    print(json.dumps(output, indent=2, default=str))
    return 0 if plan.decision == "APPROVE" else 1

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pythia-risk",
        description=(
            "Delphi risk gating (Kelly sizing + exposure / drawdown / "
            "cool-down gates). Wraps icohangar-ops/meshcfo."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    p_size = sub.add_parser("size", help="Compute a fractional-Kelly dollar stake.")
    p_size.add_argument("--consensus", type=float, required=True,
                        help="Consensus probability that YES wins, in [0, 1].")
    p_size.add_argument("--price", type=float, required=True,
                        help="Current YES share price, in [0, 1].")
    p_size.add_argument("--bankroll", type=float, required=True,
                        help="Current bankroll in USD.")
    p_size.add_argument("--fraction", type=float, default=0.25,
                        help="Kelly fraction multiplier (default: 0.25 = quarter-Kelly).")
    p_size.add_argument("--max-stake", type=float, default=50.0,
                        help="Hard cap on stake (default: $50).")
    p_size.set_defaults(func=_cmd_size)

    p_eval = sub.add_parser("evaluate", help="Run the full 6-gate pipeline on JSON inputs.")
    p_eval.add_argument("decision", type=str,
                        help="Path to ConsensusDecision JSON.")
    p_eval.add_argument("market", type=str,
                        help="Path to Market JSON.")
    p_eval.add_argument("--bankroll", type=float, required=True,
                        help="Current bankroll in USD.")
    p_eval.add_argument("--exposure", type=float, default=None,
                        help="Current open positions in USD (default: 0).")
    p_eval.add_argument("--max-stake", type=float, default=None,
                        help="Override max_stake_per_market_usd.")
    p_eval.set_defaults(func=_cmd_evaluate)

    return parser

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))

if __name__ == "__main__":
    sys.exit(main())
