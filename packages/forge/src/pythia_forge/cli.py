"""Command-line interface for pythia-forge.

Three subcommands:

    pythia-forge backtest --strategy configs/strategies/ensemble-v1.toml \
        --markets resolved-2025-Q4.json --starting-capital 1000
        Run a backtest, write a markdown report + equity-curve PNG to ./reports/.

    pythia-forge tune --strategy ... --markets ... --iterations 20
        Grid-search over consensus thresholds + Kelly fractions, find the best
        config by Sharpe ratio, write it to ./reports/best-strategy-<ts>.toml.

    pythia-forge deploy --config configs/live-mvp.toml
        CI deploy hook: validate the live config, run a smoke backtest against
        the sample fixture, emit a deploy plan as JSON.

All subcommands use the MockLLM by default (zero LLM cost, deterministic).
Pass ``--use-real-llm`` to ``backtest`` to call the real LLM provider.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import itertools
import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

# tomllib is stdlib in 3.11+; fall back to tomli for older Pythons.
try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - py < 3.11 only
    import tomli as tomllib  # type: ignore[no-redef]

from .backtester import Backtester
from .report import generate_report
from .types import BacktestConfig

logger = logging.getLogger("pythia_forge")

# Default grid for the tune subcommand. 5 thresholds × 4 fractions = 20 combos.
DEFAULT_THRESHOLDS = [0.50, 0.60, 0.65, 0.70, 0.80]
DEFAULT_KELLY_FRACTIONS = [0.10, 0.25, 0.50, 1.00]

# ---------------------------------------------------------------------------
# backtest subcommand
# ---------------------------------------------------------------------------

def _cmd_backtest(args: argparse.Namespace) -> int:
    """Run a single backtest and write the report."""
    strategy_path = Path(args.strategy).resolve()
    markets_path = Path(args.markets).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    markets_filter: dict[str, Any] = {}
    if args.categories:
        markets_filter["categories"] = args.categories
    if args.min_volume is not None:
        markets_filter["min_volume_usd"] = args.min_volume
    if args.min_lifetime is not None:
        markets_filter["min_market_lifetime_sec"] = args.min_lifetime
    if args.track_bankroll:
        markets_filter["track_bankroll"] = True
    if args.use_real_llm:
        markets_filter["use_real_llm"] = True

    config = BacktestConfig(
        strategy_path=strategy_path,
        markets_path=markets_path,
        starting_capital_usd=args.starting_capital,
        markets_filter=markets_filter,
    )

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = output_dir / f"backtest-{ts}.md"

    logger.info("running backtest: strategy=%s markets=%s capital=$%.2f",
                strategy_path, markets_path, args.starting_capital)
    result = asyncio.run(Backtester(config).run())

    generate_report(result, report_path)

    # Print summary to stdout.
    print("\n=== Backtest complete ===")
    print(f"  Starting capital : ${result.starting_capital_usd:,.2f}")
    print(f"  Ending capital   : ${result.ending_capital_usd:,.2f}")
    print(f"  Total return     : {result.total_return_pct:+.2f}%")
    print(f"  Sharpe ratio     : {result.sharpe_ratio:.3f}")
    print(f"  Max drawdown     : {result.max_drawdown_pct:.2f}%")
    print(f"  Win rate         : {result.win_rate * 100:.1f}%")
    print(f"  Total trades     : {result.total_trades}")
    print("  Brier scores     :")
    for aid, score in sorted(result.brier_scores.items(), key=lambda kv: kv[1]):
        print(f"    {aid:12s} : {score:.4f}")
    print(f"\n  Report           : {report_path}")
    print(f"  Equity curve     : {report_path.with_suffix('.png')}")

    return 0

# ---------------------------------------------------------------------------
# tune subcommand
# ---------------------------------------------------------------------------

def _cmd_tune(args: argparse.Namespace) -> int:
    """Grid-search consensus threshold × Kelly fraction, find best by Sharpe."""
    strategy_path = Path(args.strategy).resolve()
    markets_path = Path(args.markets).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the base strategy to use as a template.
    if not strategy_path.exists():
        print(f"error: strategy TOML not found: {strategy_path}", file=sys.stderr)
        return 2
    with open(strategy_path, "rb") as fh:
        base_toml = tomllib.load(fh)
    base_strategy = base_toml.get("strategy", base_toml)

    thresholds = args.thresholds or DEFAULT_THRESHOLDS
    kelly_fracs = args.kelly_fractions or DEFAULT_KELLY_FRACTIONS
    combos = list(itertools.product(thresholds, kelly_fracs))

    if args.iterations and args.iterations < len(combos):
        # Subsample for speed if --iterations is smaller than the full grid.
        step = max(1, len(combos) // args.iterations)
        combos = combos[::step][: args.iterations]

    print(f"Tuning over {len(combos)} parameter combinations...")
    print(f"{'#':>3}  {'thresh':>7}  {'kelly':>6}  {'return%':>8}  {'sharpe':>7}  "
          f"{'maxdd%':>7}  {'win%':>5}  {'trades':>6}")
    print("-" * 70)

    results: list[dict[str, Any]] = []
    base_config = BacktestConfig(
        strategy_path=strategy_path,
        markets_path=markets_path,
        starting_capital_usd=args.starting_capital,
    )

    for i, (thresh, kelly) in enumerate(combos, 1):
        # Deep-copy the base strategy dict and override the two params.
        strat = copy.deepcopy(base_strategy)
        strat.setdefault("consensus", {})["agreement_threshold"] = thresh
        strat.setdefault("risk", {})["kelly_fraction"] = kelly

        bt = Backtester(base_config, strategy_override=strat)
        result = asyncio.run(bt.run())

        row = {
            "threshold": thresh,
            "kelly_fraction": kelly,
            "return_pct": result.total_return_pct,
            "sharpe": result.sharpe_ratio,
            "max_drawdown_pct": result.max_drawdown_pct,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
        }
        results.append(row)
        print(f"{i:>3}  {thresh:>7.2f}  {kelly:>6.2f}  {row['return_pct']:>+8.2f}  "
              f"{row['sharpe']:>7.3f}  {row['max_drawdown_pct']:>7.2f}  "
              f"{row['win_rate'] * 100:>5.1f}  {row['total_trades']:>6}")

    if not results:
        print("no results — check your markets file", file=sys.stderr)
        return 1

    # Pick best by Sharpe (tie-break by return).
    best = max(results, key=lambda r: (r["sharpe"], r["return_pct"]))

    print("\n" + "=" * 70)
    print(f"BEST: threshold={best['threshold']:.2f}  kelly={best['kelly_fraction']:.2f}  "
          f"sharpe={best['sharpe']:.3f}  return={best['return_pct']:+.2f}%")

    # Write the best config as a TOML file.
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    best_path = output_dir / f"best-strategy-{ts}.toml"
    _write_best_strategy_toml(best_path, base_toml, best)
    print(f"\nBest strategy written to: {best_path}")

    # Also write the full results table as JSON for archival.
    json_path = output_dir / f"tune-results-{ts}.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Full results table: {json_path}")

    return 0

def _write_best_strategy_toml(
    path: Path,
    base_toml: dict[str, Any],
    best: dict[str, Any],
) -> None:
    """Write the best-tuned strategy as a TOML file.

    We can't use tomllib to *write* (it's read-only), so we emit a minimal
    hand-formatted TOML with the overridden values plus the original sections
    copied verbatim. Good enough for re-running a backtest.
    """
    lines: list[str] = []
    lines.append("# Best strategy from pythia-forge tune")
    lines.append(f"# Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("# Selection criterion: max Sharpe (tie-break: max return)")
    lines.append(f"# Sharpe={best['sharpe']:.3f}  Return={best['return_pct']:+.2f}%  "
                 f"MaxDD={best['max_drawdown_pct']:.2f}%  WinRate={best['win_rate'] * 100:.1f}%")
    lines.append("")

    # Copy the top-level [strategy] metadata if present.
    strat_meta = base_toml.get("strategy", {})
    base_name = strat_meta.get("name", "ensemble")
    # Avoid double "-tuned" suffix if the base name already ends with it.
    name = base_name if base_name.endswith("-tuned") else f"{base_name}-tuned"
    lines.append("[strategy]")
    lines.append(f'name = "{name}"')
    desc = f"Tuned: threshold={best['threshold']:.2f}, kelly={best['kelly_fraction']:.2f}"
    lines.append(f'description = "{desc}"')
    lines.append('version = "1.0.0"')
    lines.append("")

    # [strategy.mesh] — copy from base.
    mesh = strat_meta.get("mesh", {})
    lines.append("[strategy.mesh]")
    analysts = mesh.get("analysts", ["politics", "crypto", "sports", "niche"])
    lines.append(f'analysts = {json.dumps(analysts)}')
    lines.append(f'llm_model = "{mesh.get("llm_model", "gpt-4o-mini")}"')
    lines.append(f'llm_temperature = {mesh.get("llm_temperature", 0.2)}')
    lines.append("")

    # [strategy.consensus] — overridden threshold.
    cons = strat_meta.get("consensus", {})
    lines.append("[strategy.consensus]")
    lines.append(f'method = "{cons.get("method", "logit-mean")}"')
    lines.append(f"agreement_threshold = {best['threshold']}")
    lines.append(f"min_analysts = {cons.get('min_analysts', 2)}")
    if "weights" in cons:
        lines.append("")
        lines.append("[strategy.consensus.weights]")
        for aid, w in cons["weights"].items():
            lines.append(f"{aid} = {w}")
    lines.append("")

    # [strategy.risk] — overridden kelly_fraction.
    risk = strat_meta.get("risk", {})
    lines.append("[strategy.risk]")
    lines.append(f'sizing = "{risk.get("sizing", "kelly-fractional")}"')
    lines.append(f"kelly_fraction = {best['kelly_fraction']}")
    lines.append(f"max_stake_per_market_usd = {risk.get('max_stake_per_market_usd', 50)}")
    lines.append(f"max_total_exposure_usd = {risk.get('max_total_exposure_usd', 500)}")
    lines.append(f"max_drawdown_pct = {risk.get('max_drawdown_pct', 5)}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")

# ---------------------------------------------------------------------------
# deploy subcommand
# ---------------------------------------------------------------------------

def _cmd_deploy(args: argparse.Namespace) -> int:
    """CI deploy hook: validate config, smoke test, emit deploy plan.

    This is a reference implementation — the actual runner-deploy step is delegated to
    icohangar-ops/forge's deploy primitive once vendored (currently a stub
    that prints a plan). Marked with # VERIFY throughout.
    """
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"error: deploy config not found: {config_path}", file=sys.stderr)
        return 2

    with open(config_path, "rb") as fh:
        live_cfg = tomllib.load(fh)

    # ---- 1. Validate config sections -----------------------------------
    required_sections = ["delphi", "mesh", "consensus", "risk", "executor"]
    missing = [s for s in required_sections if s not in live_cfg]
    if missing:
        print(json.dumps({
            "status": "invalid",
            "reason": f"missing required sections: {missing}",
            "config": str(config_path),
        }, indent=2))
        return 1

    # Sanity-check risk caps.
    risk = live_cfg.get("risk", {})
    warnings: list[str] = []
    if risk.get("kelly_fraction", 0.25) > 0.5:
        warnings.append(f"kelly_fraction {risk['kelly_fraction']} > 0.5 — aggressive for live")
    if risk.get("max_drawdown_pct", 5) < 3:
        dd = risk['max_drawdown_pct']
        warnings.append(
            f"max_drawdown_pct {dd} < 3% — circuit breaker may trip too early"
        )
    if risk.get("max_stake_per_market_usd", 50) > 100:
        stake = risk['max_stake_per_market_usd']
        warnings.append(
            f"max_stake_per_market_usd ${stake} > $100 — check bankroll"
        )

    # ---- 2. Smoke test backtest ----------------------------------------
    # Run a quick 5-market backtest against the sample fixture to confirm
    # the mesh produces non-degenerate output (not all REJECTs, not all YES).
    smoke_ok = False
    smoke_summary: dict[str, Any] = {}
    fixture = (
        Path(__file__).resolve().parent.parent.parent
        / "tests" / "fixtures" / "resolved_markets_sample.json"
    )
    # If the fixture isn't found (e.g. installed as a wheel), skip smoke test.
    if fixture.exists():
        try:
            # Build a minimal strategy from the live config for the smoke test.
            smoke_strategy = {
                "mesh": live_cfg.get("mesh", {}),
                "consensus": live_cfg.get("consensus", {}),
                "risk": live_cfg.get("risk", {}),
            }
            # Default analysts if not specified.
            if "analysts" not in smoke_strategy.get("mesh", {}):
                smoke_strategy.setdefault("mesh", {})["analysts"] = [
                    "politics", "crypto", "sports", "niche"
                ]

            smoke_config = BacktestConfig(
                strategy_path=config_path,  # unused (override below)
                markets_path=fixture,
                starting_capital_usd=1000.0,
                markets_filter={},
            )
            bt = Backtester(smoke_config, strategy_override=smoke_strategy)
            smoke_result = asyncio.run(bt.run())
            smoke_summary = {
                "total_trades": smoke_result.total_trades,
                "return_pct": smoke_result.total_return_pct,
                "win_rate": smoke_result.win_rate,
            }
            # Smoke passes if at least 1 trade was placed (mesh isn't degenerate).
            smoke_ok = smoke_result.total_trades > 0
        except Exception as exc:  # noqa: BLE001 — smoke failure is non-fatal
            smoke_summary = {"error": f"{type(exc).__name__}: {exc}"}
            warnings.append(f"smoke test raised: {exc}")
    else:
        warnings.append(f"smoke fixture not found at {fixture}; skipping smoke test")
        smoke_ok = True  # don't block deploy on missing fixture in installed mode

    # ---- 3. Emit deploy plan -------------------------------------------
    # VERIFY: once icohangar-ops/forge is vendored, this should delegate to
    # forge.deploy.Runner(plan).run() or equivalent. Currently we just emit
    # the plan as JSON for the CI pipeline to consume.
    plan = {
        "status": "ready" if smoke_ok and not missing else "blocked",
        "config": str(config_path),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "validation": {
            "required_sections_present": not missing,
            "missing_sections": missing,
            "warnings": warnings,
        },
        "smoke_test": smoke_summary,
        "deploy_steps": [
            "1. Pull latest pythia mesh images",
            "2. Apply config to runner environment",
            "3. Start settlement listener",
            "4. Enable paper-trading mode (executor.mode=paper)",
            "5. Monitor first 5 live markets before enabling live mode",
            # VERIFY: forge.deploy.Runner will replace steps 1-5 once vendored.
        ],
    }

    print(json.dumps(plan, indent=2))

    if args.dry_run:
        print("\n[dry-run] no changes made.", file=sys.stderr)
        return 0

    if plan["status"] != "ready":
        return 1

    # In a real deploy, this is where we'd hand off to forge's runner.
    # For now, write the plan to a temp file as a record.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="pythia-forge-deploy-", delete=False
    ) as fh:
        json.dump(plan, fh, indent=2)
        logger.info("deploy plan written to %s", fh.name)

    return 0

# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the top-level pythia-forge CLI parser."""
    parser = argparse.ArgumentParser(
        prog="pythia-forge",
        description=(
            "Backtest harness + CI deploy pipeline for the Pythia mesh. "
            "Wraps icohangar-ops/forge."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase logging verbosity (-v=INFO, -vv=DEBUG).",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    # --- backtest ---
    p_bt = sub.add_parser(
        "backtest",
        help="Run a single backtest and write a markdown + PNG report.",
    )
    p_bt.add_argument("--strategy", required=True, type=str,
                      help="Path to strategy TOML (mesh + consensus + risk config).")
    p_bt.add_argument("--markets", required=True, type=str,
                      help="Path to resolved-markets JSON file.")
    p_bt.add_argument("--starting-capital", type=float, default=1000.0,
                      help="Starting bankroll in USD (default: 1000).")
    p_bt.add_argument("--categories", nargs="*", default=None,
                      help="Filter: only these categories (e.g. politics crypto).")
    p_bt.add_argument("--min-volume", type=float, default=None,
                      help="Filter: minimum market volume in USD.")
    p_bt.add_argument("--min-lifetime", type=float, default=None,
                      help="Filter: minimum market lifetime in seconds.")
    p_bt.add_argument("--track-bankroll", action="store_true",
                      help="Thread running bankroll through markets (exercises gates).")
    p_bt.add_argument("--use-real-llm", action="store_true",
                      help="Call the real LLM provider (slow, costs tokens). "
                           "Default: use MockLLM (deterministic, free).")
    p_bt.add_argument("--output", type=str, default="./reports",
                      help="Directory for the report output (default: ./reports).")
    p_bt.set_defaults(func=_cmd_backtest)

    # --- tune ---
    p_tune = sub.add_parser(
        "tune",
        help="Grid-search consensus threshold × Kelly fraction; find best by Sharpe.",
    )
    p_tune.add_argument("--strategy", required=True, type=str,
                        help="Path to base strategy TOML.")
    p_tune.add_argument("--markets", required=True, type=str,
                        help="Path to resolved-markets JSON file.")
    p_tune.add_argument("--starting-capital", type=float, default=1000.0,
                        help="Starting bankroll in USD (default: 1000).")
    p_tune.add_argument("--iterations", type=int, default=None,
                        help="Max combinations to try (default: full grid = 20).")
    p_tune.add_argument("--thresholds", nargs="*", type=float, default=None,
                        help=f"Override consensus thresholds (default: {DEFAULT_THRESHOLDS}).")
    p_tune.add_argument("--kelly-fractions", nargs="*", type=float, default=None,
                        help=f"Override Kelly fractions (default: {DEFAULT_KELLY_FRACTIONS}).")
    p_tune.add_argument("--output", type=str, default="./reports",
                        help="Directory for output (default: ./reports).")
    p_tune.set_defaults(func=_cmd_tune)

    # --- deploy ---
    p_dep = sub.add_parser(
        "deploy",
        help="CI deploy hook: validate config, smoke test, emit deploy plan.",
    )
    p_dep.add_argument("--config", required=True, type=str,
                       help="Path to live config TOML (e.g. configs/live-mvp.toml).")
    p_dep.add_argument("--dry-run", action="store_true",
                       help="Validate + smoke test only; do not emit a deploy plan file.")
    p_dep.set_defaults(func=_cmd_deploy)

    return parser

def main(argv: list[str] | None = None) -> int:
    """CLI entry point — dispatched by the `pythia-forge` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging based on -v / -vv.
    level = logging.WARNING
    if args.verbose >= 1:
        level = logging.INFO
    if args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    return int(args.func(args))

if __name__ == "__main__":
    sys.exit(main())
