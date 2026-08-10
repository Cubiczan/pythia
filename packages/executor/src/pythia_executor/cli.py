"""Argparse CLI for ``pythia`` — the trade-orchestration entry point.

Subcommands:

    pythia executor delphi paper-trade  --market <id> [--analysts ...] ...
    pythia executor delphi run          --config <path> [--risk-max-drawdown-pct N] ...
    pythia executor delphi replay       <audit_log_path> [--line N]

The CLI is a thin shell over ``PythiaExecutor`` + ``run_loop``. All
domain logic lives in the library; the CLI just wires config + components
together and prints results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .pipeline import PythiaExecutor
from .types import ExecutorConfig

logger = logging.getLogger("pythia_executor.cli")

# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(getattr(args, "log_level", "info"))

    if args.command == "executor":
        return _dispatch_executor(args)
    parser.print_help()
    return 0

# ---------------------------------------------------------------------------
# argparse tree
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pythia",
        description="Pythia trade-orchestration CLI (wraps icohangar-ops/metabocommand).",
    )
    sub = parser.add_subparsers(dest="command")

    # executor
    p_exec = sub.add_parser("executor", help="Trade orchestration commands.")
    exec_sub = p_exec.add_subparsers(dest="executor_command")

    # executor delphi
    p_delphi = exec_sub.add_parser("delphi", help="Delphi ATT pipeline commands.")
    delphi_sub = p_delphi.add_subparsers(dest="delphi_command")

    # executor delphi paper-trade
    p_pt = delphi_sub.add_parser("paper-trade", help="Run the pipeline once in paper mode.")
    p_pt.add_argument("--market", required=True, help="Delphi market id.")
    p_pt.add_argument(
        "--analysts", default="politics,crypto",
        help="Comma-separated analyst slugs (default: politics,crypto).",
    )
    p_pt.add_argument("--consensus-threshold", type=float, default=0.65,
                      help="agreement_score cutoff for gate=trade (default: 0.65).")
    p_pt.add_argument("--max-stake-usd", type=float, default=50.0,
                      help="Per-market stake cap (default: 50.0).")
    p_pt.add_argument("--audit-log", default="./logs/audit.jsonl",
                      help="Path to the JSONL audit log (default: ./logs/audit.jsonl).")
    p_pt.add_argument("--llm-provider", default=None,
                      help="LLM provider (default: env LLM_PROVIDER or 'openai').")
    p_pt.add_argument("--llm-model", default=None,
                      help="LLM model (default: env LLM_MODEL or 'gpt-4o-mini').")
    p_pt.add_argument("--llm-api-key", default=None,
                      help="LLM API key (default: env LLM_API_KEY).")
    p_pt.add_argument("--log-level", default="info",
                      help="Logging level (debug/info/warning/error).")

    # executor delphi run
    p_run = delphi_sub.add_parser("run", help="Run the pipeline continuously.")
    p_run.add_argument("--config", default="configs/live-mvp.toml",
                       help="TOML config path (default: configs/live-mvp.toml).")
    p_run.add_argument("--risk-max-drawdown-pct", type=float, default=None,
                       help="Override [risk].max_drawdown_pct from the TOML.")
    p_run.add_argument("--log-level", default="info",
                       help="Logging level (debug/info/warning/error).")
    p_run.add_argument("--once", action="store_true",
                       help="Run one poll iteration and exit (smoke test).")

    # executor delphi replay
    p_rep = delphi_sub.add_parser("replay", help="Replay an audit log entry.")
    p_rep.add_argument("audit_log_path", help="Path to the JSONL audit log.")
    p_rep.add_argument("--line", type=int, default=-1,
                       help="1-indexed line number; -1 = last (default: -1).")
    p_rep.add_argument("--log-level", default="info",
                       help="Logging level.")

    return parser

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def _dispatch_executor(args: argparse.Namespace) -> int:
    if args.executor_command == "delphi":
        return _dispatch_delphi(args)
    print("Usage: pythia executor delphi {paper-trade|run|replay} ...", file=sys.stderr)
    return 1

def _dispatch_delphi(args: argparse.Namespace) -> int:
    if args.delphi_command == "paper-trade":
        return asyncio.run(_cmd_paper_trade(args))
    if args.delphi_command == "run":
        return asyncio.run(_cmd_run(args))
    if args.delphi_command == "replay":
        return _cmd_replay(args)
    print("Usage: pythia executor delphi {paper-trade|run|replay} ...", file=sys.stderr)
    return 1

# ---------------------------------------------------------------------------
# paper-trade
# ---------------------------------------------------------------------------

async def _cmd_paper_trade(args: argparse.Namespace) -> int:
    from pythia_analyst_mesh import AnalystRegistry
    from pythia_consensus import ConsensusConfig, ConsensusEngine
    from pythia_delphi_adapter import DelphiClient
    from pythia_risk import MarketTypeRules, RiskConfig, RiskEngine

    api_key = os.environ.get("DELPHI_API_KEY", "")
    if not api_key:
        print(
            "warning: DELPHI_API_KEY not set; ATT calls will fail.",
            file=sys.stderr,
        )

    delphi_client = DelphiClient(api_key=api_key or "stub")

    # Build the mesh.
    analyst_slugs = [s.strip() for s in args.analysts.split(",") if s.strip()]
    llm_config = _build_llm_config(args)
    registry = AnalystRegistry()
    mesh = registry.build_mesh(analyst_slugs, llm_config)

    # Build consensus engine.
    consensus_config = ConsensusConfig(
        method="logit-mean",
        agreement_threshold=args.consensus_threshold,
        min_analysts=2,
    )
    consensus_engine = ConsensusEngine(consensus_config)

    # Build risk engine.
    risk_config = RiskConfig(
        sizing="kelly-fractional",
        kelly_fraction=0.25,
        max_stake_per_market_usd=args.max_stake_usd,
        max_total_exposure_usd=500.0,
        max_drawdown_pct=5.0,
        cool_down_min_after_loss=30.0,
        market_type_rules={
            "politics": MarketTypeRules(max_stake_usd=args.max_stake_usd, allowed=True),
            "crypto": MarketTypeRules(max_stake_usd=args.max_stake_usd, allowed=True),
            "sports": MarketTypeRules(max_stake_usd=args.max_stake_usd, allowed=True),
            "niche": MarketTypeRules(max_stake_usd=args.max_stake_usd, allowed=True),
            "subjective": MarketTypeRules(max_stake_usd=args.max_stake_usd, allowed=True),
        },
    )
    risk_engine = RiskEngine(risk_config)

    executor_config = ExecutorConfig(
        mode="paper",
        signing_key_env="DELPHI_SIGNING_KEY",
        idempotency_enabled=True,
        retry_max=3,
        retry_backoff_sec=5,
    )

    executor = PythiaExecutor(
        delphi_client=delphi_client,
        mesh=mesh,
        consensus_engine=consensus_engine,
        risk_engine=risk_engine,
        config=executor_config,
        audit_log_path=Path(args.audit_log),
    )

    try:
        result = await executor.run_for_market(args.market)
    finally:
        await delphi_client.aclose()

    # Print as JSON to stdout.
    payload = executor._result_to_jsonable(result)
    print(json.dumps(payload, indent=2, default=str))
    return 0

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

async def _cmd_run(args: argparse.Namespace) -> int:
    from .loop import run_loop

    config = _load_toml_config(args.config)

    # Build components from the TOML config.
    executor = _build_executor_from_config(config, args)

    poll_interval = int(
        config.get("delphi", {}).get("poll_interval_sec", 60)
    )
    market_filter = {"status": _resolve_market_status()}

    max_iterations = 1 if args.once else None

    try:
        await run_loop(
            executor,
            poll_interval_sec=poll_interval,
            market_filter=market_filter,
            max_iterations=max_iterations,
        )
    finally:
        # Close the delphi client if it owns an httpx pool.
        aclose = getattr(executor.delphi_client, "aclose", None)
        if aclose is not None:
            await aclose()
    return 0

def _resolve_market_status() -> Any:
    """Resolve the MarketStatus.OPEN enum without hard-importing at module load."""
    from pythia_delphi_adapter import MarketStatus

    return MarketStatus.OPEN

# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

def _cmd_replay(args: argparse.Namespace) -> int:
    log_path = Path(args.audit_log_path)
    if not log_path.exists():
        print(f"audit log not found: {log_path}", file=sys.stderr)
        return 1

    lines = log_path.read_text(encoding="utf-8").splitlines()
    # Filter empty lines.
    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        print(f"audit log is empty: {log_path}", file=sys.stderr)
        return 1

    if args.line == -1:
        idx = len(lines) - 1
    else:
        idx = args.line - 1
        if idx < 0 or idx >= len(lines):
            print(
                f"line {args.line} out of range (1..{len(lines)})",
                file=sys.stderr,
            )
            return 1

    raw = lines[idx]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"failed to parse line {args.line}: {exc}", file=sys.stderr)
        return 1

    print(_format_replay(payload))
    return 0

def _format_replay(payload: dict[str, Any]) -> str:
    """Pretty-print a single audit-log entry as a human-readable chain."""
    out: list[str] = []
    out.append(f"=== PipelineResult for market {payload.get('market_id', '?')} ===")
    out.append(f"  timestamp:      {payload.get('timestamp', '?')}")
    out.append(f"  skipped_reason: {payload.get('skipped_reason', '(none — traded)')}")
    out.append(f"  signature:      {payload.get('signature', '?')[:32]}...")
    out.append(f"  signature_algo: {payload.get('signature_algo', '?')}")
    out.append("")

    estimates = payload.get("estimates") or []
    out.append(f"-- Estimates ({len(estimates)} analyst(s)) --")
    for est in estimates:
        out.append(
            f"  [{est.get('analyst_id', '?')}] "
            f"P(YES)={est.get('probability', '?'):.4f} "
            f"conf={est.get('confidence', '?'):.3f}  "
            f"rationale: {est.get('rationale', '')[:80]}"
        )
    out.append("")

    decision = payload.get("decision")
    if decision:
        out.append("-- ConsensusDecision --")
        out.append(f"  consensus_prob:  {decision.get('consensus_prob', '?'):.4f}")
        out.append(f"  agreement_score: {decision.get('agreement_score', '?'):.4f}")
        out.append(f"  gate:            {decision.get('gate', '?')}")
        out.append(f"  method:          {decision.get('method', '?')}")
        out.append(f"  contributors:    {', '.join(decision.get('contributor_ids', []))}")
        out.append("")

    plan = payload.get("plan")
    if plan:
        out.append("-- TradePlan (risk engine) --")
        out.append(f"  decision:    {plan.get('decision', '?')}")
        out.append(f"  side:        {plan.get('side', '?')}")
        out.append(f"  size_usd:    ${plan.get('size_usd', 0.0):.2f}")
        out.append(f"  limit_price: {plan.get('limit_price', '(market order)')}")
        out.append(f"  risk_flags:  {plan.get('risk_flags', [])}")
        out.append(f"  rationale:   {plan.get('rationale', '')[:120]}")
        out.append("")

    receipt = payload.get("receipt")
    if receipt:
        out.append("-- TradeReceipt --")
        out.append(f"  att_order_id: {receipt.get('att_order_id', '?')}")
        out.append(f"  status:       {receipt.get('status', '?')}")
        out.append(f"  side:         {receipt.get('side', '?')}")
        out.append(f"  size_usd:     ${receipt.get('size_usd', 0.0):.2f}")
        out.append(f"  fill_price:   {receipt.get('fill_price', '?')}")
        out.append(f"  signed_by:    {receipt.get('signed_by', '?')}")
        out.append("")

    return "\n".join(out)

# ---------------------------------------------------------------------------
# config loading + component assembly
# ---------------------------------------------------------------------------

def _load_toml_config(path: str) -> dict[str, Any]:
    """Load a TOML config file. Uses tomllib on 3.11+, tomli otherwise."""
    try:
        import tomllib  # type: ignore[import-not-found]
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(path, "rb") as f:
        return tomllib.load(f)

def _build_executor_from_config(
    config: dict[str, Any], args: argparse.Namespace
) -> PythiaExecutor:
    """Build a PythiaExecutor from a TOML config dict + CLI args."""
    from pythia_analyst_mesh import AnalystRegistry, LLMConfig
    from pythia_consensus import ConsensusConfig, ConsensusEngine
    from pythia_delphi_adapter import DelphiClient
    from pythia_risk import MarketTypeRules, RiskConfig, RiskEngine

    # Delphi client.
    delphi_cfg = config.get("delphi", {})
    api_key_env = delphi_cfg.get("api_key_env", "DELPHI_API_KEY")
    api_key = os.environ.get(api_key_env, "")
    endpoint = delphi_cfg.get("endpoint", "https://api.delphi.gensyn.ai")
    delphi_client = DelphiClient(
        api_key=api_key or "stub", endpoint=endpoint,
    )

    # Mesh.
    mesh_cfg = config.get("mesh", {})
    analyst_slugs = mesh_cfg.get("analysts", ["politics", "crypto"])
    llm_config = LLMConfig(
        provider=mesh_cfg.get("llm_provider", "openai"),
        model=mesh_cfg.get("llm_model", "gpt-4o-mini"),
        api_key=os.environ.get(mesh_cfg.get("llm_api_key_env", "LLM_API_KEY"), ""),
        temperature=float(mesh_cfg.get("llm_temperature", 0.2)),
        max_tokens=int(mesh_cfg.get("llm_max_tokens", 800)),
    )
    registry = AnalystRegistry()
    mesh = registry.build_mesh(analyst_slugs, llm_config)

    # Consensus.
    consensus_cfg = config.get("consensus", {})
    consensus_config = ConsensusConfig(
        method=consensus_cfg.get("method", "logit-mean"),
        agreement_threshold=float(consensus_cfg.get("agreement_threshold", 0.65)),
        min_analysts=int(consensus_cfg.get("min_analysts", 2)),
        weights=consensus_cfg.get("weights"),
    )
    consensus_engine = ConsensusEngine(consensus_config)

    # Risk.
    risk_cfg = config.get("risk", {})
    market_type_rules_raw = risk_cfg.get("market_type_rules", {})
    market_type_rules = {
        k: MarketTypeRules.model_validate(v) for k, v in market_type_rules_raw.items()
    }
    max_dd = risk_cfg.get("max_drawdown_pct", 5.0)
    if getattr(args, "risk_max_drawdown_pct", None) is not None:
        max_dd = float(args.risk_max_drawdown_pct)
    risk_config = RiskConfig(
        sizing=risk_cfg.get("sizing", "kelly-fractional"),
        kelly_fraction=float(risk_cfg.get("kelly_fraction", 0.25)),
        max_stake_per_market_usd=float(risk_cfg.get("max_stake_per_market_usd", 50.0)),
        max_total_exposure_usd=float(risk_cfg.get("max_total_exposure_usd", 500.0)),
        max_drawdown_pct=float(max_dd),
        cool_down_min_after_loss=float(risk_cfg.get("cool_down_min_after_loss", 30.0)),
        market_type_rules=market_type_rules,
    )
    risk_engine = RiskEngine(risk_config)

    # Executor config.
    exec_cfg = config.get("executor", {})
    executor_config = ExecutorConfig(
        mode=exec_cfg.get("mode", "paper"),
        signing_key_env=exec_cfg.get("signing_key_env", "DELPHI_SIGNING_KEY"),
        idempotency_enabled=bool(exec_cfg.get("idempotency_enabled", True)),
        retry_max=int(exec_cfg.get("retry_max", 3)),
        retry_backoff_sec=int(exec_cfg.get("retry_backoff_sec", 5)),
    )

    # Audit log path.
    obs_cfg = config.get("observability", {})
    audit_log_path = Path(obs_cfg.get("audit_log_path", "./logs/audit.jsonl"))

    return PythiaExecutor(
        delphi_client=delphi_client,
        mesh=mesh,
        consensus_engine=consensus_engine,
        risk_engine=risk_engine,
        config=executor_config,
        audit_log_path=audit_log_path,
    )

def _build_llm_config(args: argparse.Namespace) -> Any:
    """Build an LLMConfig from CLI args + env vars."""
    from pythia_analyst_mesh import LLMConfig

    provider = args.llm_provider or os.environ.get("LLM_PROVIDER", "openai")
    model = args.llm_model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = args.llm_api_key or os.environ.get("LLM_API_KEY")
    return LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
    )

# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

def _configure_logging(level: str) -> None:
    """Configure root logging from a level name."""
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    logging.basicConfig(
        level=level_map.get(level.lower(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
