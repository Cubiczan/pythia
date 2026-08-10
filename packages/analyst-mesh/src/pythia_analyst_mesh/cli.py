"""Command-line interface: ``pythia-analyst``.

Subcommands:

    pythia-analyst list
        List all registered analyst slugs.

    pythia-analyst estimate <market_id> --analysts politics,crypto [--provider openai]
        Fetch a market via ``pythia_delphi_adapter.DelphiClient`` and run the
        mesh against it, printing estimates as JSON to stdout.

The LLM provider / model / API key are resolved from CLI flags or env vars
(``LLM_PROVIDER``, ``LLM_MODEL``, ``LLM_API_KEY``).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

from .registry import AnalystRegistry
from .runner import run_mesh
from .types import LLMConfig, MarketContext

logger = logging.getLogger("pythia_analyst_mesh.cli")


# --------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------- #


def cmd_list(args: argparse.Namespace) -> int:
    registry = AnalystRegistry()
    print("Registered analysts:")
    for name in registry.list_known():
        cls = registry.get(name)
        print(f"  {name:<10}  {cls.__name__}  ({cls.specialty})")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    # ---- Build LLMConfig from flags / env. ----
    provider = args.provider or os.environ.get("LLM_PROVIDER", "openai")
    model = args.model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
    api_key = args.api_key or os.environ.get("LLM_API_KEY")
    if provider != "ollama" and not api_key:
        print(
            f"error: no API key for provider {provider!r}. "
            "Set --api-key or LLM_API_KEY.",
            file=sys.stderr,
        )
        return 2
    llm_config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        timeout_sec=args.timeout,
    )

    # ---- Resolve analyst list. ----
    analyst_names = [s.strip() for s in args.analysts.split(",") if s.strip()]
    if not analyst_names:
        print("error: --analysts must list at least one slug", file=sys.stderr)
        return 2
    registry = AnalystRegistry()
    try:
        mesh = registry.build_mesh(analyst_names, llm_config)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # ---- Fetch market. ----
    market = asyncio.run(_fetch_market(args.market_id))

    # ---- Run mesh. ----
    estimates = asyncio.run(run_mesh(market, mesh, timeout_sec=args.timeout))

    # ---- Print as JSON. ----
    out: dict[str, Any] = {
        "market_id": market.market_id,
        "question": market.question,
        "category": market.category,
        "current_yes_price": market.current_yes_price,
        "estimates": [e.model_dump() for e in estimates],
        "dropped": len(mesh) - len(estimates),
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


async def _fetch_market(market_id: str) -> MarketContext:
    """Fetch market via ``pythia_delphi_adapter.DelphiClient``.

    Lazy import so this CLI doesn't crash if the sibling adapter isn't
    installed (e.g. when running tests). Falls back to a stub market that
    lets the mesh still be exercised.

    # VERIFY: DelphiClient.get_market return shape pending pythia-delphi-adapter
    # scaffold. Currently assumed to return a dict with at least:
    #   market_id, question, category, metadata, yes_price, no_price,
    #   volume_usd, closes_at
    """
    try:
        from pythia_delphi_adapter import DelphiClient  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "pythia_delphi_adapter not installed; using stub MarketContext "
            "for %r. Install the sibling repo for live data.",
            market_id,
        )
        return _stub_market(market_id)

    client = DelphiClient()  # type: ignore[call-arg]
    raw = await client.get_market(market_id)  # type: ignore[attr-defined]
    return MarketContext(
        market_id=str(raw.get("market_id") or market_id),
        question=str(raw.get("question") or ""),
        category=str(raw.get("category") or "unknown"),
        metadata=raw.get("metadata") or {},
        current_yes_price=raw.get("yes_price"),
        current_no_price=raw.get("no_price"),
        volume_usd=raw.get("volume_usd"),
        closes_at=raw.get("closes_at"),
    )


def _stub_market(market_id: str) -> MarketContext:
    """Minimal market so the mesh can be exercised without the adapter."""
    return MarketContext(
        market_id=market_id,
        question="(stub) Will the YES outcome occur? (install pythia-delphi-adapter)",
        category="unknown",
        metadata={},
        current_yes_price=None,
        current_no_price=None,
        volume_usd=None,
        closes_at=None,
    )


# --------------------------------------------------------------------- #
# Argparse wiring
# --------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pythia-analyst",
        description="Run specialist LLM analysts against a Delphi market.",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="-v=info, -vv=debug"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="List registered analyst slugs.")
    p_list.set_defaults(func=cmd_list)

    # estimate
    p_est = sub.add_parser("estimate", help="Run the mesh against a market.")
    p_est.add_argument("market_id", help="Delphi market id.")
    p_est.add_argument(
        "--analysts",
        required=True,
        help="Comma-separated analyst slugs, e.g. politics,crypto,niche.",
    )
    p_est.add_argument("--provider", default=None, help="openai|anthropic|gensyn|ollama")
    p_est.add_argument("--model", default=None, help="LLM model id.")
    p_est.add_argument("--api-key", default=None, help="LLM API key (or use LLM_API_KEY).")
    p_est.add_argument("--base-url", default=None, help="Override provider base URL.")
    p_est.add_argument("--temperature", type=float, default=0.2)
    p_est.add_argument("--max-tokens", type=int, default=800)
    p_est.add_argument(
        "--timeout", type=float, default=30.0, help="Per-analyst timeout (sec)."
    )
    p_est.set_defaults(func=cmd_estimate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
