"""``pythia-delphi`` CLI — one-off inspection + paper-order preview.

Usage::

    pythia-delphi markets list [--limit N] [--status STATUS] [--category CAT]
    pythia-delphi markets get <market_id>
    pythia-delphi positions
    pythia-delphi settlements [--since 24h]
    pythia-delphi paper-order <market_id> <yes|no> <size_usd> [--limit-price P]

All commands read config from env vars (``DELPHI_API_KEY``,
``DELPHI_ENDPOINT``) or ``--config <toml>``. They never log the API key.

``paper-order`` constructs the exact ``POST /orders`` payload and prints
it without submitting — useful for sanity-checking before going live.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from pythia_delphi_adapter.config import load_config
from pythia_delphi_adapter.models import MarketCategory, MarketStatus, OrderSide

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_since(since: str) -> datetime:
    """Parse a ``--since`` value like ``24h``, ``30m``, or an ISO-8601 ts."""
    since = since.strip()
    if since.endswith("h"):
        return datetime.now(timezone.utc) - timedelta(hours=int(since[:-1]))
    if since.endswith("m"):
        return datetime.now(timezone.utc) - timedelta(minutes=int(since[:-1]))
    if since.endswith("d"):
        return datetime.now(timezone.utc) - timedelta(days=int(since[:-1]))
    # Fall back: try ISO-8601.
    parsed = datetime.fromisoformat(since)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed

def _print_json(obj: Any) -> None:
    """Pretty-print a pydantic model / dict / list as JSON to stdout."""
    if hasattr(obj, "model_dump_json"):
        print(obj.model_dump_json(indent=2))
    elif isinstance(obj, list):
        items = [
            (x.model_dump(mode="json") if hasattr(x, "model_dump") else x) for x in obj
        ]
        print(json.dumps(items, indent=2, default=str))
    else:
        print(json.dumps(obj, indent=2, default=str))

def _build_client(args: argparse.Namespace):
    """Construct a ``DelphiClient`` from CLI args + env / TOML config."""
    from pythia_delphi_adapter.client import DelphiClient

    cfg = load_config(
        env_var="DELPHI_API_KEY",
        toml_path=getattr(args, "config", None),
    )
    return DelphiClient(api_key=cfg.api_key, endpoint=cfg.endpoint)

# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

async def cmd_markets_list(args: argparse.Namespace) -> int:
    status: MarketStatus | None = None
    if args.status:
        status = MarketStatus(args.status.upper())
    category: str | None = None
    if args.category:
        # Allow both raw string and enum name; validate against known categories.
        try:
            category = MarketCategory(args.category.upper()).value
        except ValueError:
            category = args.category

    async with _build_client(args) as client:
        markets = await client.list_markets(
            status=status, category=category, limit=args.limit
        )
        # Print a compact table + the full JSON.
        if markets:
            print(
                f"{'market_id':<24} {'status':<10} {'yes':<6} {'vol_usd':<12} question"
            )
            print("-" * 80)
            for m in markets:
                print(
                    f"{m.market_id:<24} {m.status.value:<10} "
                    f"{m.yes_price:<6.2f} {m.volume_usd:<12,.0f} {m.question[:40]}"
                )
            print()
        _print_json(markets)
    return 0

async def cmd_markets_get(args: argparse.Namespace) -> int:
    async with _build_client(args) as client:
        market = await client.get_market(args.market_id)
        _print_json(market)
    return 0

async def cmd_positions(args: argparse.Namespace) -> int:
    async with _build_client(args) as client:
        positions = await client.get_positions()
        _print_json(positions)
    return 0

async def cmd_settlements(args: argparse.Namespace) -> int:
    since = _parse_since(args.since) if args.since else None
    async with _build_client(args) as client:
        settlements = await client.get_settlements(since=since)
        _print_json(settlements)
    return 0

async def cmd_paper_order(args: argparse.Namespace) -> int:
    """Construct and print the order payload — do NOT submit.

    This is the single most useful command before going live: it shows
    exactly what would be POSTed to ``/orders``, including the
    idempotency key and the correlation id.
    """
    side = OrderSide(args.side.upper())
    if args.limit_price is not None and not (0.0 <= args.limit_price <= 1.0):
        print(
            f"error: --limit-price must be in [0.0, 1.0], got {args.limit_price}",
            file=sys.stderr,
        )
        return 2

    correlation_id = str(uuid.uuid4())
    payload: dict[str, Any] = {
        "endpoint": "POST /orders",
        "market_id": args.market_id,
        "side": side.value,
        "size_usd": float(args.size_usd),
        "limit_price": args.limit_price,
        "correlation_id": correlation_id,
        "idempotency_key": correlation_id,
        "headers": {
            "Idempotency-Key": correlation_id,
            "Authorization": "Bearer <redacted>",
        },
        "_note": "PAPER — not submitted. Implement --submit in pythia-executor to send live.",
    }
    _print_json(payload)
    return 0

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pythia-delphi",
        description="Inspect Delphi markets, positions, settlements, and preview orders.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a TOML config file (default: env vars only).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # markets group: `markets list` and `markets get`
    p_markets = sub.add_parser("markets", help="Market inspection commands.")
    markets_sub = p_markets.add_subparsers(dest="markets_command", required=True)

    p_list = markets_sub.add_parser("list", help="List markets (default: open).")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.add_argument("--status", default=None, help="OPEN | CLOSED | SETTLED | CANCELLED")
    p_list.add_argument(
        "--category",
        default=None,
        help="POLITICS | ECONOMICS | SPORTS | CRYPTO | SUBJECTIVE",
    )
    p_list.set_defaults(func=cmd_markets_list)

    p_get = markets_sub.add_parser("get", help="Show one market's detail.")
    p_get.add_argument("market_id")
    p_get.set_defaults(func=cmd_markets_get)

    # positions
    p_pos = sub.add_parser("positions", help="Show current open positions.")
    p_pos.set_defaults(func=cmd_positions)

    # settlements
    p_set = sub.add_parser("settlements", help="Show recent AI-arbiter settlements.")
    p_set.add_argument(
        "--since", default="24h", help="Window: e.g. 24h, 30m, 2d, or ISO-8601."
    )
    p_set.set_defaults(func=cmd_settlements)

    # paper-order
    p_paper = sub.add_parser(
        "paper-order", help="Preview an order without submitting it."
    )
    p_paper.add_argument("market_id")
    p_paper.add_argument("side", choices=["yes", "no", "YES", "NO"])
    p_paper.add_argument("size_usd", type=float)
    p_paper.add_argument(
        "--limit-price", type=float, default=None, help="Limit price 0..1."
    )
    p_paper.set_defaults(func=cmd_paper_order)

    return parser

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 1

    try:
        return asyncio.run(func(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
