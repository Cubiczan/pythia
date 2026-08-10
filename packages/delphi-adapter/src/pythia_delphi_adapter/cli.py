"""``pythia-delphi`` CLI — one-off inspection + quote preview.

Usage::

    pythia-delphi health
    pythia-delphi markets list [--limit N] [--status STATUS] [--category CAT] [--with-prices]
    pythia-delphi markets get <market_id> [--with-prices]
    pythia-delphi positions <wallet>
    pythia-delphi quote-buy <market_address> <outcome_idx> <shares_out>
    pythia-delphi balance [--token-address 0x...]

All commands read config from environment variables (``DELPHI_NETWORK``,
``DELPHI_API_ACCESS_KEY``, ``DELPHI_SIGNER_TYPE``, ``WALLET_PRIVATE_KEY``,
etc.) — see the SDK README for the full list. They never log secrets.

The ``quote-*`` and ``balance`` commands require a configured signer
(either ``WALLET_PRIVATE_KEY`` or CDP credentials) because the SDK needs
a wallet to resolve the gateway and token addresses.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from pythia_delphi_adapter.client import DelphiClient
from pythia_delphi_adapter.config import load_config
from pythia_delphi_adapter.models import MarketStatus


def _print_json(obj: Any) -> None:
    """Print a JSON object with indent + sort_keys for readability."""
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def _market_to_dict(market: Any) -> dict[str, Any]:
    """Convert a Market model to a compact dict for CLI output."""
    return {
        "market_address": market.market_address,
        "app_market_id": market.app_market_id,
        "market_url": market.market_url,
        "question": market.question,
        "category": market.category,
        "status": market.status.value if hasattr(market.status, "value") else str(market.status),
        "outcomes": market.outcomes,
        "spot_prices": market.spot_prices,
        "spot_implied_probabilities": market.spot_implied_probabilities,
        "settles_at": market.settles_at.isoformat() if market.settles_at else None,
    }


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def cmd_health(client: DelphiClient, args: argparse.Namespace) -> None:
    health = await client.health()
    _print_json({"status": health.status})


async def cmd_markets_list(client: DelphiClient, args: argparse.Namespace) -> None:
    markets = await client.list_markets(
        limit=args.limit,
        status=args.status,
        category=args.category,
        prices_and_implied_probabilities=args.with_prices,
    )
    print(json.dumps([_market_to_dict(m) for m in markets], indent=2, default=str))


async def cmd_markets_get(client: DelphiClient, args: argparse.Namespace) -> None:
    market = await client.get_market(
        args.market_id,
        prices_and_implied_probabilities=args.with_prices,
    )
    _print_json(_market_to_dict(market))


async def cmd_positions(client: DelphiClient, args: argparse.Namespace) -> None:
    positions = await client.list_positions(args.wallet)
    out = []
    for p in positions:
        out.append({
            "position_id": p.position_id,
            "market_address": p.market_address,
            "outcome_idx": p.outcome_idx,
            "shares": p.shares,
            "redeemed_or_liquidated": p.redeemed_or_liquidated,
            "market_status": p.market_status.value if hasattr(p.market_status, "value") else str(p.market_status),
        })
    _print_json(out)


async def cmd_quote_buy(client: DelphiClient, args: argparse.Namespace) -> None:
    quote = await client.quote_buy(
        market_address=args.market_address,
        outcome_idx=args.outcome_idx,
        shares_out=args.shares_out,
    )
    _print_json({"tokens_in": quote.tokens_in})


async def cmd_balance(client: DelphiClient, args: argparse.Namespace) -> None:
    if args.token_address:
        bal = await client.get_erc20_balance(token_address=args.token_address)
        _print_json({"balance": bal.balance, "decimals": bal.decimals})
    else:
        eth = await client.get_eth_balance()
        _print_json({"eth_balance": eth})


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pythia-delphi",
        description="CLI for the Gensyn Delphi SDK (via pythia-delphi-adapter)",
    )
    parser.add_argument(
        "--config",
        help="Path to a TOML config file (overrides env vars)",
        default=None,
    )
    parser.add_argument(
        "--network",
        help="Delphi network: testnet | mainnet | competition-testnet",
        default=None,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # health
    sub.add_parser("health", help="Check REST API health")

    # markets list
    p_list = sub.add_parser("markets list", help="List markets")
    p_list.add_argument("--limit", type=int, default=10)
    p_list.add_argument("--status", choices=[s.value for s in MarketStatus], default=None)
    p_list.add_argument("--category", default=None)
    p_list.add_argument("--with-prices", action="store_true", help="Fetch on-chain spot prices")

    # markets get
    p_get = sub.add_parser("markets get", help="Get a single market")
    p_get.add_argument("market_id", help="Market app UUID or contract address")
    p_get.add_argument("--with-prices", action="store_true")

    # positions
    p_pos = sub.add_parser("positions", help="List positions for a wallet")
    p_pos.add_argument("wallet", help="Wallet address (0x...)")

    # quote-buy
    p_qb = sub.add_parser("quote-buy", help="Quote collateral needed to buy shares")
    p_qb.add_argument("market_address", help="Market proxy address (0x...)")
    p_qb.add_argument("outcome_idx", type=int, help="0-based outcome index")
    p_qb.add_argument("shares_out", help="Number of shares (18-decimal string, e.g. 1000000000000000000)")

    # balance
    p_bal = sub.add_parser("balance", help="Show ETH or ERC-20 balance")
    p_bal.add_argument("--token-address", default=None, help="ERC-20 token address (default: network token)")

    return parser


COMMAND_HANDLERS = {
    "health": cmd_health,
    "markets list": cmd_markets_list,
    "markets get": cmd_markets_get,
    "positions": cmd_positions,
    "quote-buy": cmd_quote_buy,
    "balance": cmd_balance,
}


async def amain() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Load config (env vars + optional TOML)
    try:
        env_config = load_config(toml_path=args.config, require_key=False)
    except ValueError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    network = args.network or env_config.network
    env = env_config.to_env_dict()
    if network:
        env["DELPHI_NETWORK"] = network

    async with DelphiClient(env=env) as client:
        handler = COMMAND_HANDLERS[args.command]
        try:
            await handler(client, args)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    return 0


def main() -> None:
    sys.exit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
