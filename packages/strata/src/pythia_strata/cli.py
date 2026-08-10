"""CLI entrypoint for pythia-strata.

Usage
-----
    pythia-strata enrich <market_id> [--api-key KEY] [--out FILE]
    pythia-strata watch  [--interval 60] [--out FILE] [--limit 10] [--once]

``enrich`` fetches a single Delphi market, enriches it, and prints the
resulting ``EnrichedMarket`` as JSON to stdout. ``--out`` writes the
JSON to a file instead.

``watch`` polls Delphi's open-markets list every ``--interval`` seconds,
enriches each new market it hasn't seen before, and appends one JSON
line per enriched market to ``--out`` (or to stdout if ``--out`` is
omitted). ``--once`` runs a single poll and exits — useful for testing
in CI.

Both subcommands require the Delphi API key. It's resolved from
``--api-key``, ``$DELPHI_API_KEY``, or ``$DELPHI_KEY`` (first match).
Provider API keys (news, on-chain, social) are NOT required — the
default stubs return ``[]`` and the enrichment still produces a valid
``EnrichedMarket``.

The CLI exists for ad-hoc testing and for the demo replay UI. For
production use, prefer the Python API in ``pythia_strata.enricher``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from pythia_strata.enricher import MarketEnricher
from pythia_strata.providers import NewsProvider, OnChainProvider, SocialProvider

logger = logging.getLogger("pythia_strata.cli")


# ---------------------------------------------------------------------------
# Delphi client resolution — lazy so the CLI is usable for `--help` even
# when pythia_delphi_adapter isn't installed.
# ---------------------------------------------------------------------------


def _resolve_api_key(args: argparse.Namespace) -> str | None:
    """Resolve the Delphi API key from flags or env vars."""
    key = getattr(args, "api_key", None)
    if key:
        return key
    return os.environ.get("DELPHI_API_KEY") or os.environ.get("DELPHI_KEY")


def _make_delphi_client(api_key: str):  # type: ignore[no-untyped-def]
    """Instantiate a DelphiClient.

    Lazy import so the CLI can be imported without
    ``pythia_delphi_adapter`` installed (e.g. during ``--help``).
    """
    from pythia_delphi_adapter import DelphiClient  # type: ignore[import-not-found]

    return DelphiClient(api_key=api_key)


# ---------------------------------------------------------------------------
# enrich subcommand
# ---------------------------------------------------------------------------


async def _cmd_enrich_async(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    if not api_key:
        print(
            "error: no Delphi API key. Pass --api-key or set DELPHI_API_KEY.",
            file=sys.stderr,
        )
        return 2

    try:
        client = _make_delphi_client(api_key)
    except ImportError:
        print(
            "error: pythia_delphi_adapter is not installed — cannot fetch "
            "Delphi markets. Install it with `pip install pythia-delphi-adapter`.",
            file=sys.stderr,
        )
        return 2

    enricher = MarketEnricher(
        news=NewsProvider(),
        onchain=OnChainProvider(),
        social=SocialProvider(),
    )

    async with client:
        try:
            market = await client.get_market(args.market_id)
        except Exception as exc:  # noqa: BLE001 — surface any ATT error to the user
            print(f"error: failed to fetch market {args.market_id!r}: {exc}", file=sys.stderr)
            return 1

        enriched = await enricher.enrich(market)

    out = enriched.model_dump_json(indent=2)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"wrote enriched market to {args.out}", file=sys.stderr)
    print(out)
    return 0


def _cmd_enrich(args: argparse.Namespace) -> int:
    return asyncio.run(_cmd_enrich_async(args))


# ---------------------------------------------------------------------------
# watch subcommand
# ---------------------------------------------------------------------------


async def _cmd_watch_async(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    if not api_key:
        print(
            "error: no Delphi API key. Pass --api-key or set DELPHI_API_KEY.",
            file=sys.stderr,
        )
        return 2

    try:
        client = _make_delphi_client(api_key)
    except ImportError:
        print(
            "error: pythia_delphi_adapter is not installed — cannot fetch "
            "Delphi markets. Install it with `pip install pythia-delphi-adapter`.",
            file=sys.stderr,
        )
        return 2

    enricher = MarketEnricher(
        news=NewsProvider(),
        onchain=OnChainProvider(),
        social=SocialProvider(),
    )

    out_path: Path | None = Path(args.out) if args.out else None
    seen_market_ids: set[str] = set()

    async with client:
        while True:
            try:
                markets = await client.list_markets(limit=args.limit)
            except Exception as exc:  # noqa: BLE001 — log + keep polling
                logger.warning("list_markets failed: %r", exc)
                markets = []

            for market in markets:
                if market.market_id in seen_market_ids:
                    continue
                seen_market_ids.add(market.market_id)
                try:
                    enriched = await enricher.enrich(market)
                except Exception as exc:  # noqa: BLE001 — enrichment never blocks the loop
                    logger.warning(
                        "enrich failed for market=%s: %r",
                        market.market_id,
                        exc,
                    )
                    continue

                line = enriched.model_dump_json()
                if out_path is not None:
                    with out_path.open("a", encoding="utf-8") as f:
                        f.write(line + "\n")
                else:
                    print(line)

            if args.once:
                return 0
            await asyncio.sleep(args.interval)


def _cmd_watch(args: argparse.Namespace) -> int:
    try:
        return asyncio.run(_cmd_watch_async(args))
    except KeyboardInterrupt:
        print("\nwatch interrupted; exiting.", file=sys.stderr)
        return 130  # 128 + SIGINT


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pythia-strata",
        description=(
            "Stratified data ingestion + enrichment for the Pythia Delphi mesh. "
            "Fetches a Delphi market, enriches it with news / on-chain / social "
            "context, and emits the EnrichedMarket as JSON."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- enrich ---
    p_enrich = sub.add_parser(
        "enrich",
        help="Fetch and enrich a single Delphi market, print as JSON.",
    )
    p_enrich.add_argument("market_id", help="Delphi market identifier.")
    p_enrich.add_argument(
        "--api-key",
        default=None,
        help="Delphi API key. Defaults to $DELPHI_API_KEY / $DELPHI_KEY.",
    )
    p_enrich.add_argument(
        "--out",
        default=None,
        help="Optional path to write the JSON to (instead of stdout).",
    )
    p_enrich.set_defaults(func=_cmd_enrich)

    # --- watch ---
    p_watch = sub.add_parser(
        "watch",
        help="Poll Delphi for new markets, enrich each, write to stdout or --out.",
    )
    p_watch.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Seconds between polls (default: 60).",
    )
    p_watch.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max markets per poll (default: 10).",
    )
    p_watch.add_argument(
        "--out",
        default=None,
        help="Optional file to append one JSON line per enriched market.",
    )
    p_watch.add_argument(
        "--api-key",
        default=None,
        help="Delphi API key. Defaults to $DELPHI_API_KEY / $DELPHI_KEY.",
    )
    p_watch.add_argument(
        "--once",
        action="store_true",
        help="Run a single poll and exit (useful for CI / testing).",
    )
    p_watch.set_defaults(func=_cmd_watch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging: -v → WARNING, -vv → INFO, default → WARNING.
    # (We accept -v / -vv as global flags by re-parsing if necessary —
    # simpler to just check env var.)
    log_level = os.environ.get("PYTHIA_STRATA_LOG", "WARNING").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
