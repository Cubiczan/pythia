"""Concurrent mesh runner.

Runs every analyst concurrently with a per-analyst timeout. Analysts that
time out or raise are dropped from the result (with a warning), rather
than crashing the whole mesh. Downstream ``pythia-consensus`` checks
``min_analysts`` against the returned list length to decide whether to
skip the trade entirely.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from .base import BaseAnalyst
from .types import Estimate, MarketContext

logger = logging.getLogger(__name__)


async def run_mesh(
    market: MarketContext,
    analysts: Sequence[BaseAnalyst],
    timeout_sec: float = 30.0,
) -> list[Estimate]:
    """Run all analysts concurrently against ``market``.

    Parameters
    ----------
    market:
        The market to estimate.
    analysts:
        Sequence of instantiated ``BaseAnalyst`` objects.
    timeout_sec:
        Per-analyst timeout. Default 30s (matches ``live-mvp.toml``).

    Returns
    -------
    list[Estimate]
        Successful estimates, in the same order as ``analysts`` (with
        dropped analysts omitted). May be shorter than ``analysts``.
    """
    if not analysts:
        logger.warning("run_mesh called with empty analysts list")
        return []

    # Wrap each analyst.estimate in a wait_for so a single slow analyst
    # can't stall the whole mesh.
    async def _safe(analyst: BaseAnalyst) -> Estimate | None:
        try:
            return await asyncio.wait_for(
                analyst.estimate(market), timeout=timeout_sec
            )
        except TimeoutError:
            logger.warning(
                "analyst=%s market=%s timed out after %.1fs; dropping",
                analyst.analyst_id,
                market.market_id,
                timeout_sec,
            )
            return None
        except Exception as exc:  # noqa: BLE001 — mesh must keep running
            logger.warning(
                "analyst=%s market=%s raised %s: %s; dropping",
                analyst.analyst_id,
                market.market_id,
                type(exc).__name__,
                exc,
            )
            return None

    results = await asyncio.gather(*(_safe(a) for a in analysts))
    # Preserve order, drop Nones.
    return [r for r in results if r is not None]
