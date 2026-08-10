"""Polling-based settlement listener for Delphi markets.

The SDK doesn't expose a WebSocket subscription for market settlements, so
we poll the REST API on a fixed interval. This is simpler and adequate for
the Pythia mesh's cadence (settlements are rare relative to trades).

For each market the listener tracks:
  - When it transitions to ``settled`` / ``expired`` / ``failed``, emit a
    ``SettlementEvent`` that the executor / observability layer consumes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator

from .client import DelphiClient
from .models import Market, MarketStatus

logger = logging.getLogger(__name__)


@dataclass
class SettlementEvent:
    """Emitted when a tracked market reaches a terminal status."""

    market_address: str
    app_market_id: str
    question: str
    final_status: MarketStatus
    winning_outcome_idx: int | None
    timestamp: datetime


class SettlementListener:
    """Polls the Delphi REST API for market settlements.

    Usage:

        listener = SettlementListener(client, poll_interval_sec=60)
        async for event in listener.watch(markets):
            # event.final_status is settled / expired / failed
            ...
    """

    TERMINAL_STATUSES = {
        MarketStatus.SETTLED,
        MarketStatus.EXPIRED,
        MarketStatus.FAILED,
    }

    def __init__(
        self,
        client: DelphiClient,
        *,
        poll_interval_sec: int = 60,
        max_iterations: int | None = None,
    ) -> None:
        self._client = client
        self._poll_interval = max(1, poll_interval_sec)
        self._max_iterations = max_iterations

    async def watch(
        self,
        markets: list[Market],
        *,
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator[SettlementEvent]:
        """Yield settlement events for the given markets.

        Polls every ``poll_interval_sec`` until all markets reach a terminal
        status, ``stop_event`` is set, or ``max_iterations`` is reached.
        """
        pending: dict[str, Market] = {
            m.market_address: m for m in markets if m.status not in self.TERMINAL_STATUSES
        }
        if not pending:
            return

        iteration = 0
        while pending and (self._max_iterations is None or iteration < self._max_iterations):
            iteration += 1
            if stop_event is not None and stop_event.is_set():
                return

            try:
                updated = await self._refresh(list(pending.keys()))
            except Exception as exc:
                logger.warning("settlement poll failed: %s", exc)
                await asyncio.sleep(self._poll_interval)
                continue

            for market in updated:
                if market.status in self.TERMINAL_STATUSES:
                    pending.pop(market.market_address, None)
                    yield SettlementEvent(
                        market_address=market.market_address,
                        app_market_id=market.app_market_id,
                        question=market.question,
                        final_status=market.status,
                        winning_outcome_idx=market.winning_outcome_idx,
                        timestamp=datetime.now(timezone.utc),
                    )

            if pending:
                await asyncio.sleep(self._poll_interval)

    async def _refresh(self, market_addresses: list[str]) -> list[Market]:
        """Re-fetch the current state of the given markets."""
        updated: list[Market] = []
        for addr in market_addresses:
            try:
                market = await self._client.get_market(addr, prices_and_implied_probabilities=False)
                updated.append(market)
            except Exception as exc:
                logger.debug("failed to refresh market %s: %s", addr, exc)
        return updated


__all__ = ["SettlementEvent", "SettlementListener"]
