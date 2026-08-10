"""Settlement listener — polls ATT for AI-as-arbiter resolutions.

The rest of Pythia needs to know *as soon as a market settles* so that:

- ``pythia-risk`` can recompute exposure and unlock capital,
- ``pythia-observability`` can mark P&L milestones and unlock achievements,
- ``pythia-consensus`` can score the analyst mesh's calibration against
  the actual outcome.

This module provides a single ``SettlementListener`` class that runs an
async polling loop against ``DelphiClient.get_settlements`` and dispatches
each new settlement to a caller-supplied coroutine.

We poll rather than subscribe over WebSocket because the WebSocket event
stream (``MarketSettled``) is best-effort — if the listener restarts it
might miss events. Polling ``GET /settlements`` with a ``since`` cursor
gives us at-least-once delivery with a simple reconnection story.

ATT docs: https://docs.gensyn.ai/tech/agentic-trading
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pythia_delphi_adapter.models import Settlement

if TYPE_CHECKING:
    from pythia_delphi_adapter.client import DelphiClient

logger = logging.getLogger(__name__)

OnSettlement = Callable[[Settlement], Awaitable[None]]

class SettlementListener:
    """Long-running async loop that polls ATT for new settlements.

    Parameters
    ----------
    client:
        A live ``DelphiClient``. The listener does not own it — the caller
        is responsible for closing it.
    poll_interval_sec:
        Seconds between polls. Defaults to 60.
    max_backoff_sec:
        Cap on exponential backoff when ATT is erroring. After this cap is
        hit the listener keeps retrying at this interval forever (better
        to keep trying than to die silently).
    """

    def __init__(
        self,
        client: "DelphiClient",
        poll_interval_sec: int = 60,
        *,
        max_backoff_sec: int = 600,
    ) -> None:
        if poll_interval_sec < 1:
            raise ValueError("poll_interval_sec must be >= 1")
        self._client = client
        self._poll_interval_sec = poll_interval_sec
        self._max_backoff_sec = max_backoff_sec
        self._last_seen: datetime | None = None
        self._stop_event: asyncio.Event | None = None

    async def start(self, on_settlement: OnSettlement) -> None:
        """Run the polling loop until ``stop()`` is called.

        Each new settlement found is passed to ``on_settlement``. Exceptions
        raised by ``on_settlement`` are logged but do not stop the loop 
        a flaky downstream consumer should not break settlement delivery.

        ATT errors (transport, 5xx) trigger exponential backoff up to
        ``max_backoff_sec``. The loop never exits on its own except via
        ``stop()`` or a non-recoverable error.
        """
        self._stop_event = asyncio.Event()
        backoff = self._poll_interval_sec

        while self._stop_event is not None and not self._stop_event.is_set():
            try:
                settlements = await self._client.get_settlements(since=self._last_seen)
                backoff = self._poll_interval_sec  # reset on success

                if not settlements:
                    logger.debug("no new settlements since %s", self._last_seen)
                else:
                    # Sort by resolved_at so we advance _last_seen monotonically.
                    settlements.sort(key=lambda s: s.resolved_at)
                    for s in settlements:
                        if self._last_seen is not None and s.resolved_at <= self._last_seen:
                            # Defensive: ATT should respect `since`, but if it
                            # returns a stale entry we skip it.
                            continue
                        try:
                            await on_settlement(s)
                        except Exception:
                            logger.exception(
                                "on_settlement callback raised for market %s; continuing",
                                s.market_id,
                            )
                        # Advance the cursor regardless of callback outcome.
                        if self._last_seen is None or s.resolved_at > self._last_seen:
                            self._last_seen = s.resolved_at

            except Exception as exc:
                # ATT error — back off exponentially, but keep going.
                logger.warning(
                    "settlement poll failed (%r); backing off for %ds",
                    exc, backoff,
                )
                backoff = min(backoff * 2, self._max_backoff_sec)

            if self._stop_event is None:
                break
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        """Signal the polling loop to exit after the current iteration."""
        if self._stop_event is not None:
            self._stop_event.set()

    @property
    def last_seen(self) -> datetime | None:
        """Timestamp of the most recently delivered settlement (UTC)."""
        return self._last_seen

    def reset_cursor(self, to: datetime | None = None) -> None:
        """Reset the polling cursor.

        Pass ``None`` to re-poll from the beginning (next call to ``start``
        will deliver every settlement ATT returns), or pass a datetime to
        resume from a specific point.
        """
        if to is not None and to.tzinfo is None:
            to = to.replace(tzinfo=timezone.utc)
        self._last_seen = to

__all__ = ["SettlementListener", "OnSettlement"]
