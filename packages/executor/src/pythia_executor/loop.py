"""Continuous polling loop for the executor.

``run_loop`` is the long-running entry point used by ``pythia executor delphi run``.
It repeatedly:

1. lists open markets from Delphi (filtered by ``market_filter``),
2. for each market not seen in the last ``cool_down_per_market_sec`` window,
   calls ``executor.run_for_market(market_id)``,
3. sleeps ``poll_interval_sec`` between iterations,
4. exits cleanly on ``SIGINT`` / ``SIGTERM`` (drains in-flight tasks, writes
   a final audit line).

The "seen recently" tracking is in-process only — restarts re-evaluate
every open market on the first iteration. This is intentional: a fresh
process should always have a fresh view. For cross-process dedup, rely on
the audit log + idempotency keys.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from collections.abc import Sequence
from typing import Any

from .pipeline import PythiaExecutor

logger = logging.getLogger(__name__)

# How long to skip a market after evaluating it, regardless of outcome.
# Prevents the loop from spamming the same market every iteration.
_DEFAULT_COOL_DOWN_PER_MARKET_SEC = 300  # 5 minutes

async def run_loop(
    executor: PythiaExecutor,
    poll_interval_sec: int,
    market_filter: dict[str, Any] | None = None,
    *,
    cool_down_per_market_sec: int = _DEFAULT_COOL_DOWN_PER_MARKET_SEC,
    max_iterations: int | None = None,
) -> None:
    """Run the executor in a continuous poll-eval-sleep loop.

    Parameters
    ----------
    executor:
        A constructed ``PythiaExecutor``.
    poll_interval_sec:
        Seconds to sleep between polling iterations.
    market_filter:
        Optional dict of filter kwargs forwarded to
        ``delphi_client.list_markets`` (e.g. ``{"status": MarketStatus.OPEN}``).
    cool_down_per_market_sec:
        Skip markets evaluated within the last N seconds. Default 300.
    max_iterations:
        If set, stop after N iterations (used by ``--once`` and tests).
        ``None`` = run forever until signalled.
    """
    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)

    seen: dict[str, float] = {}  # market_id → last-evaluated epoch seconds
    iteration = 0

    logger.info(
        "pythia-executor loop starting (poll_interval=%ds, cool_down=%ds, filter=%s)",
        poll_interval_sec, cool_down_per_market_sec, market_filter,
    )

    while not stop_event.is_set():
        iteration += 1
        try:
            markets = await _list_open_markets(executor, market_filter)
            now = time.time()

            for market in markets:
                if stop_event.is_set():
                    break
                market_id = market.market_id
                last_seen = seen.get(market_id, 0.0)
                if now - last_seen < cool_down_per_market_sec:
                    continue
                try:
                    result = await executor.run_for_market(market_id)
                    seen[market_id] = time.time()
                    logger.info(
                        "iter=%d market=%s → %s",
                        iteration, market_id,
                        result.skipped_reason or "TRADED",
                    )
                except Exception:
                    # A single market failure must not kill the loop.
                    logger.exception(
                        "iter=%d market=%s raised; continuing",
                        iteration, market_id,
                    )
                    seen[market_id] = time.time()
        except Exception:
            # listing markets failed — log and back off.
            logger.exception(
                "iter=%d: list_markets failed; backing off %ds",
                iteration, poll_interval_sec,
            )

        if max_iterations is not None and iteration >= max_iterations:
            logger.info("reached max_iterations=%d; exiting", max_iterations)
            break

        if stop_event.is_set():
            break

        # Sleep interruptibly so SIGINT during the sleep exits promptly.
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll_interval_sec)
        except TimeoutError:
            pass  # normal: poll_interval elapsed, loop again

    logger.info(
        "pythia-executor loop exiting after %d iteration(s); %d market(s) evaluated",
        iteration, len(seen),
    )

async def _list_open_markets(
    executor: PythiaExecutor,
    market_filter: dict[str, Any] | None,
) -> Sequence[Any]:
    """List markets from Delphi, applying the optional filter dict.

    The filter dict is forwarded as ``**kwargs`` to ``list_markets``. If
    the call raises, the exception propagates to ``run_loop`` (which logs
    + backs off).
    """
    kwargs = dict(market_filter) if market_filter else {}
    markets = await executor.delphi_client.list_markets(**kwargs)  # type: ignore[attr-defined]
    return markets

def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    """Install SIGINT / SIGTERM handlers that set the stop event.

    On Windows, SIGTERM isn't available — we install SIGINT + SIGBREAK
    instead. The handler is a no-op if called from a non-main thread
    (Python's signal module raises in that case).
    """
    loop = asyncio.get_running_loop()

    def _handler(name: str) -> None:
        logger.info("received %s; draining in-flight tasks", name)
        stop_event.set()

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, _handler, sig_name)
        except (NotImplementedError, RuntimeError, ValueError):
            # add_signal_handler isn't available on all platforms / threads.
            # Fall back to the low-level signal.signal API for SIGINT.
            if sig_name == "SIGINT":
                signal.signal(sig, lambda *_: stop_event.set())

__all__ = ["run_loop"]
