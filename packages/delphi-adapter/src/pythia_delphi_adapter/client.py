"""Async client for the Gensyn Delphi Agentic Trading Toolkit (ATT).

ATT docs: https://docs.gensyn.ai/tech/agentic-trading

This module wraps every ATT HTTP + WebSocket endpoint the rest of the
Pythia mesh needs into a single ``DelphiClient`` class. The client:

- uses ``httpx.AsyncClient`` for both HTTP and WebSocket transport,
- retries idempotent GETs (and the idempotency-keyed ``POST /orders``) with
  ``tenacity`` exponential backoff,
- parses every response into the pydantic models defined in
  ``pythia_delphi_adapter.models``,
- surfaces honest errors (raises ``DelphiAPIError`` on non-2xx responses),
- supports async-context-manager usage so the underlying HTTP connection
  pool is closed cleanly.

Anywhere the live ATT response shape is uncertain, the code is annotated
with ``# VERIFY:`` and a sensible default is assumed. This is a reference implementation
— wire it up against the live ATT, run the smoke tests, fix the field
names, then go live.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from types import TracebackType
from typing import Any
from urllib.parse import quote

import httpx
import websockets
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pythia_delphi_adapter.models import (
    ATT_DOCS_URL,
    Market,
    MarketEvent,
    MarketStatus,
    OrderBook,
    OrderSide,
    Position,
    Settlement,
    TradeReceipt,
)

logger = logging.getLogger(__name__)

# Default ATT base URL. VERIFY: confirm against ATT quickstart.
DEFAULT_ENDPOINT = "https://api.delphi.gensyn.ai"

# Retry policy shared across idempotent calls.
# 3 attempts, exponential backoff 1s → 2s → 4s, retry on transient errors.
_RETRY_STOP = stop_after_attempt(3)
_RETRY_WAIT = wait_exponential(multiplier=1, min=1, max=8)

class DelphiAPIError(Exception):
    """Raised when ATT returns a non-2xx HTTP response.

    Carries the original ``httpx.Response`` so callers can inspect the body
    for ATT-specific error codes.
    """

    def __init__(self, message: str, *, response: httpx.Response | None = None) -> None:
        super().__init__(message)
        self.response = response
        self.status_code = response.status_code if response is not None else None

class DelphiAuthError(DelphiAPIError):
    """Raised on 401/403 — API key missing or invalid."""

class DelphiNotFoundError(DelphiAPIError):
    """Raised on 404 — market / order / position not found."""

class DelphiClient:
    """Async client for the Gensyn Delphi ATT API.

    Usage::

        async with DelphiClient(api_key="dphi_live_...") as client:
            markets = await client.list_markets(status=MarketStatus.OPEN)

    The client is safe to share across asyncio tasks (``httpx.AsyncClient``
    is internally concurrency-safe). It is **not** safe to use after
    ``aclose`` has been called.
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout_sec: float = 30.0,
        *,
        http_client: httpx.AsyncClient | None = None,
        user_agent: str = "pythia-delphi-adapter/0.1",
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required — set DELPHI_API_KEY or pass it explicitly.")

        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.timeout_sec = timeout_sec
        self.user_agent = user_agent

        # If the caller injected a client (used by tests via MockTransport),
        # we take ownership of closing it only if we created it.
        self._owns_http_client = http_client is None
        self._http: httpx.AsyncClient = http_client or httpx.AsyncClient(
            base_url=self.endpoint,
            timeout=timeout_sec,
            headers=self._default_headers(),
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "DelphiClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._owns_http_client:
            await self._http.aclose()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _default_headers(self) -> dict[str, str]:
        """Headers sent on every ATT request.

        # VERIFY: ATT may use ``X-Delphi-Api-Key`` instead of a Bearer token.
        We send both forms to be safe (extra headers are harmless).
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-Delphi-Api-Key": self.api_key,  # VERIFY: header name.
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Single HTTP request with error mapping. Does not retry.

        Retries are applied at the public-method level (only on idempotent
        operations) so we don't double-submit non-idempotent calls.
        """
        headers: dict[str, str] = {}
        if extra_headers:
            headers.update(extra_headers)

        try:
            response = await self._http.request(
                method,
                path,
                params=params,
                json=json_body,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            # Transport-level error (connection, DNS, timeout). Let the
            # tenacity retry layer above decide whether to retry.
            raise DelphiAPIError(f"ATT transport error: {exc!r}") from exc

        if response.status_code in (401, 403):
            raise DelphiAuthError(
                f"ATT auth failed ({response.status_code}): {response.text}",
                response=response,
            )
        if response.status_code == 404:
            raise DelphiNotFoundError(
                f"ATT resource not found: {method} {path}",
                response=response,
            )
        if response.status_code >= 400:
            raise DelphiAPIError(
                f"ATT {method} {path} → {response.status_code}: {response.text}",
                response=response,
            )
        return response

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """``_request`` wrapped in tenacity retry (3 attempts, exp backoff)."""
        try:
            async for attempt in AsyncRetrying(
                stop=_RETRY_STOP,
                wait=_RETRY_WAIT,
                retry=retry_if_exception_type((DelphiAPIError, httpx.HTTPError)),
                reraise=True,
            ):
                with attempt:
                    return await self._request(
                        method,
                        path,
                        params=params,
                        json_body=json_body,
                        extra_headers=extra_headers,
                    )
        except RetryError as exc:  # pragma: no cover — defensive
            raise DelphiAPIError("ATT retry exhausted") from exc
        return None  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    async def list_markets(
        self,
        status: MarketStatus | None = None,
        category: str | None = None,
        limit: int = 100,
    ) -> list[Market]:
        """List Delphi markets. Maps to ``GET /markets``.

        Parameters
        ----------
        status:
            Optional lifecycle filter (OPEN / CLOSED / SETTLED / CANCELLED).
        category:
            Optional category filter. Pass the raw string (e.g. ``"CRYPTO"``)
            or a ``MarketCategory`` value.
        limit:
            Page size. The client currently fetches a single page of up to
            ``limit`` markets. If ATT uses cursor-based pagination, callers
            should loop manually — see ``# VERIFY:`` note in source.

        Returns
        -------
        list[Market]
            Parsed markets, ordered by recency (ATT default).
        """
        params: dict[str, Any] = {"limit": limit}
        if status is not None:
            params["status"] = status.value
        if category is not None:
            # Accept either a MarketCategory enum or a raw string.
            from pythia_delphi_adapter.models import MarketCategory

            cat = category.value if isinstance(category, MarketCategory) else category
            params["category"] = cat

        response = await self._request_with_retry("GET", "/markets", params=params)
        payload = response.json()
        # VERIFY: ATT may wrap results in {"markets": [...]} or return a bare list.
        items = payload.get("markets", payload) if isinstance(payload, dict) else payload
        return [Market.model_validate(item) for item in items]

    async def get_market(self, market_id: str) -> Market:
        """Fetch full metadata for one market. Maps to ``GET /markets/{id}``."""
        path = f"/markets/{quote(market_id, safe='')}"
        response = await self._request_with_retry("GET", path)
        return Market.model_validate(response.json())

    async def get_orderbook(self, market_id: str) -> OrderBook:
        """Fetch the current order book for a market.

        Maps to ``GET /markets/{id}/orderbook``.
        """
        path = f"/markets/{quote(market_id, safe='')}/orderbook"
        response = await self._request_with_retry("GET", path)
        payload = response.json()
        # Inject market_id if ATT omits it in the body.
        if isinstance(payload, dict) and "market_id" not in payload:
            payload = {**payload, "market_id": market_id}
        return OrderBook.model_validate(payload)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def place_order(
        self,
        market_id: str,
        side: OrderSide,
        size_usd: float,
        limit_price: float | None = None,
        correlation_id: str | None = None,
    ) -> TradeReceipt:
        """Submit a new order. Maps to ``POST /orders``.

        Always sends an ``Idempotency-Key`` header so that a retry (after a
        timeout, for example) cannot double-submit. If ``correlation_id``
        is provided it is used as the idempotency key; otherwise a UUID4 is
        generated. ``correlation_id`` is also echoed in the request body
        so the audit trail in ``pythia-observability`` can correlate the
        Pythia decision with the ATT order.

        Parameters
        ----------
        market_id:
            Target Delphi market.
        side:
            YES or NO.
        size_usd:
            Order size in USD. Must be > 0.
        limit_price:
            Optional limit price (0..1). If omitted the order is treated as
            a market order — VERIFY: ATT's market-order semantics.
        correlation_id:
            Caller-supplied id used as the Idempotency-Key header. If
            omitted, a UUID4 is generated.
        """
        import uuid

        if size_usd <= 0:
            raise ValueError("size_usd must be > 0")
        if limit_price is not None and not (0.0 <= limit_price <= 1.0):
            raise ValueError("limit_price must be in [0.0, 1.0]")

        idempotency_key = correlation_id or str(uuid.uuid4())

        body: dict[str, Any] = {
            "market_id": market_id,
            "side": side.value,
            "size_usd": size_usd,
            "correlation_id": correlation_id or idempotency_key,
        }
        if limit_price is not None:
            body["limit_price"] = limit_price

        # Idempotency-Key is safe to retry — wrap in tenacity.
        response = await self._request_with_retry(
            "POST",
            "/orders",
            json_body=body,
            extra_headers={"Idempotency-Key": idempotency_key},  # VERIFY: header name.
        )
        return TradeReceipt.model_validate(response.json())

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a resting order. Maps to ``DELETE /orders/{id}``.

        Returns ``True`` if ATT acknowledged the cancellation. Returns
        ``False`` if the order was already terminal (filled / cancelled).
        """
        path = f"/orders/{quote(order_id, safe='')}"
        response = await self._request_with_retry("DELETE", path)
        # VERIFY: ATT's acknowledgement shape. Assume {"cancelled": true}.
        payload = response.json() if response.content else {}
        if isinstance(payload, dict):
            return bool(payload.get("cancelled", True))
        return True

    # ------------------------------------------------------------------
    # Positions & settlements
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        """List the agent's open positions. Maps to ``GET /positions``."""
        response = await self._request_with_retry("GET", "/positions")
        payload = response.json()
        items = payload.get("positions", payload) if isinstance(payload, dict) else payload
        return [Position.model_validate(item) for item in items]

    async def get_settlements(
        self,
        since: datetime | None = None,
    ) -> list[Settlement]:
        """List recent AI-as-arbiter settlements. Maps to ``GET /settlements``.

        Parameters
        ----------
        since:
            If provided, only return settlements resolved at or after this
            timestamp. Used by ``SettlementListener`` to poll incrementally.
        """
        params: dict[str, Any] = {}
        if since is not None:
            params["since"] = since.isoformat()

        response = await self._request_with_retry("GET", "/settlements", params=params)
        payload = response.json()
        items = payload.get("settlements", payload) if isinstance(payload, dict) else payload
        return [Settlement.model_validate(item) for item in items]

    # ------------------------------------------------------------------
    # WebSocket event stream
    # ------------------------------------------------------------------

    async def subscribe_events(
        self,
        market_id: str | None = None,
    ) -> AsyncIterator[MarketEvent]:
        """Subscribe to the ATT WebSocket event stream.

        Yields parsed ``MarketEvent`` instances (``MarketOpened``,
        ``PriceUpdated``, ``OrderMatched``, ``MarketSettled``) forever,
        until the caller breaks out of the loop or the connection drops.

        Reconnects automatically with exponential backoff on transient
        failures. If reconnection fails 3 times in a row, raises
        ``DelphiAPIError``.

        Parameters
        ----------
        market_id:
            Optional filter — if set, ATT should only push events for this
            market. VERIFY: ATT's per-market subscription semantics.
        """
        ws_url = self.endpoint.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_url}/events"  # VERIFY: ATT's WS path.

        adapter = _get_event_adapter()

        # Manual reconnect loop. We can't use AsyncRetrying here because
        # this is an async generator (tenacity's `with attempt:` doesn't
        # compose cleanly with `yield`).
        max_attempts = 3
        attempt = 0
        while True:
            try:
                async with websockets.connect(
                    ws_url,
                    additional_headers={
                        "Authorization": f"Bearer {self.api_key}",
                    },
                ) as ws:
                    # If filtering by market, send a subscribe message.
                    if market_id is not None:
                        await ws.send(json.dumps({"action": "subscribe",
                                                  "market_id": market_id}))
                    # Reset attempt counter after a successful connection.
                    attempt = 0
                    async for raw in ws:
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            logger.warning(
                                "ignoring non-JSON ATT event frame: %r",
                                raw[:200],
                            )
                            continue
                        try:
                            event = adapter.validate_python(data)
                        except Exception:
                            logger.exception("failed to parse ATT event: %r", data)
                            continue
                        yield event
            except (OSError, websockets.WebSocketException) as exc:
                attempt += 1
                if attempt >= max_attempts:
                    raise DelphiAPIError(
                        f"ATT WebSocket reconnect exhausted after {attempt} attempts: {exc!r}"
                    ) from exc
                backoff = min(2 ** attempt, 8)
                logger.warning(
                    "ATT WS disconnected (attempt %d/%d); reconnecting in %ds: %r",
                    attempt, max_attempts, backoff, exc,
                )
                import asyncio

                await asyncio.sleep(backoff)

# Module-level lazy TypeAdapter for the MarketEvent discriminated union.
# TypeAdapter construction is relatively expensive, so we cache it.
_EVENT_ADAPTER = None

def _get_event_adapter():
    """Return a cached ``TypeAdapter[MarketEvent]``."""
    global _EVENT_ADAPTER
    if _EVENT_ADAPTER is None:
        from pydantic import TypeAdapter

        from pythia_delphi_adapter.models import MarketEvent

        _EVENT_ADAPTER = TypeAdapter(MarketEvent)
    return _EVENT_ADAPTER

__all__ = [
    "DelphiClient",
    "DelphiAPIError",
    "DelphiAuthError",
    "DelphiNotFoundError",
    "DEFAULT_ENDPOINT",
    "ATT_DOCS_URL",
]
