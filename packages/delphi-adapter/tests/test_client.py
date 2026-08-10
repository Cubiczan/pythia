"""Smoke tests for ``DelphiClient`` using ``httpx.MockTransport``.

These tests never touch the real ATT. They assert:

- ``list_markets`` parses a representative ATT response into typed ``Market`` objects.
- ``place_order`` includes the ``Idempotency-Key`` header and parses the receipt.
- ``get_settlements`` filters by ``since`` and parses the settlement.
- ``cancel_order`` returns ``True`` on a 200 acknowledgement.
- Error mapping: 401 → ``DelphiAuthError``, 404 → ``DelphiNotFoundError``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from pythia_delphi_adapter.client import (
    DelphiAPIError,
    DelphiAuthError,
    DelphiClient,
    DelphiNotFoundError,
)
from pythia_delphi_adapter.models import MarketStatus, OrderSide

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _market_payload(
    market_id: str = "dphi_01JABC",
    *,
    status: str = "OPEN",
    yes_price: float = 0.62,
    category: str = "CRYPTO",
) -> dict:
    return {
        "market_id": market_id,
        "question": "Will BTC close above $100k on 2026-08-31?",
        "category": category,
        "status": status,
        "yes_price": yes_price,
        "no_price": round(1.0 - yes_price, 4),
        "volume_usd": 12345.67,
        "liquidity_usd": 987.65,
        "created_at": "2026-07-15T10:00:00Z",
        "closes_at": "2026-08-31T23:59:59Z",
        "settlement_at": None,
        "arbiter_model": "gensyn-arbiter-v1",  # VERIFY: real field name.
    }

def _settlement_payload(market_id: str = "dphi_01JABC") -> dict:
    return {
        "market_id": market_id,
        "outcome": "YES",
        "arbiter_model": "gensyn-arbiter-v1",
        "settlement_price": 1.0,
        "resolved_at": "2026-09-01T00:00:30Z",
        "evidence_hashes": ["bafyabc123", "bafydef456"],
    }

def _make_client(handler) -> DelphiClient:
    """Build a ``DelphiClient`` backed by a MockTransport handler."""
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(
        base_url="https://api.delphi.gensyn.ai",
        transport=transport,
        headers={
            "Authorization": "Bearer test-key",
            "X-Delphi-Api-Key": "test-key",
            "User-Agent": "pythia-delphi-adapter/test",
            "Accept": "application/json",
        },
    )
    # http_client injected → client won't own/close it (we manage via `async with`).
    return DelphiClient(
        api_key="test-key",
        endpoint="https://api.delphi.gensyn.ai",
        http_client=http,
    )

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_markets_parses_response() -> None:
    """list_markets should parse the ATT response into typed Market objects."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/markets"
        assert request.url.params.get("limit") == "50"
        assert request.url.params.get("status") == "OPEN"
        assert request.url.params.get("category") == "CRYPTO"
        return httpx.Response(
            200,
            json={"markets": [_market_payload(), _market_payload("dphi_02XYZ", yes_price=0.30)]},
        )

    async with _make_client(handler) as client:
        markets = await client.list_markets(
            status=MarketStatus.OPEN, category="CRYPTO", limit=50
        )

    assert len(markets) == 2
    assert markets[0].market_id == "dphi_01JABC"
    assert markets[0].status == MarketStatus.OPEN
    assert markets[0].yes_price == pytest.approx(0.62)
    assert markets[0].arbiter_model == "gensyn-arbiter-v1"
    assert markets[0].created_at.tzinfo is not None  # parsed as UTC-aware

@pytest.mark.asyncio
async def test_list_markets_accepts_bare_list_response() -> None:
    """ATT may return a bare list instead of {'markets': [...]}."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[_market_payload()])

    async with _make_client(handler) as client:
        markets = await client.list_markets()

    assert len(markets) == 1
    assert markets[0].market_id == "dphi_01JABC"

@pytest.mark.asyncio
async def test_place_order_includes_idempotency_header() -> None:
    """place_order must send Idempotency-Key and parse the receipt."""

    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/orders"
        captured["idempotency_key"] = request.headers.get("Idempotency-Key", "")
        captured["authorization"] = request.headers.get("Authorization", "")
        body = json.loads(request.content)
        assert body["market_id"] == "dphi_01JABC"
        assert body["side"] == "YES"
        assert body["size_usd"] == 25.0
        assert body["limit_price"] == 0.62
        return httpx.Response(
            200,
            json={
                "market_id": "dphi_01JABC",
                "side": "YES",
                "size_usd": 25.0,
                "fill_price": 0.62,
                "att_order_id": "att_ord_01J",
                "status": "FILLED",
                "signed_by": "key_0xABCD",
                "timestamp": "2026-07-15T12:00:00Z",
            },
        )

    async with _make_client(handler) as client:
        receipt = await client.place_order(
            market_id="dphi_01JABC",
            side=OrderSide.YES,
            size_usd=25.0,
            limit_price=0.62,
            correlation_id="corr-123",
        )

    # Idempotency key should equal the correlation id.
    assert captured["idempotency_key"] == "corr-123"
    assert captured["authorization"] == "Bearer test-key"

    assert receipt.att_order_id == "att_ord_01J"
    assert receipt.status.value == "FILLED"
    assert receipt.fill_price == pytest.approx(0.62)
    assert receipt.signed_by == "key_0xABCD"
    assert receipt.timestamp.tzinfo is not None

@pytest.mark.asyncio
async def test_place_order_rejects_invalid_size() -> None:
    """place_order should reject non-positive sizes client-side."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("should not have been called")

    async with _make_client(handler) as client:
        with pytest.raises(ValueError, match="size_usd"):
            await client.place_order(
                market_id="dphi_01JABC",
                side=OrderSide.YES,
                size_usd=0.0,
            )

@pytest.mark.asyncio
async def test_get_settlements_filters_by_since() -> None:
    """get_settlements should pass the `since` cursor through and parse results."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/settlements"
        assert request.url.params.get("since") is not None
        return httpx.Response(
            200,
            json={"settlements": [_settlement_payload()]},
        )

    since = datetime(2026, 8, 1, tzinfo=timezone.utc)
    async with _make_client(handler) as client:
        settlements = await client.get_settlements(since=since)

    assert len(settlements) == 1
    s = settlements[0]
    assert s.market_id == "dphi_01JABC"
    assert s.outcome.value == "YES"
    assert s.arbiter_model == "gensyn-arbiter-v1"
    assert s.settlement_price == 1.0
    assert s.evidence_hashes == ["bafyabc123", "bafydef456"]
    assert s.resolved_at.tzinfo is not None

@pytest.mark.asyncio
async def test_cancel_order_returns_true_on_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/orders/att_ord_01J"
        return httpx.Response(200, json={"cancelled": True})

    async with _make_client(handler) as client:
        result = await client.cancel_order("att_ord_01J")

    assert result is True

@pytest.mark.asyncio
async def test_auth_error_on_401() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid api key"})

    async with _make_client(handler) as client:
        with pytest.raises(DelphiAuthError):
            await client.get_market("dphi_01JABC")

@pytest.mark.asyncio
async def test_not_found_error_on_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "market not found"})

    async with _make_client(handler) as client:
        with pytest.raises(DelphiNotFoundError):
            await client.get_market("dphi_unknown")

@pytest.mark.asyncio
async def test_generic_api_error_on_500() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal server error")

    async with _make_client(handler) as client:
        # 500 is retried 3x then re-raised as DelphiAPIError.
        with pytest.raises(DelphiAPIError):
            await client.get_market("dphi_01JABC")

@pytest.mark.asyncio
async def test_get_orderbook_parses_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/markets/dphi_01JABC/orderbook"
        return httpx.Response(
            200,
            json={
                "market_id": "dphi_01JABC",
                "bids": [{"price": 0.60, "size_usd": 100.0}],
                "asks": [{"price": 0.64, "size_usd": 50.0}],
            },
        )

    async with _make_client(handler) as client:
        book = await client.get_orderbook("dphi_01JABC")

    assert book.market_id == "dphi_01JABC"
    assert len(book.bids) == 1
    assert book.bids[0].price == pytest.approx(0.60)
    assert len(book.asks) == 1
