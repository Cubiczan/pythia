"""Pydantic model validation tests for ``pythia_delphi_adapter.models``.

These tests assert the model layer parses + validates correctly, independent
of any HTTP transport. They do not touch the real ATT.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from pythia_delphi_adapter.models import (
    Market,
    MarketCategory,
    MarketOpened,
    MarketSettled,
    MarketStatus,
    OrderBook,
    OrderBookLevel,
    OrderSide,
    OrderStatus,
    Position,
    PriceUpdated,
    Settlement,
    SettlementOutcome,
    TradeReceipt,
)


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------


def _market_dict(**overrides) -> dict:
    base = {
        "market_id": "dphi_01J",
        "question": "Will X happen?",
        "category": "CRYPTO",
        "status": "OPEN",
        "yes_price": 0.62,
        "no_price": 0.38,
        "volume_usd": 1000.0,
        "liquidity_usd": 500.0,
        "created_at": "2026-07-15T10:00:00Z",
        "closes_at": "2026-08-31T23:59:59Z",
        "settlement_at": None,
        "arbiter_model": "gensyn-arbiter-v1",
    }
    base.update(overrides)
    return base


def test_market_parses_basic_payload() -> None:
    m = Market.model_validate(_market_dict())
    assert m.market_id == "dphi_01J"
    assert m.category == MarketCategory.CRYPTO
    assert m.status == MarketStatus.OPEN
    assert m.yes_price == pytest.approx(0.62)
    assert m.created_at.tzinfo is not None


def test_market_rejects_price_out_of_range() -> None:
    with pytest.raises(ValidationError):
        Market.model_validate(_market_dict(yes_price=1.5))


def test_market_coerces_naive_datetime_to_utc() -> None:
    """Naive datetimes (no Z) should be treated as UTC, not rejected."""
    m = Market.model_validate(_market_dict(created_at="2026-07-15T10:00:00"))
    assert m.created_at.tzinfo is not None
    assert m.created_at.utcoffset() == datetime.now(timezone.utc).utcoffset()


def test_market_accepts_extra_fields() -> None:
    """ATT may add fields; we tolerate them (model_config extra='allow')."""
    m = Market.model_validate({**_market_dict(), "extra_unknown_field": 123})
    assert m.market_id == "dphi_01J"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


def test_market_status_enum_roundtrip() -> None:
    assert MarketStatus("OPEN") == MarketStatus.OPEN
    assert MarketStatus.OPEN.value == "OPEN"


def test_order_side_enum_values() -> None:
    assert {s.value for s in OrderSide} == {"YES", "NO"}


def test_market_category_includes_subjective() -> None:
    """Delphi's subjective/niche category must be supported."""
    cats = {c.value for c in MarketCategory}
    assert "SUBJECTIVE" in cats
    assert "POLITICS" in cats
    assert "CRYPTO" in cats


# ---------------------------------------------------------------------------
# OrderBook
# ---------------------------------------------------------------------------


def test_orderbook_defaults_to_empty_levels() -> None:
    book = OrderBook.model_validate({"market_id": "dphi_01J"})
    assert book.bids == []
    assert book.asks == []


def test_orderbook_level_validates_price_range() -> None:
    with pytest.raises(ValidationError):
        OrderBookLevel(price=2.0, size_usd=10.0)


# ---------------------------------------------------------------------------
# TradeReceipt
# ---------------------------------------------------------------------------


def test_trade_receipt_defaults_status_to_pending() -> None:
    r = TradeReceipt.model_validate(
        {
            "market_id": "dphi_01J",
            "side": "YES",
            "size_usd": 25.0,
            "att_order_id": "att_ord_01J",
            "timestamp": "2026-07-15T12:00:00Z",
        }
    )
    assert r.status == OrderStatus.PENDING
    assert r.fill_price is None
    assert r.signed_by is None


def test_trade_receipt_parses_full_payload() -> None:
    r = TradeReceipt.model_validate(
        {
            "market_id": "dphi_01J",
            "side": "NO",
            "size_usd": 50.0,
            "fill_price": 0.40,
            "att_order_id": "att_ord_01J",
            "status": "FILLED",
            "signed_by": "key_0xABCD",
            "timestamp": "2026-07-15T12:00:00Z",
        }
    )
    assert r.side == OrderSide.NO
    assert r.status == OrderStatus.FILLED
    assert r.signed_by == "key_0xABCD"


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


def test_position_validates() -> None:
    p = Position.model_validate(
        {
            "market_id": "dphi_01J",
            "side": "YES",
            "size_usd": 25.0,
            "avg_fill_price": 0.60,
            "current_value_usd": 28.0,
            "unrealized_pnl_usd": 3.0,
        }
    )
    assert p.unrealized_pnl_usd == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------


def test_settlement_parses_evidence_hashes() -> None:
    s = Settlement.model_validate(
        {
            "market_id": "dphi_01J",
            "outcome": "YES",
            "arbiter_model": "gensyn-arbiter-v1",
            "settlement_price": 1.0,
            "resolved_at": "2026-09-01T00:00:30Z",
            "evidence_hashes": ["bafy1", "bafy2"],
        }
    )
    assert s.outcome == SettlementOutcome.YES
    assert s.evidence_hashes == ["bafy1", "bafy2"]
    assert s.resolved_at.tzinfo is not None


def test_settlement_rejects_out_of_range_price() -> None:
    with pytest.raises(ValidationError):
        Settlement.model_validate(
            {
                "market_id": "dphi_01J",
                "outcome": "YES",
                "arbiter_model": "x",
                "settlement_price": 1.5,
                "resolved_at": "2026-09-01T00:00:30Z",
            }
        )


# ---------------------------------------------------------------------------
# Market events (discriminated union)
# ---------------------------------------------------------------------------


def test_market_event_discriminated_union_dispatch() -> None:
    """Each event variant should parse via the `type` discriminator."""
    from pydantic import TypeAdapter

    from pythia_delphi_adapter.models import MarketEvent

    adapter = TypeAdapter(MarketEvent)

    opened = adapter.validate_python(
        {
            "type": "market_opened",
            "market_id": "dphi_01J",
            "timestamp": "2026-07-15T10:00:00Z",
            "question": "Will X?",
            "category": "POLITICS",
        }
    )
    assert isinstance(opened, MarketOpened)
    assert opened.category == MarketCategory.POLITICS

    updated = adapter.validate_python(
        {
            "type": "price_updated",
            "market_id": "dphi_01J",
            "timestamp": "2026-07-15T10:01:00Z",
            "yes_price": 0.65,
            "no_price": 0.35,
        }
    )
    assert isinstance(updated, PriceUpdated)
    assert updated.yes_price == pytest.approx(0.65)

    settled = adapter.validate_python(
        {
            "type": "market_settled",
            "market_id": "dphi_01J",
            "timestamp": "2026-09-01T00:00:30Z",
            "outcome": "YES",
            "arbiter_model": "gensyn-arbiter-v1",
            "settlement_price": 1.0,
        }
    )
    assert isinstance(settled, MarketSettled)
    assert settled.outcome == SettlementOutcome.YES


def test_market_event_unknown_type_raises() -> None:
    from pydantic import TypeAdapter

    from pythia_delphi_adapter.models import MarketEvent

    adapter = TypeAdapter(MarketEvent)
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "type": "unknown_event",
                "market_id": "dphi_01J",
                "timestamp": "2026-07-15T10:00:00Z",
            }
        )
